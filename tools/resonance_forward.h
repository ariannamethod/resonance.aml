/*
 * test_resonance.c — quick standalone Resonance 200M inference.
 * Used to compare SFT vs LoRA-merged checkpoints before adapting AML.
 *
 * Build: cc -O3 -Wall -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK \
 *           test_resonance.c -L/opt/homebrew/lib -lnotorch \
 *           -framework Accelerate -lm -o test_resonance
 *
 * Run: ./test_resonance <weights.bin> [prompt] [max_tokens] [temp]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <ariannamethod/notorch.h>
#include <ariannamethod/ariannamethod.h>

/* Config from RS02 header */
static int V, E, H, D, B, M, T, R;

/* Helpers — BLAS via notorch.
 *   mm_t(C, A, B, m, k, n) — cblas_sgemm (used by future prefill_batch).
 *   matvec_t(out, W, x, n, k) — cblas_sgemv, the per-token hot-loop path. */
static void mm_t(float *C, const float *A, const float *BT, int m, int k, int n) {
    nt_blas_mmT(C, A, BT, m, k, n);
}
static void matvec_t(float *out, const float *W, const float *x, int n, int k) {
    nt_blas_matvec(out, W, x, n, k);
}

/* Parametric RMSNorm: out = x * rsqrt(mean(x²) + eps) * weight */
static void rmsnorm_p(float *o, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}

static void softmax_f(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20 ? x / (1 + expf(-x)) : 0; }
static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* RoPE — even/odd interleave (model.py's _apply_rope):
 *   x1 = x[..., ::2], x2 = x[..., 1::2]
 *   out = stack([x1*cos - x2*sin, x1*sin + x2*cos]).flatten(-2)
 * I.e. pairs (i, i+1) rotate together. */
static void rope_even_odd(float *q, float *k, int pos, int dim) {
    int n_pairs = dim / 2;
    for (int i = 0; i < n_pairs; i++) {
        float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)dim);
        float val = pos * freq;
        float cs = cosf(val), sn = sinf(val);
        float qe = q[2*i], qo = q[2*i + 1];
        float ke = k[2*i], ko = k[2*i + 1];
        q[2*i]     = qe * cs - qo * sn;
        q[2*i + 1] = qe * sn + qo * cs;
        k[2*i]     = ke * cs - ko * sn;
        k[2*i + 1] = ke * sn + ko * cs;
    }
}

typedef struct {
    float *tok_emb;
    struct {
        float *norm1, *wq, *wk, *wv;
        float *wr_a, *wr_b, *gate;
        float *wo, *norm2;
        float *mlp_gate, *mlp_up, *mlp_down;
    } b[32];
    float *norm_f;
    float *out_head;
} Weights;

/* state_dict order matches PyTorch named_parameters() traversal:
 * for each Module, _parameters are yielded BEFORE recursing into sub-
 * Modules. So per ResonanceBlock the order is:
 *   1) direct Parameters of block: wr_a, wr_b, gate
 *   2) sub-Module weights in registration order:
 *      norm1.weight, wq.weight, wk.weight, wv.weight, wo.weight,
 *      norm2.weight, mlp_gate.weight, mlp_up.weight, mlp_down.weight
 *
 * The earlier (wrong) order put norm1/wq/wk/wv before wr_a/wr_b/gate
 * and shifted every per-block tensor by 1.62M floats — forward ran on
 * random data and the output came out as web-text garbage. */
static void assign(Weights *w, float *p) {
    w->tok_emb = p; p += V * E;
    for (int i = 0; i < B; i++) {
        /* direct Parameters first */
        w->b[i].wr_a     = p; p += H * E * R;
        w->b[i].wr_b     = p; p += H * R * T;
        w->b[i].gate     = p; p += H;
        /* sub-Module weights in init order */
        w->b[i].norm1    = p; p += E;
        w->b[i].wq       = p; p += E * E;
        w->b[i].wk       = p; p += E * E;
        w->b[i].wv       = p; p += E * E;
        w->b[i].wo       = p; p += E * E;
        w->b[i].norm2    = p; p += E;
        w->b[i].mlp_gate = p; p += M * E;
        w->b[i].mlp_up   = p; p += M * E;
        w->b[i].mlp_down = p; p += E * M;
    }
    w->norm_f   = p; p += E;
    w->out_head = p; p += V * E;
}

/* KV cache (per layer × T × H*D) */
static float *kv_k, *kv_v;
static int kv_len;

static void kv_init(int max_seq) {
    kv_k = calloc((size_t)B * max_seq * E, sizeof(float));
    kv_v = calloc((size_t)B * max_seq * E, sizeof(float));
    kv_len = 0;
}

/* Forward one token at position pos, using KV cache. */
static void forward_token(Weights *w, int tok, int pos,
                          float *logits, float *hidden) {
    float x[1024];      /* E ≤ 1024 */
    float xn[1024];
    float sc = 1.0f / sqrtf((float)D);

    /* Token embed */
    for (int e = 0; e < E; e++) x[e] = w->tok_emb[tok * E + e];

    for (int bl = 0; bl < B; bl++) {
        /* === norm1 → Q/K/V === */
        rmsnorm_p(xn, x, w->b[bl].norm1, E);

        float qa[1024], ka[1024], va[1024];
        matvec_t(qa, w->b[bl].wq, xn, E, E);
        matvec_t(ka, w->b[bl].wk, xn, E, E);
        matvec_t(va, w->b[bl].wv, xn, E, E);

        /* RoPE per head (even/odd interleave on each head's D dims) */
        for (int h = 0; h < H; h++)
            rope_even_odd(qa + h * D, ka + h * D, pos, D);

        /* Cache K, V at position pos */
        size_t off = ((size_t)bl * T + pos) * E;
        memcpy(kv_k + off, ka, E * sizeof(float));
        memcpy(kv_v + off, va, E * sizeof(float));

        /* === Content attention: per-head Q · K^T → softmax → · V === */
        float c_out[1024];
        memset(c_out, 0, E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *q_h = qa + h * D;
            float attn[2048];
            for (int j = 0; j <= pos; j++) {
                float *kj = kv_k + ((size_t)bl * T + j) * E + h * D;
                float s = 0;
                for (int d = 0; d < D; d++) s += q_h[d] * kj[d];
                attn[j] = s * sc;
            }
            softmax_f(attn, pos + 1);
            float out[128] = {0};
            for (int j = 0; j <= pos; j++) {
                float *vj = kv_v + ((size_t)bl * T + j) * E + h * D;
                for (int d = 0; d < D; d++) out[d] += attn[j] * vj[d];
            }
            memcpy(c_out + h * D, out, D * sizeof(float));
        }

        /* === RRPRAM low-rank attention ===
         * From model.py:
         *   xn_h = xn.unsqueeze(1).expand(B,H,T,E)        # broadcast xn over heads
         *   temp = einsum('bhie,her->bhir', xn_h, wr_a)   # [B,H,T,R]
         *   r_attn = einsum('bhir,hrj->bhij', temp, wr_b[:,:,:T])  # [B,H,T,T]
         *   r_attn *= D^-0.5
         *   r_attn[mask] = -inf, softmax, @ V (shared)
         *
         * Single token, autoregressive: temp[h,r] = sum_e xn[e] * wr_a[h,e,r]
         *   We accumulate temp[h,r] across positions in a cache, then
         *   r_attn[j] = sum_r temp_j[h,r] * wr_b[h,r,j], softmax → @ V_j.
         * For autoregressive inference we re-compute temp from xn at each
         * position (it's per-position, not accumulated like Janus). */
        float r_out[1024];
        memset(r_out, 0, E * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *wr_a_h = w->b[bl].wr_a + h * E * R;   /* [E, R] */
            float *wr_b_h = w->b[bl].wr_b + h * R * T;   /* [R, T] */

            /* temp[r] = sum_e xn[e] * wr_a[h,e,r] */
            float temp[128];
            for (int r = 0; r < R; r++) {
                float s = 0;
                for (int e = 0; e < E; e++) s += xn[e] * wr_a_h[e * R + r];
                temp[r] = s;
            }
            /* r_score[j] = sum_r temp[r] * wr_b[h,r,j], for j ≤ pos */
            float r_attn[2048];
            for (int j = 0; j <= pos; j++) {
                float s = 0;
                for (int r = 0; r < R; r++) s += temp[r] * wr_b_h[r * T + j];
                r_attn[j] = s * sc;
            }
            softmax_f(r_attn, pos + 1);
            float out[128] = {0};
            for (int j = 0; j <= pos; j++) {
                float *vj = kv_v + ((size_t)bl * T + j) * E + h * D;
                for (int d = 0; d < D; d++) out[d] += r_attn[j] * vj[d];
            }
            memcpy(r_out + h * D, out, D * sizeof(float));
        }

        /* === Per-head sigmoid gate: g·content + (1-g)·rrpram === */
        float blend[1024];
        for (int h = 0; h < H; h++) {
            float g = sigmoidf(w->b[bl].gate[h]);
            for (int d = 0; d < D; d++)
                blend[h * D + d] = g * c_out[h * D + d] + (1.0f - g) * r_out[h * D + d];
        }

        /* === WO + residual === */
        float ao[1024];
        matvec_t(ao, w->b[bl].wo, blend, E, E);
        for (int e = 0; e < E; e++) x[e] += ao[e];

        /* === norm2 → SwiGLU → residual === */
        rmsnorm_p(xn, x, w->b[bl].norm2, E);
        float mg[2048], mu[2048], mo[1024];
        matvec_t(mg, w->b[bl].mlp_gate, xn, M, E);
        matvec_t(mu, w->b[bl].mlp_up, xn, M, E);
        for (int i = 0; i < M; i++) mg[i] = siluf(mg[i]) * mu[i];
        matvec_t(mo, w->b[bl].mlp_down, mg, E, M);
        for (int e = 0; e < E; e++) x[e] += mo[e];
    }

    /* Final norm + head */
    rmsnorm_p(xn, x, w->norm_f, E);
    if (hidden) memcpy(hidden, xn, E * sizeof(float));
    matvec_t(logits, w->out_head, xn, V, E);
}

/* ── Public API used from resonance.aml ─────────────────────────────── */

typedef struct {
    Weights w;
    nt_bpe  bpe;
    float  *data;          /* owned buffer for fp32 weights */
    int   (*merges)[2];    /* owned (a, b) pair table */
} ResonanceCtx;

static int resonance_load(ResonanceCtx *ctx, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[resonance] cannot open '%s'\n", path); return 1; }

    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != 0x52533032) {
        fprintf(stderr, "[resonance] bad magic 0x%08x (expected 'RS02')\n", magic);
        fclose(f); return 1;
    }
    int hdr[9];
    fread(hdr, 4, 9, f);
    E = hdr[0]; B = hdr[1]; T = hdr[2]; H = hdr[3]; D = hdr[4];
    R = hdr[5]; M = hdr[6]; V = hdr[7];
    fprintf(stderr, "[resonance] V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
            V, E, H, D, B, M, T, R);

    uint32_t n_merges;
    fread(&n_merges, 4, 1, f);
    ctx->merges = malloc((size_t)n_merges * 2 * sizeof(int));
    for (uint32_t mi = 0; mi < n_merges; mi++) {
        int triple[3];
        fread(triple, 4, 3, f);
        ctx->merges[mi][0] = triple[0];
        ctx->merges[mi][1] = triple[1];
    }
    nt_bpe_init(&ctx->bpe, ctx->merges, (int)n_merges);
    fprintf(stderr, "[resonance] BPE vocab=%d merges=%d\n",
            ctx->bpe.vocab_size, ctx->bpe.n_merges);

    size_t np = 2 * (size_t)V * E + 1L * E;
    np += (size_t)B * (
        E + 3L * E * E + (size_t)H * E * R + (size_t)H * R * T +
        H + (size_t)E * E + E + 3L * M * E
    );
    ctx->data = malloc(np * sizeof(float));
    if (!ctx->data) { fclose(f); return 1; }
    size_t got = fread(ctx->data, sizeof(float), np, f);
    if (got != np) {
        fprintf(stderr, "[resonance] short read: %zu/%zu\n", got, np);
        fclose(f); return 1;
    }
    fclose(f);
    assign(&ctx->w, ctx->data);
    kv_init(T);
    fprintf(stderr, "[resonance] %.1fM params loaded, KV cache %d MB\n",
            np / 1e6, (int)((size_t)B * T * E * 2 * 4 / 1024 / 1024));
    return 0;
}

static void resonance_free(ResonanceCtx *ctx) {
    free(ctx->data);
    free(ctx->merges);
    free(kv_k);
    free(kv_v);
    ctx->data = NULL;
    ctx->merges = NULL;
}

/* Top-p nucleus sampler with rep_penalty + no-repeat-3-gram. */
static int resonance_sample_token(float *logits, int *cctx, int len,
                                  float temp, float top_p) {
    float rep_penalty = 1.4f;
    int window = 64;
    int start = len > window ? len - window : 0;
    for (int j = start; j < len; j++) {
        int t = cctx[j];
        if (t >= 0 && t < V)
            logits[t] = logits[t] > 0 ? logits[t] / rep_penalty
                                      : logits[t] * rep_penalty;
    }
    if (len >= 2) {
        int a = cctx[len - 2], b = cctx[len - 1];
        for (int j = 0; j + 2 < len; j++) {
            if (cctx[j] == a && cctx[j + 1] == b) {
                int forbid = cctx[j + 2];
                if (forbid >= 0 && forbid < V) logits[forbid] = -1e30f;
            }
        }
    }

    /* AML Dario field overlay (matches yent.aml's wiring). */
    am_apply_field_to_logits(logits, V);

    if (temp <= 0) temp = 1.0f;
    for (int i = 0; i < V; i++) logits[i] /= temp;
    softmax_f(logits, V);

    /* Top-p over partial-sorted top 256. */
    typedef struct { float p; int idx; } PI;
    static PI topk[256];
    int filled = 0;
    float min_in = -1;
    for (int i = 0; i < V; i++) {
        float p = logits[i];
        if (filled < 256) {
            topk[filled].p = p; topk[filled].idx = i; filled++;
            if (filled == 256) {
                for (int a = 1; a < 256; a++) {
                    PI tmp = topk[a]; int j = a;
                    while (j > 0 && topk[j-1].p < tmp.p) { topk[j] = topk[j-1]; j--; }
                    topk[j] = tmp;
                }
                min_in = topk[255].p;
            }
            continue;
        }
        if (p > min_in) {
            topk[255].p = p; topk[255].idx = i;
            int j = 255;
            while (j > 0 && topk[j-1].p < topk[j].p) {
                PI t = topk[j]; topk[j] = topk[j-1]; topk[j-1] = t; j--;
            }
            min_in = topk[255].p;
        }
    }
    if (filled < 256) {
        for (int a = 1; a < filled; a++) {
            PI tmp = topk[a]; int j = a;
            while (j > 0 && topk[j-1].p < tmp.p) { topk[j] = topk[j-1]; j--; }
            topk[j] = tmp;
        }
    }
    float cum = 0;
    int nuc = filled;
    for (int k = 0; k < filled; k++) {
        cum += topk[k].p;
        if (cum >= top_p) { nuc = k + 1; break; }
    }
    if (nuc < 1) nuc = 1;
    float total = 0;
    for (int k = 0; k < nuc; k++) total += topk[k].p;
    float r = (float)rand() / (float)RAND_MAX * total;
    float c = 0;
    for (int k = 0; k < nuc; k++) {
        c += topk[k].p;
        if (c >= r) return topk[k].idx;
    }
    return topk[nuc - 1].idx;
}

static void resonance_generate(ResonanceCtx *ctx, const char *prompt,
                               int max_gen, float temp, float top_p) {
    int cctx[4096];
    int len = nt_bpe_encode(&ctx->bpe, prompt, (int)strlen(prompt), cctx, 4096);
    fprintf(stderr, "[resonance] prompt: \"%s\" → %d tokens\n", prompt, len);

    float *logits = calloc(V, sizeof(float));
    fprintf(stderr, "[resonance] prefill %d tokens... ", len);
    fflush(stderr);
    for (int i = 0; i < len; i++)
        forward_token(&ctx->w, cctx[i], i, logits, NULL);
    fprintf(stderr, "done\n--- generation ---\n");

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    int gen_start = len;
    for (int step = 0; step < max_gen && len < T; step++) {
        int next = resonance_sample_token(logits, cctx, len, temp, top_p);
        char dec[64];
        int n = nt_bpe_decode(&ctx->bpe, &next, 1, dec, sizeof(dec) - 1);
        dec[n] = 0;
        fputs(dec, stdout);
        fflush(stdout);

        am_compute_prophecy_debt(logits, next, V);
        am_step(0.05f);

        cctx[len] = next;
        forward_token(&ctx->w, next, len, logits, NULL);
        len++;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double el = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
    int gen = len - gen_start;
    fprintf(stderr, "\n[resonance] %d tokens, %.1f tok/s (%.2fs)\n",
            gen, gen / el, el);
    free(logits);
}
