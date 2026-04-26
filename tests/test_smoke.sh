#!/usr/bin/env bash
# tests/test_smoke.sh — end-to-end smoke for resonance.aml.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== test_smoke =="

if ! command -v amlc >/dev/null 2>&1; then
    echo "  SKIP: amlc not installed"
    exit 0
fi

amlc resonance.aml -o resonance_smoke >/tmp/resonance_smoke_compile.log 2>&1
if [ ! -x resonance_smoke ]; then
    echo "  FAIL: amlc did not produce resonance_smoke binary"
    cat /tmp/resonance_smoke_compile.log
    exit 1
fi
echo "  PASS [compile]: resonance_smoke ($(wc -c < resonance_smoke) bytes)"

WEIGHTS="weights/resonance_200m/resonance_200m_lora_yent.bin"
if [ ! -f "$WEIGHTS" ]; then
    echo "  SKIP [generate]: $WEIGHTS not found"
    rm -f resonance_smoke resonance_smoke.c
    echo
    echo "smoke OK (compile only)"
    exit 0
fi

OUT=$(./resonance_smoke -p "Q: ping
A:" -n 20 -t 0.7 --top-p 0.9 2>&1)
RC=$?
if [ $RC -ne 0 ]; then
    echo "  FAIL: resonance exited $RC"
    echo "$OUT"
    exit 1
fi

GEN=$(printf '%s\n' "$OUT" | awk '/--- generation ---/{flag=1;next} flag')
GEN_TRIM=$(printf '%s' "$GEN" | tr -d '[:space:]')
if [ -z "$GEN_TRIM" ]; then
    echo "  FAIL: no generation produced"
    echo "$OUT"
    exit 1
fi
echo "  PASS [generate]: $(printf '%s' "$GEN" | wc -c) bytes"

rm -f resonance_smoke resonance_smoke.c
echo
echo "smoke OK"
