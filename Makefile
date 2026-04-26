# resonance.aml — Makefile
# Compiles resonance.aml through amlc and runs the local test suite.

.PHONY: all resonance test test-smoke clean

all: resonance

resonance: resonance.aml tools/resonance_forward.h
	amlc resonance.aml -o resonance

test: test-smoke

test-smoke:
	bash tests/test_smoke.sh

clean:
	rm -f resonance resonance.c resonance_smoke resonance_smoke.c

help:
	@echo "resonance.aml — third AML inference (Resonance 200M Yent SFT)"
	@echo
	@echo "  make            Compile resonance.aml → ./resonance"
	@echo "  make test       Run smoke test"
	@echo "  make clean      Remove build artefacts"
	@echo
	@echo "  Run after build:"
	@echo "    ./resonance -p \"Q: Who are you?\\nA:\" -n 100 -t 0.7 --top-p 0.9"
