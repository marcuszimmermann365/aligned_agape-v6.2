# Makefile for Aligned Agape Blueprint V6.2
.PHONY: test ui-test bench plot run clean

PY?=python
NPX?=npx

run:
	$(PY) main.py --scenario data/scenarios/dayton_3party.yaml --j5 neural

test:
	pytest -q

ui-test:
	$(NPX) playwright install
	$(NPX) playwright test tests/ui/j5Badge.spec.ts

bench:
	$(PY) scripts/ablation_bench.py --scenario data/scenarios/dayton_3party.yaml --j5 neural --modes none,H1,H2,H3 --seeds 10 --turns 20 --out_csv out/ablation_results.csv --out_png out/ablation_results.png --per_turn_csv out/ablation_per_turn.csv --per_turn_png out/ablation_per_turn.png

plot:
	$(PY) scripts/plot_ablation_csv.py --csv out/ablation_results.csv --png out/ablation_results.png

clean:
	rm -rf out/*.png out/*.csv
