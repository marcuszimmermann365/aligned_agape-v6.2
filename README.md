[README.md](https://github.com/user-attachments/files/22706382/README.md)

# Geopolitischer Trainings‑Simulator — **Aligned Agape V6.2**

Vollständige, lauffähige Referenz-Implementierung mit:
- Theorie-treuem **J4 (Empowerment)** via **InfoNCE** (PyTorch)
- Optionaler **Hamiltonian SDE** (symplektischer Integrator) für Hintergrund‑Dynamik (`--use-sde`)
- Einfache RAG/Rulebook/Socratic/Teleology‑Module
- Web-UI (statisches Dashboard) und PDF-Export
- Smoke-Tests

## Installation
```bash
python -m pip install -r requirements.txt
```

## Beispiel‑Run
```bash
python3 main.py --scenario data/scenarios/dayton_3party.yaml --turns 5 --ripe-proposals --use-sde
```

## Web‑UI
```bash
python3 serve.py   # http://localhost:8008
```

## PDF‑Report
```bash
python3 scripts/export_pdf.py
```

## Tests
```bash
python3 tests/sim_tests.py --turns 3
```


## J5 (Ahimsa) – Causal Modes in V6.2

CLI flag: `--j5 {neural|aipw|ridge|proxy}` (default: `neural`).

Fallback order (for `--j5=neural`): neural → aipw → ridge → proxy.
Each fallback is logged during runtime.

- **neural**: Neural-SCM (continuous, nonlinear)
- **aipw**: Doubly-Robust AIPW (discrete actions)
- **ridge**: Orthogonalized ridge (lightweight)
- **proxy**: legacy perturbation measure (non-causal)


## Ablations (V6.2)

Use `--ablation` to replicate analysis variants:

- `H1` – **Fixed Teleology**: `ripe=False`, constant `readiness=0.5` (no teleology gating).
- `H2` – **No Ahimsa**: sets `J5 = 0.0` (disables harm-avoidance signal).
- `H3` – **Static Reality**: disables SDE update in turns; zeroes actor action vectors for scoring.

Example:
```
python main.py --scenario data/scenarios/dayton_3party.yaml --j5 neural --ablation H2
```


## Ablation Benchmarking

Automatisierter Lauf über Seeds/Modi und Export als CSV/PNG:

```
python scripts/ablation_bench.py --scenario data/scenarios/dayton_3party.yaml --j5 neural --modes none,H1,H2,H3 --seeds 10 --turns 20 --out_csv out/ablation_results.csv --out_png out/ablation_results.png
```

Nur Plot aus CSV neu erzeugen:
```
python scripts/plot_ablation_csv.py --csv out/ablation_results.csv --png out/ablation_results.png
```
