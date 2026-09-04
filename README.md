# DVXR clinical-risk extract

Research-stage pipeline for **when learned multimodal fusion helps and when it harms** on small wearable / BCI / CGM cohorts.

This is a public-facing, single-entry extract. It is **not** a fusion product, **not** a diagnosis, and **not** a claim that LLMs beat specialists.

## Reproduce (offline — no raw data)

These commands support every quantitative paper claim from committed scoreboards:

```bash
python3 -m pip install -r requirements.txt
python3 main.py --profile paper-validate   # claim → board PASS/FAIL
python3 main.py --profile layer-tour        # CACMF layer shapes + hyperparams
python3 -m pytest -q tests/
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full claim → artifact map.

## Retrain (optional — needs lab + public cohorts)

```bash
cp config.example.json config.json
export DVXR_LAB_ROOT=/path/to/pipelinedvxr   # preferred over hard-coded paths
python3 main.py --profile smoke              # bounded WESAD walk
python3 main.py --profile mh                 # 5×5 mental-health board
python3 main.py --profile glucose            # CGMacros 30-min forecast
```

Raw PhysioNet / DEAP / CGMacros recordings are **not** shipped. Fetch notes: [`docs/DATA.md`](docs/DATA.md).

## What the research shows

Learned cross-modal fusion (CACMF) loses to the best single-modality specialist on every mental-health / BCI task in the paper protocol (Holm *p* = 1.0). The one reproducible integration win is a shallow add: meal timing on top of CGM (RMSE 13.33 → 12.99), not a learned fusion architecture. `ridge_raw_sequence` (MAE ≈ 10.82) beats fused MAE on CGMacros. Frozen foundation-model probes stay off by default. See [`docs/RESEARCH_SYNTHESIS.md`](docs/RESEARCH_SYNTHESIS.md).

Specialists remain primary. Fusion strategies (early, intermediate, late, attention, cross-modal, DNH) are **comparators**.

## Profiles

| Flag | What it runs |
|---|---|
| `--profile paper-validate` | Offline validation of committed boards (public CI) |
| `--profile layer-tour` | Print Layer 0–5 shapes / `CACMFConfig` defaults |
| `--profile smoke` | WESAD stress, 1×2 folds, specialist vs fusion comparator |
| `--profile mh` | WESAD, DEAP, EEGMAT, Mumtaz — paper 5×5 protocol |
| `--profile glucose` | CGMacros 30-min forecast |
| `--profile pow` | Lab POW harness + blocked complication/progression |

`--dry-run` loads config and task builders without training.

## Endpoints

| Endpoint | Status |
|---|---|
| Stress (WESAD) | real labels; specialist path (AUROC ≈ 0.955 on committed board) |
| Anxiety (DEAP) | at chance; excluded from product claims |
| Depression (Mumtaz) | real labels, **identity-leakage-confounded** |
| Cognitive overload (EEGMAT) | real rest-vs-task labels |
| Glucose forecast (CGMacros) | real prospective CGM; `ridge_raw_sequence` is the current floor |
| Complication / progression | **blocked** — no dated labels |

EMOTIV and Galea loaders exist for **validation/testing only**. They refuse pairing with CGMacros glucose (not the same people).

## Layout

```
main.py                 # staged entry: config → load/validate → train → scoreboard → infer
config.example.json     # portable template (use DVXR_LAB_ROOT)
scripts/validate_paper_claims.py
outputs/scoreboards/    # committed paper boards (cite these)
docs/EXPERIMENTS.md     # claim → artifact map
tests/                  # honesty + blocked endpoints
.github/workflows/      # offline CI
```

## Status

**RESEARCH-STAGE — NOT A DIAGNOSIS.** GPU training for CACMF is CPU-locked. Optional FM probes stay off (`fm_probes: false`).
