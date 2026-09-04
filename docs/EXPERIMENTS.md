# Experiments mapped to paper claims

Every quantitative statement in the enhanced Summer 2026 manuscript
(`pipelinedvxr/paper/summer2026_enhanced/`) is tied to a committed artifact
under `outputs/scoreboards/` and a command that either **validates** it offline
or **regenerates** it when the private lab + public cohorts are available.

## Offline gate (no raw data, no lab)

```bash
python3 main.py --profile paper-validate
python3 main.py --profile layer-tour
```

Exit code 0 from `paper-validate` means every claimed number below is present
and within disclosed tolerance in the committed boards.

## Artifact validation vs. reproduction

These are two different questions, kept structurally separate:

```bash
python3 scripts/validate_paper_claims.py    # ARTIFACT VALIDATION — checks committed boards, no lab, seconds
DVXR_LAB_ROOT=… python3 scripts/reproduce_from_raw.py   # REPRODUCTION — retrains from raw cohorts through the lab
```

`reproduce_from_raw.py` chains the three new generator scripts below plus
`main.py --profile mh`/`--profile glucose`, writing NEW artifacts under
`outputs/scoreboards/regenerated/` — it never overwrites the committed evidence, so
committed-vs-regenerated stays diffable. `validate_paper_claims.py` automatically picks
up `outputs/scoreboards/regenerated/glycemic_warning.csv` when present and compares real
recomputed AUROC instead of only checking that the committed summary contains the right
substring.

## Claim → artifact → regenerate

| Paper claim | Committed artifact | Offline check | Regenerate (needs lab + data) |
|---|---|---|---|
| WESAD specialist AUROC ≈ 0.955; fusion RER negative | `outputs/scoreboards/paper_mh_5x5/benchmark_scoreboard.csv` | `paper-validate` | `DVXR_LAB_ROOT=… python3 main.py --profile mh` (5×5) or lab `scripts/run_benchmark.py --profile mh` |
| Fusion loses on 6 MH tasks (Holm p=1) | same | `paper-validate` | same |
| EEGMAT floor ≈ 0.740 | same (+ comparative CSV) | `paper-validate` | same |
| `ridge_raw_sequence` MAE 10.817 beats fused 11.678 | `outputs/scoreboards/paper_cgmacros_5x5/` | `paper-validate` | `python3 main.py --profile glucose` |
| Meal context 13.33 → 12.99 RMSE; drop CGM → ~34 | `outputs/scoreboards/glucose_ablation/leave_one_modality_out_cgmacros.csv` | `paper-validate` | `DVXR_LAB_ROOT=… python3 scripts/build_meal_ablation_table.py` (writes `outputs/scoreboards/regenerated/meal_ablation.csv`) |
| Five fusion strategies ranking (AUROC primary; F1@0.5 caveat) | `outputs/scoreboards/fusion_strategies/` | `paper-validate` | lab `scripts/run_fusion_strategy_table.py` |
| GBM 30-min RMSE 12.48; deep lowest at 60/90/120 | `glucose_ablation/glucose_model_ladder.csv`, `deep_tabular_result.csv` | `paper-validate` | `DVXR_LAB_ROOT=… python3 scripts/build_glucose_model_ladder.py` reproduces the classical-model half (persistence/ridge/decision_tree/random_forest/gradient_boosting/mlp @ 30/60/90/120 min); the `deep_v2`/temporal-net row remains lab Goal-2 (artifact-cited, not re-derived in this extract) |
| Hypo/hyper warning 0.976 / 0.981 | `abstract_summary/best_models_summary.csv` | `paper-validate` (now checks a real recomputed AUROC when present, not a substring — see `validate_paper_claims.py::check_glucose_ladder_and_warnings`) | `DVXR_LAB_ROOT=… python3 scripts/build_glycemic_warning_table.py` (writes `outputs/scoreboards/regenerated/glycemic_warning.csv`) |
| PhysioNet stress 0.892 | `glucose_ablation/comparative_performance.csv` | `paper-validate` | `mh` board includes `stress` when data present |

| Complication / progression blocked | `pipeline/outcomes.py` | `paper-validate` | N/A |
| EMOTIV / Galea val-only; no cross-cohort pairing | `data_loaders/emotiv.py`, `galea.py`, `pairing.py` | `pytest tests/` | N/A |

## What this extract does **not** claim

- Learned CACMF / LLM as product winners
- EMOTIV / Galea predictive metrics
- Mumtaz AUROC as a portable biomarker (identity-leakage upper bound)
- EEG×CGM fusion on public data
- Complication / progression scores
- Strategy-table linear probes ≡ full 5×5 `rep:fused` protocol

## Protocol constants

- Seed **7**
- Primary board: **5 repeats × 5 folds**, subject-held-out
- Strategy table: subject 70/30 split, seed 7, frozen encoder + linear probe
- Device: **CPU** for CACMF training in this extract (`fm_probes: false`)

## Status of evidence

| Kind | Meaning |
|---|---|
| Committed board | Numbers the paper may cite; validated by `paper-validate` |
| Retrain profile | Reproduces a board when lab + public cohorts exist |
| Blocked | Structural absence of labels — must not emit a score |
