# POW Goal 3 — specialist ablation (generated)

Classification error is **1−AUROC** (lower better). Numbers are cited from
`BENCHMARK_FINDINGS.md` (dnh_gated vs best single vs CACMF `rep:fused`).
**Held-out do-no-harm** is true only when DNH error is not worse than the best single.
This table does **not** include DAF unmatched EEG×CGMacros fused glycemic F1.

| task | metric | dnh_gated | best single | cacmf_fused | held-out do-no-harm |
|---|---|---|---|---|---|
| stress | 1-AUROC | 0.1154 | single:motion 0.167 | 0.1294 | yes |
| wesad_stress | 1-AUROC | 0.0929 | single:resp 0.1243 | 0.1294 | yes |
| deap_anxiety | 1-AUROC | 0.4825 | single:physiology 0.4658 | 0.4688 | no |
| deap_arousal | 1-AUROC | 0.4726 | single:physiology 0.4522 | 0.4575 | no |
| eegmat_workload | 1-AUROC | 0.3003 | single:physiology 0.2598 | 0.3649 | no |
| mumtaz_depression | 1-AUROC | 0.0964 | single:eeg 0.1121 | 0.2046 | yes |

## Glucose forecast (separate block — not mixed into 1−AUROC)

Source: `neuroglycemic-runtime/runs/cgmacros-devices-v1/test_metrics.json`. CGMacros devices, CGM history in. Model vs persistence.

| horizon_min | model_rmse | persistence_rmse | model_mae | persistence_mae |
|---|---|---|---|---|
| 30 | 12.768140518686218 | 17.396680484909407 | 8.429810590965564 | 11.308860064196883 |
| 60 | 21.918432025202378 | 26.787755097043753 | 14.49642479706843 | 17.52674784989415 |
| 90 | 26.609039246889967 | 32.64406834756311 | 18.158165173520235 | 21.801482299920536 |
| 120 | 29.061851544649578 | 36.45189775158726 | 20.44166615980211 | 24.853970888029522 |

## Footnote — wearable-only glucose without CGM-in (closed negative)

Source: `neuroglycemic-runtime/runs/bigideas-v61-seed-42/test_metrics.json`. MAE@30 model=23.608911061155204 vs persistence=11.922651933701658. ambient_no_cgm wearable-only; closed negative (not a POW headline)
