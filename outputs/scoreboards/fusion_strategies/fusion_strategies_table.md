# Five fusion strategies — real same-person cohorts

**Protocol:** subject-held-out 70/30, seed=7; frozen CACMF encoder + linear probe (logistic for classification, ridge for forecast). Not the paper 5×5 `rep:fused` board.

**Strategies:** `early` | `intermediate` | `late_weighted` | `attention` | `cross_modal` (from `dvxr.config.FUSION_STRATEGIES` = ['early', 'intermediate', 'late_weighted', 'attention', 'cross_modal']).

## Metric caveat (why WESAD F1/accuracy look broken)

Classification F1 and accuracy use a **fixed 0.5 probability threshold** (`dvxr.eval.metrics.classification_metrics`). On WESAD the positive (stress) rate is ~22%, so majority-class accuracy is **0.781**. Every fusion strategy on this split emits all-negative hard labels at 0.5 → F1=0.0 and accuracy exactly 0.781, while AUROC/AUPRC still rank discrimination correctly. **Use AUROC (and AUPRC) as the primary ranking metrics** for this table; treat F1/accuracy as secondary and threshold-sensitive.

Skipped (not multi-stream fusion): Mumtaz (EEG-only), Shanghai CGM (CGM-only), MIMIC mortality (EHR-only). EMOTIV/Galea remain val/test-only.

## wesad_stress (`wesad`)

_WESAD chest+wrist wearable streams, same subject/session_

Best single-modality probe for context: **motion** AUROC=0.9511.

| fusion_strategy   |   auroc |   auprc |     f1 |   accuracy |    ece |   n_train |   n_test |
|:------------------|--------:|--------:|-------:|-----------:|-------:|----------:|---------:|
| late_weighted     |  0.9244 |  0.7851 | 0.0000 |     0.7812 | 0.1358 |       296 |       96 |
| intermediate      |  0.8619 |  0.6828 | 0.0000 |     0.7812 | 0.0590 |       296 |       96 |
| early             |  0.8483 |  0.6859 | 0.0000 |     0.7812 | 0.0205 |       296 |       96 |
| attention         |  0.7873 |  0.5315 | 0.0000 |     0.7812 | 0.0874 |       296 |       96 |
| cross_modal       |  0.5003 |  0.4816 | 0.0000 |     0.7812 | 0.0009 |       296 |       96 |

Best among the five: **late_weighted** (AUROC=0.9244).
Does **not** beat the best single-modality probe on this split.

## eegmat_workload (`eegmat`)

_EEGMAT EEG+ECG, same subject/session_

Best single-modality probe for context: **eeg** AUROC=0.5013.

| fusion_strategy   |   auroc |   auprc |     f1 |   accuracy |    ece |   n_train |   n_test |
|:------------------|--------:|--------:|-------:|-----------:|-------:|----------:|---------:|
| early             |  0.6455 |  0.5931 | 0.6776 |     0.6488 | 0.1062 |       392 |      168 |
| attention         |  0.6105 |  0.5825 | 0.6774 |     0.6429 | 0.0593 |       392 |      168 |
| intermediate      |  0.5614 |  0.6258 | 0.5422 |     0.5476 | 0.0314 |       392 |      168 |
| late_weighted     |  0.5488 |  0.5428 | 0.6564 |     0.6012 | 0.1110 |       392 |      168 |
| cross_modal       |  0.5220 |  0.5128 | 0.6267 |     0.5179 | 0.0143 |       392 |      168 |

Best among the five: **early** (AUROC=0.6455).

## deap_anxiety (`deap_anxiety`)

_DEAP EEG+peripheral, high-arousal+low-valence (real SAM)_

Best single-modality probe for context: **physiology** AUROC=0.6232.

| fusion_strategy   |   auroc |   auprc |     f1 |   accuracy |    ece |   n_train |   n_test |
|:------------------|--------:|--------:|-------:|-----------:|-------:|----------:|---------:|
| intermediate      |  0.5385 |  0.2366 | 0.0000 |     0.7750 | 0.0659 |      1680 |      560 |
| attention         |  0.5318 |  0.2689 | 0.0000 |     0.7750 | 0.0627 |      1680 |      560 |
| early             |  0.5112 |  0.2232 | 0.0000 |     0.7750 | 0.0751 |      1680 |      560 |
| cross_modal       |  0.4820 |  0.2222 | 0.0000 |     0.7750 | 0.0758 |      1680 |      560 |
| late_weighted     |  0.4647 |  0.2216 | 0.0000 |     0.7750 | 0.0788 |      1680 |      560 |

Best among the five: **intermediate** (AUROC=0.5385).
Does **not** beat the best single-modality probe on this split.

## deap_arousal (`deap_arousal`)

_DEAP EEG+peripheral, high vs low SAM arousal_

Best single-modality probe for context: **eeg** AUROC=0.6018.

| fusion_strategy   |   auroc |   auprc |     f1 |   accuracy |    ece |   n_train |   n_test |
|:------------------|--------:|--------:|-------:|-----------:|-------:|----------:|---------:|
| intermediate      |  0.5670 |  0.2510 | 0.0000 |     0.7750 | 0.0574 |      1680 |      560 |
| attention         |  0.5506 |  0.2861 | 0.0000 |     0.7750 | 0.0554 |      1680 |      560 |
| early             |  0.5436 |  0.2540 | 0.0000 |     0.7750 | 0.0664 |      1680 |      560 |
| late_weighted     |  0.5029 |  0.2304 | 0.0000 |     0.7750 | 0.0520 |      1680 |      560 |
| cross_modal       |  0.4678 |  0.2167 | 0.0000 |     0.7750 | 0.0636 |      1680 |      560 |

Best among the five: **intermediate** (AUROC=0.5670).
Does **not** beat the best single-modality probe on this split.

## stress (`noneeg`)

_PhysioNet noneeg peripheral stress, same subject/session_

Best single-modality probe for context: **motion** AUROC=0.8241.

| fusion_strategy   |   auroc |   auprc |     f1 |   accuracy |    ece |   n_train |   n_test |
|:------------------|--------:|--------:|-------:|-----------:|-------:|----------:|---------:|
| late_weighted     |  0.8244 |  0.8410 | 0.7196 |     0.7670 | 0.1536 |      1079 |      455 |
| attention         |  0.7650 |  0.7800 | 0.5240 |     0.6725 | 0.0791 |      1079 |      455 |
| intermediate      |  0.7538 |  0.7655 | 0.5714 |     0.6703 | 0.1160 |      1079 |      455 |
| early             |  0.7148 |  0.7490 | 0.5707 |     0.6330 | 0.0650 |      1079 |      455 |
| cross_modal       |  0.5623 |  0.5663 | 0.3355 |     0.5560 | 0.0300 |      1079 |      455 |

Best among the five: **late_weighted** (AUROC=0.8244).

## cgmacros_glucose_mm (`cgmacros`)

_CGMacros same-subject CGM + Fitbit HR + meal carbs (causal lookback); not EEG×CGM_

Best single-modality probe for context: **cgm** MAE=11.1004.

| fusion_strategy   |     mae |   coverage |   interval_radius |   n_train |   n_test |
|:------------------|--------:|-----------:|------------------:|----------:|---------:|
| late_weighted     | 20.7693 |     0.9105 |           41.6920 |     11308 |     4716 |
| attention         | 20.7755 |     0.9099 |           41.6791 |     11308 |     4716 |
| cross_modal       | 20.7799 |     0.9099 |           41.7125 |     11308 |     4716 |
| early             | 20.7807 |     0.9090 |           41.6993 |     11308 |     4716 |
| intermediate      | 20.8009 |     0.9046 |           40.0981 |     11308 |     4716 |

Best among the five: **late_weighted** (MAE=20.7693).
Does **not** beat the best single-modality probe on this split.

## Target glucose distribution (CGMacros multimodal forecast)

30-min-ahead `target_glucose` (mg/dL), n=16024: mean=125.2, std=27.8, p10/p50/p90=98.0/120.0/159.0, range=[40.0, 309.0].

For comparison, the paper-style 5×5 CGMacros CGM-only board (`outputs/_e2e_cgmacros_full_20260904/`) reports **ridge_raw_sequence MAE=10.817** vs **rep:fused MAE=11.678** (fusion does not beat the floor).

---

Source CSV: `outputs/fusion_strategies_table.csv`. Synthetic predecessor `outputs/ablation_table.csv` remains harness-only (see `paper/harness/phase1_honesty/QUARANTINE.md`).
