# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=5, folds=5, seed=7, sota=True

**Protocol:** repeated subject/patient-held-out grouped CV (single-level; opponent selection and RER share folds — nested CV deferred; exact repeats×folds in the Run params line above)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** stress = MULTIMODAL (4 peripheral-physiology streams, one wearable); wesad_stress = MULTIMODAL (chest+wrist wearable physiology: ECG/EDA/EMG/resp/temp/ACC); deap_anxiety = MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label); deap_arousal = MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label); eegmat_workload = MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label); mumtaz_depression = EEG-BCI single-modality (19-ch resting EEG @64 Hz, real MDD-vs-control diagnosis label). Multimodal-fusion evidence spans the peripheral-physiology stress task(s) and the DEAP EEG+peripheral affective/BCI tasks; no single dataset co-registers EEG+CGM+EHR per subject.

| task              | metric   | best_baseline     |   base_err |   prop_err |   delta_abs |   RER_pct |   RER_CI_low |   RER_CI_high |   p_wilcoxon |   p_holm |   cliffs_delta |   n_folds | meets_>=50%   |
|:------------------|:---------|:------------------|-----------:|-----------:|------------:|----------:|-------------:|--------------:|-------------:|---------:|---------------:|----------:|:--------------|
| stress            | 1-AUROC  | rep:pca           |     0.1079 |     0.1294 |     -0.0214 |    -19.86 |       -28.55 |        -13.51 |      1       |        1 |         -0.258 |        25 | False         |
| wesad_stress      | 1-AUROC  | xgboost           |     0.0453 |     0.1295 |     -0.0842 |   -186.05 |      -442.68 |        -80.54 |      0.99982 |        1 |         -0.667 |        25 | False         |
| deap_anxiety      | 1-AUROC  | single:physiology |     0.4658 |     0.4696 |     -0.0038 |     -0.81 |        -6.49 |          5.12 |      0.77846 |        1 |         -0.034 |        25 | False         |
| deap_arousal      | 1-AUROC  | single:physiology |     0.4522 |     0.4576 |     -0.0054 |     -1.19 |        -7.07 |          4.63 |      0.8017  |        1 |         -0.027 |        25 | False         |
| eegmat_workload   | 1-AUROC  | single:physiology |     0.2598 |     0.3644 |     -0.1046 |    -40.25 |       -56.57 |        -26.74 |      1       |        1 |         -0.69  |        25 | False         |
| mumtaz_depression | 1-AUROC  | labram            |     0.041  |     0.2052 |     -0.1642 |   -400.2  |      -582.17 |       -279.22 |      1       |        1 |         -0.805 |        25 | False         |

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

| task              | metric   | floor             |   floor_err | floor_ECE   | sota   |   sota_err | sota_ECE   | proposed   |   proposed_err | proposed_ECE   |
|:------------------|:---------|:------------------|------------:|:------------|:-------|-----------:|:-----------|:-----------|---------------:|:---------------|
| stress            | 1-AUROC  | rep:pca           |      0.1079 | 0.059/0.035 | —      |        nan | —          | rep:fused  |         0.1294 | 0.048/0.033    |
| wesad_stress      | 1-AUROC  | xgboost           |      0.0453 | 0.059/0.029 | —      |        nan | —          | cacmf_e2e  |         0.1115 | 0.248/0.158    |
| deap_anxiety      | 1-AUROC  | single:physiology |      0.4658 | 0.285/0.308 | —      |        nan | —          | rep:fused  |         0.4696 | 0.311/0.319    |
| deap_arousal      | 1-AUROC  | single:physiology |      0.4522 | 0.273/0.298 | —      |        nan | —          | rep:fused  |         0.4576 | 0.302/0.311    |
| eegmat_workload   | 1-AUROC  | single:physiology |      0.2598 | 0.059/0.092 | —      |        nan | —          | rep:fused  |         0.3644 | 0.122/0.028    |
| mumtaz_depression | 1-AUROC  | labram            |      0.041  | 0.061/0.020 | —      |        nan | —          | cacmf_e2e  |         0.1923 | 0.177/0.087    |

- **stress**: vs floor (rep:pca 0.1079): proposed rep:fused 0.1294 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **wesad_stress**: vs floor (xgboost 0.0453): proposed cacmf_e2e 0.1115 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **deap_anxiety**: vs floor (single:physiology 0.4658): proposed rep:fused 0.4696 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **deap_arousal**: vs floor (single:physiology 0.4522): proposed rep:fused 0.4576 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **eegmat_workload**: vs floor (single:physiology 0.2598): proposed rep:fused 0.3644 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **mumtaz_depression**: vs floor (labram 0.0410): proposed cacmf_e2e 0.1923 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).

## Verdict

- **stress** (1-AUROC, MULTIMODAL (4 peripheral-physiology streams, one wearable)): fused 0.1294 vs rep:pca 0.1079 -> RER -19.9% (95% CI -28.6..-13.5, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **wesad_stress** (1-AUROC, MULTIMODAL (chest+wrist wearable physiology: ECG/EDA/EMG/resp/temp/ACC)): fused 0.1295 vs xgboost 0.0453 -> RER -186.1% (95% CI -442.7..-80.5, Wilcoxon p=0.9998, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **deap_anxiety** (1-AUROC, MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label)): fused 0.4696 vs single:physiology 0.4658 -> RER -0.8% (95% CI -6.5..5.1, Wilcoxon p=0.7785, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **deap_arousal** (1-AUROC, MULTIMODAL affective/BCI (EEG band-power + peripheral physiology, real SAM label)): fused 0.4576 vs single:physiology 0.4522 -> RER -1.2% (95% CI -7.1..4.6, Wilcoxon p=0.8017, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **eegmat_workload** (1-AUROC, MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label)): fused 0.3644 vs single:physiology 0.2598 -> RER -40.2% (95% CI -56.6..-26.7, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **mumtaz_depression** (1-AUROC, EEG-BCI single-modality (19-ch resting EEG @64 Hz, real MDD-vs-control diagnosis label)): fused 0.2052 vs labram 0.0410 -> RER -400.2% (95% CI -582.2..-279.2, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- **stress**: failures by config = {'sota': 25}; unstable (NaN >20% folds) = ['sota']
- **wesad_stress**: failures by config = {'hubert_ecg': 25, 'sota': 25}; unstable (NaN >20% folds) = ['hubert_ecg', 'sota']
- **deap_anxiety**: failures by config = {'sota': 25}; unstable (NaN >20% folds) = ['sota']
- **deap_arousal**: failures by config = {'sota': 25}; unstable (NaN >20% folds) = ['sota']
- **eegmat_workload**: failures by config = {'sota': 25}; unstable (NaN >20% folds) = ['sota']
- **mumtaz_depression**: failures by config = {'sota': 25}; unstable (NaN >20% folds) = ['sota']

## Per-configuration CV error (lower is better)


### stress
| config        |   1-AUROC |
|:--------------|----------:|
| rep:pca       |    0.1079 |
| rep:raw       |    0.1113 |
| xgboost       |    0.1141 |
| dnh_gated     |    0.1162 |
| classical_gbm |    0.1222 |
| rep:fused     |    0.1294 |
| cacmf_e2e     |    0.1666 |
| single:motion |    0.167  |
| single:ppg    |    0.2505 |
| rep:vq        |    0.3244 |
| rep:neural    |    0.3266 |
| single:eda    |    0.3416 |
| single:temp   |    0.3599 |
| majority      |    0.5    |
| sota          |  nan      |

### wesad_stress
| config        |   1-AUROC |
|:--------------|----------:|
| xgboost       |    0.0453 |
| rep:raw       |    0.0528 |
| classical_gbm |    0.0595 |
| dnh_gated     |    0.0936 |
| rep:pca       |    0.1042 |
| cacmf_e2e     |    0.1115 |
| single:resp   |    0.1243 |
| rep:fused     |    0.1295 |
| single:motion |    0.1369 |
| rep:neural    |    0.2053 |
| rep:vq        |    0.2278 |
| single:eda    |    0.2407 |
| single:ppg    |    0.2577 |
| single:ecg    |    0.2623 |
| single:temp   |    0.2691 |
| single:emg    |    0.3852 |
| majority      |    0.5    |
| hubert_ecg    |  nan      |
| sota          |  nan      |

### deap_anxiety
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.4658 |
| rep:fused         |    0.4696 |
| rep:vq            |    0.4709 |
| rep:pca           |    0.4738 |
| rep:neural        |    0.4771 |
| classical_gbm     |    0.4807 |
| xgboost           |    0.4828 |
| labram            |    0.4918 |
| majority          |    0.5    |
| raw_cnn           |    0.5059 |
| dnh_gated         |    0.5113 |
| cacmf_e2e         |    0.5149 |
| rep:raw           |    0.5246 |
| single:eeg        |    0.5464 |
| sota              |  nan      |

### deap_arousal
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.4522 |
| rep:fused         |    0.4576 |
| rep:pca           |    0.4652 |
| classical_gbm     |    0.4742 |
| xgboost           |    0.4759 |
| rep:vq            |    0.4773 |
| rep:neural        |    0.4786 |
| dnh_gated         |    0.4864 |
| labram            |    0.4872 |
| majority          |    0.5    |
| raw_cnn           |    0.5049 |
| cacmf_e2e         |    0.5182 |
| rep:raw           |    0.5292 |
| single:eeg        |    0.5531 |
| sota              |  nan      |

### eegmat_workload
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.2598 |
| dnh_gated         |    0.2872 |
| xgboost           |    0.341  |
| raw_cnn           |    0.343  |
| labram            |    0.3449 |
| rep:raw           |    0.3487 |
| classical_gbm     |    0.3533 |
| rep:fused         |    0.3644 |
| single:eeg        |    0.365  |
| cacmf_e2e         |    0.3858 |
| rep:pca           |    0.3864 |
| rep:vq            |    0.4565 |
| rep:neural        |    0.4876 |
| majority          |    0.5    |
| sota              |  nan      |

### mumtaz_depression
| config        |   1-AUROC |
|:--------------|----------:|
| labram        |    0.041  |
| dnh_gated     |    0.0413 |
| xgboost       |    0.0827 |
| classical_gbm |    0.0909 |
| rep:raw       |    0.1121 |
| single:eeg    |    0.1121 |
| raw_cnn       |    0.1659 |
| rep:pca       |    0.1757 |
| cacmf_e2e     |    0.1923 |
| rep:fused     |    0.2052 |
| rep:neural    |    0.211  |
| rep:vq        |    0.2314 |
| majority      |    0.5    |
| sota          |  nan      |
