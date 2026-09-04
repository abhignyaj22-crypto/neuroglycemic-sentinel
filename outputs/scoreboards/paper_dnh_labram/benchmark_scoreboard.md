# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=3, folds=5, seed=7, sota=False

**Protocol:** repeated subject/patient-held-out grouped CV (single-level; opponent selection and RER share folds — nested CV deferred; exact repeats×folds in the Run params line above)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** mumtaz_depression = EEG-BCI single-modality (19-ch resting EEG @64 Hz, real MDD-vs-control diagnosis label); eegmat_workload = MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label). Multimodal-fusion conclusions rest on the **stress** task; no single dataset co-registers EEG+CGM+EHR per subject.

| task              | metric   | best_baseline     |   base_err |   prop_err |   delta_abs |   RER_pct |   RER_CI_low |   RER_CI_high |   p_wilcoxon |   p_holm |   cliffs_delta |   n_folds | meets_>=50%   |
|:------------------|:---------|:------------------|-----------:|-----------:|------------:|----------:|-------------:|--------------:|-------------:|---------:|---------------:|----------:|:--------------|
| mumtaz_depression | 1-AUROC  | labram            |     0.0392 |     0.2134 |     -0.1742 |   -444.51 |      -682.59 |       -310.66 |      1       |        1 |         -0.893 |        15 | False         |
| eegmat_workload   | 1-AUROC  | single:physiology |     0.2565 |     0.3543 |     -0.0978 |    -38.13 |       -58.3  |        -21.01 |      0.99994 |        1 |         -0.6   |        15 | False         |

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

| task              | metric   | floor             |   floor_err | floor_ECE   | sota   |   sota_err | sota_ECE   | proposed   |   proposed_err | proposed_ECE   |
|:------------------|:---------|:------------------|------------:|:------------|:-------|-----------:|:-----------|:-----------|---------------:|:---------------|
| mumtaz_depression | 1-AUROC  | labram            |      0.0392 | 0.057/0.021 | —      |        nan | —          | cacmf_e2e  |         0.1974 | 0.183/0.091    |
| eegmat_workload   | 1-AUROC  | single:physiology |      0.2565 | 0.058/0.090 | —      |        nan | —          | rep:fused  |         0.3543 | 0.105/0.028    |

- **mumtaz_depression**: vs floor (labram 0.0392): proposed cacmf_e2e 0.1974 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).
- **eegmat_workload**: vs floor (single:physiology 0.2565): proposed rep:fused 0.3543 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).

## Verdict

- **mumtaz_depression** (1-AUROC, EEG-BCI single-modality (19-ch resting EEG @64 Hz, real MDD-vs-control diagnosis label)): fused 0.2134 vs labram 0.0392 -> RER -444.5% (95% CI -682.6..-310.7, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**
- **eegmat_workload** (1-AUROC, MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label)): fused 0.3543 vs single:physiology 0.2565 -> RER -38.1% (95% CI -58.3..-21.0, Wilcoxon p=0.9999, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

## Per-configuration CV error (lower is better)


### mumtaz_depression
| config        |   1-AUROC |
|:--------------|----------:|
| labram        |    0.0392 |
| dnh_gated     |    0.0394 |
| xgboost       |    0.07   |
| classical_gbm |    0.081  |
| rep:raw       |    0.1112 |
| single:eeg    |    0.1112 |
| raw_cnn       |    0.16   |
| rep:pca       |    0.1735 |
| cacmf_e2e     |    0.1974 |
| rep:fused     |    0.2134 |
| rep:vq        |    0.2175 |
| rep:neural    |    0.2285 |
| majority      |    0.5    |

### eegmat_workload
| config            |   1-AUROC |
|:------------------|----------:|
| single:physiology |    0.2565 |
| dnh_gated         |    0.2954 |
| xgboost           |    0.3228 |
| raw_cnn           |    0.3366 |
| labram            |    0.3373 |
| classical_gbm     |    0.3423 |
| rep:raw           |    0.3468 |
| rep:fused         |    0.3543 |
| single:eeg        |    0.3635 |
| cacmf_e2e         |    0.3833 |
| rep:pca           |    0.3869 |
| rep:neural        |    0.431  |
| rep:vq            |    0.4365 |
| majority          |    0.5    |
