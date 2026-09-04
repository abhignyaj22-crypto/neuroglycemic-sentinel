# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=5, folds=5, seed=7, sota=False

**Protocol:** repeated subject/patient-held-out grouped CV (single-level; opponent selection and RER share folds — nested CV deferred; exact repeats×folds in the Run params line above)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** eegmat_workload = MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label). Multimodal-fusion conclusions rest on the **stress** task; no single dataset co-registers EEG+CGM+EHR per subject.

| task            | metric   | best_baseline     |   base_err |   prop_err |   delta_abs |   RER_pct |   RER_CI_low |   RER_CI_high |   p_wilcoxon |   p_holm |   cliffs_delta |   n_folds | meets_>=50%   |
|:----------------|:---------|:------------------|-----------:|-----------:|------------:|----------:|-------------:|--------------:|-------------:|---------:|---------------:|----------:|:--------------|
| eegmat_workload | 1-AUROC  | single:physiology |     0.2598 |     0.3643 |     -0.1045 |    -40.21 |        -56.5 |        -26.72 |            1 |        1 |          -0.69 |        25 | False         |

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

| task            | metric   | floor             |   floor_err | floor_ECE   | sota   |   sota_err | sota_ECE   | proposed   |   proposed_err | proposed_ECE   |
|:----------------|:---------|:------------------|------------:|:------------|:-------|-----------:|:-----------|:-----------|---------------:|:---------------|
| eegmat_workload | 1-AUROC  | single:physiology |      0.2598 | 0.059/0.092 | —      |        nan | —          | rep:fused  |         0.3643 | 0.123/0.027    |

- **eegmat_workload**: vs floor (single:physiology 0.2598): proposed rep:fused 0.3643 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).

## Verdict

- **eegmat_workload** (1-AUROC, MULTIMODAL EEG-BCI (19-ch EEG + ECG @64 Hz, real rest-vs-arithmetic workload label)): fused 0.3643 vs single:physiology 0.2598 -> RER -40.2% (95% CI -56.5..-26.7, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

## Per-configuration CV error (lower is better)


### eegmat_workload
| config                             |   1-AUROC |
|:-----------------------------------|----------:|
| single:physiology                  |    0.2598 |
| dnh_gated                          |    0.3046 |
| xgboost                            |    0.341  |
| raw_cnn                            |    0.343  |
| labram                             |    0.3448 |
| rep:raw                            |    0.3487 |
| classical_gbm                      |    0.3562 |
| rep:fused                          |    0.3643 |
| single:eeg                         |    0.365  |
| cacmf_e2e                          |    0.3858 |
| rep:pca                            |    0.3864 |
| llm:moment1-linear-eegmat-workload |    0.4394 |
| rep:neural                         |    0.4499 |
| rep:vq                             |    0.4611 |
| majority                           |    0.5    |
