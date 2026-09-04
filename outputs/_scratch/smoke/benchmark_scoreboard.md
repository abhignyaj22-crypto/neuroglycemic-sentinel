# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** profile=smoke, seed=7, repeats=1, folds=2, no_sota=True, device=cpu

**Protocol:** repeated subject/patient-held-out grouped CV (single-level; opponent selection and RER share folds — nested CV deferred; exact repeats×folds in the Run params line above)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** wesad_stress = MULTIMODAL (chest+wrist wearable physiology: ECG/EDA/EMG/resp/temp/ACC). Multimodal-fusion conclusions rest on the **stress** task; no single dataset co-registers EEG+CGM+EHR per subject.

| task         | metric   | best_baseline   |   base_err |   prop_err |   delta_abs |   RER_pct |   RER_CI_low |   RER_CI_high |   p_wilcoxon |   p_holm |   cliffs_delta |   n_folds | meets_>=50%   |
|:-------------|:---------|:----------------|-----------:|-----------:|------------:|----------:|-------------:|--------------:|-------------:|---------:|---------------:|----------:|:--------------|
| wesad_stress | 1-AUROC  | rep:raw         |      0.094 |      0.187 |      -0.093 |    -98.95 |      -148.63 |        -78.22 |            1 |      nan |             -1 |         2 | False         |

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

| task         | metric   | floor   |   floor_err | floor_ECE   | sota   |   sota_err | sota_ECE   | proposed   |   proposed_err | proposed_ECE   |
|:-------------|:---------|:--------|------------:|:------------|:-------|-----------:|:-----------|:-----------|---------------:|:---------------|
| wesad_stress | 1-AUROC  | rep:raw |       0.094 | 0.342/0.259 | —      |        nan | —          | cacmf_e2e  |         0.1254 | 0.281/0.275    |

- **wesad_stress**: vs floor (rep:raw 0.0940): proposed cacmf_e2e 0.1254 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).

## Verdict

- **wesad_stress** (1-AUROC, MULTIMODAL (chest+wrist wearable physiology: ECG/EDA/EMG/resp/temp/ACC)): fused 0.1870 vs rep:raw 0.0940 -> RER -98.9% (95% CI -148.6..-78.2, Wilcoxon p=1.0000, Holm p=nan) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

## Per-configuration CV error (lower is better)


### wesad_stress
| config        |   1-AUROC |
|:--------------|----------:|
| rep:raw       |    0.094  |
| cacmf_e2e     |    0.1254 |
| single:eda    |    0.1509 |
| dnh_gated     |    0.1521 |
| xgboost       |    0.1522 |
| classical_gbm |    0.1534 |
| rep:fused     |    0.187  |
| single:resp   |    0.2007 |
| single:motion |    0.2235 |
| single:ecg    |    0.2756 |
| single:ppg    |    0.2929 |
| single:temp   |    0.3491 |
| hubert_ecg    |    0.3688 |
| single:emg    |    0.4286 |
| majority      |    0.5    |
