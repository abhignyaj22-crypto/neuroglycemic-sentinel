# CACMF relativity scoreboard — real labels, held-out subjects

**Run params:** repeats=5, folds=5, seed=7, sota=False

**Protocol:** repeated subject/patient-held-out grouped CV (single-level; opponent selection and RER share folds — nested CV deferred; exact repeats×folds in the Run params line above)

Proposed = CACMF fused (cross-modal transformer + VQ) as a swappable representation into a shared head. Baseline = the single strongest NON-fused opponent on the same folds (trivial floor, classical GBM, best single modality, or a real pretrained SOTA encoder — unstable configs excluded). Error metric per task; RER = (base_err - prop_err)/base_err. No configuration is assumed to win.


**Modality labeling (M4):** cgmacros_glucose = single-modality (CGM only). Multimodal-fusion conclusions rest on the **stress** task; no single dataset co-registers EEG+CGM+EHR per subject.

| task             | metric   | best_baseline      |   base_err |   prop_err |   delta_abs |   RER_pct |   RER_CI_low |   RER_CI_high |   p_wilcoxon |   p_holm |   cliffs_delta |   n_folds | meets_>=50%   |
|:-----------------|:---------|:-------------------|-----------:|-----------:|------------:|----------:|-------------:|--------------:|-------------:|---------:|---------------:|----------:|:--------------|
| cgmacros_glucose | MAE      | ridge_raw_sequence |     10.817 |    11.6784 |     -0.8614 |     -7.96 |        -8.93 |         -6.86 |            1 |        1 |         -0.552 |        25 | False         |

## Triangulation — floor vs SOTA vs proposed

For each task: the strongest **floor** opponent you must not lose to (tuned GBM / TabPFN / Riemannian / single-modality / PCA->logistic / persistence), the strongest open-weight **SOTA** encoder that actually ran here, and the **proposed** model. `err` is the task error (1-AUROC or MAE, lower better); `ECE` is calibration (raw / after temperature scaling). A win must beat BOTH floor and SOTA.

| task             | metric   | floor              |   floor_err | floor_ECE   | sota   |   sota_err | sota_ECE   | proposed   |   proposed_err | proposed_ECE   |
|:-----------------|:---------|:-------------------|------------:|:------------|:-------|-----------:|:-----------|:-----------|---------------:|:---------------|
| cgmacros_glucose | MAE      | ridge_raw_sequence |      10.817 | —           | —      |        nan | —          | rep:fused  |        11.6784 | —              |

- **cgmacros_glucose**: vs floor (ridge_raw_sequence 10.8170): proposed rep:fused 11.6784 -> does NOT beat; SOTA encoder: not runnable in this environment (labeled, not faked).

## Verdict

- **cgmacros_glucose** (MAE, single-modality (CGM only)): fused 11.6784 vs ridge_raw_sequence 10.8170 -> RER -8.0% (95% CI -8.9..-6.9, Wilcoxon p=1.0000, Holm p=1.0000) -> **does NOT meet the >=50% RER bar.**

## Stability (M2)

- No config/fold failures; no unstable configs.

## Per-configuration CV error (lower is better)


### cgmacros_glucose
| config                                   |     MAE |
|:-----------------------------------------|--------:|
| ridge_raw_sequence                       | 10.817  |
| dnh_gated                                | 10.8305 |
| rep:raw                                  | 10.8763 |
| single:cgm                               | 10.8763 |
| ridge_history                            | 10.8763 |
| rep:pca                                  | 10.8783 |
| xgboost                                  | 10.9807 |
| classical_gbm                            | 11.0159 |
| persistence                              | 11.2958 |
| rep:fused                                | 11.6784 |
| rep:vq                                   | 11.8672 |
| rep:neural                               | 12.0269 |
| cacmf_e2e                                | 13.2647 |
| llm:chronos2-timeseries-cgmacros-glucose | 20.2273 |
