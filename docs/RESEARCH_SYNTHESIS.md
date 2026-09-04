# What does the research collectively reveal?

*Full-repo synthesis, produced by a self-paced `/loop` analysis covering all 17 major areas of pipelinedvxr: `src/dvxr/` (core + subpackages), `diabetes-attention-fusion/`, `neuroglycemic-sentinel/`, `neuroglycemic-sentinel-claude-integration/`, `dvxr-vr-realtime/`, the UI/deployment patch bundles, `paper/`, `outputs/`, `presentation/`, `scripts/`, `docs/`, `tasks/`, `configs/`, `data/`, `third_party/`, `web/`, and `tests/`. Every claim below traces to a file read during that pass; no number is invented.*

## Headline answer

**Across every task family in this repo, learned multimodal fusion does not reliably beat the best single-modality specialist in small clinical cohorts.** The one reproducible exception is a shallow, clinically-motivated feature add — meal timing on top of continuous glucose monitoring (CGM) — not a learned cross-modal architecture. The repo's substantive research contribution is not a fusion method that wins; it is the empirical demonstration of *when* fusion helps and when it harms, backed by a structural discipline (two honesty-audit test suites, an ADR, CI gates, per-prediction abstention) that prevents the codebase from quietly reporting a loss as a win.

This finding replicates across three **independently engineered** fusion systems in this repo — the paper's CACMF, the separate `diabetes-attention-fusion` port of a Nature architecture, and the `neuroglycemic-sentinel` CogWear arm — which is what makes it a genuine finding rather than one failed experiment.

## The core evidence, by task family

### 1. Mental-health / clinical BCI tasks (the paper's primary study)

Six tasks, subject-held-out CV (5×5, seed=7), IEEE paper "When Learned Cross-Modal Fusion Helps and When It Harms":

| Task | Best specialist AUROC | Learned CACMF fusion RER | DNH (do-no-harm gated fusion) |
|---|---|---|---|
| Mumtaz depression | 0.961 window / 0.986 subject (**identity-leakage-confounded**, decode acc 0.888 vs 1/58 chance) | −148.2% (Holm p=1.0) | beats CACMF, +52.9% RER |
| WESAD stress | 0.955 (n=8) | −185.9% | beats CACMF, +28.2% RER |
| EEGMAT cognitive workload | 0.740 (ECG), 0.663 (LaBraM EEG) | −40.4% | **fails held-out**, −15.6% |
| Non-EEG/peripheral stress | 0.892 | −19.9% | beats CACMF, +10.8% |
| DEAP anxiety | 0.534 (chance) | −0.6% | **fails held-out**, −3.6% |
| DEAP arousal | 0.548 (chance) | −1.2% | **fails held-out**, −4.5% |

Fusion loses on **all 6** tasks (Holm-corrected p=1.0 in every case). The gated do-no-harm (DNH) alternative beats CACMF on 4/6, but its inner-CV floor is not a safe held-out guarantee at N≤60 — it fails on EEGMAT and both DEAP tasks. DEAP anxiety/arousal sit at chance regardless of method (honest negatives, excluded from product claims).

### 2. BCI raw-signal decode

Most raw-signal decode tasks sit at or below chance under strict cross-validation: Welch-feature 5-class balanced accuracy 0.2651 (chance 0.20), engaged-vs-neutral AUROC 0.489 (chance 0.50), lateralization AUROC 0.5412. The one real signal — 4-class command decoding — degrades hard from 0.8227 (trial-grouped CV, since demoted as leaky) to 0.4452 under leave-one-block-out (chance 0.25), showing most of the apparent signal was a CV-design artifact, not real decode.

### 3. Glucose forecasting — the one place integration wins

- Model ladder (CGMacros, 30-min RMSE): persistence 17.40 → gradient boosting 12.48 (best point-RMSE at every horizon) → NeuroGlycemicNet (deep net) 12.99. **GBM beats the in-house deep net.**
- CGM + meal events: RMSE 12.99 vs. CGM-only 13.33 — a 0.34 mg/dL improvement, the **only** integrated-fusion win found anywhere in the repo.
- Removing CGM entirely collapses accuracy (RMSE→34.09) and forces abstention on ~49.7% of windows — the system declines to predict rather than fabricate.
- Separately, an un-merged SSL package (`diabetes-attention-fusion/docs/SSL_GLUCOSE_FORECASTING.md`) found that a **zero-shot, never-fine-tuned foundation model** (`chronos-bolt-tiny`, 9M params) beats every in-house glucose model, including the project's own SSL copy-head and ridge/GBM/LSTM baselines — a finding that quietly outranks the repo's custom modeling effort on this task.

### 3a. Phase-I LLM/foundation-model re-test — two additional negatives

Two independently audited candidates were run through the unchanged repeated grouped-CV
protocol (5 repeats × 5 folds, seed 7) and neither was promoted:

- **CGMacros / Chronos-2 frozen embedding + linear probe**: MAE 20.2273 mg/dL, the worst
  configuration in the run, versus 10.8170 for `ridge_raw_sequence`, 10.8305 for
  `dnh_gated`, and 11.2958 for persistence. The useful result is the additive causal
  raw-history sidecar: plain ridge on that sidecar became the new floor. Wrapping the same
  history in the frozen Chronos-2 embedding discarded predictive signal in this
  configuration. This result does not contradict the separate native zero-shot
  Chronos-Bolt experiment above; the model, adapter, and forecasting path differ.
- **EEGMAT / MOMENT-1 frozen embedding + linear probe**: error (`1-AUROC`) 0.4394 versus
  0.2598 for the best physiology specialist, 0.3430 for the same-raw-input CNN, and
  0.3448 for LaBraM. This is an honest negative, but it is not a definitive test of a
  channel-aware EEG foundation model: the pilot flattened 19 EEG channels and truncated
  the flattened vector to 512 samples. A future multichannel `[N,C,T]` candidate must
  receive a new ID and preregistration rather than rewriting this result.

The raw scoreboards are frozen at
`outputs/_llm_pilot_v1_eegmat_workload/benchmark_scoreboard.csv` and
`outputs/_llm_pilot_v2_cgmacros_glucose/benchmark_scoreboard.csv`. The result preserves the
repo-wide conclusion: foundation-model infrastructure can be useful without making the
foundation model a better predictor, and failed candidates remain experimental opponents
rather than product representations.

### 4. EHR / clinical notes

MTSamples (4,499 notes): Bio_ClinicalBERT wins the binary surgery-vs-rest task (AUROC 0.910 vs. TF-IDF 0.823); **TF-IDF wins the 40-way specialty task** (macro-AUROC 0.961 vs. ClinicalBERT 0.931) — the fancier transformer does not win everywhere. MIMIC-IV mortality AUROC 0.813 is explicitly called "indicative rather than validated" (only 15/252 positive events). Diabetes-complication risk has **no model** — the repo explicitly abstains rather than fabricate a label that doesn't exist. Note: the EHR results section and the full clinical-notes numbers are present in the codebase but the corresponding `.tex` file is not actually `\input` into the compiled IEEE paper — the finding exists in the repo, just not in the paper artifact.

### 5. diabetes-attention-fusion (separate 3-modality port of a Nature architecture)

Ports a real Nature *Sci Rep* attention-fusion design (95.7% acc on EEG+ECG stress, same-person data) to a 3-modality diabetes task where **no co-registered EEG+CGM+EHR cohort exists** — every "fused" triple is cross-cohort by construction. Across every board variant (committed, IRM base/consistency/smoketest): fusion beats EEG-only and EHR-only but **loses to wearable/CGM-only** every time, by −0.067 to −0.144 macro-F1. A disclosed bug (fixed in commit `7d196f7`) meant the EEG pathway was silently dropped whenever cached LaBraM embeddings were used — so the committed 0.772 fused headline predates the fix and is flagged "NOT YET RE-VERIFIED"; a reduced post-fix re-run shows fused macro-F1 dropping further (e.g. 0.705→0.667). An LLM-as-predictor triangulation arm scores near chance (macro-F1 0.333).

### 6. neuroglycemic-sentinel (standalone causal forecaster) + CogWear arm

Architecturally isolated research package (external runtime workspace, subprocess-only integration, no live deployment — every checkpoint self-labels `release_recommendation: "research_only_do_not_deploy"`). MIMIC EHR glucose forecasting: RMSE 29.06, MASE vs. naive persistence = 1.016 — **essentially tied with the dumbest possible baseline**. Its CogWear EEG+wearable cognitive-load fusion arm (10-patient cohort) is the cleanest replication of the whole-repo pattern: EEG-only AUROC = 0.5 (chance/degenerate head), wearable-only AUROC = 0.972–1.0, and the **learned late-fusion weights collapsed to `{eeg: 0.0, wearable: 1.0}`** — the model itself learned to discard EEG and just copy the wearable head.

### 7. Product / demo / VR layers

`outputs/product/` screeners (subject-held-out CV): depression AUROC 0.9608 (peak decision-curve net benefit 0.441 @55%), stress 0.9485, cognitive workload 0.6627 — all carrying "research-grade screening, not a diagnosis" caveats. The VR HUD, the Claude-Artifact bridge, and both UI patch bundles were all checked directly for oversold numbers and none were found: illustrative values are explicitly labeled `illustrative: true` / "No live model was executed," the fused `stress_glucose_risk` product path abstains by construction because no synchronized cohort exists, and the public `web/signal/` research-pitch site's copy matches the real scoreboard numbers, including presenting the fusion-losing RER figures on the page itself ("we report it plainly").

## The most important caveat in the repo

The single best-looking number — Mumtaz depression AUROC 0.961 (subject-level 0.986) — is **self-qualified in the paper itself, repeatedly**, as "identity-leakage-confounded, not a validated biomarker": subject identity is decodable from the same feature space at 88.8% accuracy (52× chance), meaning the classifier may be partly recognizing *who the subject is* rather than detecting depression. This caveat appears in the abstract, the results section, and the headline table — it is not a hidden flaw found only during this audit.

## What enforces this honesty (why the finding can be trusted)

- **Two honesty-audit test suites** (`tests/test_honesty_audit.py` at root, `diabetes-attention-fusion/tests/test_diabetes_honesty_audit.py`) structurally block reintroducing a fabricated win: no cross-modal claim across unsynchronized subjects, no leakage fields in training, no LLM-as-predictor headline, every product number must trace to a committed scoreboard file (verified by `scripts/build_dnh_labram_scoreboard.py::verify()`, tolerance 5e-3).
- **`docs/adr/0002-specialist-not-merged-fusion.md`** is an explicit Architecture Decision Record for *not* merging the diabetes-attention-fusion model into the main product, quoting the rejection reason verbatim: *"fused loses to wearable-only; stale 0.772 — Rejected: scientifically invalid and empirically worse."*
- **`.github/workflows/audit.yml`** runs `make audit` as a blocking CI check on every push/PR to `main`.
- **~120 additional root-level tests** guard specific failure modes: no row-level leakage, no future-event leakage, no hallucinated numbers in LLM explanations, no synthetic subjects in scientific runs, no cross-joining of public cohorts, missing-data distinguishable from zero.
- Every deployed/demo surface checked (VR HUD, Claude-Artifact bridge, both UI patches, the public web pitch site) discloses research-stage/non-clinical status and none were found presenting an unvalidated number as fact.

## Real vs. synthetic data — the map

Almost everything driving a *reported metric* is real public or lab data: WESAD, DEAP, EEGMAT, Mumtaz MDD, MIMIC-IV demo, CGMacros, MTSamples, Zenodo hypoglycemic-clamp EEG (34 T1D patients), and committed single-subject EMOTIV/Galea recordings (schema-validation only, never training/validation). Two explicit exceptions exist and are labeled as such: `src/dvxr/omics.py` generates synthetic omics rows (`source_system: "synthetic_omics"`), and several paper-adjacent tables (`ablation.tex`, `clinical_metrics.tex`, `codebook.tex`) are marked "HARNESS ONLY... Not a scientific result" and contain fixture data (e.g., a synthetic stress-detection AUROC of 1.000 that must never be confused with a real result).

## Bottom line

This is a research program that ran the multimodal-fusion bet several different ways — a custom cross-modal transformer (CACMF), a literature-ported attention architecture (diabetes-attention-fusion), and a late-fusion cognitive-load classifier (CogWear) — and got the same answer each time: **in small (n=8–58), non-co-registered clinical cohorts, a well-tuned single-modality specialist (frozen foundation-model EEG embeddings, or gradient-boosted wearable/CGM features) beats learned fusion.** The single robust win for combining signals is adding a cheap, clinically meaningful feature (meal timing) to an already-strong CGM model — not a learned cross-modal architecture. The repo's actual deliverable is the infrastructure that makes this negative result impossible to quietly reverse: abstention-by-default when data isn't co-registered, audit tests that fail the build if a forbidden "winner" reappears, and an ADR record of the decision to report the loss rather than merge it into the product.
