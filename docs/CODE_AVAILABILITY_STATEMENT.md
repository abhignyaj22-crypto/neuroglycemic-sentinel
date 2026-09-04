# Draft Code Availability statement

For insertion into the manuscript alongside the existing Data Availability section.
Fill in `[repository URL]` and `[commit hash]` once you push this repo and are ready to
cite it — nothing here should be pasted with placeholders still in it. No `repository-code`
remote is configured yet (checked via `git remote -v`).

---

**CODE AVAILABILITY**

Source code supporting the preprocessing, modality-specific representation, multimodal
integration, evaluation, and ablation analyses reported in this manuscript is publicly
available at [repository URL] (commit `[commit hash]`), under the MIT license. Two
reproduction paths are provided and documented in `docs/EXPERIMENTS.md`: `python3
main.py --profile paper-validate` and `python3 scripts/validate_paper_claims.py`
validate every reported number against committed evidence offline, with no external
dependency; `python3 scripts/reproduce_from_raw.py` retrains from the cited public
cohorts and regenerates fresh evidence for comparison. The private-lab dependency for
retrain profiles (`smoke`/`mh`/`glucose`/`pow`) is disclosed explicitly in
`docs/DATA.md` and `README.md`, not hidden.

Raw EMOTIV and Galea/OpenBCI recordings are not redistributed. The corresponding
acquisition, synchronization, and representation-verification code
(`data_loaders/emotiv.py`, `data_loaders/galea.py`) is included; both loaders are
restricted to validation/test splits only and reject any attempt to pair them with
CGMacros subjects (`data_loaders/pairing.py::CrossCohortPairingForbidden`), matching the
manuscript's own framing of these devices as acquisition/synchronization verification,
not headline predictive evidence.

---

## Before pasting this in

- [ ] Push this repository somewhere public and fill in the URL + commit hash.
- [ ] Consider whether to cut an immutable tagged release (e.g. `v1.0.0`) so the
      manuscript points at a fixed commit rather than a moving branch — not done here,
      per your prior decision not to commit/tag without being asked directly.
- [ ] If you later decide to archive the tagged release on Zenodo or OSF for a DOI (out
      of scope for this pass), add a third sentence citing the DOI alongside the
      repository URL.
