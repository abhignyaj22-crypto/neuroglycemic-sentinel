# Review gate — public publish checklist

Before first public push:

1. [ ] `python3 main.py --profile paper-validate` exits 0
2. [ ] `python3 main.py --profile layer-tour` prints L0–L5
3. [ ] `pytest -q tests/` passes
4. [ ] No `data/real`, MIMIC, device zips, or `venv` in the tree
5. [ ] `config.json` is local-only or scrubbed; ship `config.example.json`
6. [ ] Prefer `DVXR_LAB_ROOT` over absolute home paths in docs
7. [ ] Do not add `diabetes-attention-fusion` (ADR 0002)
8. [ ] Do not enable CACMF GPU training or Phase I FM probes as product floors
9. [ ] Committed boards under `outputs/scoreboards/` match `docs/EXPERIMENTS.md`

Retrain still wraps the private lab via `DVXR_LAB_ROOT` / `lab_root`. Offline validation does **not** require the lab.
