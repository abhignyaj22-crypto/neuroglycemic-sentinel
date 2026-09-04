# Data that is not in this repository

This extract ships **code, committed scoreboards, and docs**. It does **not** ship raw recordings.

## Offline reproduction (no downloads)

```bash
python3 main.py --profile paper-validate
python3 main.py --profile layer-tour
```

## Public cohorts (for optional retrain)

Set `DVXR_LAB_ROOT` to a checkout of the lab that already has `data/real/`, **or** place fetches under this extract's `data/real/` (gitignored):

| Task | Typical path | Source |
|---|---|---|
| `wesad_stress` | `data/real/WESAD` | WESAD (UBFC; mirrors exist) |
| `eegmat_workload` | `data/real/eegmat` | EEGMAT (PhysioNet) |
| `mumtaz_depression` | `data/real/mumtaz_mdd` | Mumtaz MDD EEG |
| `deap_anxiety` / `deap_arousal` | `data/real/deap` | DEAP (license / signup required) |
| `cgmacros_glucose` | `data/real/cgmacros` | CGMacros |
| `stress` (PhysioNet Non-EEG) | `data/real/noneeg` | PhysioNet Non-EEG |

Lab helper (private tree): `scripts/fetch_data.py`. Do not commit the resulting files.

## Must never go public

- MIMIC-IV (DUA)
- Raw EMOTIV / Galea zips and CSVs (n=1 device recordings; val/test only, not training)
- `neuroglycemic-runtime/` model weights and aligned windows
- Absolute home-directory paths in committed config (use `config.example.json` + `DVXR_LAB_ROOT`)

## EMOTIV and Galea

Loaders in `data_loaders/emotiv.py` and `data_loaders/galea.py` accept `split` in `{val, test, val_only}` only. Pairing either source with CGMacros raises `CrossCohortPairingForbidden`.
