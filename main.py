#!/usr/bin/env python3
"""Single public entry — config → validate or train → scoreboard → demo infer.

Profiles
  smoke | mh | glucose | pow     — require lab ``src/`` on PYTHONPATH (retrain)
  paper-validate                   — offline: check committed boards vs paper claims
  layer-tour                       — print CACMF layer shapes/hyperparams from snapshot

Fusion is a comparator, never assumed to win. CACMF trains on CPU. FM probes stay off
unless config.fm_probes is true. Complication/progression stay blocked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api.predict import predict
from data_loaders import load_task
from models.evaluators import write_scoreboard
from models.fusion import describe as describe_fusion
from models.trainer import train_and_eval
from pipeline.outcomes import OUTCOMES, serve_outcome
from utils.logger import log
from utils.paths import ROOT, ensure_lab_on_path, load_config


def _banner(title: str) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)


def _print_layer_tour(cfg: dict) -> int:
    """Print methods-section layer defaults from the committed snapshot (no torch)."""
    snap_path = ROOT / "outputs" / "scoreboards" / "cacmf_config_defaults.json"
    _banner("[main] Stage 0 — layer tour (CACMFConfig defaults)")
    if not snap_path.is_file():
        log(f"missing {snap_path}", level="error")
        return 2
    d = json.loads(snap_path.read_text())
    print("[trace] L0  events → aligned windows (subject-held-out splits)")
    print(f"[trace] L1  modality encoders → z_m in R^{{B x d}}  d={d['d']}")
    print(
        f"[trace] L2  VQ codebooks → q_m  K={d['codebook_size']} "
        f"beta={d['commitment_beta']} (absent token, not zero-fill)"
    )
    print(
        f"[trace] L3  fusion → h in R^{{B x d_f}}  d_f={d['d_f']} "
        f"L={d['n_fusion_layers']} heads={d['n_heads']} dropout={d['dropout']}"
    )
    print(f"[trace]     strategies={d['fusion_strategies']}")
    print(
        f"[trace] L4  task heads → p in [0,1] or y_hat mg/dL  "
        f"(epochs={d['epochs']} bs={d['batch_size']} "
        f"lr_enc={d['lr_encoder']} lr_fus={d['lr_fusion']})"
    )
    print(
        f"[trace] L5  honesty gate → argmin held-out error; "
        f"refuse unmatched EEG x CGM; EMOTIV/Galea val-only"
    )
    print(f"[trace] seed={d['seed']} source={d.get('source')}")
    fusion = describe_fusion()
    log(f"fusion comparators registered: {fusion.get('strategies')}")
    blocked = {
        k: serve_outcome(k).evidence_status.value
        for k, spec in OUTCOMES.items()
        if spec.structural_label_absent
    }
    print(json.dumps({"blocked_endpoints": blocked, "fm_probes": cfg.get("fm_probes", False)}, indent=2))
    return 0


def _run_paper_validate() -> int:
    _banner("[main] Stage — paper-validate (committed boards, no lab required)")
    # Prefer package-relative import path
    script = ROOT / "scripts" / "validate_paper_claims.py"
    if not script.is_file():
        log(f"missing {script}", level="error")
        return 2
    import runpy

    ns = runpy.run_path(str(script), run_name="__validate__")
    return int(ns.get("main", lambda: 1)())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="DVXR clinical-risk extract — public entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 main.py --profile paper-validate\n"
            "  python3 main.py --profile layer-tour\n"
            "  python3 main.py --profile smoke --dry-run\n"
            "  DVXR_LAB_ROOT=/path/to/pipelinedvxr python3 main.py --profile smoke\n"
        ),
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="smoke | mh | glucose | pow | paper-validate | layer-tour",
    )
    ap.add_argument("--dry-run", action="store_true", help="load config + builders, do not train")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args(argv)

    # ------------------------------------------------------------------ Stage 1
    _banner("[main] Stage 1 — load config")
    cfg = load_config(Path(args.config) if args.config else None)
    profile_name = args.profile or cfg.get("default_profile", "smoke")
    seed = int(cfg.get("seed", 7))
    log(f"profile={profile_name} device={cfg.get('device', 'cpu')} seed={seed}")
    log(f"root={ROOT}")

    # Offline profiles — no lab import
    if profile_name == "paper-validate":
        return _run_paper_validate()
    if profile_name == "layer-tour":
        return _print_layer_tour(cfg)

    # ------------------------------------------------------------------ Stage 2
    _banner("[main] Stage 2 — resolve lab src/ (retrain profiles)")
    try:
        lab_src = ensure_lab_on_path(cfg)
        log(f"lab src on path: {lab_src}")
    except Exception as exc:
        log(f"lab not available: {exc}", level="error")
        log("Set DVXR_LAB_ROOT or config lab_root, or use --profile paper-validate", level="error")
        return 2

    log(f"fusion comparators: {describe_fusion()['strategies']}")

    if profile_name == "pow":
        from pipeline.pow_run import run_pow

        _banner("[main] Stage 3 — POW harness walk")
        return run_pow(cfg, dry_run=bool(args.dry_run))

    if profile_name not in cfg.get("profiles", {}):
        log(f"unknown profile {profile_name!r}", level="error")
        return 2
    profile = cfg["profiles"][profile_name]

    # ------------------------------------------------------------------ Stage 3
    _banner("[main] Stage 3 — load public cohorts")
    tasks = {}
    for name in profile["tasks"]:
        log(f"Loading {name}...")
        if args.dry_run:
            tasks[name] = name
            continue
        try:
            tasks[name] = load_task(name, cfg=cfg)
        except Exception as exc:
            log(f"could not load {name}: {exc}", level="error")
            log("See docs/DATA.md — raw cohorts are not shipped in this extract.", level="error")
            return 2
        t = tasks[name]
        log(f"  n={t.n} modalities={t.modalities} kind={getattr(t, 'kind', '?')}")
        print(f"[trace] task={name} shapes=" + json.dumps(
            {m: list(t.features[m].shape) for m in t.modalities}, default=str
        ))

    if args.dry_run:
        log("dry-run: skipping train/eval")
        blocked = {
            k: serve_outcome(k).evidence_status.value
            for k, spec in OUTCOMES.items()
            if spec.structural_label_absent
        }
        print(json.dumps({"profile": profile_name, "tasks": list(tasks), "blocked": blocked}, indent=2))
        demo = predict({"endpoint": "stress_detection", "eeg_source": "emotiv", "wearable_source": "cgmacros"})
        print(json.dumps(demo, indent=2, default=str))
        return 0

    # ------------------------------------------------------------------ Stage 4
    _banner("[main] Stage 4 — specialists + fusion comparators (CPU)")
    if cfg.get("fm_probes"):
        log("fm_probes requested but stay experimental; not enabled in this extract", level="warning")

    results = []
    out_dir = str(ROOT / "outputs" / "_scratch" / profile_name)
    for name, task in tasks.items():
        log(
            f"Training/evaluating {name} "
            f"({profile['repeats']}x{profile['folds']}, no_sota={profile['no_sota']})..."
        )
        result = train_and_eval(
            task,
            repeats=int(profile["repeats"]),
            folds=int(profile["folds"]),
            seed=seed,
            no_sota=bool(profile["no_sota"]),
            representations=list(profile.get("reps") or ["raw"]),
            out_dir=out_dir,
            cfg=cfg,
        )
        results.append(result)
        print(f"[trace] finished task={name}")

    # ------------------------------------------------------------------ Stage 5
    _banner("[main] Stage 5 — write scoreboard")
    meta = {
        "profile": profile_name,
        "seed": seed,
        "repeats": profile["repeats"],
        "folds": profile["folds"],
        "no_sota": profile["no_sota"],
        "device": "cpu",
    }
    write_scoreboard(results, out_dir=out_dir, meta=meta)
    log(f"wrote scoreboard under {out_dir}")

    # ------------------------------------------------------------------ Stage 6
    _banner("[main] Stage 6 — fail-closed demo infer")
    demo = predict({"endpoint": "stress_detection"})
    print(json.dumps(demo, indent=2, default=str))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
