#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_hash(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_legacy_command(cfg: dict, stage: str, out_dir: Path) -> list[str]:
    t = cfg["training"]
    d = cfg["transformer"]
    f = cfg["features"]
    runner = cfg["runner"]
    cmd = [
        sys.executable, str(ROOT / "legacy" / "model_pipeline_v4_fastdev.py"),
        "--stage", stage,
        "--project-dir", str(ROOT),
        "--output-dir", str(out_dir),
        "--device", str(runner.get("device", "auto")),
    ]
    # locked/final load the frozen manifest created during dev inside the V4 pipeline.
    if stage in {"prepare", "dev"}:
        cmd += [
            "--temporal-history-scope", str(f.get("temporal_history_scope", "all")),
            "--sample-weight-mode", str(t.get("sample_weight_mode", "none")),
            "--old-f-weight", str(t.get("old_f_weight", 0.25)),
            "--dev-2023-weight", str(t.get("dev_2023_weight", 2.0)),
            "--catboost-locked-candidates", str(t.get("catboost_locked_candidates", 3)),
            "--seeds", ",".join(map(str, t.get("seeds", [41, 42, 43]))),
            "--d-model", str(d.get("d_model", 64)),
            "--n-heads", str(d.get("n_heads", 4)),
            "--n-layers", str(d.get("n_layers", 2)),
            "--ffn-dim", str(d.get("ffn_dim", 128)),
            "--dropout", str(d.get("dropout", 0.15)),
            "--token-dropout", str(d.get("token_dropout", 0.10)),
            "--dl-batch-size", str(d.get("batch_size", 512)),
            "--dl-lr", str(d.get("lr", 1e-3)),
            "--dl-weight-decay", str(d.get("weight_decay", 1e-4)),
            "--dl-max-epochs", str(d.get("max_epochs", 60)),
        ]
    return cmd


def freeze_or_validate_config(config_path: Path, cfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "config.frozen.yaml"
    frozen_hash = run_dir / "config.sha256"
    digest = canonical_hash(cfg)
    if frozen.exists():
        old = load_yaml(frozen)
        if canonical_hash(old) != digest:
            raise RuntimeError(
                f"Experiment {run_dir.name} already exists with a different config. "
                "Create a new experiment id instead of overwriting it."
            )
    else:
        shutil.copy2(config_path, frozen)
        frozen_hash.write_text(digest + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=["prepare", "dev", "locked", "final"], required=True)
    args = p.parse_args()

    config_path = args.config.resolve()
    cfg = load_yaml(config_path)
    exp_id = cfg["experiment"]["id"]
    run_dir = ROOT / "runs" / exp_id
    freeze_or_validate_config(config_path, cfg, run_dir)

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "experiment_id": exp_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": canonical_hash(cfg),
            "git_commit": git_commit(),
            "implementation": cfg["runner"].get("implementation", "legacy_v4"),
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    impl = cfg["runner"].get("implementation", "legacy_v4")
    if impl != "legacy_v4":
        raise NotImplementedError(
            "Only legacy_v4 is wired initially. Migrate one module at a time while keeping this anchor reproducible."
        )

    cmd = build_legacy_command(cfg, args.stage, run_dir)
    print("[experiment]", exp_id)
    print("[stage]     ", args.stage)
    print("[run_dir]   ", run_dir)
    print("[command]   ", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
