from pathlib import Path


def test_reproduction_anchor_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "legacy" / "model_pipeline_v4.py").exists()
    assert (root / "configs" / "experiments" / "exp_000_v4_repro.yaml").exists()
