from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_dir(experiment_id: str) -> Path:
    return project_root() / "runs" / experiment_id
