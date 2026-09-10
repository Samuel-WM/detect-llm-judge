"""Long-format result writer and run manifest. Every phase appends rows matching the schema in
CLAUDE_CODE_PROMPT.md Section 3 to a per-phase parquet file. Every run writes a manifest.json
with the full resolved config, git commit hash, package versions, and seed. A result without a
manifest does not count.
"""
import importlib.metadata as md
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RESULT_COLUMNS = [
    "phase", "run_id", "seed", "base_policy", "true_judge", "candidate_judge", "rule",
    "n_probes", "probe_selection", "beta_dpo", "epochs", "delta_stat", "judge_margin",
    "top1_correct", "score", "wall_clock_s", "timestamp",
]

MANIFEST_PACKAGES = [
    "torch", "transformers", "trl", "peft", "datasets", "accelerate",
    "numpy", "scipy", "pandas", "pyarrow", "matplotlib", "tqdm",
]


def append_results(rows: list[dict[str, Any]], path: Path) -> None:
    """Append long-format rows to a parquet file at `path`, creating it if absent. Rows may omit
    columns not relevant to that phase; missing schema columns are filled with None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[RESULT_COLUMNS + [c for c in df.columns if c not in RESULT_COLUMNS]]

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def write_manifest(
    run_id: str, config: dict[str, Any], results_dir: Path, seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    versions = {}
    for pkg in MANIFEST_PACKAGES:
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None

    manifest = {
        "run_id": run_id,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "package_versions": versions,
        "config": config,
    }
    if extra:
        manifest["extra"] = extra

    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest_path
