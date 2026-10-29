"""Collect everything the demo app needs into the deploy folder.

Run ONCE from the project root (after the experiments and src/analysis.py):
    python deploy/prepare_demo_data.py
Re-run it whenever results or figures change.

It copies (nothing is recomputed):
  data/processed/windows.npz (test part only) -> deploy/data/demo_windows.npz
  data/processed/scaler.json                  -> deploy/data/scaler.json
  results/models/fedavg_{iid,noniid}_seed0.npz -> deploy/data/
  results/comparison_table.md, communication_analysis.json -> deploy/data/
  figures/*.png                                -> deploy/static/figures/
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _need(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    return path


def prepare(project_root: Path, deploy_dir: Path) -> dict:
    processed = project_root / "data" / "processed"
    results = project_root / "results"
    out = deploy_dir / "data"
    fig_out = deploy_dir / "static" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    windows = np.load(_need(processed / "windows.npz", "Run: python src/preprocess.py"))
    np.savez_compressed(out / "demo_windows.npz",
                        X=windows["X_test"].astype(np.float32),
                        y=windows["y_test"].astype(np.float32),
                        times=windows["times_test"].astype("datetime64[h]"))
    shutil.copy2(_need(processed / "scaler.json", "Run: python src/preprocess.py"), out / "scaler.json")

    for scheme in ("iid", "noniid"):
        name = f"fedavg_{scheme}_seed0.npz"
        hint = ("Run python src/federated.py --scheme " + scheme + " (or git pull if your teammate ran it).")
        shutil.copy2(_need(results / "models" / name, hint), out / name)
    for name in ("comparison_table.md", "communication_analysis.json"):
        shutil.copy2(_need(results / name, "Run: python src/analysis.py"), out / name)

    figures = sorted((project_root / "figures").glob("*.png"))
    for f in figures:
        shutil.copy2(f, fig_out / f.name)

    total = sum(p.stat().st_size for p in out.iterdir()) + sum(p.stat().st_size for p in fig_out.glob("*.png"))
    return {"test_windows": int(len(windows["y_test"])), "figures": [f.name for f in figures],
            "files_in_data": sorted(p.name for p in out.iterdir()), "total_bytes": int(total)}


def main() -> int:
    try:
        summary = prepare(ROOT, ROOT / "deploy")
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1
    print("Demo data prepared in deploy/")
    print(json.dumps(summary, indent=2))
    if not any(n.startswith("convergence") for n in summary["figures"]):
        print("NOTE: no analysis figures found; run python src/analysis.py first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
