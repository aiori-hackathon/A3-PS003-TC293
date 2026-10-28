"""Secondary check (decided in advance): average of the LAST 10 epochs/rounds.

Run from the project root: python src/secondary_check.py

The primary number in every table is the FINAL epoch/round. Because the error
wobbles a little from epoch to epoch, this check averages the last 10 instead,
so one lucky or unlucky final step cannot decide a comparison. It reads the
saved curves; no training is repeated.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 10


def _stat(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def final_value(curves: dict, key: str) -> dict:
    """Value at the last epoch/round, mean and sample std across seeds."""
    return _stat([curve[-1][key] for curve in curves.values()])


def last_k_mean(curves: dict, key: str, k: int = K) -> dict:
    """Average of the last k epochs/rounds per seed, then mean/std across seeds."""
    return _stat([np.mean([p[key] for p in curve[-k:]]) for curve in curves.values()])


SOURCES = [("Centralized (epochs)", "centralized_results.json", "per_seed_curves", "mae", "rmse"),
           ("FedAvg, IID (rounds)", "federated_iid.json", "per_seed_history", "global_mae", "global_rmse"),
           ("FedAvg, Non-IID (rounds)", "federated_noniid.json", "per_seed_history", "global_mae", "global_rmse")]


def main() -> int:
    table, saved = [], {}
    for label, file, curves_key, mae_key, rmse_key in SOURCES:
        path = RESULTS / file
        if not path.exists():
            print(f"(skipping {label}: results/{file} not found)")
            continue
        curves = json.loads(path.read_text())[curves_key]
        row = {"final_mae": final_value(curves, mae_key), "last10_mae": last_k_mean(curves, mae_key),
               "final_rmse": final_value(curves, rmse_key), "last10_rmse": last_k_mean(curves, rmse_key)}
        saved[label] = row
        f = lambda s: f"{s['mean']:.1f} ± {s['std']:.1f}"
        table.append(f"| {label} | {f(row['final_mae'])} | {f(row['last10_mae'])} | "
                     f"{f(row['final_rmse'])} | {f(row['last10_rmse'])} |")
    if not table:
        print("No result files found.")
        return 1
    print(f"\nPrimary (final epoch/round) vs secondary (average of the last {K}); vehicles per hour\n")
    print("| Method | Final MAE | Last-10 MAE | Final RMSE | Last-10 RMSE |")
    print("|---|---|---|---|---|")
    print("\n".join(table))
    out = RESULTS / "secondary_check.json"
    out.write_text(json.dumps({"k": K, "rows": saved}, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
