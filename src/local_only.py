"""Phase 7a: LOCAL-ONLY baseline. Every client trains ALONE, nothing is shared.

Run from the project root:
  python src/local_only.py --quick            # 3 epochs, 1 seed, nothing saved
  python src/local_only.py --scheme iid
  python src/local_only.py --scheme noniid
  python src/local_only.py                    # both

Budget matches federated training: rounds x local_epochs = 30 x 2 = 60 epochs
over the client's own data. Each client's model is judged two ways:
  * on the shared TEST period (can it forecast all kinds of traffic?)
  * on its own local test slice (can it forecast its own kind of traffic?)
No bytes are sent.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import preprocess as pp
from federated import mean_std
from metrics import evaluate_scaled
from model import TrafficMLP, predict, set_seed, train_epochs
from partition import build_clients

ROOT = Path(__file__).resolve().parents[1]


def run_local_only(arrays: dict, scaler: dict, scheme: str, seed: int,
                   cfg: dict, epochs: int) -> list:
    clients = build_clients(arrays, scheme, cfg["clients"]["num_clients"], seed)
    input_size, hidden = arrays["X_train"].shape[1], cfg["model"]["hidden_sizes"]
    rows = []
    for c in clients:
        set_seed(seed * 100 + c["id"])
        model = TrafficMLP(input_size, hidden)
        train_epochs(model, c["X_train"], c["y_train"], epochs,
                     cfg["training"]["learning_rate"], cfg["training"]["batch_size"],
                     seed * 10_000 + c["id"])
        overall = evaluate_scaled(arrays["y_test"], predict(model, arrays["X_test"]), scaler)
        local = evaluate_scaled(c["y_test_local"], predict(model, c["X_test_local"]), scaler)
        rows.append({"client": c["id"], "name": c["name"], "n_train": c["n_train"],
                     "global_mae": overall["mae"], "global_rmse": overall["rmse"],
                     "local_mae": local["mae"], "local_rmse": local["rmse"]})
    return rows


def summarize(per_seed: dict) -> dict:
    def stat(key):
        per_seed_means = [np.mean([r[key] for r in rows]) for rows in per_seed.values()]
        return mean_std(per_seed_means)
    return {"global_test_mae": stat("global_mae"), "global_test_rmse": stat("global_rmse"),
            "local_test_mae": stat("local_mae"), "local_test_rmse": stat("local_rmse")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["iid", "noniid", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="fast smoke test")
    args = parser.parse_args()

    cfg = pp.load_config()
    processed = ROOT / cfg["data"]["processed_dir"]
    if not (processed / "windows.npz").exists():
        print("Processed data not found. Run: python src/preprocess.py")
        return 1
    arrays, scaler = pp.load_processed(processed)

    epochs = 3 if args.quick else cfg["federated"]["rounds"] * cfg["federated"]["local_epochs"]
    seeds = cfg["seeds"][:1] if args.quick else cfg["seeds"]
    schemes = ["iid", "noniid"] if args.scheme == "both" else [args.scheme]

    for scheme in schemes:
        print(f"\n=== LOCAL-ONLY, {scheme.upper()} clients: {epochs} epochs each, seeds {seeds} ===")
        per_seed, started = {}, time.time()
        for seed in seeds:
            per_seed[str(seed)] = run_local_only(arrays, scaler, scheme, seed, cfg, epochs)
            print(f"  seed {seed} done")
        summary = summarize(per_seed)

        print(f"\nResults (vehicles per hour); {time.time() - started:.0f}s")
        print(f"  On the shared TEST period (mean of the 4 client models): "
              f"MAE {summary['global_test_mae']['mean']:.1f} +/- {summary['global_test_mae']['std']:.1f}, "
              f"RMSE {summary['global_test_rmse']['mean']:.1f} +/- {summary['global_test_rmse']['std']:.1f}")
        print(f"  On each client's OWN test slice: MAE {summary['local_test_mae']['mean']:.1f}")
        first = per_seed[str(seeds[0])]
        print(f"\n  Per client (seed {seeds[0]}):  shared-test MAE | own-slice MAE")
        for r in first:
            print(f"    {r['name']:<36} {r['global_mae']:8.1f} | {r['local_mae']:8.1f}")

        if args.quick:
            print("\n(--quick run: results NOT saved)")
            continue
        out = ROOT / "results" / f"local_only_{scheme}.json"
        out.write_text(json.dumps({
            "setting": f"local_only_{scheme}", "epochs_per_client": epochs, "seeds": seeds,
            "summary": summary, "std_type": "sample standard deviation across seeds",
            "bytes_sent": 0, "config": cfg, "per_seed": per_seed}, indent=2))
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
