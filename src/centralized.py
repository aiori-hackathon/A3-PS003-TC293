"""Phase 4b: centralized baseline = ALL training data in one place.

Run from the project root:
  python src/centralized.py --quick   # 3 epochs, 1 seed, nothing saved (smoke test)
  python src/centralized.py           # full run: 3 seeds, results saved
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import preprocess as pp
from metrics import evaluate_scaled
from model import TrafficMLP, count_parameters, predict, set_seed, train_epochs

ROOT = Path(__file__).resolve().parents[1]


def run_one_seed(seed: int, arrays: dict, scaler: dict, cfg: dict, epochs: int):
    set_seed(seed)
    model = TrafficMLP(arrays["X_train"].shape[1], cfg["model"]["hidden_sizes"])
    curve = []

    def after_epoch(epoch: int, train_loss: float) -> None:
        # The test set is only MONITORED here (for the learning curve).
        # We always report the FINAL epoch; we never pick the best epoch on test data.
        pred = predict(model, arrays["X_test"])
        metrics = evaluate_scaled(arrays["y_test"], pred, scaler)
        curve.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  seed {seed} | epoch {epoch:3d}/{epochs} | "
                  f"MAE {metrics['mae']:7.1f} | RMSE {metrics['rmse']:7.1f}")

    train_epochs(model, arrays["X_train"], arrays["y_train"], epochs,
                 cfg["training"]["learning_rate"], cfg["training"]["batch_size"],
                 seed, on_epoch_end=after_epoch)
    return curve, count_parameters(model)


def mean_std(values) -> dict:
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0   # sample std
    return {"mean": float(values.mean()), "std": std}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke test")
    args = parser.parse_args()

    cfg = pp.load_config()
    processed = ROOT / cfg["data"]["processed_dir"]
    if not (processed / "windows.npz").exists():
        print("Processed data not found. Run: python src/preprocess.py")
        return 1
    arrays, scaler = pp.load_processed(processed)

    epochs = 3 if args.quick else cfg["training"]["centralized_epochs"]
    seeds = cfg["seeds"][:1] if args.quick else cfg["seeds"]
    print(f"Centralized training: {epochs} epochs, seeds {seeds}, "
          f"{len(arrays['y_train'])} training windows")

    runs, started = {}, time.time()
    for seed in seeds:
        curve, n_params = run_one_seed(seed, arrays, scaler, cfg, epochs)
        runs[str(seed)] = curve
    elapsed = time.time() - started

    finals = [runs[str(s)][-1] for s in seeds]
    summary = {
        "mae": mean_std([f["mae"] for f in finals]),
        "rmse": mean_std([f["rmse"] for f in finals]),
    }
    print("\nFinal-epoch results on the TEST period (vehicles per hour)")
    print(f"  MAE  = {summary['mae']['mean']:.1f} +/- {summary['mae']['std']:.1f}")
    print(f"  RMSE = {summary['rmse']['mean']:.1f} +/- {summary['rmse']['std']:.1f}")
    print(f"  model parameters: {n_params}, time: {elapsed:.0f}s")

    if args.quick:
        print("\n(--quick run: results NOT saved)")
        return 0

    out = ROOT / "results" / "centralized_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "setting": "centralized", "epochs": epochs, "seeds": seeds,
        "model_parameters": n_params, "summary_final_epoch": summary,
        "std_type": "sample standard deviation across seeds",
        "config": cfg, "per_seed_curves": runs,
    }, indent=2))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
