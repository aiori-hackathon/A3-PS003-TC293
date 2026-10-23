"""Phase 4a: simple 'no learning' baselines. A trained model must beat these.

Run from the project root:  python src/baselines.py
(No PyTorch needed.)
"""
import json
import sys
from pathlib import Path

import preprocess as pp
from metrics import evaluate_scaled

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTIONS = {
    "training_average": "always predict the average traffic of the training period",
    "last_hour": "predict that next hour equals the previous hour",
    "same_hour_yesterday": "predict that next hour equals the same hour 24 hours earlier",
}


def naive_predictions(X, window_length: int) -> dict:
    """Predictions (in scaled units) from the input windows alone."""
    return {
        "training_average": X[:, 0] * 0.0,       # 0 in scaled space = training mean
        "last_hour": X[:, window_length - 1],    # most recent hour in the window
        "same_hour_yesterday": X[:, 0],          # oldest hour in the window = t-24
    }


def main() -> int:
    cfg = pp.load_config()
    processed = ROOT / cfg["data"]["processed_dir"]
    if not (processed / "windows.npz").exists():
        print("Processed data not found. Run: python src/preprocess.py")
        return 1

    arrays, scaler = pp.load_processed(processed)
    predictions = naive_predictions(arrays["X_test"], cfg["data"]["window_length"])

    results = {}
    print("Naive baselines on the TEST period (vehicles per hour, lower is better)")
    for name, pred in predictions.items():
        metrics = evaluate_scaled(arrays["y_test"], pred, scaler)
        results[name] = {**metrics, "description": DESCRIPTIONS[name]}
        print(f"  {name:<22} MAE = {metrics['mae']:8.1f}   RMSE = {metrics['rmse']:8.1f}")

    out = ROOT / "results" / "naive_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
