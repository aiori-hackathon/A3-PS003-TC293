"""Tests for src/metrics.py and the naive baselines (no PyTorch needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baselines  # noqa: E402
import metrics  # noqa: E402
import preprocess as pp  # noqa: E402


def test_mae_and_rmse_known_values():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 5.0])
    assert np.isclose(metrics.mae(y_true, y_pred), 2.0)
    assert np.isclose(metrics.rmse(y_true, y_pred), np.sqrt(7.0))


def test_rmse_is_never_smaller_than_mae():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert metrics.rmse(a, b) >= metrics.mae(a, b)


def test_evaluate_scaled_reports_original_units():
    scaler = {"mean": 100.0, "std": 10.0}
    result = metrics.evaluate_scaled([0.0, 0.0], [0.5, -0.5], scaler)
    assert np.isclose(result["mae"], 5.0) and np.isclose(result["rmse"], 5.0)


def test_same_hour_yesterday_is_perfect_on_a_perfect_daily_cycle():
    profile = np.array([10, 5, 3, 3, 4, 8, 20, 40, 35, 25, 22, 24,
                        26, 25, 27, 30, 38, 45, 40, 30, 22, 18, 14, 11], dtype=float)
    idx = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    series = pd.Series(np.tile(profile, 10), index=idx)
    scaler = {"mean": 0.0, "std": 1.0}
    X, y, _ = pp.build_windows(series, scaler, window_length=24)
    preds = baselines.naive_predictions(X, 24)
    assert metrics.mae(y, preds["same_hour_yesterday"]) == 0.0
    assert metrics.mae(y, preds["last_hour"]) > 0.0
