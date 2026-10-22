"""Tests for src/preprocess.py. Run from project root:  pytest tests/ -v

These use small made-up data so the rules can be checked exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as pp  # noqa: E402


def make_series(n=100, start="2020-01-01"):
    """Traffic = 0, 1, 2, ... so every hour has a unique, easy-to-check value."""
    idx = pd.date_range(start, periods=n, freq="h")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_duplicate_timestamps_become_one_row():
    t = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00"])
    df = pd.DataFrame({"date_time": t, "traffic_volume": [500.0, 500.0, 700.0]})
    series = pp.one_reading_per_hour(df)
    assert len(series) == 2
    assert series.loc["2020-01-01 00:00"] == 500.0


def test_short_gap_filled_long_gap_left_empty():
    s = make_series(60)
    s = s.drop(s.index[10:12])     # 2-hour gap  -> should be filled
    s = s.drop(s.index[30:40])     # 10-hour gap -> must stay empty
    filled, n = pp.fill_short_gaps(s, max_fill_hours=3)
    assert n == 2
    assert filled.isna().sum() == 10
    assert filled.loc["2020-01-01 10:00"] == 10.0   # straight line between 9 and 12


def test_zero_max_fill_never_fills():
    s = make_series(30).drop(make_series(30).index[5:7])
    filled, n = pp.fill_short_gaps(s, max_fill_hours=0)
    assert n == 0 and filled.isna().sum() == 2


def test_split_is_chronological_and_has_no_overlap():
    train, test, cutoff = pp.chronological_split(make_series(100), 0.8)
    assert train.index.max() < test.index.min()
    assert test.index.min() == cutoff
    assert len(train) == 80 and len(test) == 20


def test_scaler_uses_training_data_only():
    train, test, _ = pp.chronological_split(make_series(100), 0.8)
    scaler = pp.fit_scaler(train)
    assert np.isclose(scaler["mean"], train.mean())
    assert not np.isclose(scaler["mean"], make_series(100).mean())


def test_scale_then_inverse_returns_original_units():
    scaler = {"mean": 100.0, "std": 20.0}
    original = np.array([40.0, 100.0, 160.0])
    assert np.allclose(pp.inverse_scale(pp.scale(original, scaler), scaler), original)


def test_window_contents_are_correct():
    s = make_series(50)
    scaler = {"mean": 0.0, "std": 1.0}          # identity scaling to read values easily
    X, y, times = pp.build_windows(s, scaler, window_length=24)
    assert X.shape == (26, 28)                  # 24 history + 4 calendar features
    assert np.allclose(X[0, :24], np.arange(24))   # hours 0..23
    assert y[0] == 24.0                            # the next hour
    assert times[0] == s.index[24].to_datetime64()


def test_windows_never_span_a_gap():
    s = make_series(80)
    s = s.drop(s.index[40:45])                  # 5-hour gap
    filled, _ = pp.fill_short_gaps(s, max_fill_hours=0)
    scaler = {"mean": 0.0, "std": 1.0}
    X, y, _ = pp.build_windows(filled, scaler, window_length=10)
    assert not np.isnan(X).any() and not np.isnan(y).any()
    # every window is 11 consecutive hours, so y == last history value + 1
    assert np.allclose(y, X[:, 9] + 1)
    # windows touching the gap are gone: 80-10 = 70 possible, minus those crossing it
    assert len(y) < 70


def test_test_windows_only_use_test_period():
    train, test, cutoff = pp.chronological_split(make_series(100), 0.8)
    scaler = pp.fit_scaler(train)
    X_te, y_te, t_te = pp.build_windows(test, scaler, window_length=10)
    assert pd.Timestamp(t_te.min()) >= cutoff
    X_tr, y_tr, t_tr = pp.build_windows(train, scaler, window_length=10)
    assert pd.Timestamp(t_tr.max()) < cutoff
