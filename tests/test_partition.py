"""Tests for src/partition.py (no PyTorch needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import partition as pt  # noqa: E402


def fake_arrays(days=60, seed=0):
    """Hourly traffic with a daily cycle that is different on weekends."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-06", periods=24 * days, freq="h")  # starts on a Monday
    hour, weekend = np.asarray(times.hour), np.asarray(times.day_of_week) >= 5
    y = 3000 + 2500 * np.sin(2 * np.pi * (hour - 6) / 24) - 1500 * weekend
    y = (y + rng.normal(0, 100, len(times))).astype(np.float32)
    X = rng.normal(size=(len(times), 28)).astype(np.float32)
    half = len(times) // 2
    return {"X_train": X[:half], "y_train": y[:half], "times_train": times.to_numpy()[:half],
            "X_test": X[half:], "y_test": y[half:], "times_test": times.to_numpy()[half:]}


def test_iid_partition_is_disjoint_and_complete():
    parts = pt.iid_partition(1000, 4, seed=0)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(1000))


def test_iid_sizes_differ_by_at_most_one():
    sizes = [len(p) for p in pt.iid_partition(1003, 4, seed=0)]
    assert max(sizes) - min(sizes) <= 1


def test_iid_is_reproducible_and_seed_dependent():
    a, b = pt.iid_partition(500, 4, 1), pt.iid_partition(500, 4, 1)
    c = pt.iid_partition(500, 4, 2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_noniid_partition_is_disjoint_and_complete():
    times = fake_arrays()["times_train"]
    parts = pt.noniid_partition(times, 4)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(times)))
    assert all(len(p) > 0 for p in parts)


def test_noniid_clients_hold_only_their_own_regime():
    times = pd.DatetimeIndex(fake_arrays()["times_train"])
    parts = pt.noniid_partition(times.to_numpy(), 4)
    for k, idx in enumerate(parts):
        hour, dow = times[idx].hour, times[idx].day_of_week
        if k == 0:      # weekday morning
            assert (dow < 5).all() and ((hour >= 5) & (hour < 12)).all()
        elif k == 1:    # weekday afternoon/evening
            assert (dow < 5).all() and ((hour >= 12) & (hour < 20)).all()
        elif k == 2:    # weekday night
            assert (dow < 5).all() and ((hour >= 20) | (hour < 5)).all()
        else:           # weekend
            assert (dow >= 5).all()


def test_noniid_requires_exactly_four_clients():
    with pytest.raises(ValueError):
        pt.noniid_partition(fake_arrays()["times_train"], 3)


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError):
        pt.partition_indices(fake_arrays()["times_train"], "banana", 4, 0)


def test_build_clients_loses_and_duplicates_nothing():
    arrays = fake_arrays()
    for scheme in ["iid", "noniid"]:
        clients = pt.build_clients(arrays, scheme, 4, seed=0)
        assert sum(c["n_train"] for c in clients) == len(arrays["y_train"])
        merged = np.sort(np.concatenate([c["y_train"] for c in clients]))
        assert np.array_equal(merged, np.sort(arrays["y_train"]))
        assert sum(len(c["y_test_local"]) for c in clients) == len(arrays["y_test"])


def test_noniid_is_much_more_heterogeneous_than_iid():
    arrays = fake_arrays(days=120)
    scaler = {"mean": 0.0, "std": 1.0}
    spread = {s: pt.describe_clients(pt.build_clients(arrays, s, 4, 0), scaler)
              ["spread_of_client_means"] for s in ["iid", "noniid"]}
    assert spread["noniid"] > 5 * spread["iid"]
