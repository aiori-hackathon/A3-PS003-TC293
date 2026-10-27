"""Integration test for src/local_only.py. Needs PyTorch (skipped if missing)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import local_only as lo  # noqa: E402

CFG = {"clients": {"num_clients": 4}, "model": {"hidden_sizes": [16, 16]},
       "training": {"learning_rate": 0.01, "batch_size": 64}}


def toy_arrays(days=60):
    rng = np.random.default_rng(0)
    times = pd.date_range("2020-01-06", periods=24 * days, freq="h").to_numpy()
    X = rng.normal(size=(len(times), 28)).astype(np.float32)
    y = (0.5 * X[:, 0] + X[:, 1]).astype(np.float32)
    half = len(times) // 2
    return {"X_train": X[:half], "y_train": y[:half], "times_train": times[:half],
            "X_test": X[half:], "y_test": y[half:], "times_test": times[half:]}


def test_each_client_gets_its_own_trained_model_and_finite_metrics():
    rows = lo.run_local_only(toy_arrays(), {"mean": 0.0, "std": 1.0}, "noniid", seed=0, cfg=CFG, epochs=2)
    assert len(rows) == 4
    assert sum(r["n_train"] for r in rows) == len(toy_arrays()["y_train"])
    for r in rows:
        assert all(np.isfinite(r[k]) for k in ("global_mae", "global_rmse", "local_mae", "local_rmse"))


def test_local_only_is_reproducible_for_the_same_seed():
    a = lo.run_local_only(toy_arrays(), {"mean": 0.0, "std": 1.0}, "iid", 1, CFG, epochs=2)
    b = lo.run_local_only(toy_arrays(), {"mean": 0.0, "std": 1.0}, "iid", 1, CFG, epochs=2)
    assert [r["global_mae"] for r in a] == [r["global_mae"] for r in b]


def test_summary_averages_over_clients_then_seeds():
    per_seed = {"0": [{"global_mae": 10.0, "global_rmse": 1.0, "local_mae": 1.0, "local_rmse": 1.0},
                      {"global_mae": 20.0, "global_rmse": 1.0, "local_mae": 1.0, "local_rmse": 1.0}],
                "1": [{"global_mae": 30.0, "global_rmse": 1.0, "local_mae": 1.0, "local_rmse": 1.0},
                      {"global_mae": 50.0, "global_rmse": 1.0, "local_mae": 1.0, "local_rmse": 1.0}]}
    s = lo.summarize(per_seed)["global_test_mae"]
    assert np.isclose(s["mean"], 27.5)                    # seed means 15 and 40
    assert np.isclose(s["std"], np.std([15.0, 40.0], ddof=1))
