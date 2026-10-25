"""Integration tests for client.py + federated.py. Needs PyTorch (skipped if missing)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import comm  # noqa: E402
import federated as fed  # noqa: E402
import partition as pt  # noqa: E402
import server  # noqa: E402
from client import local_train, weights_to_numpy  # noqa: E402
from model import TrafficMLP, set_seed  # noqa: E402

CFG = {"clients": {"num_clients": 4}, "model": {"hidden_sizes": [16, 16]},
       "training": {"learning_rate": 0.01, "batch_size": 64}}
SCALER = {"mean": 0.0, "std": 1.0}


def toy_arrays(days=60):
    """A LEARNABLE toy problem: y depends on X, with real timestamps for the partitions."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2020-01-06", periods=24 * days, freq="h").to_numpy()
    X = rng.normal(size=(len(times), 28)).astype(np.float32)
    y = (0.5 * X[:, 0] + X[:, 1]).astype(np.float32)
    half = len(times) // 2
    return {"X_train": X[:half], "y_train": y[:half], "times_train": times[:half],
            "X_test": X[half:], "y_test": y[half:], "times_test": times[half:]}


def initial_global(seed=0):
    set_seed(seed)
    return weights_to_numpy(TrafficMLP(28, CFG["model"]["hidden_sizes"]))


def test_zero_epochs_returns_exactly_the_received_global_weights():
    clients = pt.build_clients(toy_arrays(), "iid", 4, seed=0)
    global_w = initial_global()
    local_w, n_k, loss = local_train(global_w, clients[0], [16, 16], 0, 0.01, 64, seed=0)
    assert loss is None and n_k == clients[0]["n_train"]
    assert all(np.array_equal(local_w[k], global_w[k]) for k in global_w)


def test_training_changes_the_weights():
    clients = pt.build_clients(toy_arrays(), "iid", 4, seed=0)
    global_w = initial_global()
    local_w, _, _ = local_train(global_w, clients[0], [16, 16], 2, 0.01, 64, seed=0)
    assert any(not np.array_equal(local_w[k], global_w[k]) for k in global_w)


def test_a_round_equals_a_manual_fedavg_of_the_client_updates():
    clients = pt.build_clients(toy_arrays(), "noniid", 4, seed=0)
    global_w = initial_global()
    new_global, _ = fed.run_round(global_w, clients, round_number=1, seed=0,
                                  cfg=CFG, local_epochs=1)
    updates, counts = [], []
    for c in clients:
        w, n, _ = local_train(global_w, c, [16, 16], 1, 0.01, 64,
                              fed.client_seed(0, 1, c["id"]))
        updates.append(w)
        counts.append(n)
    manual = server.fedavg(updates, counts)
    assert all(np.allclose(new_global[k], manual[k], atol=1e-6) for k in manual)


def test_round_reports_the_real_payload_sizes():
    clients = pt.build_clients(toy_arrays(), "iid", 4, seed=0)
    global_w = initial_global()
    _, stats = fed.run_round(global_w, clients, 1, 0, CFG, local_epochs=1)
    down_one = len(comm.weights_to_bytes(global_w))
    up_one = len(comm.weights_to_bytes(global_w, n_examples=123))   # same size for any n
    assert stats["bytes_down"] == down_one * 4
    assert stats["bytes_up"] == up_one * 4


def test_federated_training_improves_the_global_model():
    result = fed.run_federated(toy_arrays(), SCALER, "iid", seed=0, cfg=CFG,
                               rounds=8, local_epochs=2)
    history = result["history"]
    assert history[-1]["global_mae"] < history[0]["global_mae"]
    assert history[-1]["cumulative_bytes"] > history[0]["cumulative_bytes"]
    assert len(history[0]["client_mae"]) == 4


def test_same_seed_gives_the_same_federated_result():
    a = fed.run_federated(toy_arrays(), SCALER, "noniid", 1, CFG, rounds=3, local_epochs=1)
    b = fed.run_federated(toy_arrays(), SCALER, "noniid", 1, CFG, rounds=3, local_epochs=1)
    assert [h["global_mae"] for h in a["history"]] == [h["global_mae"] for h in b["history"]]
