"""Tests for src/analysis.py and src/numpy_model.py (NumPy only, except one PyTorch check)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analysis as an  # noqa: E402
import numpy_model as nm  # noqa: E402


def fake_fed(seeds=(0, 1), rounds=30, mae_start=300.0):
    """Result structure with the same keys federated.py saves."""
    history = lambda offset: [{
        "round": r, "global_mae": mae_start / r + offset, "global_rmse": 1.5 * mae_start / r + offset,
        "client_mae": [1.0, 2.0, 3.0, 4.0], "bytes_up": 104_520, "bytes_down": 103_480,
        "cumulative_bytes": 208_000 * r} for r in range(1, rounds + 1)]
    return {"per_seed_history": {str(s): history(10.0 * s) for s in seeds},
            "summary_final_round": {"mae": {"mean": 148.2, "std": 1.9}, "rmse": {"mean": 229.0, "std": 1.0}},
            "raw_dataset_file_bytes": 3_237_208}


# ---------------------------------------------------------- numpy forward pass
def test_numpy_forward_matches_a_hand_calculation():
    weights = {"net.0.weight": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
               "net.0.bias": np.zeros(2, dtype=np.float32),
               "net.2.weight": np.array([[1.0, 1.0]], dtype=np.float32),
               "net.2.bias": np.array([0.5], dtype=np.float32)}
    out = nm.forward(weights, np.array([[2.0, -3.0]], dtype=np.float32))
    assert np.allclose(out, [2.5])        # relu([2,-3]) = [2,0]; 2 + 0 + 0.5


def test_numpy_forward_equals_the_pytorch_model():
    pytest.importorskip("torch")
    from client import weights_to_numpy
    from model import TrafficMLP, predict, set_seed
    set_seed(0)
    net = TrafficMLP(28, (64, 64))
    X = np.random.default_rng(0).normal(size=(50, 28)).astype(np.float32)
    assert np.allclose(nm.forward(weights_to_numpy(net), X), predict(net, X), atol=1e-5)


# --------------------------------------------------------------- communication
def test_communication_arithmetic_uses_the_measured_numbers():
    result = an.communication_analysis(fake_fed(), raw_bytes=3_237_208)
    m = result["measured"]
    assert m["bytes_per_round"] == 208_000 and m["total_bytes"] == 6_240_000
    assert np.isclose(m["total_over_raw_file"], 6_240_000 / 3_237_208)
    assert np.isclose(m["break_even_round_vs_raw_file"], 3_237_208 / 208_000)
    one_mbit = result["assumption_based_scenarios"]["rows"][1]
    # 1,560,000 bytes per client / 125,000 bytes per second = 12.48 seconds
    assert np.isclose(one_mbit["seconds_of_streaming_equal_to_one_clients_whole_federated_run"], 12.48)
    assert "ASSUMPTION" in result["assumption_based_scenarios"]["label"]


# ----------------------------------------------------------------- statistics
def test_curve_stats_gives_mean_and_sample_std_across_seeds():
    mean, std = an.curve_stats(fake_fed(seeds=(0, 1)), "global_mae")
    assert np.isclose(mean[0], 300.0 + 5.0)                 # seeds differ by +0 and +10
    assert np.isclose(std[0], np.std([300.0, 310.0], ddof=1))


def test_comparison_rows_work_with_missing_files():
    naive = {k: {"mae": 1.0, "rmse": 2.0} for k in ("training_average", "last_hour", "same_hour_yesterday")}
    assert len(an.comparison_rows(naive, None, {}, {})) == 3
    rows = an.comparison_rows(naive, None, {}, {"iid": fake_fed()})
    assert rows[-1][0] == "FedAvg, IID clients" and rows[-1][3] == "6,240,000"
    assert an.markdown_table(rows).count("\n") == len(rows) + 1


def test_find_consecutive_run_skips_over_gaps():
    first = pd.date_range("2020-01-01", periods=100, freq="h")
    second = pd.date_range("2020-02-01", periods=200, freq="h")        # starts after a gap
    start, size = an.find_consecutive_run(first.append(second), length=168)
    assert (start, size) == (100, 168)


def test_find_consecutive_run_falls_back_to_the_longest_run():
    times = pd.date_range("2020-01-01", periods=50, freq="h").append(
        pd.date_range("2020-03-01", periods=80, freq="h"))
    assert an.find_consecutive_run(times, length=168) == (50, 80)
