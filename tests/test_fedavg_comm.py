"""Tests for src/server.py (FedAvg) and src/comm.py. NumPy only, no PyTorch."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import comm  # noqa: E402
import server  # noqa: E402


def mlp_shaped_weights(fill=None, seed=0):
    """Arrays with the exact shapes of our MLP (6081 numbers in total)."""
    rng = np.random.default_rng(seed)
    shapes = {"net.0.weight": (64, 28), "net.0.bias": (64,), "net.2.weight": (64, 64),
              "net.2.bias": (64,), "net.4.weight": (1, 64), "net.4.bias": (1,)}
    make = (lambda s: np.full(s, fill, dtype=np.float32)) if fill is not None else \
           (lambda s: rng.normal(size=s).astype(np.float32))
    return {name: make(shape) for name, shape in shapes.items()}


# ---------------------------------------------------------------- FedAvg
def test_fedavg_matches_a_hand_calculation():
    w1 = {"a": np.array([0.0, 0.0], dtype=np.float32), "b": np.array([[10.0]], dtype=np.float32)}
    w2 = {"a": np.array([4.0, 8.0], dtype=np.float32), "b": np.array([[2.0]], dtype=np.float32)}
    result = server.fedavg([w1, w2], [1, 3])           # shares 25% and 75%
    assert np.allclose(result["a"], [3.0, 6.0])        # (0*1 + 4*3)/4, (0*1 + 8*3)/4
    assert np.allclose(result["b"], [[4.0]])           # (10*1 + 2*3)/4


def test_fedavg_with_equal_sizes_is_the_plain_mean():
    ws = [mlp_shaped_weights(seed=s) for s in range(3)]
    result = server.fedavg(ws, [500, 500, 500])
    for key in result:
        assert np.allclose(result[key], np.mean([w[key] for w in ws], axis=0), atol=1e-6)


def test_fedavg_gives_more_influence_to_the_bigger_client():
    small, big = mlp_shaped_weights(fill=0.0), mlp_shaped_weights(fill=10.0)
    result = server.fedavg([small, big], [100, 900])
    assert np.allclose(result["net.0.bias"], 9.0)      # 90% of the way to the big client


def test_fedavg_of_identical_weights_changes_nothing():
    ws = [mlp_shaped_weights(fill=7.0) for _ in range(4)]
    result = server.fedavg(ws, [10, 20, 30, 40])
    assert all(np.allclose(v, 7.0) for v in result.values())   # shares sum to 1


def test_fedavg_with_one_client_returns_that_clients_weights():
    w = mlp_shaped_weights(seed=5)
    result = server.fedavg([w], [123])
    assert all(np.allclose(result[k], w[k]) for k in w)


def test_fedavg_keeps_shapes_and_dtype_and_does_not_modify_inputs():
    ws = [mlp_shaped_weights(seed=1), mlp_shaped_weights(seed=2)]
    before = [{k: v.copy() for k, v in w.items()} for w in ws]
    result = server.fedavg(ws, [1, 2])
    for key in result:
        assert result[key].shape == ws[0][key].shape and result[key].dtype == np.float32
    for w, b in zip(ws, before):
        assert all(np.array_equal(w[k], b[k]) for k in w)


def test_fedavg_rejects_bad_input():
    good = mlp_shaped_weights()
    bad_shape = {**good, "net.0.bias": np.zeros(65, dtype=np.float32)}
    bad_keys = {k: v for k, v in good.items() if k != "net.4.bias"}
    with pytest.raises(ValueError):
        server.fedavg([], [])
    with pytest.raises(ValueError):
        server.fedavg([good, bad_shape], [1, 1])
    with pytest.raises(ValueError):
        server.fedavg([good, bad_keys], [1, 1])
    with pytest.raises(ValueError):
        server.fedavg([good, good], [1, 0])
    with pytest.raises(ValueError):
        server.fedavg([good, good], [1])


# ------------------------------------------------------------------ comm
def test_serialization_round_trip_is_exact():
    w = mlp_shaped_weights(seed=3)
    decoded, n = comm.bytes_to_weights(comm.weights_to_bytes(w, n_examples=4321))
    assert n == 4321
    assert decoded.keys() == w.keys()
    assert all(np.array_equal(decoded[k], w[k]) for k in w)


def test_a_message_contains_only_weights_and_the_example_count():
    """Structural check that no raw records are inside a client update."""
    w = mlp_shaped_weights()
    payload = comm.weights_to_bytes(w, n_examples=7000)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        assert set(data.files) == set(w.keys()) | {"n_examples"}


def test_message_size_does_not_depend_on_how_much_data_the_client_has():
    w = mlp_shaped_weights()
    assert len(comm.weights_to_bytes(w, n_examples=10)) == \
           len(comm.weights_to_bytes(w, n_examples=1_000_000))


def test_message_size_is_close_to_the_raw_parameter_bytes():
    w = mlp_shaped_weights()
    param_bytes = sum(v.nbytes for v in w.values())
    assert param_bytes == 6081 * 4
    size = len(comm.weights_to_bytes(w, n_examples=1))
    assert param_bytes <= size < param_bytes + 4000


def test_pickled_objects_are_refused():
    buffer = io.BytesIO()
    np.savez(buffer, evil=np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError):
        comm.bytes_to_weights(buffer.getvalue())


def test_reserved_name_cannot_be_used_as_a_weight_name():
    with pytest.raises(ValueError):
        comm.weights_to_bytes({"n_examples": np.zeros(1)})


# --------------------------------------------------------- server + comm
def test_server_aggregates_from_bytes_like_plain_fedavg():
    ws = [mlp_shaped_weights(seed=1), mlp_shaped_weights(seed=2)]
    payloads = [comm.weights_to_bytes(w, n_examples=n) for w, n in zip(ws, [100, 300])]
    from_bytes = server.aggregate_updates(payloads)
    direct = server.fedavg(ws, [100, 300])
    assert all(np.array_equal(from_bytes[k], direct[k]) for k in direct)


def test_server_rejects_an_update_without_an_example_count():
    with pytest.raises(ValueError):
        server.aggregate_updates([comm.weights_to_bytes(mlp_shaped_weights())])
