"""Tests for src/model.py. Needs PyTorch (skipped automatically if missing)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model as m  # noqa: E402


def toy_data(n=512):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 28)).astype(np.float32)
    y = (0.5 * X[:, 0] + X[:, 1]).astype(np.float32)
    return X, y


def test_parameter_count_is_6081():
    # (28*64+64) + (64*64+64) + (64*1+1) = 6081
    assert m.count_parameters(m.TrafficMLP(28, (64, 64))) == 6081


def test_output_has_one_number_per_example():
    assert m.TrafficMLP()(torch.zeros(5, 28)).shape == (5,)


def test_weights_roundtrip_and_get_weights_returns_a_copy():
    a, b = m.TrafficMLP(), m.TrafficMLP()
    weights = m.get_weights(a)
    m.set_weights(b, weights)
    x = torch.randn(4, 28)
    assert torch.allclose(a(x), b(x))
    for key in weights:                       # damage the saved copy ...
        weights[key].zero_()
    assert not torch.allclose(a(x), torch.zeros(4))   # ... the model is unaffected


def test_training_reduces_the_loss():
    X, y = toy_data()
    m.set_seed(0)
    net = m.TrafficMLP()
    losses = m.train_epochs(net, X, y, epochs=15, lr=0.01, batch_size=64, seed=0)
    assert losses[-1] < losses[0] * 0.5


def test_same_seed_gives_same_result():
    X, y = toy_data()

    def run():
        m.set_seed(1)
        net = m.TrafficMLP()
        m.train_epochs(net, X, y, epochs=3, lr=0.01, batch_size=64, seed=1)
        return m.predict(net, X[:10])

    assert np.allclose(run(), run())


def test_predict_returns_one_value_per_row():
    X, _ = toy_data(100)
    assert m.predict(m.TrafficMLP(), X, batch_size=32).shape == (100,)
