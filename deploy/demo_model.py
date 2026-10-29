"""The trained MLP evaluated with plain NumPy (the demo app needs no PyTorch).

This is a self-contained copy of src/numpy_model.py so the deploy folder can be
hosted on its own. A test checks that both copies give identical outputs.
"""
import numpy as np


def load_weights(path) -> dict:
    """Read a saved weights file (an uncompressed .npz). Refuses pickled objects."""
    with np.load(path, allow_pickle=False) as data:
        weights = {key: data[key] for key in data.files}
    weights.pop("n_examples", None)      # present only in client update messages
    return weights


def forward(weights: dict, X) -> np.ndarray:
    """Linear -> ReLU -> Linear -> ReLU -> Linear; one number per input row."""
    layer_ids = sorted({int(name.split(".")[1]) for name in weights})   # e.g. 0, 2, 4
    h = np.asarray(X, dtype=np.float32)
    for position, layer in enumerate(layer_ids):
        h = h @ weights[f"net.{layer}.weight"].T + weights[f"net.{layer}.bias"]
        if position < len(layer_ids) - 1:        # no ReLU after the last layer
            h = np.maximum(h, 0.0)
    return h.squeeze(-1)
