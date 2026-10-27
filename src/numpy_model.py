"""The trained MLP evaluated with plain NumPy (no PyTorch needed).

Used for plotting predictions and, later, by the web demo: free hosting
plans cannot hold PyTorch, but our model is only 6,081 numbers.
"""
from pathlib import Path

import numpy as np

from comm import bytes_to_weights


def load_weights(path) -> dict:
    """Read a weights file written by federated.py (the same format as a message)."""
    weights, _ = bytes_to_weights(Path(path).read_bytes())
    return weights


def forward(weights: dict, X) -> np.ndarray:
    """Linear -> ReLU -> Linear -> ReLU -> Linear, returns one value per row."""
    layer_ids = sorted({int(name.split(".")[1]) for name in weights})   # e.g. 0, 2, 4
    h = np.asarray(X, dtype=np.float32)
    for position, layer in enumerate(layer_ids):
        h = h @ weights[f"net.{layer}.weight"].T + weights[f"net.{layer}.bias"]
        if position < len(layer_ids) - 1:        # no ReLU after the last layer
            h = np.maximum(h, 0.0)
    return h.squeeze(-1)
