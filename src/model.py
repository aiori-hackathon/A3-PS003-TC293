"""The forecasting model (a small MLP) and the training helpers.

The same functions are reused later by the federated clients and server.
"""
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Make random choices repeatable (initial weights, batch order)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrafficMLP(nn.Module):
    """Input: 24 past hours + 4 calendar features (28 numbers). Output: 1 number."""

    def __init__(self, input_size: int = 28, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_weights(model: nn.Module) -> dict:
    """A COPY of all weights (what a client would send to the server)."""
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def set_weights(model: nn.Module, weights: dict) -> None:
    """Load weights into the model (what a client does with the global model)."""
    model.load_state_dict(weights)


def train_epochs(model, X, y, epochs: int, lr: float, batch_size: int,
                 seed: int, on_epoch_end=None) -> list:
    """Train with Adam + MSE loss. Returns the average loss of each epoch.

    Shuffling the order of TRAINING examples is fine. Leakage is about test data.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t, y_t = torch.as_tensor(X), torch.as_tensor(y)
    generator = torch.Generator().manual_seed(seed)
    n = len(y_t)

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        epoch_losses.append(total / n)
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_losses[-1])
    return epoch_losses


def predict(model, X, batch_size: int = 4096) -> np.ndarray:
    """Predictions (in scaled units) as a NumPy array."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start + batch_size])
            outputs.append(model(batch).numpy())
    return np.concatenate(outputs)
