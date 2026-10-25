"""One simulated intersection: trains the model on ITS OWN data only."""
import torch

from model import TrafficMLP, get_weights, set_weights, train_epochs


def weights_to_numpy(model) -> dict:
    """Model weights as plain NumPy arrays (copies)."""
    return {key: value.numpy().copy() for key, value in get_weights(model).items()}


def load_numpy_weights(model, weights: dict) -> None:
    set_weights(model, {key: torch.as_tensor(value) for key, value in weights.items()})


def local_train(global_weights: dict, client: dict, hidden_sizes, epochs: int,
                lr: float, batch_size: int, seed: int):
    """Start from the global weights, train on local data, return the new weights.

    Returns (local_weights, n_k, last_epoch_loss). The client's data
    (client["X_train"], client["y_train"]) is used here and goes nowhere else.
    """
    model = TrafficMLP(client["X_train"].shape[1], hidden_sizes)
    load_numpy_weights(model, global_weights)
    losses = []
    if epochs > 0:
        losses = train_epochs(model, client["X_train"], client["y_train"],
                              epochs, lr, batch_size, seed)
    return weights_to_numpy(model), client["n_train"], (losses[-1] if losses else None)
