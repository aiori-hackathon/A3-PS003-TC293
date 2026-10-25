"""The aggregation server: Federated Averaging (FedAvg).

IMPORTANT DESIGN RULE: the server only ever receives BYTES that contain
model weights and one integer (n_examples). It never receives, and has no
code path for, raw traffic records.

    w_global = sum_k (n_k / N) * w_k
"""
import numpy as np

from comm import bytes_to_weights


def fedavg(weight_list: list, n_list: list) -> dict:
    """Weighted average of several clients' weights.

    weight_list : one dict {parameter name: NumPy array} per client
    n_list      : number of training examples each client used (n_k)
    """
    if len(weight_list) == 0:
        raise ValueError("fedavg needs at least one client update.")
    if len(weight_list) != len(n_list):
        raise ValueError("weight_list and n_list must have the same length.")
    counts = np.asarray(n_list, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError("Every client must report n_k > 0.")

    reference = weight_list[0]
    for weights in weight_list[1:]:
        if weights.keys() != reference.keys():
            raise ValueError("Clients sent different parameter names.")
    for key in reference:
        for weights in weight_list:
            if weights[key].shape != reference[key].shape:
                raise ValueError(f"Shape mismatch for '{key}': "
                                 f"{weights[key].shape} vs {reference[key].shape}")

    shares = counts / counts.sum()                       # n_k / N
    averaged = {}
    for key in reference:
        stacked = np.stack([w[key].astype(np.float64) for w in weight_list])
        factor = shares.reshape((-1,) + (1,) * (stacked.ndim - 1))
        averaged[key] = (stacked * factor).sum(axis=0).astype(reference[key].dtype)
    return averaged


def aggregate_updates(update_payloads: list) -> dict:
    """Receive client updates as BYTES, decode them, and run FedAvg."""
    weights, counts = [], []
    for payload in update_payloads:
        client_weights, n_k = bytes_to_weights(payload)
        if n_k is None:
            raise ValueError("A client update is missing its example count.")
        weights.append(client_weights)
        counts.append(n_k)
    return fedavg(weights, counts)
