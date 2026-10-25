"""Phase 6: federated training with FedAvg on the simulated clients.

Run from the project root:
  python src/federated.py --quick                 # 3 rounds, 1 seed, nothing saved
  python src/federated.py --scheme iid            # full run, IID,     3 seeds
  python src/federated.py --scheme noniid         # full run, Non-IID, 3 seeds
  python src/federated.py                         # both schemes

One ROUND:
  1. server  -> every client : the global weights (as bytes)
  2. each client trains locally on its own data
  3. each client -> server   : its new weights + n_k (as bytes)
  4. server runs FedAvg and produces the next global weights
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import preprocess as pp
from client import load_numpy_weights, local_train, weights_to_numpy
from comm import bytes_to_weights, weights_to_bytes
from metrics import evaluate_scaled
from model import TrafficMLP, predict, set_seed
from partition import build_clients
from server import aggregate_updates

ROOT = Path(__file__).resolve().parents[1]


def client_seed(seed: int, round_number: int, client_id: int) -> int:
    """A different, repeatable seed for every (run, round, client)."""
    return seed * 10_000 + round_number * 100 + client_id


def run_round(global_weights: dict, clients: list, round_number: int, seed: int,
              cfg: dict, local_epochs: int):
    """One federated round. Returns (new_global_weights, communication stats)."""
    hidden = cfg["model"]["hidden_sizes"]
    lr, batch_size = cfg["training"]["learning_rate"], cfg["training"]["batch_size"]

    global_bytes = weights_to_bytes(global_weights)             # server -> clients
    bytes_down = len(global_bytes) * len(clients)               # one copy per client

    payloads, losses = [], []
    for client in clients:
        received, _ = bytes_to_weights(global_bytes)            # client decodes it
        local_weights, n_k, loss = local_train(
            received, client, hidden, local_epochs, lr, batch_size,
            client_seed(seed, round_number, client["id"]))
        payloads.append(weights_to_bytes(local_weights, n_examples=n_k))  # client -> server
        if loss is not None:
            losses.append(loss)

    new_global = aggregate_updates(payloads)                    # server sees ONLY bytes
    stats = {"bytes_down": bytes_down,
             "bytes_up": sum(len(p) for p in payloads),
             "mean_local_loss": float(np.mean(losses)) if losses else None}
    return new_global, stats


def _evaluate(model, X, y, scaler):
    if len(y) == 0:
        return {"mae": None, "rmse": None}
    return evaluate_scaled(y, predict(model, X), scaler)


def run_federated(arrays: dict, scaler: dict, scheme: str, seed: int, cfg: dict,
                  rounds: int, local_epochs: int, verbose: bool = False) -> dict:
    set_seed(seed)
    clients = build_clients(arrays, scheme, cfg["clients"]["num_clients"], seed)
    input_size, hidden = arrays["X_train"].shape[1], cfg["model"]["hidden_sizes"]

    global_weights = weights_to_numpy(TrafficMLP(input_size, hidden))   # shared start
    eval_model = TrafficMLP(input_size, hidden)

    history, cumulative = [], 0
    for r in range(1, rounds + 1):
        global_weights, stats = run_round(global_weights, clients, r, seed, cfg, local_epochs)
        cumulative += stats["bytes_up"] + stats["bytes_down"]

        load_numpy_weights(eval_model, global_weights)
        overall = _evaluate(eval_model, arrays["X_test"], arrays["y_test"], scaler)
        per_client = [_evaluate(eval_model, c["X_test_local"], c["y_test_local"], scaler)
                      for c in clients]
        history.append({
            "round": r, "global_mae": overall["mae"], "global_rmse": overall["rmse"],
            "client_mae": [m["mae"] for m in per_client],
            "client_rmse": [m["rmse"] for m in per_client],
            "cumulative_bytes": cumulative, **stats})
        if verbose and (r == 1 or r % 5 == 0 or r == rounds):
            print(f"  [{scheme}] seed {seed} | round {r:3d}/{rounds} | "
                  f"MAE {overall['mae']:7.1f} | RMSE {overall['rmse']:7.1f} | "
                  f"bytes so far {cumulative:,}")
    return {"history": history, "final_weights": global_weights,
            "client_sizes": [c["n_train"] for c in clients],
            "client_names": [c["name"] for c in clients]}


def mean_std(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["iid", "noniid", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="fast smoke test")
    args = parser.parse_args()

    cfg = pp.load_config()
    processed = ROOT / cfg["data"]["processed_dir"]
    if not (processed / "windows.npz").exists():
        print("Processed data not found. Run: python src/preprocess.py")
        return 1
    arrays, scaler = pp.load_processed(processed)

    rounds = 3 if args.quick else cfg["federated"]["rounds"]
    local_epochs = cfg["federated"]["local_epochs"]
    seeds = cfg["seeds"][:1] if args.quick else cfg["seeds"]
    schemes = ["iid", "noniid"] if args.scheme == "both" else [args.scheme]
    raw_path = ROOT / cfg["data"]["raw_file"]
    raw_bytes = raw_path.stat().st_size if raw_path.exists() else None

    for scheme in schemes:
        print(f"\n=== FedAvg, {scheme.upper()} clients: {rounds} rounds x "
              f"{local_epochs} local epochs, seeds {seeds} ===")
        runs, started = {}, time.time()
        for seed in seeds:
            result = run_federated(arrays, scaler, scheme, seed, cfg, rounds,
                                   local_epochs, verbose=True)
            runs[str(seed)] = result["history"]
            if not args.quick:
                model_path = ROOT / "results" / "models" / f"fedavg_{scheme}_seed{seed}.npz"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(weights_to_bytes(result["final_weights"]))
        elapsed = time.time() - started

        finals = [runs[str(s)][-1] for s in seeds]
        summary = {"mae": mean_std([f["global_mae"] for f in finals]),
                   "rmse": mean_std([f["global_rmse"] for f in finals])}
        first = runs[str(seeds[0])][0]
        total_bytes = finals[0]["cumulative_bytes"]
        print(f"\nFinal-round results on the TEST period (vehicles per hour)")
        print(f"  MAE  = {summary['mae']['mean']:.1f} +/- {summary['mae']['std']:.1f}")
        print(f"  RMSE = {summary['rmse']['mean']:.1f} +/- {summary['rmse']['std']:.1f}")
        print(f"  time: {elapsed:.0f}s")
        print("\nCommunication (actual serialized payload bytes)")
        print(f"  per round : {first['bytes_up']:,} up + {first['bytes_down']:,} down "
              f"= {first['bytes_up'] + first['bytes_down']:,}")
        print(f"  all {rounds} rounds : {total_bytes:,} bytes ({total_bytes / 1e6:.2f} MB)")
        if raw_bytes:
            print(f"  raw dataset file on disk: {raw_bytes:,} bytes ({raw_bytes / 1e6:.2f} MB)")
            print(f"  federated total / raw file size = {total_bytes / raw_bytes:.2f}")

        if args.quick:
            print("\n(--quick run: results NOT saved)")
            continue
        out = ROOT / "results" / f"federated_{scheme}.json"
        out.write_text(json.dumps({
            "setting": f"fedavg_{scheme}", "rounds": rounds, "local_epochs": local_epochs,
            "seeds": seeds, "summary_final_round": summary,
            "std_type": "sample standard deviation across seeds",
            "raw_dataset_file_bytes": raw_bytes,
            "config": cfg, "per_seed_history": runs}, indent=2))
        print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
