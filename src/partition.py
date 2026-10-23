"""Phase 5: give each simulated intersection (client) its own slice of the data.

Run from the project root:  python src/partition.py

Two ways to split the TRAINING windows among 4 clients:
  IID      : random, equal shares -> every client sees all hours and days.
  Non-IID  : by traffic regime    -> every client sees a different kind of traffic.
The TEST set stays one shared set, so all models are judged on the same data.
Each client also gets a "local test slice" (for client-level reports only).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocess as pp

ROOT = Path(__file__).resolve().parents[1]

NONIID_NAMES = [
    "weekday morning (05-12h)",
    "weekday afternoon/evening (12-20h)",
    "weekday night (20-05h)",
    "weekend (Sat/Sun)",
]


def regime_of(times) -> np.ndarray:
    """Label each window 0..3 using the calendar time of the hour being predicted."""
    ts = pd.DatetimeIndex(times)
    hour = ts.hour.to_numpy()
    weekday = ts.day_of_week.to_numpy() < 5
    regime = np.full(len(ts), -1)
    regime[weekday & (hour >= 5) & (hour < 12)] = 0
    regime[weekday & (hour >= 12) & (hour < 20)] = 1
    regime[weekday & ((hour >= 20) | (hour < 5))] = 2
    regime[~weekday] = 3
    assert (regime >= 0).all(), "every window must belong to exactly one regime"
    return regime


def iid_partition(n: int, num_clients: int, seed: int) -> list:
    """Shuffle all indices (with a fixed seed) and deal them out in equal shares."""
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(order, num_clients)]


def noniid_partition(times, num_clients: int) -> list:
    """One client per traffic regime. Deterministic: no seed needed."""
    if num_clients != 4:
        raise ValueError("The Non-IID rule is defined for exactly 4 clients.")
    regimes = regime_of(times)
    return [np.where(regimes == k)[0] for k in range(4)]


def partition_indices(times, scheme: str, num_clients: int, seed: int) -> list:
    if scheme == "iid":
        return iid_partition(len(times), num_clients, seed)
    if scheme == "noniid":
        return noniid_partition(times, num_clients)
    raise ValueError(f"Unknown scheme '{scheme}' (use 'iid' or 'noniid').")


def build_clients(arrays: dict, scheme: str, num_clients: int, seed: int) -> list:
    """Return one dict per client. This data never leaves the client in Phase 6."""
    train_idx = partition_indices(arrays["times_train"], scheme, num_clients, seed)
    test_idx = partition_indices(arrays["times_test"], scheme, num_clients, seed + 1000)
    clients = []
    for k, (tr, te) in enumerate(zip(train_idx, test_idx)):
        clients.append({
            "id": k,
            "name": NONIID_NAMES[k] if scheme == "noniid" else f"client {k}",
            "X_train": arrays["X_train"][tr],
            "y_train": arrays["y_train"][tr],
            "times_train": arrays["times_train"][tr],
            "n_train": int(len(tr)),
            "X_test_local": arrays["X_test"][te],
            "y_test_local": arrays["y_test"][te],
        })
    return clients


def describe_clients(clients: list, scaler: dict) -> dict:
    """Numbers that show how similar or different the clients are (vehicles/hour)."""
    total = sum(c["n_train"] for c in clients)
    rows = []
    for c in clients:
        y = pp.inverse_scale(c["y_train"].astype(np.float64), scaler)
        rows.append({"client": c["id"], "name": c["name"], "n_train": c["n_train"],
                     "share_percent": round(100 * c["n_train"] / total, 1),
                     "mean_traffic": float(y.mean()), "std_traffic": float(y.std()),
                     "n_test_local": int(len(c["y_test_local"]))})
    means = [r["mean_traffic"] for r in rows]
    return {"clients": rows, "spread_of_client_means": float(np.std(means))}


def plot_distributions(by_scheme: dict, scaler: dict, path: Path) -> None:
    """Average traffic by hour of day, one line per client, IID vs Non-IID."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    titles = {"iid": "IID: every client sees all hours",
              "noniid": "Non-IID: each client sees a different regime"}
    for ax, scheme in zip(axes, ["iid", "noniid"]):
        for c in by_scheme[scheme]:
            hours = pd.DatetimeIndex(c["times_train"]).hour.to_numpy()
            y = pp.inverse_scale(c["y_train"].astype(np.float64), scaler)
            mean_by_hour = pd.Series(y).groupby(hours).mean().reindex(range(24))
            ax.plot(range(24), mean_by_hour.to_numpy(), marker="o", ms=3,
                    label=c["name"])
        ax.set_title(titles[scheme])
        ax.set_xlabel("Hour of day (of the hour being predicted)")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Average traffic volume (vehicles/hour)")
    fig.suptitle("Simulated clients: training data by hour of day", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = pp.load_config()
    processed = ROOT / cfg["data"]["processed_dir"]
    if not (processed / "windows.npz").exists():
        print("Processed data not found. Run: python src/preprocess.py")
        return 1
    arrays, scaler = pp.load_processed(processed)
    num_clients, seed = cfg["clients"]["num_clients"], cfg["seeds"][0]

    summary, by_scheme = {}, {}
    for scheme in ["iid", "noniid"]:
        clients = build_clients(arrays, scheme, num_clients, seed)
        by_scheme[scheme] = clients
        summary[scheme] = describe_clients(clients, scaler)
        print(f"\n{scheme.upper()} partition (training windows, traffic in vehicles/hour)")
        print(f"  {'client':<38}{'n_train':>8}{'share':>8}{'mean':>9}{'std':>8}")
        for r in summary[scheme]["clients"]:
            print(f"  {r['name']:<38}{r['n_train']:>8}{r['share_percent']:>7.1f}%"
                  f"{r['mean_traffic']:>9.0f}{r['std_traffic']:>8.0f}")
        print(f"  spread of client means (std): {summary[scheme]['spread_of_client_means']:.0f}")

    out = ROOT / "results" / "partition_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed_used": seed, "num_clients": num_clients,
                               "summary": summary}, indent=2))
    fig_path = ROOT / "figures" / "client_distributions.png"
    plot_distributions(by_scheme, scaler, fig_path)
    print(f"\nSaved: {out}\nSaved: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
