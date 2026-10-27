"""Phase 7b: comparison table, communication analysis and the five figures.

Run from the project root:  python src/analysis.py
Reads the saved result files in results/ (no PyTorch needed). Anything that is
missing is skipped with a message, so you can run it at any time.

Writes:
  results/comparison_table.md          one table: all methods side by side
  results/communication_analysis.json  measured bytes + break-even + labeled assumptions
  figures/convergence.png              error vs communication round
  figures/method_comparison.png        all methods side by side
  figures/communication.png            cumulative bytes vs the raw file size
  figures/client_level.png             per-client errors
  figures/actual_vs_predicted.png      one week: real traffic vs model forecasts
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocess as pp
from numpy_model import forward, load_weights
from partition import NONIID_NAMES

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
BLUE, ORANGE, GRAY, RED, GREEN = "#1f77b4", "#ff7f0e", "#7f7f7f", "#d62728", "#2ca02c"
SCHEMES = (("iid", "IID"), ("noniid", "Non-IID"))


# ------------------------------------------------------------------ helpers
def load_json(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt(stat: dict, digits: int = 1) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def curve_stats(fed: dict, key: str):
    """Mean and sample std across seeds, for every round."""
    runs = np.array([[h[key] for h in history] for history in fed["per_seed_history"].values()])
    std = runs.std(axis=0, ddof=1) if runs.shape[0] > 1 else np.zeros(runs.shape[1])
    return runs.mean(axis=0), std


def total_bytes(fed: dict) -> int:
    return next(iter(fed["per_seed_history"].values()))[-1]["cumulative_bytes"]


def find_consecutive_run(times, length: int = 168):
    """Start index and size of the first run of `length` consecutive hourly
    targets. If no run is that long, returns the longest run instead."""
    t = pd.DatetimeIndex(times)
    if len(t) == 0:
        return 0, 0
    breaks = np.where(np.diff(t.values) != np.timedelta64(1, "h"))[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(t)]])
    lengths = ends - starts
    long_enough = np.where(lengths >= length)[0]
    i = int(long_enough[0]) if len(long_enough) else int(np.argmax(lengths))
    return int(starts[i]), int(min(length, lengths[i]))


# ------------------------------------------------------------ table + bytes
def comparison_rows(naive, central, local, fed) -> list:
    rows = []
    if naive:
        for key, label in (("training_average", "Always predict the training average"),
                           ("last_hour", "Repeat the last hour"),
                           ("same_hour_yesterday", "Same hour yesterday")):
            rows.append((label, f"{naive[key]['mae']:.1f}", f"{naive[key]['rmse']:.1f}", "0"))
    if central:
        s = central["summary_final_epoch"]
        rows.append(("Centralized MLP (all data in one place)", fmt(s["mae"]), fmt(s["rmse"]),
                     "raw data must be uploaded"))
    for scheme, label in SCHEMES:
        if local.get(scheme):
            s = local[scheme]["summary"]
            rows.append((f"Local-only, {label} clients (mean of 4 models)",
                         fmt(s["global_test_mae"]), fmt(s["global_test_rmse"]), "0"))
    for scheme, label in SCHEMES:
        if fed.get(scheme):
            s = fed[scheme]["summary_final_round"]
            rows.append((f"FedAvg, {label} clients", fmt(s["mae"]), fmt(s["rmse"]),
                         f"{total_bytes(fed[scheme]):,}"))
    return rows


def markdown_table(rows: list) -> str:
    lines = ["| Method | MAE (veh/h) | RMSE (veh/h) | Bytes sent over training |",
             "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    return "\n".join(lines)


def communication_analysis(fed: dict, raw_bytes: int, num_clients: int = 4) -> dict:
    history = next(iter(fed["per_seed_history"].values()))
    per_round = history[0]["bytes_up"] + history[0]["bytes_down"]
    total = history[-1]["cumulative_bytes"]
    per_client = total / num_clients
    scenarios = []
    for mbit in (0.5, 1.0, 2.0):
        bytes_per_second = mbit * 1e6 / 8
        scenarios.append({
            "assumed_stream_mbit_per_s": mbit,
            "bytes_per_hour_per_intersection": bytes_per_second * 3600,
            "seconds_of_streaming_equal_to_one_clients_whole_federated_run": per_client / bytes_per_second})
    return {
        "measured": {
            "bytes_up_per_round": history[0]["bytes_up"],
            "bytes_down_per_round": history[0]["bytes_down"],
            "bytes_per_round": per_round, "rounds": len(history), "total_bytes": total,
            "raw_dataset_file_bytes": raw_bytes,
            "total_over_raw_file": total / raw_bytes,
            "break_even_round_vs_raw_file": raw_bytes / per_round,
            "bytes_per_client_whole_run": per_client},
        "assumption_based_scenarios": {
            "label": "ASSUMPTION, not measured: each intersection would otherwise stream "
                     "raw video-like data continuously at the stated bit rate",
            "rows": scenarios}}


# ------------------------------------------------------------------- plots
def plot_convergence(fed, central, naive, path) -> bool:
    if not any(fed.values()):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, name in ((axes[0], "global_mae", "MAE"), (axes[1], "global_rmse", "RMSE")):
        for (scheme, label), color in zip(SCHEMES, (BLUE, ORANGE)):
            if not fed.get(scheme):
                continue
            mean, std = curve_stats(fed[scheme], key)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=color, label=f"FedAvg, {label}")
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        if central:
            ax.axhline(central["summary_final_epoch"][name.lower()]["mean"], color=GRAY,
                       ls="--", label="Centralized (final epoch)")
        if naive:
            ax.axhline(min(v[name.lower()] for v in naive.values()), color=RED, ls=":",
                       label="Best simple baseline")
        ax.set_xlabel("Communication round")
        ax.set_ylabel(f"{name} on the test period (vehicles/hour)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Federated training: error after each round (mean and spread over 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_method_comparison(naive, central, local, fed, path) -> bool:
    entries = []   # (label, mae, mae_std, rmse, rmse_std, color)
    if naive:
        entries.append(("Best simple baseline", min(v["mae"] for v in naive.values()), 0.0,
                        min(v["rmse"] for v in naive.values()), 0.0, RED))
    for scheme, label in SCHEMES:
        if local.get(scheme):
            s = local[scheme]["summary"]
            entries.append((f"Local-only, {label}", s["global_test_mae"]["mean"], s["global_test_mae"]["std"],
                            s["global_test_rmse"]["mean"], s["global_test_rmse"]["std"], GRAY))
    for scheme, label in reversed(SCHEMES):
        if fed.get(scheme):
            s = fed[scheme]["summary_final_round"]
            entries.append((f"FedAvg, {label}", s["mae"]["mean"], s["mae"]["std"],
                            s["rmse"]["mean"], s["rmse"]["std"], BLUE if scheme == "iid" else ORANGE))
    if central:
        s = central["summary_final_epoch"]
        entries.append(("Centralized", s["mae"]["mean"], s["mae"]["std"],
                        s["rmse"]["mean"], s["rmse"]["std"], GREEN))
    if not entries:
        return False
    y = np.arange(len(entries))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.65 * len(entries) + 2.6), sharey=True)
    for ax, (vi, si, name) in zip(axes, ((1, 2, "MAE"), (3, 4, "RMSE"))):
        ax.barh(y, [e[vi] for e in entries], xerr=[e[si] for e in entries],
                color=[e[5] for e in entries], capsize=3)
        xmax = max(e[vi] + e[si] for e in entries)
        ax.set_xlim(0, xmax * 1.18)
        for yi, e in zip(y, entries):
            ax.text(e[vi] + e[si] + xmax * 0.015, yi, f"{e[vi]:.0f}", va="center", fontsize=9)
        ax.set_xlabel(f"{name} on the test period (vehicles/hour, lower is better)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([e[0] for e in entries])
    axes[0].invert_yaxis()
    fig.suptitle("All methods on the same test period")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_communication(fed, analysis, path) -> bool:
    if analysis is None:
        return False
    m = analysis["measured"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (scheme, label), color in zip(SCHEMES, (BLUE, ORANGE)):
        if fed.get(scheme):
            history = next(iter(fed[scheme]["per_seed_history"].values()))
            ax.plot([h["round"] for h in history], [h["cumulative_bytes"] / 1e6 for h in history],
                    color=color, label=f"FedAvg, {label} (measured)", lw=2 if scheme == "iid" else 1.2,
                    ls="-" if scheme == "iid" else "--")
    ax.axhline(m["raw_dataset_file_bytes"] / 1e6, color=RED, ls=":",
               label=f"Raw dataset file, uploaded once ({m['raw_dataset_file_bytes'] / 1e6:.2f} MB)")
    ax.axvline(m["break_even_round_vs_raw_file"], color=GRAY, ls=":")
    ax.annotate(f"federated traffic passes the raw file\nsize after about round {m['break_even_round_vs_raw_file']:.0f}",
                xy=(m["break_even_round_vs_raw_file"], m["raw_dataset_file_bytes"] / 1e6),
                xytext=(m["break_even_round_vs_raw_file"] + 1.5, m["raw_dataset_file_bytes"] / 1e6 * 0.45),
                fontsize=9, arrowprops={"arrowstyle": "->", "color": GRAY})
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative bytes sent, all clients (MB)")
    ax.set_title("Measured communication cost of federated training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_client_level(fed, local, path) -> bool:
    panels = [(s, l) for s, l in SCHEMES if fed.get(s)]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 4.9), squeeze=False)
    for ax, (scheme, label) in zip(axes[0], panels):
        fed_mae = np.mean([h[-1]["client_mae"] for h in fed[scheme]["per_seed_history"].values()], axis=0)
        names = NONIID_NAMES if scheme == "noniid" else [f"client {i}" for i in range(len(fed_mae))]
        x, width = np.arange(len(names)), 0.38
        bars = [ax.bar(x - width / 2, fed_mae, width, color=BLUE if scheme == "iid" else ORANGE,
                       label="FedAvg global model")]
        if local.get(scheme):
            local_mae = np.mean([[c["local_mae"] for c in rows]
                                 for rows in local[scheme]["per_seed"].values()], axis=0)
            bars.append(ax.bar(x + width / 2, local_mae, width, color=GRAY, label="Local-only (own model)"))
        for group in bars:
            ax.bar_label(group, fmt="%.0f", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(n, 16, break_long_words=False) for n in names], fontsize=8)
        ax.set_ylabel("MAE on the client's own test slice (vehicles/hour)")
        ax.set_title(f"{label} clients")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Client-level performance (final round, mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_actual_vs_predicted(arrays, scaler, path) -> bool:
    models = [(s, l, RESULTS / "models" / f"fedavg_{s}_seed0.npz") for s, l in SCHEMES]
    models = [m for m in models if m[2].exists()]
    if not models:
        return False
    times = pd.DatetimeIndex(arrays["times_test"])
    start, size = find_consecutive_run(times, 168)
    sl = slice(start, start + size)
    X = arrays["X_test"][sl]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(times[sl], pp.inverse_scale(arrays["y_test"][sl].astype(np.float64), scaler),
            color="black", lw=1.8, label="Actual traffic")
    for (scheme, label, file), color in zip(models, (BLUE, ORANGE)):
        pred = pp.inverse_scale(forward(load_weights(file), X).astype(np.float64), scaler)
        ax.plot(times[sl], pred, color=color, lw=1.2, ls="-" if scheme == "iid" else "--",
                label=f"FedAvg, {label} (seed 0)")
    ax.set_ylabel("Traffic volume (vehicles/hour)")
    ax.set_title(f"{size} consecutive test hours starting {times[start]:%d %b %Y}: "
                 f"actual vs forecast one hour ahead")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- main
def main() -> int:
    naive, central = load_json("naive_baselines.json"), load_json("centralized_results.json")
    fed = {s: load_json(f"federated_{s}.json") for s, _ in SCHEMES}
    local = {s: load_json(f"local_only_{s}.json") for s, _ in SCHEMES}
    for name, obj in (("naive_baselines", naive), ("centralized_results", central),
                      *((f"federated_{s}", fed[s]) for s, _ in SCHEMES),
                      *((f"local_only_{s}", local[s]) for s, _ in SCHEMES)):
        if obj is None:
            print(f"  (skipping: results/{name}.json not found yet)")

    rows = comparison_rows(naive, central, local, fed)
    if rows:
        table = markdown_table(rows)
        (RESULTS / "comparison_table.md").write_text(table + "\n")
        print("\nComparison table (test period, vehicles per hour)\n")
        print(table)

    analysis = None
    any_fed = next((f for f in fed.values() if f), None)
    if any_fed and any_fed.get("raw_dataset_file_bytes"):
        analysis = communication_analysis(any_fed, any_fed["raw_dataset_file_bytes"])
        (RESULTS / "communication_analysis.json").write_text(json.dumps(analysis, indent=2))
        m = analysis["measured"]
        print(f"\nCommunication (measured): {m['bytes_per_round']:,} bytes per round, "
              f"{m['total_bytes']:,} bytes over {m['rounds']} rounds")
        print(f"  = {m['total_over_raw_file']:.2f} x the raw dataset file "
              f"({m['raw_dataset_file_bytes']:,} bytes); break-even at round "
              f"{m['break_even_round_vs_raw_file']:.1f}")
        print("  " + analysis["assumption_based_scenarios"]["label"])
        for r in analysis["assumption_based_scenarios"]["rows"]:
            print(f"    at {r['assumed_stream_mbit_per_s']} Mbit/s, one client's whole federated "
                  f"run equals {r['seconds_of_streaming_equal_to_one_clients_whole_federated_run']:.0f} "
                  f"seconds of streaming")

    FIGURES.mkdir(parents=True, exist_ok=True)
    arrays = scaler = None
    processed = ROOT / pp.load_config()["data"]["processed_dir"]
    if (processed / "windows.npz").exists():
        arrays, scaler = pp.load_processed(processed)
    jobs = [("convergence.png", lambda p: plot_convergence(fed, central, naive, p)),
            ("method_comparison.png", lambda p: plot_method_comparison(naive, central, local, fed, p)),
            ("communication.png", lambda p: plot_communication(fed, analysis, p)),
            ("client_level.png", lambda p: plot_client_level(fed, local, p)),
            ("actual_vs_predicted.png",
             lambda p: arrays is not None and plot_actual_vs_predicted(arrays, scaler, p))]
    print()
    for name, job in jobs:
        print(f"  figures/{name}: {'saved' if job(FIGURES / name) else 'skipped (inputs missing)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
