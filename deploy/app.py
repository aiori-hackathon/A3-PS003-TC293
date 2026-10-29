"""Demo web app: replays saved HISTORICAL test data through the saved federated models.

Nothing here is live data and nothing is trained here. The app loads the final global
weights produced by src/federated.py (6,081 numbers per model) and runs them with NumPy.

Local use (from the project root):
    python deploy/prepare_demo_data.py     # once
    python deploy/app.py                   # then open http://127.0.0.1:5000
Hosting (for example Render): root directory deploy, build command
    pip install -r requirements.txt, start command  gunicorn app:app
"""
import datetime
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from markupsafe import escape

from demo_model import forward, load_weights

HERE = Path(__file__).resolve().parent
ALLOWED_HOURS = (24, 72, 168)
MODELS = (("iid", "FedAvg trained on IID clients"), ("noniid", "FedAvg trained on Non-IID clients"))
NEEDED = ["demo_windows.npz", "scaler.json", "fedavg_iid_seed0.npz", "fedavg_noniid_seed0.npz",
          "comparison_table.md", "communication_analysis.json"]
CAPTIONS = {
    "convergence.png": "Test error after each communication round (mean and spread over 3 seeds).",
    "method_comparison.png": "All methods on the same test period.",
    "communication.png": "Measured bytes sent by federated training compared with the raw dataset file.",
    "client_level.png": "Error on each simulated client's own test slice.",
    "actual_vs_predicted.png": "One week of real traffic and the two federated models' one-hour-ahead forecasts.",
    "client_distributions.png": "Average traffic by hour of day for each simulated client (IID vs Non-IID).",
}


def read_table(path: Path):
    """Parse a markdown table into (header cells, list of row cells)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):          # the |---|---| separator row
            continue
        rows.append(cells)
    return rows[0], rows[1:]


def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def create_app(data_dir=None) -> Flask:
    data_dir = Path(data_dir) if data_dir else HERE / "data"
    missing = [n for n in NEEDED if not (data_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"Demo data missing in {data_dir}: {', '.join(missing)}. "
                                f"Run: python deploy/prepare_demo_data.py")

    app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))

    with np.load(data_dir / "demo_windows.npz", allow_pickle=False) as d:
        X, y, times = d["X"], d["y"], d["times"].astype("datetime64[h]")
    scaler = json.loads((data_dir / "scaler.json").read_text())
    weights = {key: load_weights(data_dir / f"fedavg_{key}_seed0.npz") for key, _ in MODELS}
    table_header, table_rows = read_table(data_dir / "comparison_table.md")
    comm = json.loads((data_dir / "communication_analysis.json").read_text())
    n_params = int(sum(v.size for v in weights["iid"].values()))

    days = times.astype("datetime64[D]")
    unique_days, counts = np.unique(days, return_counts=True)
    complete = unique_days[counts == 24]
    default_day = str(complete[0] if len(complete) else unique_days[0])
    first_day, last_day = str(unique_days[0]), str(unique_days[-1])

    def to_units(values):
        return np.asarray(values, dtype=np.float64) * scaler["std"] + scaler["mean"]

    @app.route("/")
    def index():
        fig_dir = Path(app.static_folder) / "figures"
        figures = [(n, CAPTIONS[n]) for n in CAPTIONS if (fig_dir / n).exists()]
        return render_template("index.html", table_header=table_header, table_rows=table_rows,
                               comm=comm["measured"], scenarios=comm["assumption_based_scenarios"],
                               figures=figures, first_day=first_day, last_day=last_day,
                               default_day=default_day, n_params=n_params, models=MODELS)

    @app.route("/healthz")
    def healthz():
        return "ok"

    @app.route("/api/info")
    def info():
        return jsonify({"first_day": first_day, "last_day": last_day, "default_day": default_day,
                        "test_windows": int(len(y)), "model_parameters": n_params,
                        "allowed_hours": list(ALLOWED_HOURS)})

    @app.route("/api/forecast")
    def forecast():
        raw_date = request.args.get("date", "")
        try:
            day = datetime.date.fromisoformat(raw_date)
        except ValueError:
            return jsonify({"error": "Please give a date as YYYY-MM-DD."}), 400
        try:
            hours = int(request.args.get("hours", 24))
        except ValueError:
            return jsonify({"error": "hours must be a whole number."}), 400
        if hours not in ALLOWED_HOURS:
            return jsonify({"error": f"hours must be one of {list(ALLOWED_HOURS)}."}), 400

        start = np.datetime64(day.isoformat(), "h")
        mask = (times >= start) & (times < start + np.timedelta64(hours, "h"))
        count = int(mask.sum())
        if count == 0:
            return jsonify({"error": f"No test data for {day.isoformat()}. The test period runs from "
                                     f"{first_day} to {last_day}; a few hours inside it are missing in the "
                                     f"original data."}), 404

        actual = to_units(y[mask])
        predictions = {key: to_units(forward(weights[key], X[mask])) for key, _ in MODELS}
        stamps = [str(t).replace("T", " ") + ":00" for t in times[mask]]
        note = "" if count == hours else (f"Only {count} of {hours} hours are available for this period "
                                          f"(some hours are missing in the original data).")
        return jsonify({
            "date": day.isoformat(), "hours_requested": hours, "count": count, "times": stamps,
            "actual": np.round(actual, 1).tolist(),
            "predictions": {k: np.round(v, 1).tolist() for k, v in predictions.items()},
            "metrics": {k: {"mae": round(mae(actual, v), 1), "rmse": round(rmse(actual, v), 1)}
                        for k, v in predictions.items()},
            "note": note})

    return app


try:
    app = create_app()
except FileNotFoundError as error:                    # lets the host show a clear message
    app = Flask(__name__)
    _message = str(error)

    @app.route("/")
    def missing_data():
        return f"<h1>Demo data missing</h1><p>{escape(_message)}</p>", 500

    @app.route("/healthz")
    def missing_health():
        return "demo data missing", 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
