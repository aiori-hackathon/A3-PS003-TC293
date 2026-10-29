"""Tests for deploy/app.py, deploy/demo_model.py and deploy/prepare_demo_data.py.

They build their own small FAKE demo data in a temporary folder, so they do not
depend on your real results. Needs Flask (skipped if it is not installed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "src"))
import app as demo_app  # noqa: E402
import demo_model  # noqa: E402
import numpy_model  # noqa: E402
import prepare_demo_data as prep  # noqa: E402

SHAPES = {"net.0.weight": (64, 28), "net.0.bias": (64,), "net.2.weight": (64, 64),
          "net.2.bias": (64,), "net.4.weight": (1, 64), "net.4.bias": (1,)}


def fake_weights(seed):
    rng = np.random.default_rng(seed)
    return {k: (rng.normal(size=s) * 0.2).astype(np.float32) for k, s in SHAPES.items()}


def fake_times(days=5, drop=(30, 31, 32)):
    start = np.datetime64("2018-01-01T00", "h")
    times = start + np.arange(24 * days).astype("timedelta64[h]")
    return np.delete(times, list(drop))          # 3 missing hours on 2 Jan (06:00-08:00)


def write_demo_dir(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    times = fake_times()
    np.savez(folder / "demo_windows.npz", X=rng.normal(size=(len(times), 28)).astype(np.float32),
             y=rng.normal(size=len(times)).astype(np.float32), times=times)
    (folder / "scaler.json").write_text(json.dumps({"mean": 3000.0, "std": 1900.0}))
    for scheme, seed in (("iid", 1), ("noniid", 2)):
        np.savez(folder / f"fedavg_{scheme}_seed0.npz", **fake_weights(seed))
    (folder / "comparison_table.md").write_text(
        "| Method | MAE | RMSE | Bytes |\n|---|---|---|---|\n| Method A | 1.0 | 2.0 | 0 |\n| Method B | 3.0 | 4.0 | 5 |\n")
    (folder / "communication_analysis.json").write_text(json.dumps({
        "measured": {"bytes_up_per_round": 104520, "bytes_down_per_round": 103480, "bytes_per_round": 208000,
                     "rounds": 30, "total_bytes": 6240000, "raw_dataset_file_bytes": 3237208,
                     "total_over_raw_file": 1.93, "break_even_round_vs_raw_file": 15.6,
                     "bytes_per_client_whole_run": 1560000},
        "assumption_based_scenarios": {"label": "ASSUMPTION, not measured: example", "rows": [
            {"assumed_stream_mbit_per_s": 1.0,
             "seconds_of_streaming_equal_to_one_clients_whole_federated_run": 12.48}]}}))


@pytest.fixture()
def client(tmp_path):
    write_demo_dir(tmp_path / "data")
    return demo_app.create_app(data_dir=tmp_path / "data").test_client()


def test_health_and_info(client):
    assert client.get("/healthz").data == b"ok"
    info = client.get("/api/info").get_json()
    assert info["first_day"] == "2018-01-01" and info["last_day"] == "2018-01-05"
    assert info["model_parameters"] == 6081 and info["default_day"] == "2018-01-01"


def test_forecast_returns_aligned_series_and_metrics(client):
    data = client.get("/api/forecast?date=2018-01-03&hours=24").get_json()
    assert data["count"] == 24 and data["note"] == ""
    assert len(data["times"]) == len(data["actual"]) == len(data["predictions"]["iid"]) == 24
    assert set(data["predictions"]) == {"iid", "noniid"}
    assert all(m["mae"] >= 0 and m["rmse"] >= m["mae"] for m in data["metrics"].values())
    assert data["times"][0] == "2018-01-03 00:00"


def test_forecast_equals_the_numpy_model_on_the_same_rows(tmp_path):
    write_demo_dir(tmp_path / "data")
    app = demo_app.create_app(data_dir=tmp_path / "data")
    data = app.test_client().get("/api/forecast?date=2018-01-04&hours=24").get_json()
    saved = np.load(tmp_path / "data" / "demo_windows.npz")
    mask = (saved["times"] >= np.datetime64("2018-01-04T00", "h")) & (saved["times"] < np.datetime64("2018-01-05T00", "h"))
    manual = demo_model.forward(demo_model.load_weights(tmp_path / "data" / "fedavg_iid_seed0.npz"), saved["X"][mask])
    manual = manual.astype(np.float64) * 1900.0 + 3000.0
    assert np.allclose(data["predictions"]["iid"], manual, atol=0.06)       # values are rounded to 0.1
    assert np.allclose(data["actual"], saved["y"][mask].astype(np.float64) * 1900.0 + 3000.0, atol=0.06)


def test_missing_hours_are_reported_honestly(client):
    data = client.get("/api/forecast?date=2018-01-02&hours=24").get_json()
    assert data["count"] == 21 and "21 of 24" in data["note"]


def test_longer_windows_work(client):
    data = client.get("/api/forecast?date=2018-01-01&hours=72").get_json()
    assert data["count"] == 72 - 3        # 3 hours missing on 2 Jan


def test_bad_requests_get_clear_errors(client):
    assert client.get("/api/forecast").status_code == 400
    assert client.get("/api/forecast?date=banana").status_code == 400
    assert client.get("/api/forecast?date=2018-01-02&hours=5").status_code == 400
    assert client.get("/api/forecast?date=2018-01-02&hours=abc").status_code == 400
    assert client.get("/api/forecast?date=2030-01-01&hours=24").status_code == 404
    assert "error" in client.get("/api/forecast?date=2030-01-01").get_json()


def test_home_page_shows_the_honesty_notes_and_results(client):
    page = client.get("/").get_data(as_text=True)
    assert "simulated" in page and "historical" in page and "not full privacy" in page
    assert "Method A" in page and "208,000" in page and "1.93" in page


def test_missing_data_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        demo_app.create_app(data_dir=tmp_path)
    assert "prepare_demo_data.py" in str(error.value)


def test_demo_model_matches_the_src_numpy_model():
    weights = fake_weights(3)
    X = np.random.default_rng(1).normal(size=(40, 28)).astype(np.float32)
    assert np.array_equal(demo_model.forward(weights, X), numpy_model.forward(weights, X))


def test_prepare_script_collects_everything_and_the_app_can_use_it(tmp_path):
    project, deploy = tmp_path / "project", tmp_path / "deploy"
    (project / "data" / "processed").mkdir(parents=True)
    (project / "results" / "models").mkdir(parents=True)
    (project / "figures").mkdir()
    rng = np.random.default_rng(0)
    times = fake_times()
    np.savez(project / "data" / "processed" / "windows.npz",
             X_train=np.zeros((3, 28), np.float32), y_train=np.zeros(3, np.float32), times_train=times[:3],
             X_test=rng.normal(size=(len(times), 28)).astype(np.float32),
             y_test=rng.normal(size=len(times)).astype(np.float32), times_test=times.astype("datetime64[ns]"))
    (project / "data" / "processed" / "scaler.json").write_text(json.dumps({"mean": 3000.0, "std": 1900.0}))
    for scheme, seed in (("iid", 1), ("noniid", 2)):
        np.savez(project / "results" / "models" / f"fedavg_{scheme}_seed0.npz", **fake_weights(seed))
    source = tmp_path / "source"
    write_demo_dir(source)
    for name in ("comparison_table.md", "communication_analysis.json"):
        (project / "results" / name).write_text((source / name).read_text())
    (project / "figures" / "convergence.png").write_bytes(b"not-a-real-png")

    summary = prep.prepare(project, deploy)
    assert summary["test_windows"] == len(times) and summary["figures"] == ["convergence.png"]
    assert (deploy / "static" / "figures" / "convergence.png").exists()
    app = demo_app.create_app(data_dir=deploy / "data")
    assert app.test_client().get("/api/forecast?date=2018-01-03&hours=24").status_code == 200


def test_prepare_script_explains_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        prep.prepare(tmp_path, tmp_path / "deploy")
    assert "preprocess.py" in str(error.value)
