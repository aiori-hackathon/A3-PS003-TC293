"""Phase 3b: clean, split, scale and window the traffic data.

Run from the project root:  python src/preprocess.py

Pipeline (the order matters):
  1. load_raw                -> read CSV, fix timestamps, sort by time
  2. one_reading_per_hour    -> merge duplicate timestamps
  3. fill_short_gaps         -> fill tiny gaps only, leave long gaps empty
  4. chronological_split     -> earlier 80% = train, later 20% = test
  5. fit_scaler (TRAIN only) -> mean/std from training hours only
  6. build_windows           -> (previous 24 hours + calendar) -> next hour
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "default.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- 1. load
def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV, parse timestamps, drop unusable rows, sort by time."""
    # keep_default_na=False: the word "None" in `holiday` is a real label.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    n_start = len(df)

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"])
    df = df[df["traffic_volume"] >= 0]  # a vehicle count cannot be negative

    dropped = n_start - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with bad timestamp or invalid traffic value")
    return df.sort_values("date_time").reset_index(drop=True)


# ------------------------------------------------------------ 2. duplicates
def one_reading_per_hour(df: pd.DataFrame) -> pd.Series:
    """Return ONE traffic_volume per hour, indexed by the hour.

    Duplicate timestamps in this dataset have identical traffic_volume (only
    the weather label differs), so averaging them keeps the same number.
    """
    hour = df["date_time"].dt.floor("h")
    off_hour = int((df["date_time"] != hour).sum())
    if off_hour:
        print(f"  {off_hour} timestamps were not exactly on the hour; rounded down")
    series = df.groupby(hour)["traffic_volume"].mean().astype(float)
    series.index.name = "date_time"
    return series


# ----------------------------------------------------------------- 3. gaps
def fill_short_gaps(series: pd.Series, max_fill_hours: int):
    """Put the series on a complete hourly grid; fill only SHORT gaps.

    A gap is filled (straight line between neighbours) only if the whole gap
    is <= max_fill_hours long. Longer gaps stay empty (NaN) so we never
    invent days of traffic.  Returns (series, number_of_hours_filled).
    """
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    s = series.reindex(full_index)

    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()           # label each run
    run_length = missing.groupby(run_id).transform("sum")    # length of each gap
    fillable = missing & (run_length <= max_fill_hours)

    interpolated = s.interpolate(method="linear", limit_area="inside")
    s = s.where(~fillable, interpolated)
    return s, int(fillable.sum())


# ---------------------------------------------------------------- 4. split
def chronological_split(series: pd.Series, train_fraction: float):
    """Split by time. Returns (train, test, cutoff_timestamp)."""
    valid = series.dropna()
    cut_position = int(len(valid) * train_fraction)
    cutoff = valid.index[cut_position]  # first hour of the test period
    return series[series.index < cutoff], series[series.index >= cutoff], cutoff


# --------------------------------------------------------------- 5. scaling
def fit_scaler(train: pd.Series) -> dict:
    """Learn mean and std from TRAINING hours only (prevents leakage)."""
    std = float(train.std())
    return {"mean": float(train.mean()), "std": std if std > 0 else 1.0}


def scale(values, scaler: dict):
    return (values - scaler["mean"]) / scaler["std"]


def inverse_scale(values, scaler: dict):
    """Convert scaled values back to vehicles per hour."""
    return values * scaler["std"] + scaler["mean"]


# --------------------------------------------------------------- 6. windows
def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Hour-of-day and day-of-week as sin/cos pairs (so 23:00 is near 00:00)."""
    hour = index.hour.to_numpy()
    day = index.dayofweek.to_numpy()
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7),
    ])


def build_windows(series: pd.Series, scaler: dict, window_length: int):
    """Turn a series into (X, y, target_times).

    X row = [previous `window_length` scaled hours] + [4 calendar features of
    the hour being predicted].  y = the scaled traffic of the NEXT hour.
    A window containing any missing hour is skipped, so no example ever
    spans a gap.
    """
    values = scale(series.to_numpy(dtype=float), scaler)
    if len(values) <= window_length:
        raise ValueError("Series is shorter than one window.")

    # each row: `window_length` inputs followed by 1 target
    windows = sliding_window_view(values, window_length + 1)
    valid = ~np.isnan(windows).any(axis=1)

    inputs = windows[valid, :window_length]
    targets = windows[valid, window_length]
    target_times = series.index[window_length:][valid]

    X = np.hstack([inputs, calendar_features(target_times)]).astype(np.float32)
    return X, targets.astype(np.float32), target_times.to_numpy()


# --------------------------------------------------------------------- main
def main() -> int:
    cfg = load_config()["data"]
    raw_path = ROOT / cfg["raw_file"]
    if not raw_path.exists():
        print(f"File not found: {raw_path}\nRun: python src/download_data.py")
        return 1

    print("1. Loading raw data ...")
    raw = load_raw(raw_path)
    print("2. Merging duplicate timestamps ...")
    series = one_reading_per_hour(raw)
    print("3. Handling gaps ...")
    hourly, n_filled = fill_short_gaps(series, cfg["max_gap_fill_hours"])
    print("4. Chronological split ...")
    train, test, cutoff = chronological_split(hourly, cfg["train_fraction"])
    print("5. Fitting scaler on training data only ...")
    scaler = fit_scaler(train)
    print("6. Building windows ...")
    L = cfg["window_length"]
    X_tr, y_tr, t_tr = build_windows(train, scaler, L)
    X_te, y_te, t_te = build_windows(test, scaler, L)

    out_dir = ROOT / cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X_train=X_tr, y_train=y_tr,
                        times_train=t_tr, X_test=X_te, y_test=y_te, times_test=t_te)
    (out_dir / "scaler.json").write_text(json.dumps(scaler, indent=2))

    summary = {
        "raw_rows": int(len(raw)),
        "distinct_hours_present": int(series.shape[0]),
        "hours_in_full_span": int(hourly.shape[0]),
        "missing_hours_before_filling": int(hourly.shape[0] - series.shape[0]),
        "hours_filled_by_interpolation": n_filled,
        "hours_still_missing": int(hourly.isna().sum()),
        "max_gap_fill_hours": cfg["max_gap_fill_hours"],
        "cutoff_first_test_hour": str(cutoff),
        "train_valid_hours": int(train.notna().sum()),
        "test_valid_hours": int(test.notna().sum()),
        "window_length": L,
        "train_windows": int(len(y_tr)),
        "test_windows": int(len(y_te)),
        "features_per_window": int(X_tr.shape[1]),
        "train_window_targets_from": str(t_tr[0]),
        "train_window_targets_to": str(t_tr[-1]),
        "test_window_targets_from": str(t_te[0]),
        "test_window_targets_to": str(t_te[-1]),
        "scaler_fit_on": "training hours only",
        "scaler": scaler,
        "features_used": "previous traffic volumes + hour/day-of-week encodings "
                         "(weather and holiday columns not used in the MVP)",
    }
    summary_path = ROOT / cfg["summary_file"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nPreprocessing summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {out_dir / 'windows.npz'}, {out_dir / 'scaler.json'}, {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
