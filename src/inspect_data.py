"""Phase 3a: look at the raw data BEFORE cleaning it.

Run from the project root:  python src/inspect_data.py
This script only reads and prints. It does not change any file.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    if not RAW_FILE.exists():
        print(f"File not found: {RAW_FILE}\nRun: python src/download_data.py")
        return 1

    # keep_default_na=False: the word "None" in the holiday column is a real label
    # (meaning "not a holiday"), not a missing value. Only truly empty cells become NaN.
    df = pd.read_csv(RAW_FILE, keep_default_na=False, na_values=[""])
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    section("1. Size")
    print(f"Rows: {len(df)}")
    print(f"Distinct timestamps: {df['date_time'].nunique()}")

    section("2. Duplicate timestamps")
    counts = df.groupby("date_time").size()
    repeated = counts[counts > 1]
    print(f"Timestamps that appear more than once: {len(repeated)}")
    print(f"Most times one timestamp repeats: {counts.max()}")
    differing = df[df["date_time"].isin(repeated.index)].groupby("date_time")["traffic_volume"].nunique()
    print(f"Repeated timestamps whose traffic_volume values DIFFER: {(differing > 1).sum()}")
    print(f"Repeated timestamps whose traffic_volume values are IDENTICAL: {(differing == 1).sum()}")
    if len(repeated) > 0:
        print("\nExample of one repeated timestamp:")
        example = repeated.index[0]
        print(df[df["date_time"] == example][["date_time", "weather_main", "weather_description", "traffic_volume"]])

    section("3. Missing hours (gaps in time)")
    times = pd.Series(df["date_time"].unique()).sort_values().reset_index(drop=True)
    expected = pd.date_range(times.iloc[0], times.iloc[-1], freq="h")
    print(f"Hours between first and last timestamp: {len(expected)}")
    print(f"Hours actually present: {len(times)}")
    print(f"Missing hours: {len(expected) - len(times)}")
    gaps = times.diff()
    big = gaps[gaps > pd.Timedelta(hours=1)]
    print(f"Number of gaps (jumps longer than 1 hour): {len(big)}")
    print("Five longest gaps (ends at / length):")
    for idx in big.sort_values(ascending=False).head(5).index:
        print(f"  gap ends at {times.iloc[idx]}  length {gaps.iloc[idx]}")

    section("4. Rows per year (check how much data each year has)")
    print(df.groupby(df["date_time"].dt.year).size().to_string())

    section("5. Suspicious values")
    print(f"traffic_volume == 0 : {(df['traffic_volume'] == 0).sum()} rows")
    print(f"temp == 0 (Kelvin)  : {(df['temp'] == 0).sum()} rows")
    print(f"rain_1h maximum     : {df['rain_1h'].max()}")
    print(f"snow_1h maximum     : {df['snow_1h'].max()}")
    print(f"Missing values per column:\n{df.isna().sum().to_string()}")
    print(f"\nHoliday values (top): \n{df['holiday'].value_counts().head(5).to_string()}")

    print("\nDone. Paste this whole output back before we write the cleaning code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
