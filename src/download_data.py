"""Download the Metro Interstate Traffic Volume dataset (UCI) into data/raw/.

Run from the project root:  python src/download_data.py

If the automatic download fails, the script prints manual instructions.
Dataset page: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
"""
import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZIP_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = RAW_DIR / "Metro_Interstate_Traffic_Volume.csv"
REQUIRED_COLUMNS = {"date_time", "traffic_volume", "temp", "rain_1h",
                    "snow_1h", "clouds_all", "weather_main", "holiday"}


def download_zip(url: str = ZIP_URL, timeout: int = 60) -> bytes:
    """Download the zip file and return its raw bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_csv_bytes(zip_bytes: bytes) -> bytes:
    """Find the CSV (plain or .gz) inside the zip and return its text bytes."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        candidates = [n for n in names if n.lower().endswith((".csv", ".csv.gz"))]
        if not candidates:
            raise FileNotFoundError(f"No CSV file found inside the zip. Contents: {names}")
        name = candidates[0]
        data = archive.read(name)
    if name.lower().endswith(".gz"):
        data = gzip.decompress(data)
    return data


def validate_csv(path: Path) -> pd.DataFrame:
    """Check that the saved file has the columns we expect."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"The file is missing expected columns: {sorted(missing)}. "
                         f"Columns found: {df.columns.tolist()}")
    return df


def print_manual_help() -> None:
    print("\nAutomatic download failed. Please download manually:")
    print(f"  1. Open {DATASET_PAGE}")
    print("  2. Click 'Download' and unzip the file.")
    print(f"  3. Save the CSV as: {OUT_FILE}")
    print("     (if it is a .csv.gz file, unzip it first, or keep the .gz and tell your mentor)")
    print("  4. Run this script again; it will validate the file.")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        print(f"File already exists: {OUT_FILE}. Skipping download.")
    else:
        try:
            print(f"Downloading from {ZIP_URL} ...")
            zip_bytes = download_zip()
            OUT_FILE.write_bytes(extract_csv_bytes(zip_bytes))
            print(f"Saved to {OUT_FILE}")
        except Exception as error:  # network, zip, or file problems
            print(f"ERROR: {error}")
            print_manual_help()
            return 1

    try:
        df = validate_csv(OUT_FILE)
    except Exception as error:
        print(f"ERROR while checking the file: {error}")
        return 1

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    print("\nDataset check")
    print(f"  Rows              : {len(df)}")
    print(f"  Columns           : {df.columns.tolist()}")
    print(f"  First timestamp   : {df['date_time'].min()}")
    print(f"  Last timestamp    : {df['date_time'].max()}")
    print(f"  Bad timestamps    : {df['date_time'].isna().sum()}")
    print(f"  Duplicate times   : {df['date_time'].duplicated().sum()}")
    print(f"  Traffic volume    : min={df['traffic_volume'].min()}, max={df['traffic_volume'].max()}")
    print("\nDone. Paste this output to your mentor/assistant before Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
