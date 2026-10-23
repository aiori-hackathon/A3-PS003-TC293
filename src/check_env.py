"""Check that Python and all required libraries are installed correctly.

Run from the project root:  python src/check_env.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FOLDERS = ["configs", "src", "tests", "results", "figures", "docs"]
# These are ignored by Git (they hold downloaded/generated data), so a fresh
# clone does not contain them. The script simply creates them.
AUTO_CREATE_FOLDERS = ["data/raw", "data/processed"]


def main() -> int:
    problems = []
    print(f"Python version : {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is recommended.")

    for name in ["numpy", "pandas", "sklearn", "matplotlib", "torch", "yaml", "pytest"]:
        try:
            module = __import__(name)
            print(f"{name:<14} : OK (version {getattr(module, '__version__', 'unknown')})")
        except ImportError:
            print(f"{name:<14} : MISSING")
            problems.append(f"Library '{name}' is not installed. Run: pip install -r requirements.txt")

    print()
    for folder in AUTO_CREATE_FOLDERS:
        path = ROOT / folder
        if path.is_dir():
            print(f"folder {folder:<16}: OK")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"folder {folder:<16}: was missing, CREATED (normal after git clone)")
    for folder in REQUIRED_FOLDERS:
        if (ROOT / folder).is_dir():
            print(f"folder {folder:<16}: OK")
        else:
            print(f"folder {folder:<16}: MISSING")
            problems.append(f"Create the folder '{folder}' in the project root.")

    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(" -", p)
        return 1
    print("Everything looks good. You can move on to downloading the data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
