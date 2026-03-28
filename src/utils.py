import os
import json
from pathlib import Path
import pandas as pd


def ensure_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def find_csv_files(folder):
    return [str(p) for p in Path(folder).glob("*.csv")]


def load_csv_safe(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def save_text(text, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)