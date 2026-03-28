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


def safe_numeric_conversion(series):
    return pd.to_numeric(series, errors="coerce")


def convert_boolean_like_columns(df):
    converted = df.copy()
    true_values = {"yes", "true", "y", "1", "t"}
    false_values = {"no", "false", "n", "0", "f"}

    for col in converted.columns:
        series = converted[col]
        if series.dtype == "object":
            lowered = series.astype(str).str.strip().str.lower()
            valid_mask = lowered.isin(true_values.union(false_values))

            # Only convert when most non-null values are boolean-like.
            if valid_mask.sum() > 0 and valid_mask.sum() >= 0.8 * series.notna().sum():
                converted[col] = lowered.map(
                    lambda x: 1 if x in true_values else (0 if x in false_values else None)
                )

    return converted


def choose_primary_dataset(datasets):
    if not datasets:
        return None

    # Prefer the dataset with the most rows so modeling has enough samples.
    ranked = sorted(
        datasets.items(),
        key=lambda item: int(item[1].shape[0]) if item[1] is not None else 0,
        reverse=True,
    )
    return ranked[0][0]