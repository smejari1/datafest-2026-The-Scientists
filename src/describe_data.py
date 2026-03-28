import os
import pandas as pd
from utils import find_csv_files, load_csv_safe, save_json, save_text
from config import DATA_FOLDER, DESCRIPTION_FOLDER, SAMPLE_ROWS


def describe_dataframe(df, filename):
    description = {
        "file_name": filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(SAMPLE_ROWS).to_dict(orient="records"),
    }
    return description


def compare_datasets(descriptions):
    shared_columns = {}
    files = list(descriptions.keys())

    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            f1, f2 = files[i], files[j]
            cols1 = set(descriptions[f1]["column_names"])
            cols2 = set(descriptions[f2]["column_names"])
            overlap = list(cols1.intersection(cols2))
            if overlap:
                shared_columns[f"{f1} <-> {f2}"] = overlap

    return shared_columns


def generate_background(descriptions, shared_columns):
    lines = []
    lines.append("# Dataset Background Report\n")
    lines.append("## Overview\n")
    lines.append(
        f"This project contains {len(descriptions)} CSV datasets that were scanned and profiled."
    )

    lines.append("\n## Individual Dataset Descriptions\n")
    for filename, desc in descriptions.items():
        lines.append(f"### {filename}")
        lines.append(
            f"- Rows: {desc['rows']}, Columns: {desc['columns']}"
        )
        lines.append(f"- Columns: {', '.join(desc['column_names'])}")
        lines.append(f"- Duplicate rows: {desc['duplicate_rows']}")

        missing_cols = [k for k, v in desc["missing_values"].items() if v > 0]
        if missing_cols:
            lines.append(f"- Missing data appears in: {', '.join(missing_cols)}")
        else:
            lines.append("- No missing data detected.")

    lines.append("\n## Cross-Dataset Relationships\n")
    if shared_columns:
        for pair, overlap in shared_columns.items():
            lines.append(f"- {pair}: shared columns -> {', '.join(overlap)}")
    else:
        lines.append("- No obvious overlapping columns were found.")

    lines.append("\n## Pre-Development Notes\n")
    lines.append(
        "Before moving into development, these datasets should be reviewed for consistency, joinability, missing values, and possible cleaning needs."
    )

    return "\n".join(lines)


def run_description():
    csv_files = find_csv_files(DATA_FOLDER)
    descriptions = {}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        df = load_csv_safe(filepath)
        if df is not None:
            descriptions[filename] = describe_dataframe(df, filename)

    shared_columns = compare_datasets(descriptions)
    background_report = generate_background(descriptions, shared_columns)

    save_json(descriptions, f"{DESCRIPTION_FOLDER}/dataset_description.json")
    save_json(shared_columns, f"{DESCRIPTION_FOLDER}/dataset_relationships.json")
    save_text(background_report, f"{DESCRIPTION_FOLDER}/dataset_background.md")

    return descriptions, shared_columns