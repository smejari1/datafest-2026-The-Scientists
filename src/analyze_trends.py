import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import find_csv_files, load_csv_safe, save_json, save_text
from config import DATA_FOLDER, ANALYSIS_FOLDER, GRAPHS_FOLDER


def analyze_dataframe(df, filename):
    results = {
        "file_name": filename,
        "numeric_columns": [],
        "categorical_columns": [],
        "summary_statistics": {},
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    results["numeric_columns"] = numeric_cols
    results["categorical_columns"] = categorical_cols

    if numeric_cols:
        results["summary_statistics"] = df[numeric_cols].describe().to_dict()

    return results


def create_graphs(df, filename):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols[:3]:
        plt.figure(figsize=(8, 5))
        df[col].dropna().hist()
        plt.title(f"{filename} - Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        safe_name = filename.replace(".csv", "")
        plt.savefig(f"{GRAPHS_FOLDER}/{safe_name}_{col}_hist.png")
        plt.close()


def generate_analysis_report(results):
    lines = []
    lines.append("# Trend and Analysis Report\n")

    for filename, result in results.items():
        lines.append(f"## {filename}")
        lines.append(f"- Numeric columns: {', '.join(result['numeric_columns']) or 'None'}")
        lines.append(f"- Categorical columns: {', '.join(result['categorical_columns']) or 'None'}")

        if result["numeric_columns"]:
            lines.append("- Basic descriptive statistics were generated for numeric fields.")
        else:
            lines.append("- No numeric fields detected for statistical trend analysis.")

    return "\n".join(lines)


def run_analysis():
    csv_files = find_csv_files(DATA_FOLDER)
    analysis_results = {}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        df = load_csv_safe(filepath)
        if df is not None:
            analysis_results[filename] = analyze_dataframe(df, filename)
            create_graphs(df, filename)

    report = generate_analysis_report(analysis_results)
    save_json(analysis_results, f"{ANALYSIS_FOLDER}/analysis_results.json")
    save_text(report, f"{ANALYSIS_FOLDER}/trend_report.md")

    return analysis_results