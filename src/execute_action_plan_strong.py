from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

ACTION_DIR = OUTPUT_DIR / "action"
SYNTHESIS_DIR = OUTPUT_DIR / "synthesis"
DESCRIPTIONS_DIR = OUTPUT_DIR / "descriptions"

ACTION_PLAN_JSON = ACTION_DIR / "action_plan.json"
VARIABLE_MAP_JSON = SYNTHESIS_DIR / "variable_knowledge_map.json"
CONNECTIONS_JSON = SYNTHESIS_DIR / "dataset_connection_recommendations.json"
DESCRIPTION_JSON = DESCRIPTIONS_DIR / "dataset_description.json"

EXECUTION_DIR = OUTPUT_DIR / "execution_strong"
GRAPHS_DIR = EXECUTION_DIR / "graphs"
STANDARDIZED_DIR = EXECUTION_DIR / "standardized_datasets"
MERGED_DIR = EXECUTION_DIR / "merged_datasets"

MERGE_RESULTS_JSON = EXECUTION_DIR / "merge_results.json"
STANDARDIZATION_RESULTS_JSON = EXECUTION_DIR / "standardization_results.json"
GRAPH_RESULTS_JSON = EXECUTION_DIR / "graph_results.json"
EXECUTION_REPORT_MD = EXECUTION_DIR / "execution_report.md"


# -----------------------------
# JSON / file helpers
# -----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"Warning: missing file -> {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


# -----------------------------
# CSV helpers
# -----------------------------

def normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    encodings = ["utf-8", "latin1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    print(f"Could not load CSV: {path}")
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    return df


def find_csv_for_dataset(dataset_name: str) -> Optional[Path]:
    direct = DATA_DIR / dataset_name
    if direct.exists():
        return direct

    target = normalize_name(dataset_name)
    for path in DATA_DIR.glob("*.csv"):
        if normalize_name(path.name) == target:
            return path
        if normalize_name(path.stem) == target.replace("_csv", ""):
            return path
    return None


def load_referenced_datasets(description_json: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}

    # First, load any explicitly referenced datasets from descriptions.
    for dataset_name in description_json.keys():
        path = find_csv_for_dataset(dataset_name)
        if not path:
            continue

        df = load_csv_safe(path)
        if df is None:
            continue

        datasets[path.name] = normalize_columns(df)

    # Then, ensure every CSV in data/ is available for downstream graph/merge actions.
    for path in DATA_DIR.glob("*.csv"):
        if path.name in datasets:
            continue

        df = load_csv_safe(path)
        if df is None:
            continue

        datasets[path.name] = normalize_columns(df)

    print(f"Loaded {len(datasets)} dataset(s) from data/.")

    return datasets


# -----------------------------
# Standardization logic
# -----------------------------

def build_rename_map_from_action_plan(action_plan: Dict[str, Any]) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}

    for item in safe_list(action_plan.get("standardization_actions")):
        concept_root = item.get("concept_root")
        variables = safe_list(item.get("variables"))

        if not concept_root:
            continue

        canonical = normalize_name(concept_root)
        for var in variables:
            rename_map[normalize_name(var)] = canonical

    return rename_map


def standardize_dataset_columns(
    datasets: Dict[str, pd.DataFrame],
    rename_map: Dict[str, str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    standardized: Dict[str, pd.DataFrame] = {}
    results: Dict[str, Any] = {}

    for dataset_name, df in datasets.items():
        original_cols = list(df.columns)
        df_std = df.copy()

        # apply rename map
        applied = {}
        new_cols = []
        seen = {}

        for col in df_std.columns:
            new_col = rename_map.get(col, col)

            # avoid duplicate resulting names by suffixing
            if new_col in seen:
                seen[new_col] += 1
                deduped = f"{new_col}__dup{seen[new_col]}"
                applied[col] = deduped
                new_cols.append(deduped)
            else:
                seen[new_col] = 0
                applied[col] = new_col
                new_cols.append(new_col)

        df_std.columns = new_cols
        standardized[dataset_name] = df_std

        outpath = STANDARDIZED_DIR / dataset_name
        df_std.to_csv(outpath, index=False)

        results[dataset_name] = {
            "original_columns": original_cols,
            "standardized_columns": list(df_std.columns),
            "rename_applied": applied,
            "saved_path": str(outpath),
        }

    return standardized, results


# -----------------------------
# Merge logic
# -----------------------------

def parse_dataset_pair(pair: str) -> Optional[Tuple[str, str]]:
    parts = [p.strip() for p in str(pair).split("<->")]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def choose_join_keys(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    possible_join_keys: List[str],
    shared_columns: List[str],
) -> List[str]:
    candidate_keys = [normalize_name(k) for k in possible_join_keys if normalize_name(k) in left_df.columns and normalize_name(k) in right_df.columns]
    if candidate_keys:
        return candidate_keys

    fallback = [normalize_name(k) for k in shared_columns if normalize_name(k) in left_df.columns and normalize_name(k) in right_df.columns]
    if fallback:
        # Prefer smaller number of columns for safer merge
        preferred = [k for k in fallback if k.endswith("_id") or k in {"id", "year", "date", "zip_code", "zipcode"}]
        return preferred[:3] if preferred else fallback[:3]

    return []


def assess_key_quality(left_df: pd.DataFrame, right_df: pd.DataFrame, keys: List[str]) -> Dict[str, Any]:
    if not keys:
        return {"valid": False, "reason": "No join keys available."}

    left_non_null = left_df[keys].dropna()
    right_non_null = right_df[keys].dropna()

    if left_non_null.empty or right_non_null.empty:
        return {"valid": False, "reason": "Join keys become empty after dropping nulls."}

    left_unique_ratio = len(left_non_null.drop_duplicates()) / max(len(left_non_null), 1)
    right_unique_ratio = len(right_non_null.drop_duplicates()) / max(len(right_non_null), 1)

    return {
        "valid": True,
        "left_non_null_rows": int(len(left_non_null)),
        "right_non_null_rows": int(len(right_non_null)),
        "left_unique_ratio": round(float(left_unique_ratio), 4),
        "right_unique_ratio": round(float(right_unique_ratio), 4),
    }


def attempt_merge(
    left_name: str,
    right_name: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    keys: List[str],
) -> Dict[str, Any]:
    quality = assess_key_quality(left_df, right_df, keys)
    if not quality.get("valid"):
        return {
            "merged": False,
            "reason": quality.get("reason", "Invalid key quality."),
        }

    try:
        merged_inner = pd.merge(
            left_df,
            right_df,
            on=keys,
            how="inner",
            suffixes=(f"__{normalize_name(Path(left_name).stem)}", f"__{normalize_name(Path(right_name).stem)}"),
        )

        merged_left = pd.merge(
            left_df,
            right_df,
            on=keys,
            how="left",
            suffixes=(f"__{normalize_name(Path(left_name).stem)}", f"__{normalize_name(Path(right_name).stem)}"),
        )

        output_name = f"{Path(left_name).stem}__MERGED_WITH__{Path(right_name).stem}.csv"
        outpath = MERGED_DIR / output_name

        chosen_merge = merged_inner if len(merged_inner) > 0 else merged_left
        chosen_merge.to_csv(outpath, index=False)

        return {
            "merged": True,
            "keys_used": keys,
            "left_rows": int(len(left_df)),
            "right_rows": int(len(right_df)),
            "inner_rows": int(len(merged_inner)),
            "left_join_rows": int(len(merged_left)),
            "saved_path": str(outpath),
            "key_quality": quality,
        }
    except Exception as e:
        return {
            "merged": False,
            "reason": f"Merge failed: {e}",
            "key_quality": quality,
        }


def execute_merges(
    action_plan: Dict[str, Any],
    standardized_datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    results = {
        "successful_merges": [],
        "failed_merges": [],
    }

    for item in safe_list(action_plan.get("merge_actions")):
        if item.get("recommended_action") not in {"test_merge_first", "evaluate_merge_after_standardization"}:
            continue

        pair = item.get("dataset_pair")
        parsed = parse_dataset_pair(pair)
        if not parsed:
            results["failed_merges"].append({
                "dataset_pair": pair,
                "reason": "Could not parse dataset pair.",
            })
            continue

        left_name, right_name = parsed
        left_df = standardized_datasets.get(left_name)
        right_df = standardized_datasets.get(right_name)

        if left_df is None or right_df is None:
            results["failed_merges"].append({
                "dataset_pair": pair,
                "reason": "One or both datasets were not loaded.",
            })
            continue

        keys = choose_join_keys(
            left_df=left_df,
            right_df=right_df,
            possible_join_keys=safe_list(item.get("possible_join_keys")),
            shared_columns=safe_list(item.get("shared_columns")),
        )

        result = attempt_merge(left_name, right_name, left_df, right_df, keys)
        result["dataset_pair"] = pair

        if result.get("merged"):
            results["successful_merges"].append(result)
        else:
            results["failed_merges"].append(result)

    return results


# -----------------------------
# Graph logic
# -----------------------------

def choose_dataset_for_variable(
    variable_name: str,
    variable_map: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> Optional[str]:
    info = safe_dict(variable_map.get(variable_name))
    preferred = safe_list(info.get("datasets"))

    # Columns in loaded datasets are normalized; normalize the lookup key too.
    norm_var = normalize_name(variable_name)

    for dataset_name in preferred:
        if dataset_name in datasets and norm_var in datasets[dataset_name].columns:
            return dataset_name

    for dataset_name, df in datasets.items():
        if norm_var in df.columns:
            return dataset_name

    return None


def create_numeric_graph(df: pd.DataFrame, column: str, outpath: Path) -> bool:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=20)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def create_categorical_graph(df: pd.DataFrame, column: str, outpath: Path) -> bool:
    vc = df[column].astype(str).value_counts().head(10)
    if vc.empty:
        return False

    plt.figure(figsize=(10, 5))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(f"Top categories for {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def create_time_graph(df: pd.DataFrame, column: str, outpath: Path) -> bool:
    parsed = pd.to_datetime(df[column], errors="coerce")
    parsed = parsed.dropna()
    if parsed.empty:
        return False

    counts = parsed.dt.to_period("M").astype(str).value_counts().sort_index()
    if counts.empty:
        return False

    plt.figure(figsize=(10, 5))
    plt.plot(counts.index.astype(str), counts.values)
    plt.title(f"Counts over time for {column}")
    plt.xlabel("Period")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def create_relationship_graph(
    variable_a: str,
    variable_b: str,
    dataset_name: str,
    df: pd.DataFrame,
    outpath: Path,
) -> bool:
    variable_a = normalize_name(variable_a)
    variable_b = normalize_name(variable_b)
    if variable_a not in df.columns or variable_b not in df.columns:
        return False

    a_num = pd.to_numeric(df[variable_a], errors="coerce")
    b_num = pd.to_numeric(df[variable_b], errors="coerce")
    sub = pd.DataFrame({variable_a: a_num, variable_b: b_num}).dropna()

    if len(sub) < 5:
        return False

    plt.figure(figsize=(8, 5))
    plt.scatter(sub[variable_a], sub[variable_b])
    plt.title(f"{variable_a} vs {variable_b}")
    plt.xlabel(variable_a)
    plt.ylabel(variable_b)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def execute_graphs(
    action_plan: Dict[str, Any],
    variable_map: Dict[str, Any],
    standardized_datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    results = {
        "created_graphs": [],
        "skipped_graphs": [],
    }

    for item in safe_list(action_plan.get("graph_actions"))[:25]:
        variable_name = item.get("variable_name")
        graph_type = item.get("recommended_graph", "relationship_plot")

        if not variable_name:
            continue

        dataset_name = choose_dataset_for_variable(variable_name, variable_map, standardized_datasets)
        if not dataset_name:
            results["skipped_graphs"].append({
                "variable_name": variable_name,
                "reason": "No dataset found containing this variable.",
            })
            continue

        df = standardized_datasets[dataset_name]
        norm_var = normalize_name(variable_name)
        outpath = GRAPHS_DIR / f"{Path(dataset_name).stem}__{norm_var}.png"

        if norm_var not in df.columns:
            results["skipped_graphs"].append({
                "variable_name": variable_name,
                "dataset_name": dataset_name,
                "reason": f"Normalized column '{norm_var}' not found in dataset.",
            })
            continue

        try:
            success = False
            if graph_type == "histogram_or_boxplot":
                success = create_numeric_graph(df, norm_var, outpath)
            elif graph_type == "bar_chart":
                success = create_categorical_graph(df, norm_var, outpath)
            elif graph_type == "time_series_line_chart":
                success = create_time_graph(df, norm_var, outpath)
            else:
                roles = set(safe_list(safe_dict(variable_map.get(variable_name)).get("roles")))
                if "numeric" in roles:
                    success = create_numeric_graph(df, norm_var, outpath)
                else:
                    success = create_categorical_graph(df, norm_var, outpath)

            if success:
                results["created_graphs"].append({
                    "variable_name": variable_name,
                    "dataset_name": dataset_name,
                    "graph_type": graph_type,
                    "file_path": str(outpath),
                })
            else:
                results["skipped_graphs"].append({
                    "variable_name": variable_name,
                    "dataset_name": dataset_name,
                    "reason": "Graph function returned no output.",
                })
        except Exception as e:
            results["skipped_graphs"].append({
                "variable_name": variable_name,
                "dataset_name": dataset_name,
                "reason": f"Graph creation failed: {e}",
            })

    return results


# -----------------------------
# Report
# -----------------------------

def build_report(
    standardization_results: Dict[str, Any],
    merge_results: Dict[str, Any],
    graph_results: Dict[str, Any],
) -> str:
    lines = []
    lines.append("# Strong Action Execution Report\n")

    lines.append("## Standardization")
    lines.append(f"- Standardized datasets written: {len(standardization_results)}")
    for dataset_name, info in list(standardization_results.items())[:10]:
        lines.append(
            f"- `{dataset_name}` standardized and saved to `{info['saved_path']}`"
        )
    lines.append("")

    lines.append("## Merges")
    successful = safe_list(merge_results.get("successful_merges"))
    failed = safe_list(merge_results.get("failed_merges"))
    lines.append(f"- Successful merges: {len(successful)}")
    lines.append(f"- Failed merges: {len(failed)}")
    for item in successful[:10]:
        lines.append(
            f"- `{item['dataset_pair']}` merged on {', '.join(item['keys_used'])}; "
            f"inner rows={item['inner_rows']}, left-join rows={item['left_join_rows']}; "
            f"saved to `{item['saved_path']}`"
        )
    if failed:
        lines.append("")
        lines.append("### Merge failures")
        for item in failed[:10]:
            lines.append(
                f"- `{item.get('dataset_pair', 'unknown')}` failed: {item.get('reason', 'unknown reason')}"
            )
    lines.append("")

    lines.append("## Graphs")
    created = safe_list(graph_results.get("created_graphs"))
    skipped = safe_list(graph_results.get("skipped_graphs"))
    lines.append(f"- Graphs created: {len(created)}")
    lines.append(f"- Graphs skipped: {len(skipped)}")
    for item in created[:15]:
        lines.append(
            f"- Created `{item['graph_type']}` for `{item['variable_name']}` "
            f"from `{item['dataset_name']}`"
        )
    if skipped:
        lines.append("")
        lines.append("### Skipped graphs")
        for item in skipped[:10]:
            lines.append(
                f"- `{item.get('variable_name', 'unknown')}` skipped: {item.get('reason', 'unknown reason')}"
            )
    lines.append("")

    lines.append("## Result")
    lines.append(
        "This script applied the action plan directly: it standardized datasets, attempted real merges, "
        "and generated focused graphs without rerunning the earlier slow pipeline."
    )

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    action_plan = load_json(ACTION_PLAN_JSON)
    variable_map = load_json(VARIABLE_MAP_JSON)
    description_json = load_json(DESCRIPTION_JSON)

    if not action_plan:
        print("Missing action plan. Run src/act_on_synthesis.py first.")
        return

    if not variable_map:
        print("Missing variable map. Run src/synthesize_from_outputs.py first.")
        return

    datasets = load_referenced_datasets(description_json)
    if not datasets:
        print("No datasets could be loaded from data/.")
        return

    rename_map = build_rename_map_from_action_plan(action_plan)
    standardized_datasets, standardization_results = standardize_dataset_columns(datasets, rename_map)

    merge_results = execute_merges(action_plan, standardized_datasets)
    graph_results = execute_graphs(action_plan, variable_map, standardized_datasets)

    save_json(standardization_results, STANDARDIZATION_RESULTS_JSON)
    save_json(merge_results, MERGE_RESULTS_JSON)
    save_json(graph_results, GRAPH_RESULTS_JSON)

    report = build_report(
        standardization_results=standardization_results,
        merge_results=merge_results,
        graph_results=graph_results,
    )
    save_text(report, EXECUTION_REPORT_MD)

    print("Strong action execution complete.")
    print(f"Saved: {STANDARDIZATION_RESULTS_JSON}")
    print(f"Saved: {MERGE_RESULTS_JSON}")
    print(f"Saved: {GRAPH_RESULTS_JSON}")
    print(f"Saved: {EXECUTION_REPORT_MD}")


if __name__ == "__main__":
    main()