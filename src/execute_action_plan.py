from __future__ import annotations

import json
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
RELATIONSHIPS_DIR = OUTPUT_DIR / "relationships"

ACTION_PLAN_JSON = ACTION_DIR / "action_plan.json"
VARIABLE_MAP_JSON = SYNTHESIS_DIR / "variable_knowledge_map.json"
CONNECTIONS_JSON = SYNTHESIS_DIR / "dataset_connection_recommendations.json"
DESCRIPTION_JSON = DESCRIPTIONS_DIR / "dataset_description.json"
DISCOVERED_RELATIONSHIPS_JSON = RELATIONSHIPS_DIR / "discovered_relationships.json"

EXECUTION_DIR = OUTPUT_DIR / "execution"
GRAPHS_DIR = EXECUTION_DIR / "graphs"
MERGE_PLAN_JSON = EXECUTION_DIR / "merge_execution_plan.json"
STANDARDIZATION_PLAN_JSON = EXECUTION_DIR / "standardization_execution_plan.json"
GRAPH_PLAN_JSON = EXECUTION_DIR / "graph_execution_plan.json"
EXECUTION_REPORT_MD = EXECUTION_DIR / "execution_report.md"


# -----------------------------
# Helpers
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


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


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
    df.columns = [normalize_name(c).replace(" ", "_") for c in df.columns]
    return df


def find_csv_for_dataset(dataset_name: str) -> Optional[Path]:
    direct = DATA_DIR / dataset_name
    if direct.exists():
        return direct

    dataset_norm = normalize_name(dataset_name)
    for path in DATA_DIR.glob("*.csv"):
        if normalize_name(path.name) == dataset_norm:
            return path
    return None


def infer_role_from_variable_map(variable_map: Dict[str, Any], var_name: str) -> List[str]:
    info = safe_dict(variable_map.get(var_name))
    return safe_list(info.get("roles"))


# -----------------------------
# Load raw data once
# -----------------------------

def load_referenced_datasets(description_json: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}

    for dataset_name in description_json.keys():
        path = find_csv_for_dataset(dataset_name)
        if not path:
            continue

        df = load_csv_safe(path)
        if df is None:
            continue

        datasets[dataset_name] = df

    return datasets


# -----------------------------
# Merge plan execution
# -----------------------------

def build_merge_execution_plan(
    action_plan: Dict[str, Any],
    description_json: Dict[str, Any],
) -> Dict[str, Any]:
    merge_actions = safe_list(action_plan.get("merge_actions"))
    plan = {
        "ready_to_test": [],
        "needs_review": [],
    }

    for item in merge_actions:
        pair = str(item.get("dataset_pair", ""))
        parts = [p.strip() for p in pair.split("<->")]
        if len(parts) != 2:
            continue

        left, right = parts
        left_exists = left in description_json
        right_exists = right in description_json

        entry = {
            "dataset_pair": pair,
            "left_dataset": left,
            "right_dataset": right,
            "left_exists": left_exists,
            "right_exists": right_exists,
            "shared_columns": safe_list(item.get("shared_columns")),
            "possible_join_keys": safe_list(item.get("possible_join_keys")),
            "join_strength": item.get("join_strength"),
            "recommended_action": item.get("recommended_action"),
        }

        if left_exists and right_exists and entry["recommended_action"] in {
            "test_merge_first",
            "evaluate_merge_after_standardization",
        }:
            plan["ready_to_test"].append(entry)
        else:
            plan["needs_review"].append(entry)

    return plan


# -----------------------------
# Standardization execution
# -----------------------------

def build_standardization_execution_plan(action_plan: Dict[str, Any]) -> Dict[str, Any]:
    std_actions = safe_list(action_plan.get("standardization_actions"))
    rename_map = {}
    clusters = []

    for item in std_actions:
        concept = item.get("concept_root")
        variables = safe_list(item.get("variables"))
        if not concept or not variables:
            continue

        canonical_name = str(concept)
        cluster_entry = {
            "concept_root": concept,
            "canonical_name": canonical_name,
            "variables": variables,
        }
        clusters.append(cluster_entry)

        for var in variables:
            rename_map[var] = canonical_name

    return {
        "concept_clusters": clusters,
        "proposed_rename_map": rename_map,
    }


# -----------------------------
# Graph planning
# -----------------------------

def choose_dataset_for_variable(
    variable_name: str,
    variable_map: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> Optional[str]:
    info = safe_dict(variable_map.get(variable_name))
    preferred = safe_list(info.get("datasets"))

    for dataset_name in preferred:
        df = datasets.get(dataset_name)
        if df is not None and variable_name in df.columns:
            return dataset_name

    for dataset_name, df in datasets.items():
        if variable_name in df.columns:
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
    series = df[column].dropna()
    if series.empty:
        return False

    # Skip ID-like / free-text columns with too many unique values — useless as a bar chart
    if series.nunique() > 50:
        return False

    series = series.astype(str)
    vc = series.value_counts().head(10)
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


def build_graph_execution_plan(
    action_plan: Dict[str, Any],
    variable_map: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    graph_actions = safe_list(action_plan.get("graph_actions"))
    created_graphs = []
    skipped_graphs = []

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for item in graph_actions[:20]:
        variable_name = item.get("variable_name")
        recommended_graph = item.get("recommended_graph", "relationship_plot")

        if not variable_name:
            continue

        dataset_name = choose_dataset_for_variable(variable_name, variable_map, datasets)
        if not dataset_name:
            skipped_graphs.append({
                "variable_name": variable_name,
                "reason": "No dataset found containing this variable.",
            })
            continue

        df = datasets[dataset_name]
        if variable_name not in df.columns:
            skipped_graphs.append({
                "variable_name": variable_name,
                "dataset_name": dataset_name,
                "reason": "Variable not found in dataset columns.",
            })
            continue

        safe_file = f"{dataset_name.replace('.csv', '')}__{variable_name}.png"
        outpath = GRAPHS_DIR / safe_file

        success = False
        try:
            if recommended_graph == "histogram_or_boxplot":
                success = create_numeric_graph(df, variable_name, outpath)
            elif recommended_graph == "bar_chart":
                success = create_categorical_graph(df, variable_name, outpath)
            elif recommended_graph == "time_series_line_chart":
                success = create_time_graph(df, variable_name, outpath)
            else:
                roles = infer_role_from_variable_map(variable_map, variable_name)
                if "numeric" in roles:
                    success = create_numeric_graph(df, variable_name, outpath)
                else:
                    success = create_categorical_graph(df, variable_name, outpath)
        except Exception as e:
            skipped_graphs.append({
                "variable_name": variable_name,
                "dataset_name": dataset_name,
                "reason": f"Graph creation failed: {e}",
            })
            continue

        if success:
            created_graphs.append({
                "variable_name": variable_name,
                "dataset_name": dataset_name,
                "graph_type": recommended_graph,
                "file_path": str(outpath),
            })
        else:
            skipped_graphs.append({
                "variable_name": variable_name,
                "dataset_name": dataset_name,
                "reason": "Graph function returned no output.",
            })

    return {
        "created_graphs": created_graphs,
        "skipped_graphs": skipped_graphs,
    }


# -----------------------------
# Report
# -----------------------------

def build_execution_report(
    merge_plan: Dict[str, Any],
    standardization_plan: Dict[str, Any],
    graph_plan: Dict[str, Any],
) -> str:
    lines = []
    lines.append("# Executed Action Plan Report\n")

    lines.append("## Merge Execution")
    ready = safe_list(merge_plan.get("ready_to_test"))
    review = safe_list(merge_plan.get("needs_review"))
    lines.append(f"- Merge candidates ready to test: {len(ready)}")
    lines.append(f"- Merge candidates needing review: {len(review)}")
    for item in ready[:10]:
        lines.append(
            f"- Ready: {item['dataset_pair']} using keys "
            f"{', '.join(item['possible_join_keys'] or item['shared_columns'][:3])}."
        )
    lines.append("")

    lines.append("## Standardization Execution")
    concept_clusters = safe_list(standardization_plan.get("concept_clusters"))
    lines.append(f"- Concept clusters identified for standardization: {len(concept_clusters)}")
    for item in concept_clusters[:10]:
        lines.append(
            f"- `{item['concept_root']}` -> canonical `{item['canonical_name']}` "
            f"for variables: {', '.join(item['variables'])}"
        )
    lines.append("")

    lines.append("## Graph Execution")
    created = safe_list(graph_plan.get("created_graphs"))
    skipped = safe_list(graph_plan.get("skipped_graphs"))
    lines.append(f"- Graphs created: {len(created)}")
    lines.append(f"- Graphs skipped: {len(skipped)}")
    for item in created[:15]:
        lines.append(
            f"- Created {item['graph_type']} for `{item['variable_name']}` "
            f"from `{item['dataset_name']}`"
        )
    lines.append("")

    if skipped:
        lines.append("## Skipped Graphs")
        for item in skipped[:15]:
            lines.append(
                f"- `{item.get('variable_name', 'unknown')}` skipped: {item.get('reason', 'unknown reason')}"
            )
        lines.append("")

    lines.append("## Next Use")
    lines.append(
        "You can now use the merge plan, rename map, and generated graphs as the next concrete layer "
        "without rerunning the slow earlier pipeline."
    )

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

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

    merge_plan = build_merge_execution_plan(action_plan, description_json)
    standardization_plan = build_standardization_execution_plan(action_plan)
    graph_plan = build_graph_execution_plan(action_plan, variable_map, datasets)

    save_json(merge_plan, MERGE_PLAN_JSON)
    save_json(standardization_plan, STANDARDIZATION_PLAN_JSON)
    save_json(graph_plan, GRAPH_PLAN_JSON)

    report = build_execution_report(
        merge_plan=merge_plan,
        standardization_plan=standardization_plan,
        graph_plan=graph_plan,
    )
    save_text(report, EXECUTION_REPORT_MD)

    print("Action execution complete.")
    print(f"Saved: {MERGE_PLAN_JSON}")
    print(f"Saved: {STANDARDIZATION_PLAN_JSON}")
    print(f"Saved: {GRAPH_PLAN_JSON}")
    print(f"Saved: {EXECUTION_REPORT_MD}")


if __name__ == "__main__":
    main()