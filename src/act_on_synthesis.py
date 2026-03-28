from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

SYNTHESIS_DIR = OUTPUT_DIR / "synthesis"
VARIABLE_MAP_JSON = SYNTHESIS_DIR / "variable_knowledge_map.json"
CONNECTIONS_JSON = SYNTHESIS_DIR / "dataset_connection_recommendations.json"

ACTION_DIR = OUTPUT_DIR / "action"
ACTION_PLAN_JSON = ACTION_DIR / "action_plan.json"
ACTION_REPORT_MD = ACTION_DIR / "action_report.md"


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
        json.dump(data, f, indent=4)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


# -----------------------------
# Action builders
# -----------------------------

def build_feature_actions(variable_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    variables = sorted(
        variable_map.values(),
        key=lambda x: x.get("importance_score", 0),
        reverse=True,
    )

    actions = []
    for var in variables[:25]:
        reasons = []

        if var.get("in_model_features"):
            reasons.append("already useful in modeling")
        if var.get("in_top_relationships"):
            reasons.append("appears in strong discovered relationships")
        if len(var.get("datasets", [])) >= 2:
            reasons.append("appears in multiple datasets")
        if var.get("shared_across_files"):
            reasons.append("may help connect files")

        actions.append({
            "variable_name": var["variable_name"],
            "concept_root": var.get("concept_root"),
            "priority": var.get("importance_bucket"),
            "importance_score": var.get("importance_score", 0),
            "recommended_action": "prioritize_for_feature_engineering",
            "why": reasons or ["high overall synthesis score"],
        })

    return actions


def build_merge_actions(connections: Dict[str, Any]) -> List[Dict[str, Any]]:
    join_candidates = safe_list(connections.get("join_candidates"))

    actions = []
    for jc in join_candidates:
        strength = jc.get("join_strength", "weak")

        if strength == "strong":
            action = "test_merge_first"
        elif strength == "moderate":
            action = "evaluate_merge_after_standardization"
        else:
            action = "hold_for_manual_review"

        actions.append({
            "dataset_pair": jc.get("dataset_pair"),
            "shared_columns": safe_list(jc.get("shared_columns")),
            "possible_join_keys": safe_list(jc.get("possible_join_keys")),
            "join_strength": strength,
            "recommended_action": action,
        })

    return actions


def build_standardization_actions(connections: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = safe_list(connections.get("standardization_candidates"))

    actions = []
    for item in candidates:
        vars_in_cluster = safe_list(item.get("variables"))
        if len(vars_in_cluster) < 2:
            continue

        actions.append({
            "concept_root": item.get("concept_root"),
            "variables": vars_in_cluster,
            "recommended_action": "standardize_variable_names_and_meaning",
            "reason": item.get("reason", ""),
        })

    return actions


def build_graph_actions(variable_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    variables = sorted(
        variable_map.values(),
        key=lambda x: x.get("importance_score", 0),
        reverse=True,
    )

    graph_actions = []
    for var in variables[:20]:
        role_set = set(var.get("roles", []))
        analysis_tags = set(var.get("analysis_tags", []))

        if "numeric" in role_set:
            graph_type = "histogram_or_boxplot"
        elif "categorical" in role_set:
            graph_type = "bar_chart"
        else:
            graph_type = "relationship_plot"

        if "time_candidate" in analysis_tags:
            graph_type = "time_series_line_chart"

        graph_actions.append({
            "variable_name": var["variable_name"],
            "concept_root": var.get("concept_root"),
            "recommended_graph": graph_type,
            "priority": var.get("importance_bucket"),
        })

    return graph_actions


def build_model_actions(variable_map: Dict[str, Any]) -> Dict[str, Any]:
    variables = sorted(
        variable_map.values(),
        key=lambda x: x.get("importance_score", 0),
        reverse=True,
    )

    target_candidates = []
    feature_candidates = []

    for var in variables:
        if var.get("is_model_target"):
            target_candidates.append(var["variable_name"])
        elif var.get("in_model_features") or var.get("in_top_relationships"):
            feature_candidates.append(var["variable_name"])

    if not target_candidates:
        for var in variables:
            if "target" in var["variable_name"].lower() or "outcome" in var["variable_name"].lower():
                target_candidates.append(var["variable_name"])

    return {
        "recommended_targets": target_candidates[:5],
        "recommended_features": feature_candidates[:15],
        "recommended_action": "use_for_next_model_iteration",
    }


def build_action_plan(variable_map: Dict[str, Any], connections: Dict[str, Any]) -> Dict[str, Any]:
    feature_actions = build_feature_actions(variable_map)
    merge_actions = build_merge_actions(connections)
    standardization_actions = build_standardization_actions(connections)
    graph_actions = build_graph_actions(variable_map)
    model_actions = build_model_actions(variable_map)

    return {
        "feature_actions": feature_actions,
        "merge_actions": merge_actions,
        "standardization_actions": standardization_actions,
        "graph_actions": graph_actions,
        "model_actions": model_actions,
    }


def build_action_report(action_plan: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Action Report From Synthesis\n")
    lines.append("## Overview")
    lines.append(
        "This script turns synthesis outputs into concrete next-step actions without rerunning the expensive earlier pipeline."
    )
    lines.append("")

    lines.append("## Top Feature Actions")
    for item in safe_list(action_plan.get("feature_actions"))[:10]:
        lines.append(
            f"- `{item['variable_name']}` ({item['concept_root']}, score={item['importance_score']}) "
            f"→ {item['recommended_action']} because {', '.join(item['why'])}."
        )
    lines.append("")

    lines.append("## Merge Actions")
    for item in safe_list(action_plan.get("merge_actions"))[:10]:
        lines.append(
            f"- {item['dataset_pair']} → {item['recommended_action']} "
            f"(strength={item['join_strength']}, keys={', '.join(item['possible_join_keys']) or 'none'})."
        )
    lines.append("")

    lines.append("## Standardization Actions")
    for item in safe_list(action_plan.get("standardization_actions"))[:10]:
        lines.append(
            f"- Concept `{item['concept_root']}` → standardize: {', '.join(item['variables'])}."
        )
    lines.append("")

    lines.append("## Graph Actions")
    for item in safe_list(action_plan.get("graph_actions"))[:10]:
        lines.append(
            f"- `{item['variable_name']}` → suggested graph: {item['recommended_graph']}."
        )
    lines.append("")

    model_actions = safe_dict(action_plan.get("model_actions"))
    lines.append("## Modeling Actions")
    lines.append(
        f"- Recommended targets: {', '.join(safe_list(model_actions.get('recommended_targets'))) or 'none'}"
    )
    lines.append(
        f"- Recommended features: {', '.join(safe_list(model_actions.get('recommended_features'))) or 'none'}"
    )
    lines.append("")

    lines.append("## Next Move")
    lines.append(
        "Use this action plan to drive a later script that actually builds merges, standardized schemas, focused graphs, or revised models."
    )

    return "\n".join(lines)


def main() -> None:
    ACTION_DIR.mkdir(parents=True, exist_ok=True)

    variable_map = load_json(VARIABLE_MAP_JSON)
    connections = load_json(CONNECTIONS_JSON)

    if not variable_map or not connections:
        print("Missing synthesis outputs. Run synthesize_from_outputs.py first.")
        return

    action_plan = build_action_plan(variable_map, connections)
    report = build_action_report(action_plan)

    save_json(action_plan, ACTION_PLAN_JSON)
    save_text(report, ACTION_REPORT_MD)

    print("Action plan complete.")
    print(f"Saved: {ACTION_PLAN_JSON}")
    print(f"Saved: {ACTION_REPORT_MD}")


if __name__ == "__main__":
    main()