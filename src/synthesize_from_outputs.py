from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

DESCRIPTION_JSON = OUTPUT_DIR / "descriptions" / "dataset_description.json"
SHARED_RELATIONSHIPS_JSON = OUTPUT_DIR / "descriptions" / "dataset_relationships.json"
ANALYSIS_JSON = OUTPUT_DIR / "analysis" / "analysis_results.json"
DISCOVERED_RELATIONSHIPS_JSON = OUTPUT_DIR / "relationships" / "discovered_relationships.json"
MODEL_RESULTS_JSON = OUTPUT_DIR / "models" / "likelihood_results.json"

SYNTHESIS_DIR = OUTPUT_DIR / "synthesis"
VARIABLE_MAP_JSON = SYNTHESIS_DIR / "variable_knowledge_map.json"
CONNECTIONS_JSON = SYNTHESIS_DIR / "dataset_connection_recommendations.json"
SYNTHESIS_REPORT_MD = SYNTHESIS_DIR / "synthesized_insights.md"


# -----------------------------
# Helpers
# -----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"Warning: file not found -> {path}")
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


def parse_shared_pair_info(pair_info: Any) -> Dict[str, List[Any]]:
    # Support both formats:
    # 1) {"shared_columns": [...], "possible_join_keys": [...]} (new)
    # 2) [...] (legacy list of shared columns)
    if isinstance(pair_info, dict):
        return {
            "shared_columns": safe_list(pair_info.get("shared_columns")),
            "possible_join_keys": safe_list(pair_info.get("possible_join_keys")),
        }

    if isinstance(pair_info, list):
        return {
            "shared_columns": pair_info,
            "possible_join_keys": [],
        }

    return {
        "shared_columns": [],
        "possible_join_keys": [],
    }


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def concept_root(name: str) -> str:
    n = normalize_name(name)

    replacements = [
        ("_flag", ""),
        ("_status", ""),
        ("_score", ""),
        ("_group", ""),
        ("_level", ""),
        ("_rate", ""),
        ("_count", ""),
    ]
    for old, new in replacements:
        n = n.replace(old, new)

    aliases = {
        "birth_year": "age",
        "year_of_birth": "age",
        "dob_year": "age",
        "age_group": "age",
        "age": "age",
        "income_level": "income",
        "low_income": "income",
        "household_income": "income",
        "food_insecure": "food_insecurity",
        "food_insecurity": "food_insecurity",
        "social_determinant_score": "social_determinants",
        "high_social_determinant_burden": "social_determinants",
        "housing_instability": "housing",
        "housing_status": "housing",
        "transportation_barrier": "transportation",
        "transport_access": "transportation",
        "employment_status": "employment",
        "unemployed": "employment",
        "insurance_status": "insurance",
        "no_health_insurance": "insurance",
        "zip_code": "location",
        "zipcode": "location",
        "county": "location",
        "city": "location",
        "state": "location",
        "gender": "gender",
        "sex": "gender",
        "race": "race_ethnicity",
        "ethnicity": "race_ethnicity",
    }
    return aliases.get(n, n)


def importance_bucket(score: float) -> str:
    if score >= 8:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def why_variable_matters(info: Dict[str, Any]) -> str:
    reasons = []

    if info.get("is_model_target"):
        reasons.append("it is the modeled outcome")
    if info.get("in_model_features"):
        reasons.append("it is used in the likelihood model")
    if info.get("group_comparison_column"):
        reasons.append("it is used in group comparisons")
    if info.get("in_top_relationships"):
        reasons.append("it appears in strong discovered relationships")
    if len(info.get("datasets", [])) >= 2:
        reasons.append("it appears across multiple datasets")
    if info.get("shared_across_files"):
        reasons.append("it may help connect files")

    return "; ".join(reasons) if reasons else "it appears repeatedly in the generated outputs"


# -----------------------------
# Core synthesis logic
# -----------------------------

def build_variable_knowledge_map(
    descriptions: Dict[str, Any],
    shared_relationships: Dict[str, Any],
    analysis_results: Dict[str, Any],
    discovered_relationships: Dict[str, Any],
    model_results: Dict[str, Any],
) -> Dict[str, Any]:
    variable_map: Dict[str, Dict[str, Any]] = {}

    def ensure_var(name: str) -> Dict[str, Any]:
        if name not in variable_map:
            variable_map[name] = {
                "variable_name": name,
                "concept_root": concept_root(name),
                "datasets": [],
                "roles": [],
                "missingness_by_dataset": {},
                "unique_count_by_dataset": {},
                "in_top_relationships": [],
                "relationship_partners": [],
                "relationship_types": [],
                "in_model_features": False,
                "is_model_target": False,
                "model_dataset": None,
                "group_comparison_column": False,
                "shared_across_files": [],
                "analysis_tags": [],
                "notes": [],
                "importance_score": 0.0,
                "importance_bucket": "low",
            }
        return variable_map[name]

    # Description layer
    for dataset_name, desc in safe_dict(descriptions).items():
        column_names = safe_list(desc.get("column_names"))
        numeric_columns = set(safe_list(desc.get("numeric_columns")))
        categorical_columns = set(safe_list(desc.get("categorical_columns")))
        missing_by_col = safe_dict(desc.get("missing_percent_by_column"))
        column_profiles = safe_dict(desc.get("column_profiles"))

        for col in column_names:
            entry = ensure_var(col)

            if dataset_name not in entry["datasets"]:
                entry["datasets"].append(dataset_name)

            if col in numeric_columns and "numeric" not in entry["roles"]:
                entry["roles"].append("numeric")
            if col in categorical_columns and "categorical" not in entry["roles"]:
                entry["roles"].append("categorical")

            if col in missing_by_col:
                entry["missingness_by_dataset"][dataset_name] = missing_by_col[col]

            profile = safe_dict(column_profiles.get(col))
            if "unique_count" in profile:
                entry["unique_count_by_dataset"][dataset_name] = profile["unique_count"]

            entry["importance_score"] += 1.0

    # Analysis layer
    for dataset_name, analysis in safe_dict(analysis_results).items():
        for col in safe_list(analysis.get("numeric_columns")):
            entry = ensure_var(col)
            if "numeric_analysis" not in entry["analysis_tags"]:
                entry["analysis_tags"].append("numeric_analysis")
            entry["importance_score"] += 0.5

        for col in safe_list(analysis.get("categorical_columns")):
            entry = ensure_var(col)
            if "categorical_analysis" not in entry["analysis_tags"]:
                entry["analysis_tags"].append("categorical_analysis")
            entry["importance_score"] += 0.25

        for col in safe_list(analysis.get("potential_date_columns")):
            entry = ensure_var(col)
            if "time_candidate" not in entry["analysis_tags"]:
                entry["analysis_tags"].append("time_candidate")
            entry["importance_score"] += 0.5

    # Discovered relationships
    for dataset_name, rel_info in safe_dict(discovered_relationships).items():
        top_relationships = safe_list(rel_info.get("top_relationships"))
        for rel in top_relationships:
            a = rel.get("column_a")
            b = rel.get("column_b")
            rel_type = rel.get("comparison_type")
            score = float(rel.get("score", 0) or 0)

            if not a or not b:
                continue

            for col, partner in [(a, b), (b, a)]:
                entry = ensure_var(col)
                entry["in_top_relationships"].append(
                    {
                        "dataset": dataset_name,
                        "partner": partner,
                        "relationship_type": rel_type,
                        "score": score,
                    }
                )
                if partner not in entry["relationship_partners"]:
                    entry["relationship_partners"].append(partner)
                if rel_type and rel_type not in entry["relationship_types"]:
                    entry["relationship_types"].append(rel_type)

                entry["importance_score"] += 1.5 + min(score, 2.0)

    # Model layer
    feature_cols = safe_list(model_results.get("feature_columns"))
    target_col = model_results.get("target_column")
    model_dataset = model_results.get("dataset_used")
    group_col = model_results.get("group_column")

    for col in feature_cols:
        entry = ensure_var(col)
        entry["in_model_features"] = True
        entry["model_dataset"] = model_dataset
        entry["notes"].append("Used as a model feature.")
        entry["importance_score"] += 2.5

    if target_col:
        entry = ensure_var(target_col)
        entry["is_model_target"] = True
        entry["model_dataset"] = model_dataset
        entry["notes"].append("Used as the modeled target/outcome.")
        entry["importance_score"] += 4.0

    if group_col:
        entry = ensure_var(group_col)
        entry["group_comparison_column"] = True
        entry["notes"].append("Used in group-rate comparisons.")
        entry["importance_score"] += 1.5

    # Shared across files
    for pair_name, pair_info in safe_dict(shared_relationships).items():
        parsed_pair_info = parse_shared_pair_info(pair_info)
        shared_cols = parsed_pair_info["shared_columns"]
        for col in shared_cols:
            entry = ensure_var(col)
            if pair_name not in entry["shared_across_files"]:
                entry["shared_across_files"].append(pair_name)
            entry["importance_score"] += 1.25

    # Final cleanup
    for entry in variable_map.values():
        entry["datasets"] = sorted(set(entry["datasets"]))
        entry["roles"] = sorted(set(entry["roles"]))
        entry["relationship_partners"] = sorted(set(entry["relationship_partners"]))
        entry["relationship_types"] = sorted(set(entry["relationship_types"]))
        entry["shared_across_files"] = sorted(set(entry["shared_across_files"]))
        entry["analysis_tags"] = sorted(set(entry["analysis_tags"]))
        entry["notes"] = sorted(set(entry["notes"]))

        if entry["missingness_by_dataset"]:
            min_missing = min(entry["missingness_by_dataset"].values())
            if min_missing <= 10:
                entry["importance_score"] += 1.0
            elif min_missing <= 25:
                entry["importance_score"] += 0.5

        if len(entry["datasets"]) >= 2:
            entry["importance_score"] += 2.0

        if entry["in_model_features"] or entry["is_model_target"] or entry["in_top_relationships"]:
            entry["importance_score"] += 0.5

        entry["importance_score"] = round(entry["importance_score"], 2)
        entry["importance_bucket"] = importance_bucket(entry["importance_score"])

    return variable_map


def build_connection_recommendations(
    descriptions: Dict[str, Any],
    shared_relationships: Dict[str, Any],
    variable_map: Dict[str, Any],
) -> Dict[str, Any]:
    recommendations = {
        "join_candidates": [],
        "standardization_candidates": [],
        "concept_clusters": [],
        "high_priority_variables": [],
        "dataset_priority_notes": [],
    }

    # Join candidates
    for pair_name, pair_info in safe_dict(shared_relationships).items():
        parsed_pair_info = parse_shared_pair_info(pair_info)
        shared_cols = parsed_pair_info["shared_columns"]
        possible_keys = parsed_pair_info["possible_join_keys"]

        strength = "weak"
        if possible_keys:
            strength = "strong"
        elif len(shared_cols) >= 2:
            strength = "moderate"

        recommendations["join_candidates"].append(
            {
                "dataset_pair": pair_name,
                "shared_columns": shared_cols,
                "possible_join_keys": possible_keys,
                "join_strength": strength,
                "recommendation": (
                    f"Consider testing a merge for {pair_name} using "
                    f"{', '.join(possible_keys or shared_cols[:3])}."
                ),
            }
        )

    # Concept clusters
    concept_clusters = defaultdict(list)
    for var_name, info in variable_map.items():
        concept_clusters[info["concept_root"]].append(
            {
                "variable_name": var_name,
                "datasets": info["datasets"],
                "importance_score": info["importance_score"],
            }
        )

    for concept, vars_in_cluster in concept_clusters.items():
        if len(vars_in_cluster) >= 2:
            vars_in_cluster = sorted(
                vars_in_cluster,
                key=lambda x: x["importance_score"],
                reverse=True,
            )
            recommendations["concept_clusters"].append(
                {
                    "concept_root": concept,
                    "variables": vars_in_cluster,
                    "recommendation": (
                        f"Standardize or harmonize the variables in the '{concept}' cluster."
                    ),
                }
            )
            recommendations["standardization_candidates"].append(
                {
                    "concept_root": concept,
                    "variables": [v["variable_name"] for v in vars_in_cluster],
                    "reason": "These variables likely represent the same or related concepts.",
                }
            )

    # High-priority variables
    sorted_vars = sorted(
        variable_map.values(),
        key=lambda x: x["importance_score"],
        reverse=True,
    )

    recommendations["high_priority_variables"] = [
        {
            "variable_name": info["variable_name"],
            "concept_root": info["concept_root"],
            "datasets": info["datasets"],
            "importance_score": info["importance_score"],
            "importance_bucket": info["importance_bucket"],
            "why_it_matters": why_variable_matters(info),
        }
        for info in sorted_vars[:20]
    ]

    # Dataset notes
    for dataset_name, desc in safe_dict(descriptions).items():
        high_value_cols = [
            item["variable_name"]
            for item in recommendations["high_priority_variables"]
            if dataset_name in item["datasets"]
        ][:8]

        recommendations["dataset_priority_notes"].append(
            {
                "dataset_name": dataset_name,
                "priority_variables": high_value_cols,
                "note": (
                    f"{dataset_name} should prioritize work around: "
                    f"{', '.join(high_value_cols) if high_value_cols else 'no standout variables identified yet'}."
                ),
            }
        )

    recommendations["join_candidates"] = sorted(
        recommendations["join_candidates"],
        key=lambda x: {"strong": 2, "moderate": 1, "weak": 0}[x["join_strength"]],
        reverse=True,
    )

    recommendations["concept_clusters"] = sorted(
        recommendations["concept_clusters"],
        key=lambda x: len(x["variables"]),
        reverse=True,
    )

    return recommendations


def build_report(
    variable_map: Dict[str, Any],
    connections: Dict[str, Any],
) -> str:
    lines = []
    lines.append("# Synthesized Insights Report\n")
    lines.append("## Overview")
    lines.append(
        "This script used previously generated output files only. "
        "It did not rerun the expensive CSV profiling, graphing, relationship discovery, or modeling steps."
    )
    lines.append("")

    top_vars = sorted(
        variable_map.values(),
        key=lambda x: x["importance_score"],
        reverse=True,
    )[:15]

    lines.append("## Highest-Priority Variables")
    for info in top_vars:
        lines.append(
            f"- `{info['variable_name']}` "
            f"(concept: `{info['concept_root']}`, score: {info['importance_score']}, "
            f"datasets: {', '.join(info['datasets']) or 'none'}) — {why_variable_matters(info)}."
        )
    lines.append("")

    lines.append("## Join Candidates")
    join_candidates = connections.get("join_candidates", [])
    if join_candidates:
        for jc in join_candidates[:10]:
            lines.append(
                f"- {jc['dataset_pair']} — join strength: {jc['join_strength']}; "
                f"shared columns: {', '.join(jc['shared_columns']) or 'none'}; "
                f"possible keys: {', '.join(jc['possible_join_keys']) or 'none'}."
            )
    else:
        lines.append("- No join candidates were identified.")
    lines.append("")

    lines.append("## Standardization Opportunities")
    for item in connections.get("standardization_candidates", [])[:10]:
        lines.append(
            f"- Concept `{item['concept_root']}`: {', '.join(item['variables'])}. {item['reason']}"
        )
    lines.append("")

    lines.append("## Concept Clusters")
    for cluster in connections.get("concept_clusters", [])[:10]:
        var_names = [v["variable_name"] for v in cluster["variables"]]
        lines.append(
            f"- `{cluster['concept_root']}`: {', '.join(var_names)}. {cluster['recommendation']}"
        )
    lines.append("")

    lines.append("## Recommended Next Moves")
    lines.append(
        "Use the high-priority variables, concept clusters, and join candidates to guide future feature engineering, "
        "dataset merging, graph focus, and model refinement."
    )

    return "\n".join(lines)


def main() -> None:
    SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)

    descriptions = load_json(DESCRIPTION_JSON)
    shared_relationships = load_json(SHARED_RELATIONSHIPS_JSON)
    analysis_results = load_json(ANALYSIS_JSON)
    discovered_relationships = load_json(DISCOVERED_RELATIONSHIPS_JSON)
    model_results = load_json(MODEL_RESULTS_JSON)

    variable_map = build_variable_knowledge_map(
        descriptions=descriptions,
        shared_relationships=shared_relationships,
        analysis_results=analysis_results,
        discovered_relationships=discovered_relationships,
        model_results=model_results,
    )

    connections = build_connection_recommendations(
        descriptions=descriptions,
        shared_relationships=shared_relationships,
        variable_map=variable_map,
    )

    report = build_report(variable_map, connections)

    save_json(variable_map, VARIABLE_MAP_JSON)
    save_json(connections, CONNECTIONS_JSON)
    save_text(report, SYNTHESIS_REPORT_MD)

    print("Synthesis complete.")
    print(f"Saved: {VARIABLE_MAP_JSON}")
    print(f"Saved: {CONNECTIONS_JSON}")
    print(f"Saved: {SYNTHESIS_REPORT_MD}")


if __name__ == "__main__":
    main()