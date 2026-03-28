from typing import Dict

from src.config import FINAL_SUMMARY_MD
from src.utils import save_text


def build_final_summary(
    descriptions: Dict[str, dict],
    analysis_results: Dict[str, dict],
    relationship_results: Dict[str, dict],
    model_results: dict,
    shared_columns: dict,
) -> str:
    lines = []
    lines.append("# Final Data Summary\n")

    lines.append("## Overview")
    lines.append(
        f"This project analyzed **{len(descriptions)} dataset(s)**. "
        "The workflow profiled the datasets, explored trends and graphs, automatically discovered "
        "relationships, and then estimated likelihoods for an identified target outcome."
    )
    lines.append("")

    lines.append("## Description Stage")
    lines.append(
        "The description stage documented dataset structure, likely purpose, missingness, duplicates, "
        "and possible relationships between files."
    )
    lines.append(f"- Dataset relationships identified from shared columns: {len(shared_columns)}")
    lines.append("")

    lines.append("## Trend Analysis Stage")
    lines.append(
        "The trend analysis stage generated distributions, category counts, and basic trend graphs "
        "to identify visible patterns and candidate variables for later modeling."
    )
    lines.append("")

    lines.append("## Automated Relationship Discovery Stage")
    total_relationships = sum(
        dataset_result.get("valid_relationship_count", 0)
        for dataset_result in relationship_results.values()
    )
    lines.append(
        "The relationship discovery stage automatically classified columns, filtered unusable fields, "
        "compared sensible variable pairs, and ranked stronger associations."
    )
    lines.append(f"- Relationships retained across datasets: {total_relationships}")
    lines.append("")

    lines.append("## Likelihood Modeling Stage")
    if "error" in model_results:
        lines.append(f"Likelihood modeling could not be fully completed: {model_results['error']}")
    else:
        lines.append(f"- Dataset used for modeling: {model_results['dataset_used']}")
        lines.append(f"- Target column: {model_results['target_column']}")
        lines.append(f"- Feature columns used: {', '.join(model_results['feature_columns'])}")
        if model_results.get("group_column"):
            lines.append(f"- Group comparison column: {model_results['group_column']}")
    lines.append("")

    lines.append("## Final Interpretation")
    lines.append(
        "This pipeline now provides a strong exploratory and analytical foundation. "
        "The automatically discovered relationships can guide which variables should be emphasized "
        "in later graphing, domain interpretation, and predictive modeling. The next improvements "
        "would be domain-specific feature engineering, merged cross-file datasets, stronger statistical tests, "
        "and more targeted narrative generation."
    )

    return "\n".join(lines)


def run_summary(
    descriptions: Dict[str, dict],
    analysis_results: Dict[str, dict],
    relationship_results: Dict[str, dict],
    model_results: dict,
    shared_columns: dict,
) -> str:
    summary = build_final_summary(
        descriptions=descriptions,
        analysis_results=analysis_results,
        relationship_results=relationship_results,
        model_results=model_results,
        shared_columns=shared_columns,
    )
    save_text(summary, FINAL_SUMMARY_MD)
    return summary