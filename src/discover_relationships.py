from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import time
import warnings

import pandas as pd

from src.config import (
    CATEGORY_RATE_DIFF_THRESHOLD,
    CORRELATION_STRENGTH_THRESHOLD,
    CURRENT_YEAR,
    DISCOVERED_RELATIONSHIPS_JSON,
    DISCOVERED_RELATIONSHIPS_MD,
    MAX_CARDINALITY_FOR_CATEGORICAL,
    MAX_MISSINGNESS_PERCENT_FOR_RELATIONSHIP,
    MEAN_DIFFERENCE_MIN_EFFECT,
    MIN_ROWS_FOR_COMPARISON,
    TOP_RELATIONSHIPS_PER_DATASET,
    TOP_RELATIONSHIPS_TO_REPORT,
)
from src.utils import save_json, save_text


# -----------------------------
# Basic helpers
# -----------------------------

def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_bool_like(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_vals = set(non_null.unique().tolist())
    return unique_vals.issubset({0, 1, True, False})


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return 0


def _missing_percent(series: pd.Series) -> float:
    try:
        return float(series.isna().mean() * 100)
    except Exception:
        return 100.0


def _maybe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_for_role_checks(col_name: str) -> str:
    return str(col_name).strip().lower()


def _is_identifier_column(col_name: str, series: pd.Series) -> bool:
    name = _normalize_for_role_checks(col_name)
    nunique = _safe_nunique(series)
    non_null_count = int(series.notna().sum())

    if name == "id" or name.endswith("_id"):
        return True

    if "identifier" in name or "record_id" in name or "person_id" in name or "patient_id" in name:
        return True

    # If nearly every row is unique, it behaves like an ID.
    if non_null_count > 0 and nunique / max(non_null_count, 1) > 0.95 and nunique > 20:
        return True

    return False


def _is_probable_text_column(series: pd.Series) -> bool:
    if _is_numeric(series):
        return False

    nunique = _safe_nunique(series)
    if nunique == 0:
        return False

    # High-cardinality object columns often behave like free text / notes / names.
    return nunique > 50


def _looks_like_time_column(col_name: str, series: pd.Series) -> bool:
    name = _normalize_for_role_checks(col_name)
    if any(token in name for token in ["date", "time", "timestamp", "month", "year"]):
        return True

    # Avoid expensive full-column parsing for clearly non-temporal dtypes.
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    # Sample a subset to keep discovery responsive on large datasets.
    sample = non_null.astype(str).head(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")

    return parsed.notna().sum() >= max(5, int(0.25 * len(sample)))


def _column_role(col_name: str, series: pd.Series) -> str:
    if _is_identifier_column(col_name, series):
        return "identifier"

    if _looks_like_time_column(col_name, series):
        if "year" in _normalize_for_role_checks(col_name):
            return "year_like"
        return "time_like"

    if _is_bool_like(series):
        return "binary"

    if _is_numeric(series):
        nunique = _safe_nunique(series)
        if nunique <= 10:
            return "numeric_discrete"
        return "numeric_continuous"

    if _is_probable_text_column(series):
        return "text"

    nunique = _safe_nunique(series)
    if nunique <= MAX_CARDINALITY_FOR_CATEGORICAL:
        return "categorical"

    return "text"


def _is_usable_column(col_name: str, series: pd.Series) -> Tuple[bool, Optional[str]]:
    if _missing_percent(series) > MAX_MISSINGNESS_PERCENT_FOR_RELATIONSHIP:
        return False, "too_much_missingness"

    if _safe_nunique(series) <= 1:
        return False, "no_variation"

    role = _column_role(col_name, series)
    if role in {"identifier", "text"}:
        return False, role

    return True, None


def _effect_label_from_abs_corr(abs_corr: float) -> str:
    if abs_corr >= 0.7:
        return "very strong"
    if abs_corr >= 0.5:
        return "strong"
    if abs_corr >= 0.3:
        return "moderate"
    if abs_corr >= 0.15:
        return "weak"
    return "very weak"


def _safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _series_std(series: pd.Series) -> Optional[float]:
    try:
        val = series.std()
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


# -----------------------------
# Pair comparison methods
# -----------------------------

def compare_numeric_numeric(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> Optional[dict]:
    sub = df[[col_a, col_b]].copy()
    sub[col_a] = _maybe_to_numeric(sub[col_a])
    sub[col_b] = _maybe_to_numeric(sub[col_b])
    sub = sub.dropna()

    if len(sub) < MIN_ROWS_FOR_COMPARISON:
        return None

    corr = sub[col_a].corr(sub[col_b], method="pearson")
    if pd.isna(corr):
        return None

    abs_corr = abs(float(corr))
    if abs_corr < CORRELATION_STRENGTH_THRESHOLD:
        return None

    direction = "positive" if corr > 0 else "negative"
    strength = _effect_label_from_abs_corr(abs_corr)

    return {
        "comparison_type": "numeric_vs_numeric",
        "column_a": col_a,
        "column_b": col_b,
        "rows_used": int(len(sub)),
        "score": round(abs_corr, 4),
        "effect_size": round(abs_corr, 4),
        "metric_name": "pearson_correlation",
        "metric_value": round(float(corr), 4),
        "strength_label": strength,
        "direction": direction,
        "narrative": (
            f"`{col_a}` and `{col_b}` show a {strength} {direction} linear association "
            f"(correlation = {corr:.3f}, n = {len(sub)})."
        ),
    }


def compare_categorical_numeric(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
) -> Optional[dict]:
    sub = df[[cat_col, num_col]].copy()
    sub[num_col] = _maybe_to_numeric(sub[num_col])
    sub = sub.dropna()

    if len(sub) < MIN_ROWS_FOR_COMPARISON:
        return None

    group_stats = sub.groupby(cat_col)[num_col].agg(["mean", "median", "count"]).reset_index()
    group_stats = group_stats[group_stats["count"] >= max(3, MIN_ROWS_FOR_COMPARISON // 5)]

    if len(group_stats) < 2:
        return None

    global_std = _series_std(sub[num_col])
    if global_std is None or global_std == 0:
        return None

    max_row = group_stats.loc[group_stats["mean"].idxmax()]
    min_row = group_stats.loc[group_stats["mean"].idxmin()]
    mean_diff = float(max_row["mean"] - min_row["mean"])
    normalized_diff = abs(mean_diff) / global_std

    if normalized_diff < MEAN_DIFFERENCE_MIN_EFFECT:
        return None

    return {
        "comparison_type": "categorical_vs_numeric",
        "column_a": cat_col,
        "column_b": num_col,
        "rows_used": int(len(sub)),
        "score": round(float(normalized_diff), 4),
        "effect_size": round(float(normalized_diff), 4),
        "metric_name": "normalized_mean_difference",
        "metric_value": round(float(normalized_diff), 4),
        "group_summary": group_stats.to_dict(orient="records"),
        "top_group": str(max_row[cat_col]),
        "bottom_group": str(min_row[cat_col]),
        "top_group_mean": round(float(max_row["mean"]), 4),
        "bottom_group_mean": round(float(min_row["mean"]), 4),
        "narrative": (
            f"`{num_col}` differs across `{cat_col}` groups. "
            f"The highest average appears in `{max_row[cat_col]}` ({max_row['mean']:.3f}) "
            f"and the lowest in `{min_row[cat_col]}` ({min_row['mean']:.3f}), "
            f"with a normalized difference of {normalized_diff:.3f}."
        ),
    }


def compare_binary_categorical(
    df: pd.DataFrame,
    cat_col: str,
    binary_col: str,
) -> Optional[dict]:
    sub = df[[cat_col, binary_col]].copy()
    sub[binary_col] = _maybe_to_numeric(sub[binary_col])
    sub = sub.dropna()

    if len(sub) < MIN_ROWS_FOR_COMPARISON:
        return None

    if set(sub[binary_col].dropna().unique().tolist()) - {0, 1}:
        return None

    group_rates = sub.groupby(cat_col)[binary_col].agg(["mean", "count", "sum"]).reset_index()
    group_rates = group_rates[group_rates["count"] >= max(3, MIN_ROWS_FOR_COMPARISON // 5)]

    if len(group_rates) < 2:
        return None

    max_row = group_rates.loc[group_rates["mean"].idxmax()]
    min_row = group_rates.loc[group_rates["mean"].idxmin()]
    rate_diff = float(max_row["mean"] - min_row["mean"])

    if abs(rate_diff) < CATEGORY_RATE_DIFF_THRESHOLD:
        return None

    return {
        "comparison_type": "categorical_vs_binary",
        "column_a": cat_col,
        "column_b": binary_col,
        "rows_used": int(len(sub)),
        "score": round(abs(rate_diff), 4),
        "effect_size": round(abs(rate_diff), 4),
        "metric_name": "rate_difference",
        "metric_value": round(rate_diff, 4),
        "group_rates": [
            {
                cat_col: row[cat_col],
                "count": int(row["count"]),
                "positive_cases": int(row["sum"]),
                "rate": round(float(row["mean"]), 4),
                "percent_rate": round(float(row["mean"]) * 100, 2),
            }
            for _, row in group_rates.iterrows()
        ],
        "top_group": str(max_row[cat_col]),
        "bottom_group": str(min_row[cat_col]),
        "top_group_rate": round(float(max_row["mean"]), 4),
        "bottom_group_rate": round(float(min_row["mean"]), 4),
        "narrative": (
            f"The observed rate of `{binary_col}` varies across `{cat_col}` groups. "
            f"`{max_row[cat_col]}` has the highest observed rate ({max_row['mean'] * 100:.2f}%), "
            f"while `{min_row[cat_col]}` has the lowest ({min_row['mean'] * 100:.2f}%)."
        ),
    }


def compare_binary_numeric(
    df: pd.DataFrame,
    binary_col: str,
    num_col: str,
) -> Optional[dict]:
    sub = df[[binary_col, num_col]].copy()
    sub[binary_col] = _maybe_to_numeric(sub[binary_col])
    sub[num_col] = _maybe_to_numeric(sub[num_col])
    sub = sub.dropna()

    if len(sub) < MIN_ROWS_FOR_COMPARISON:
        return None

    if set(sub[binary_col].dropna().unique().tolist()) - {0, 1}:
        return None

    group_stats = sub.groupby(binary_col)[num_col].agg(["mean", "median", "count"]).reset_index()
    if len(group_stats) < 2:
        return None

    global_std = _series_std(sub[num_col])
    if global_std is None or global_std == 0:
        return None

    mean_0 = float(group_stats.loc[group_stats[binary_col] == 0, "mean"].iloc[0]) if (group_stats[binary_col] == 0).any() else None
    mean_1 = float(group_stats.loc[group_stats[binary_col] == 1, "mean"].iloc[0]) if (group_stats[binary_col] == 1).any() else None
    if mean_0 is None or mean_1 is None:
        return None

    normalized_diff = abs(mean_1 - mean_0) / global_std
    if normalized_diff < MEAN_DIFFERENCE_MIN_EFFECT:
        return None

    direction = "higher" if mean_1 > mean_0 else "lower"

    return {
        "comparison_type": "binary_vs_numeric",
        "column_a": binary_col,
        "column_b": num_col,
        "rows_used": int(len(sub)),
        "score": round(float(normalized_diff), 4),
        "effect_size": round(float(normalized_diff), 4),
        "metric_name": "normalized_mean_difference",
        "metric_value": round(float(normalized_diff), 4),
        "mean_when_0": round(mean_0, 4),
        "mean_when_1": round(mean_1, 4),
        "narrative": (
            f"`{num_col}` is {direction} on average when `{binary_col}` = 1 "
            f"than when `{binary_col}` = 0 "
            f"(means: {mean_1:.3f} vs {mean_0:.3f})."
        ),
    }


def compare_time_numeric(
    df: pd.DataFrame,
    time_col: str,
    num_col: str,
) -> Optional[dict]:
    sub = df[[time_col, num_col]].copy()
    sub[num_col] = _maybe_to_numeric(sub[num_col])

    if "year" in time_col.lower():
        sub[time_col] = _maybe_to_numeric(sub[time_col])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sub[time_col], errors="coerce", format="mixed")
        if parsed.notna().sum() < MIN_ROWS_FOR_COMPARISON:
            return None
        sub[time_col] = parsed.map(pd.Timestamp.toordinal)

    sub = sub.dropna()

    if len(sub) < MIN_ROWS_FOR_COMPARISON:
        return None

    corr = sub[time_col].corr(sub[num_col])
    if pd.isna(corr):
        return None

    abs_corr = abs(float(corr))
    if abs_corr < CORRELATION_STRENGTH_THRESHOLD:
        return None

    direction = "increase" if corr > 0 else "decrease"
    strength = _effect_label_from_abs_corr(abs_corr)

    return {
        "comparison_type": "time_vs_numeric",
        "column_a": time_col,
        "column_b": num_col,
        "rows_used": int(len(sub)),
        "score": round(abs_corr, 4),
        "effect_size": round(abs_corr, 4),
        "metric_name": "time_correlation",
        "metric_value": round(float(corr), 4),
        "strength_label": strength,
        "direction": direction,
        "narrative": (
            f"`{num_col}` shows a {strength} tendency to {direction} over `{time_col}` "
            f"(correlation = {corr:.3f}, n = {len(sub)})."
        ),
    }


# -----------------------------
# Dataset-level discovery
# -----------------------------

def classify_columns(df: pd.DataFrame) -> Dict[str, dict]:
    info = {}

    for col in df.columns:
        role = _column_role(col, df[col])
        usable, exclusion_reason = _is_usable_column(col, df[col])
        info[col] = {
            "role": role,
            "usable_for_relationships": usable,
            "exclusion_reason": exclusion_reason,
            "missing_percent": round(_missing_percent(df[col]), 2),
            "unique_count": _safe_nunique(df[col]),
        }

    return info


def generate_candidate_pairs(df: pd.DataFrame, column_info: Dict[str, dict]) -> List[Tuple[str, str, str]]:
    """
    Returns tuples:
    (comparison_method_key, col_a, col_b)
    """
    usable_cols = [
        col for col, meta in column_info.items()
        if meta["usable_for_relationships"]
    ]

    pairs: List[Tuple[str, str, str]] = []

    for i in range(len(usable_cols)):
        for j in range(i + 1, len(usable_cols)):
            a = usable_cols[i]
            b = usable_cols[j]
            role_a = column_info[a]["role"]
            role_b = column_info[b]["role"]

            role_set = {role_a, role_b}

            if role_a in {"numeric_continuous", "numeric_discrete"} and role_b in {"numeric_continuous", "numeric_discrete"}:
                pairs.append(("numeric_numeric", a, b))
                continue

            if role_set == {"categorical", "binary"}:
                cat_col = a if role_a == "categorical" else b
                bin_col = a if role_a == "binary" else b
                pairs.append(("categorical_binary", cat_col, bin_col))
                continue

            if role_set == {"binary", "numeric_continuous"} or role_set == {"binary", "numeric_discrete"}:
                bin_col = a if role_a == "binary" else b
                num_col = a if role_a in {"numeric_continuous", "numeric_discrete"} else b
                pairs.append(("binary_numeric", bin_col, num_col))
                continue

            if role_set == {"categorical", "numeric_continuous"} or role_set == {"categorical", "numeric_discrete"}:
                cat_col = a if role_a == "categorical" else b
                num_col = a if role_a in {"numeric_continuous", "numeric_discrete"} else b
                pairs.append(("categorical_numeric", cat_col, num_col))
                continue

            if role_a in {"year_like", "time_like"} and role_b in {"numeric_continuous", "numeric_discrete"}:
                pairs.append(("time_numeric", a, b))
                continue

            if role_b in {"year_like", "time_like"} and role_a in {"numeric_continuous", "numeric_discrete"}:
                pairs.append(("time_numeric", b, a))
                continue

    return pairs


def run_pair_method(df: pd.DataFrame, method_key: str, col_a: str, col_b: str) -> Optional[dict]:
    if method_key == "numeric_numeric":
        return compare_numeric_numeric(df, col_a, col_b)
    if method_key == "categorical_numeric":
        return compare_categorical_numeric(df, col_a, col_b)
    if method_key == "categorical_binary":
        return compare_binary_categorical(df, col_a, col_b)
    if method_key == "binary_numeric":
        return compare_binary_numeric(df, col_a, col_b)
    if method_key == "time_numeric":
        return compare_time_numeric(df, col_a, col_b)
    return None


def discover_relationships_for_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    description: Optional[dict] = None,
) -> dict:
    started_at = time.perf_counter()
    column_info = classify_columns(df)
    candidate_pairs = generate_candidate_pairs(df, column_info)
    usable_columns = sum(1 for meta in column_info.values() if meta["usable_for_relationships"])
    excluded_columns_count = len(column_info) - usable_columns

    print(
        f"    > {dataset_name}: rows={len(df)}, cols={df.shape[1]}, "
        f"usable_cols={usable_columns}, excluded_cols={excluded_columns_count}"
    )
    print(f"    > {dataset_name}: evaluating {len(candidate_pairs)} candidate pair(s)...")

    findings = []
    progress_interval = max(100, len(candidate_pairs) // 10) if candidate_pairs else 100

    for idx, (method_key, col_a, col_b) in enumerate(candidate_pairs, start=1):
        if idx % progress_interval == 0 or idx == len(candidate_pairs):
            elapsed = time.perf_counter() - started_at
            percent = (idx / len(candidate_pairs) * 100) if candidate_pairs else 100.0
            print(
                f"    > {dataset_name}: processed {idx}/{len(candidate_pairs)} pairs "
                f"({percent:.1f}%) in {elapsed:.1f}s"
            )
        try:
            result = run_pair_method(df, method_key, col_a, col_b)
            if result:
                result["dataset_name"] = dataset_name
                findings.append(result)
        except Exception as exc:
            findings.append({
                "dataset_name": dataset_name,
                "comparison_type": method_key,
                "column_a": col_a,
                "column_b": col_b,
                "error": str(exc),
                "score": -1,
            })

    valid_findings = [f for f in findings if "error" not in f]
    valid_findings.sort(key=lambda x: x.get("score", 0), reverse=True)

    elapsed_total = time.perf_counter() - started_at
    print(
        f"    > {dataset_name}: retained {len(valid_findings)} relationship(s) from "
        f"{len(candidate_pairs)} candidate pair(s) in {elapsed_total:.1f}s."
    )

    top_findings = valid_findings[:TOP_RELATIONSHIPS_PER_DATASET]

    excluded_columns = {
        col: meta for col, meta in column_info.items()
        if not meta["usable_for_relationships"]
    }

    return {
        "dataset_name": dataset_name,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "column_info": column_info,
        "excluded_columns": excluded_columns,
        "candidate_pair_count": len(candidate_pairs),
        "valid_relationship_count": len(valid_findings),
        "top_relationships": top_findings,
        "all_relationships": valid_findings,
    }


# -----------------------------
# Narrative generation
# -----------------------------

def _group_findings_by_type(findings: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for finding in findings:
        grouped.setdefault(finding["comparison_type"], []).append(finding)
    return grouped


def _build_dataset_narrative(dataset_result: dict) -> str:
    name = dataset_result["dataset_name"]
    column_info = dataset_result["column_info"]
    top_findings = dataset_result["top_relationships"]

    usable_count = sum(1 for m in column_info.values() if m["usable_for_relationships"])
    excluded_count = sum(1 for m in column_info.values() if not m["usable_for_relationships"])

    lines = []
    lines.append(f"## {name}")
    lines.append(f"- Rows: {dataset_result['row_count']}")
    lines.append(f"- Columns: {dataset_result['column_count']}")
    lines.append(f"- Usable relationship columns: {usable_count}")
    lines.append(f"- Excluded columns: {excluded_count}")
    lines.append(f"- Candidate comparisons evaluated: {dataset_result['candidate_pair_count']}")
    lines.append(f"- Relationships retained: {dataset_result['valid_relationship_count']}")
    lines.append("")

    if not top_findings:
        lines.append("No meaningful relationships met the current thresholds for reporting.")
        lines.append("")
        return "\n".join(lines)

    grouped = _group_findings_by_type(top_findings)

    lines.append("### Strongest findings")
    for finding in top_findings[:6]:
        lines.append(f"- {finding['narrative']}")
    lines.append("")

    if "numeric_vs_numeric" in grouped:
        lines.append("### Numeric patterns")
        for finding in grouped["numeric_vs_numeric"][:3]:
            lines.append(f"- {finding['narrative']}")
        lines.append("")

    if "categorical_vs_numeric" in grouped:
        lines.append("### Group differences in numeric measures")
        for finding in grouped["categorical_vs_numeric"][:3]:
            lines.append(f"- {finding['narrative']}")
        lines.append("")

    if "categorical_vs_binary" in grouped:
        lines.append("### Group differences in outcome rates")
        for finding in grouped["categorical_vs_binary"][:3]:
            lines.append(f"- {finding['narrative']}")
        lines.append("")

    if "time_vs_numeric" in grouped:
        lines.append("### Time-related trends")
        for finding in grouped["time_vs_numeric"][:3]:
            lines.append(f"- {finding['narrative']}")
        lines.append("")

    lines.append("### Interpretation")
    lines.append(
        "These findings were selected automatically based on variable type, data availability, "
        "and effect-size thresholds. They should be treated as association patterns that may guide "
        "later modeling, graph generation, and deeper domain-specific interpretation."
    )
    lines.append("")

    return "\n".join(lines)


def build_relationship_report(all_results: Dict[str, dict]) -> str:
    all_findings = []
    for dataset_name, dataset_result in all_results.items():
        for finding in dataset_result["all_relationships"]:
            enriched = dict(finding)
            enriched["dataset_name"] = dataset_name
            all_findings.append(enriched)

    all_findings.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_global = all_findings[:TOP_RELATIONSHIPS_TO_REPORT]

    lines = []
    lines.append("# Automated Relationship Discovery Report\n")
    lines.append("## Overview")
    lines.append(
        "This report summarizes automatically discovered relationships across the datasets. "
        "The system classified variables by role, filtered out low-value columns such as identifiers "
        "and high-cardinality text, selected statistically sensible pairings, and retained the "
        "strongest findings using effect-size thresholds."
    )
    lines.append("")

    if top_global:
        lines.append("## Top findings across all datasets")
        for finding in top_global[:10]:
            lines.append(f"- [{finding['dataset_name']}] {finding['narrative']}")
        lines.append("")

    for dataset_name, dataset_result in all_results.items():
        lines.append(_build_dataset_narrative(dataset_result))

    lines.append("## Caution")
    lines.append(
        "Automatically discovered relationships can highlight patterns, but they do not prove causation. "
        "Some findings may still reflect confounding variables, uneven group sizes, or chance patterns from "
        "many comparisons. These results are best used as a guided discovery layer before targeted modeling."
    )

    return "\n".join(lines)


# -----------------------------
# Main runner
# -----------------------------

def run_relationship_discovery(
    datasets: Dict[str, pd.DataFrame],
    descriptions: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    started_at = time.perf_counter()
    descriptions = descriptions or {}
    results: Dict[str, dict] = {}

    print(f"Running relationship discovery for {len(datasets)} dataset(s)...")

    total = len(datasets)
    for index, (dataset_name, df) in enumerate(datasets.items(), start=1):
        print(f"  - [{index}/{total}] Discovering relationships in {dataset_name}...")
        dataset_description = descriptions.get(dataset_name)
        results[dataset_name] = discover_relationships_for_dataset(
            dataset_name=dataset_name,
            df=df,
            description=dataset_description,
        )

    report = build_relationship_report(results)
    save_json(results, DISCOVERED_RELATIONSHIPS_JSON)
    save_text(report, DISCOVERED_RELATIONSHIPS_MD)

    total_elapsed = time.perf_counter() - started_at
    print(f"Relationship discovery complete in {total_elapsed:.1f}s.")

    return results