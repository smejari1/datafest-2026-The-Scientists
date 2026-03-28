from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    CURRENT_YEAR,
    DEFAULT_FEATURE_CANDIDATES,
    DEFAULT_TARGET_CANDIDATES,
    MODEL_REPORT_MD,
    MODEL_RESULTS_JSON,
    SOCIAL_DETERMINANT_INDICATOR_CANDIDATES,
)
from src.utils import (
    choose_primary_dataset,
    convert_boolean_like_columns,
    safe_numeric_conversion,
    save_json,
    save_text,
)


# -----------------------------------
# Feature engineering
# -----------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = convert_boolean_like_columns(df)

    # birth_year -> age
    if "birth_year" in df.columns:
        df["birth_year"] = safe_numeric_conversion(df["birth_year"])
        if df["birth_year"].notna().sum() > 0:
            df["age"] = CURRENT_YEAR - df["birth_year"]

    # age grouping
    if "age" in df.columns:
        df["age"] = safe_numeric_conversion(df["age"])
        if df["age"].notna().sum() > 0:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 18, 25, 35, 50, 65, 120],
                labels=["0-18", "19-25", "26-35", "36-50", "51-65", "66+"],
                include_lowest=True,
            )

    # numeric income -> low_income
    if "income" in df.columns:
        df["income"] = safe_numeric_conversion(df["income"])
        median_income = df["income"].median()
        if pd.notna(median_income):
            df["low_income"] = (df["income"] < median_income).astype(int)

    # unemployment from employment_status
    if "employment_status" in df.columns and "unemployed" not in df.columns:
        lowered = df["employment_status"].astype(str).str.strip().str.lower()
        df["unemployed"] = lowered.isin(
            ["unemployed", "not employed", "jobless", "none"]
        ).astype(int)

    # insurance_status -> no_health_insurance
    if "insurance_status" in df.columns and "no_health_insurance" not in df.columns:
        lowered = df["insurance_status"].astype(str).str.strip().str.lower()
        df["no_health_insurance"] = lowered.isin(
            ["none", "no", "uninsured", "no insurance"]
        ).astype(int)

    # standardize determinant-like columns to numeric
    for col in SOCIAL_DETERMINANT_INDICATOR_CANDIDATES:
        if col in df.columns:
            df[col] = safe_numeric_conversion(df[col])

    # derived social determinant score
    determinant_columns = [
        col for col in SOCIAL_DETERMINANT_INDICATOR_CANDIDATES
        if col in df.columns
    ]
    if determinant_columns:
        df["social_determinant_score"] = (
            df[determinant_columns].fillna(0).sum(axis=1)
        )
        df["high_social_determinant_burden"] = (
            df["social_determinant_score"] >= 2
        ).astype(int)

    # try to map common text target values
    for possible_target in ["food_insecurity", "food_insecure", "target", "outcome", "label"]:
        if possible_target in df.columns and df[possible_target].dtype == "object":
            lowered = df[possible_target].astype(str).str.strip().str.lower()
            mapping = {
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "food insecure": 1,
                "food secure": 0,
                "positive": 1,
                "negative": 0,
                "1": 1,
                "0": 0,
            }
            mapped = lowered.map(mapping)
            if mapped.notna().sum() > 0:
                df[possible_target] = mapped

    return df


# -----------------------------------
# Target / feature detection
# -----------------------------------

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    for col in DEFAULT_TARGET_CANDIDATES:
        if col in df.columns:
            return col

    # backup: find binary-like columns, prefer semantically meaningful names
    preferred_keywords = [
        "risk", "outcome", "target", "label", "insecurity", "burden", "flag", "status"
    ]

    binary_candidates = []
    for col in df.columns:
        values = sorted(df[col].dropna().unique().tolist())
        if len(values) == 2 and set(values).issubset({0, 1}):
            binary_candidates.append(col)

    if not binary_candidates:
        return None

    for keyword in preferred_keywords:
        for col in binary_candidates:
            if keyword in col.lower():
                return col

    return binary_candidates[0]


def _filter_bad_features(df: pd.DataFrame, columns: List[str], target_col: str) -> List[str]:
    good = []

    for col in columns:
        if col == target_col:
            continue
        if col not in df.columns:
            continue

        series = df[col]

        if series.isna().mean() >= 0.80:
            continue

        if series.nunique(dropna=True) <= 1:
            continue

        # avoid ID-like columns
        lower = col.lower()
        if lower == "id" or lower.endswith("_id") or "identifier" in lower:
            continue

        # avoid extreme-cardinality object columns
        if series.dtype == "object" and series.nunique(dropna=True) > 40:
            continue

        good.append(col)

    return good


def detect_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    relationship_results: Optional[Dict[str, dict]] = None,
    dataset_name: Optional[str] = None,
) -> List[str]:
    chosen: List[str] = []

    # 1. pull features from strongest discovered relationships if available
    if relationship_results and dataset_name and dataset_name in relationship_results:
        rels = relationship_results[dataset_name].get("top_relationships", [])
        for rel in rels:
            a = rel.get("column_a")
            b = rel.get("column_b")
            for col in [a, b]:
                if col and col != target_col and col in df.columns and col not in chosen:
                    chosen.append(col)

    # 2. add default candidates
    for col in DEFAULT_FEATURE_CANDIDATES:
        if col in df.columns and col != target_col and col not in chosen:
            chosen.append(col)

    # 3. fallback: grab reasonable numeric + low-cardinality categorical columns
    if not chosen:
        for col in df.columns:
            if col == target_col:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                chosen.append(col)
            elif series.nunique(dropna=True) <= 15:
                chosen.append(col)

    chosen = _filter_bad_features(df, chosen, target_col)

    # prefer a moderate number of features for interpretability
    return chosen[:12]


def pick_best_group_column(df: pd.DataFrame, target_col: str) -> Optional[str]:
    preferred = [
        "age_group",
        "birth_year",
        "income_level",
        "employment_status",
        "education_level",
        "race",
        "ethnicity",
        "gender",
        "sex",
        "zip_code",
    ]

    for col in preferred:
        if col in df.columns and col != target_col:
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 20:
                return col

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        nunique = series.nunique(dropna=True)
        if (series.dtype == "object" or str(series.dtype).startswith("category")) and 2 <= nunique <= 15:
            return col

    return None


# -----------------------------------
# Group comparison
# -----------------------------------

def calculate_group_rates(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    subset = df[[group_col, target_col]].dropna()
    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset.groupby(group_col)[target_col]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    grouped["percent_rate"] = grouped["mean"] * 100
    grouped = grouped[grouped["count"] >= 3]
    return grouped.sort_values("percent_rate", ascending=False)


# -----------------------------------
# Logistic modeling
# -----------------------------------

def run_logistic_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[Optional[Pipeline], dict]:
    working = df[feature_cols + [target_col]].copy()
    working = working.dropna(subset=[target_col])

    if working.empty:
        return None, {"error": "No rows available after dropping missing target values."}

    X = working[feature_cols]
    y = working[target_col]

    if y.nunique(dropna=True) < 2:
        return None, {"error": "Target column does not have at least two classes."}

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    if not categorical_cols and not numeric_cols:
        return None, {"error": "No usable feature columns found."}

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_cols,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    test_size = 0.2 if len(working) >= 20 else 0.3

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    sample_predictions = X_test.copy()
    sample_predictions["actual"] = y_test.values
    sample_predictions["predicted_probability"] = y_prob
    sample_predictions["predicted_class"] = y_pred

    return model, {
        "classification_report": report,
        "sample_predictions": sample_predictions.head(20).to_dict(orient="records"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }


# -----------------------------------
# Narrative generation
# -----------------------------------

def _best_rate_summary(group_rates: pd.DataFrame, group_col: str) -> Optional[dict]:
    if group_rates.empty or len(group_rates) < 2:
        return None

    top_row = group_rates.iloc[0]
    bottom_row = group_rates.iloc[-1]

    return {
        "top_group": top_row[group_col],
        "top_rate": float(top_row["percent_rate"]),
        "bottom_group": bottom_row[group_col],
        "bottom_rate": float(bottom_row["percent_rate"]),
        "difference_points": float(top_row["percent_rate"] - bottom_row["percent_rate"]),
    }


def generate_likelihood_narrative(
    dataset_name: str,
    target_col: str,
    feature_cols: List[str],
    group_col: Optional[str],
    group_rates: pd.DataFrame,
    model_output: dict,
) -> str:
    lines = []
    lines.append("# Likelihood Modeling Report\n")
    lines.append(f"## Dataset Used\n{dataset_name}\n")
    lines.append(f"## Outcome Modeled\n`{target_col}`\n")
    lines.append(f"## Feature Columns Used\n{', '.join(feature_cols) if feature_cols else 'None'}\n")

    if group_col and not group_rates.empty:
        lines.append(f"## Observed Group Rates by `{group_col}`\n")
        for _, row in group_rates.iterrows():
            lines.append(
                f"- {row[group_col]}: {row['percent_rate']:.2f}% observed rate "
                f"(n={int(row['count'])}, positive cases={int(row['sum'])})"
            )
        lines.append("")

        summary = _best_rate_summary(group_rates, group_col)
        if summary:
            lines.append("## Group Comparison Highlight")
            lines.append(
                f"The highest observed rate appears in `{summary['top_group']}` "
                f"at {summary['top_rate']:.2f}%, while the lowest appears in "
                f"`{summary['bottom_group']}` at {summary['bottom_rate']:.2f}%. "
                f"This is a difference of {summary['difference_points']:.2f} percentage points."
            )
            lines.append("")

    if "error" in model_output:
        lines.append("## Model Status")
        lines.append(f"Model could not be fit: {model_output['error']}")
    else:
        lines.append("## Model Status")
        lines.append("A logistic regression model was fit to estimate outcome likelihood from the selected features.\n")

        lines.append("## Data Split")
        lines.append(f"- Training rows: {model_output['train_rows']}")
        lines.append(f"- Testing rows: {model_output['test_rows']}\n")

        lines.append("## Evaluation Summary")
        clf_report = model_output["classification_report"]
        if "accuracy" in clf_report:
            lines.append(f"- Accuracy: {clf_report['accuracy']:.3f}")
        if "1" in clf_report:
            lines.append(f"- Positive class precision: {clf_report['1']['precision']:.3f}")
            lines.append(f"- Positive class recall: {clf_report['1']['recall']:.3f}")
            lines.append(f"- Positive class F1-score: {clf_report['1']['f1-score']:.3f}")
        lines.append("")

        lines.append("## Example Predicted Profiles")
        for row in model_output["sample_predictions"][:10]:
            prob = row.get("predicted_probability")
            actual = row.get("actual")
            profile_parts = []
            for k, v in row.items():
                if k not in {"predicted_probability", "actual", "predicted_class"}:
                    profile_parts.append(f"{k}={v}")
            lines.append(
                f"- Predicted probability: {prob:.3f}, actual outcome: {actual}, profile: "
                + ", ".join(profile_parts)
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append(
        "The observed rates describe how often the outcome appears in different groups within the dataset. "
        "The logistic model estimates the likelihood of the outcome using the selected features, which were "
        "chosen automatically from discovered relationships and candidate variables where possible. "
        "These results should be interpreted as associations in the available data, not proof of causation."
    )

    return "\n".join(lines)


# -----------------------------------
# Main runner
# -----------------------------------

def run_likelihood_modeling(
    datasets: Dict[str, pd.DataFrame],
    relationship_results: Optional[Dict[str, dict]] = None,
) -> dict:
    primary_name = choose_primary_dataset(datasets)
    if primary_name is None:
        result = {"error": "No datasets available for modeling."}
        save_json(result, MODEL_RESULTS_JSON)
        save_text("# Likelihood Modeling Report\n\nNo datasets available for modeling.", MODEL_REPORT_MD)
        return result

    df = datasets[primary_name].copy()
    df = create_features(df)

    target_col = detect_target_column(df)
    if target_col is None:
        result = {
            "error": (
                "No target column was detected. Add a binary target like "
                "'food_insecurity', 'food_insecure', or 'high_social_determinant_burden'."
            ),
            "dataset_used": primary_name,
        }
        save_json(result, MODEL_RESULTS_JSON)
        save_text(
            "# Likelihood Modeling Report\n\n"
            f"Dataset used: {primary_name}\n\n"
            "No target column was detected.",
            MODEL_REPORT_MD,
        )
        return result

    df[target_col] = safe_numeric_conversion(df[target_col])

    feature_cols = detect_feature_columns(
        df=df,
        target_col=target_col,
        relationship_results=relationship_results,
        dataset_name=primary_name,
    )

    if not feature_cols:
        result = {
            "error": "No usable feature columns were available for modeling.",
            "dataset_used": primary_name,
            "target_column": target_col,
        }
        save_json(result, MODEL_RESULTS_JSON)
        save_text(
            "# Likelihood Modeling Report\n\n"
            f"Dataset used: {primary_name}\n\n"
            f"Outcome modeled: {target_col}\n\n"
            "No usable feature columns were available for modeling.",
            MODEL_REPORT_MD,
        )
        return result

    group_col = pick_best_group_column(df, target_col)

    group_rates = pd.DataFrame()
    if group_col:
        group_rates = calculate_group_rates(df, group_col, target_col)

    _, model_output = run_logistic_model(df, feature_cols, target_col)

    result = {
        "dataset_used": primary_name,
        "target_column": target_col,
        "feature_columns": feature_cols,
        "group_column": group_col,
        "group_rates": group_rates.to_dict(orient="records") if not group_rates.empty else [],
        "model_output": model_output,
    }

    narrative = generate_likelihood_narrative(
        dataset_name=primary_name,
        target_col=target_col,
        feature_cols=feature_cols,
        group_col=group_col,
        group_rates=group_rates,
        model_output=model_output,
    )

    save_json(result, MODEL_RESULTS_JSON)
    save_text(narrative, MODEL_REPORT_MD)

    return result