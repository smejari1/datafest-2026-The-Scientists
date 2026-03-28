from pathlib import Path

# Base folders
DATA_FOLDER = Path("data")
OUTPUT_FOLDER = Path("output")

# Output subfolders
DESCRIPTION_FOLDER = OUTPUT_FOLDER / "descriptions"
ANALYSIS_FOLDER = OUTPUT_FOLDER / "analysis"
GRAPHS_FOLDER = OUTPUT_FOLDER / "graphs"
MODELS_FOLDER = OUTPUT_FOLDER / "models"
RELATIONSHIPS_FOLDER = OUTPUT_FOLDER / "relationships"
SUMMARIES_FOLDER = OUTPUT_FOLDER / "summaries"

# General settings
SAMPLE_ROWS = 5
TOP_UNIQUE_VALUES = 10
MAX_GRAPH_COLUMNS_PER_FILE = 5
CURRENT_YEAR = 2026

# Modeling config
DEFAULT_TARGET_CANDIDATES = [
    "food_insecurity",
    "food_insecure",
    "high_social_determinant_burden",
    "target",
    "outcome",
    "label",
]

DEFAULT_FEATURE_CANDIDATES = [
    "birth_year",
    "age",
    "age_group",
    "income",
    "income_level",
    "employment_status",
    "household_size",
    "housing_instability",
    "transportation_barrier",
    "snap_participation",
    "education_level",
    "insurance_status",
    "zip_code",
    "sex",
    "gender",
    "race",
    "ethnicity",
    "social_determinant_score",
    "low_income",
    "unemployed",
]

SOCIAL_DETERMINANT_INDICATOR_CANDIDATES = [
    "low_income",
    "housing_instability",
    "transportation_barrier",
    "unemployed",
    "no_health_insurance",
    "snap_participation",
    "food_access_barrier",
    "utility_insecurity",
]

# Relationship discovery thresholds
MAX_MISSINGNESS_PERCENT_FOR_RELATIONSHIP = 60.0
MAX_CARDINALITY_FOR_CATEGORICAL = 20
MIN_ROWS_FOR_COMPARISON = 15
TOP_RELATIONSHIPS_TO_REPORT = 25
TOP_RELATIONSHIPS_PER_DATASET = 12
CORRELATION_STRENGTH_THRESHOLD = 0.15
MEAN_DIFFERENCE_MIN_EFFECT = 0.10
CATEGORY_RATE_DIFF_THRESHOLD = 0.05

# Files
DESCRIPTION_JSON = DESCRIPTION_FOLDER / "dataset_description.json"
RELATIONSHIPS_JSON = DESCRIPTION_FOLDER / "dataset_relationships.json"
BACKGROUND_MD = DESCRIPTION_FOLDER / "dataset_background.md"

ANALYSIS_JSON = ANALYSIS_FOLDER / "analysis_results.json"
TREND_REPORT_MD = ANALYSIS_FOLDER / "trend_report.md"

MODEL_RESULTS_JSON = MODELS_FOLDER / "likelihood_results.json"
MODEL_REPORT_MD = MODELS_FOLDER / "likelihood_report.md"

DISCOVERED_RELATIONSHIPS_JSON = RELATIONSHIPS_FOLDER / "discovered_relationships.json"
DISCOVERED_RELATIONSHIPS_MD = RELATIONSHIPS_FOLDER / "relationship_report.md"

FINAL_SUMMARY_MD = SUMMARIES_FOLDER / "final_summary.md"