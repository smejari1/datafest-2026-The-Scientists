import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import ensure_directories
from config import (
    DESCRIPTION_FOLDER,
    ANALYSIS_FOLDER,
    GRAPHS_FOLDER,
    SUMMARIES_FOLDER,
)
from describe_data import run_description
from analyze_trends import run_analysis
from summarize_results import run_summary


def main():
    ensure_directories([
        DESCRIPTION_FOLDER,
        ANALYSIS_FOLDER,
        GRAPHS_FOLDER,
        SUMMARIES_FOLDER,
    ])

    descriptions, shared_columns = run_description()
    analysis_results = run_analysis()
    run_summary(descriptions, analysis_results, shared_columns)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()