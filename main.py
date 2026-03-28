from src.config import (
    ANALYSIS_FOLDER,
    DESCRIPTION_FOLDER,
    GRAPHS_FOLDER,
    MODELS_FOLDER,
    RELATIONSHIPS_FOLDER,
    SUMMARIES_FOLDER,
)
from src.describe_data import run_description
from src.analyze_trends import run_analysis
from src.discover_relationships import run_relationship_discovery
from src.model_likelihoods import run_likelihood_modeling
from src.summarize_results import run_summary
from src.utils import ensure_directories


def main() -> None:
    ensure_directories([
        DESCRIPTION_FOLDER,
        ANALYSIS_FOLDER,
        GRAPHS_FOLDER,
        MODELS_FOLDER,
        RELATIONSHIPS_FOLDER,
        SUMMARIES_FOLDER,
    ])

    print("Step 1: Describing datasets...")
    descriptions, datasets, shared_columns = run_description()

    print("Step 2: Analyzing trends and generating graphs...")
    analysis_results = run_analysis(datasets)

    print("Step 3: Discovering relationships automatically...")
    relationship_results = run_relationship_discovery(
        datasets=datasets,
        descriptions=descriptions,
    )

    print("Step 4: Running likelihood modeling...")
    model_results = run_likelihood_modeling(
        datasets=datasets,
        relationship_results=relationship_results,
    )

    print("Step 5: Writing final summary...")
    run_summary(
        descriptions=descriptions,
        analysis_results=analysis_results,
        relationship_results=relationship_results,
        model_results=model_results,
        shared_columns=shared_columns,
    )

    print("Pipeline complete.")
    print("Check the output/ folder for reports, graphs, and results.")


if __name__ == "__main__":
    main()