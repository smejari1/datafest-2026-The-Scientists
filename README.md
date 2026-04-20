# DataFest 2026 - The Scientists

An end-to-end Python analytics project for exploring healthcare and social determinants of health (SDOH) data. This repository contains two main workflows:

1. A general-purpose exploratory pipeline that profiles CSV files, summarizes trends, discovers relationships, and fits a simple likelihood model.
2. A research-question workflow focused on SDOH, emergency department utilization, geography, diagnosis groups, and time trends.

The code is designed for rapid exploratory analysis on local CSV files and produces Markdown reports, JSON artifacts, CSV summaries, and graphs.

## What This Project Does

The repository helps a team move from raw CSV files to interpretable outputs in a structured way:

- Profiles each dataset in the `data/` folder.
- Detects missing values, duplicate rows, column types, and shared columns across files.
- Generates descriptive statistics and histogram-style graphs for numeric features.
- Automatically searches for meaningful relationships between columns.
- Builds a logistic regression model on the largest available dataset when a binary target can be detected.
- Supports deeper research-question analysis for SDOH and healthcare utilization patterns.

## Repository Structure

```text
datafest-2026-The-Scientists-main/
├── main.py
├── helloworld.py
├── methods_used.txt
├── requirements.txt
├── SETUP.md
├── setup_env.ps1
├── README.md
└── src/
    ├── config.py
    ├── utils.py
    ├── describe_data.py
    ├── analyze_trends.py
    ├── discover_relationships.py
    ├── model_likelihoods.py
    ├── summarize_results.py
    ├── questions.py
    ├── execute_action_plan.py
    ├── execute_action_plan_strong.py
    ├── synthesize_from_outputs.py
    └── act_on_synthesis.py
```

## Expected Inputs

This repo expects local CSV files in a top-level `data/` folder:

```text
data/
├── file1.csv
├── file2.csv
└── ...
```

Important notes:

- `data/` is ignored by Git and is not currently checked into the repository.
- `output/` and `research_output/` are also Git-ignored and are generated locally when analyses run.
- The main exploratory pipeline scans all `*.csv` files directly inside `data/`.

For the research-question workflow in `src/questions.py`, the code is especially oriented around these files when available:

- `encounters.csv`
- `social_determinants.csv`
- `diagnosis.csv`
- `departments.csv`
- `tigercensuscodes.csv`

## Environment Setup

Use Python 3.11 for best compatibility.

### Windows Quick Start

```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Install Verification

```bash
python -c "import pandas, numpy, matplotlib, scipy, sklearn, seaborn, plotly, openpyxl, requests; print('All libraries installed successfully!')"
```

## Dependencies

Core libraries from `requirements.txt`:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `jupyterlab`
- `ipykernel`
- `seaborn`
- `plotly`
- `openpyxl`
- `requests`

The research-question workflow also optionally uses `folium` for interactive mapping if it is installed.

## Main Exploratory Pipeline

The primary entry point is `main.py`.

Run it from the repository root:

```bash
python main.py
```

### Pipeline Stages

#### 1. Dataset Description

Implemented in `src/describe_data.py`.

This stage:

- Finds CSV files under `data/`
- Loads them into pandas
- Records row counts, column counts, dtypes, missing values, duplicate rows, and sample rows
- Detects overlapping column names across files
- Writes a background summary of the datasets

#### 2. Trend Analysis and Graphing

Implemented in `src/analyze_trends.py`.

This stage:

- Separates numeric and categorical columns
- Produces descriptive statistics for numeric fields
- Generates histogram PNGs for up to the first three numeric columns in each dataset
- Writes a Markdown trend report

#### 3. Automated Relationship Discovery

Implemented in `src/discover_relationships.py`.

This stage:

- Classifies columns into roles such as identifier, binary, categorical, continuous numeric, discrete numeric, year-like, time-like, or text
- Excludes low-value columns such as IDs, text-like fields, columns with too much missingness, or columns with no variation
- Evaluates sensible pair types only
- Retains relationships that meet effect-size thresholds

Relationship types currently supported include:

- Numeric vs numeric via Pearson correlation
- Categorical vs numeric via normalized mean differences
- Categorical vs binary via rate differences
- Binary vs numeric via normalized mean differences
- Time vs numeric via correlation with year/date values

#### 4. Likelihood Modeling

Implemented in `src/model_likelihoods.py`.

This stage:

- Chooses the largest dataset as the primary modeling dataset
- Performs lightweight feature engineering
- Detects a binary target column if possible
- Selects candidate features from discovered relationships and predefined defaults
- Fits a logistic regression model
- Produces evaluation metrics and example predicted profiles

Feature engineering currently includes:

- `birth_year -> age`
- `age -> age_group`
- `income -> low_income`
- `employment_status -> unemployed`
- `insurance_status -> no_health_insurance`
- Aggregation of SDOH indicator columns into:
  - `social_determinant_score`
  - `high_social_determinant_burden`

Likely target columns include:

- `food_insecurity`
- `food_insecure`
- `high_social_determinant_burden`
- `target`
- `outcome`
- `label`

#### 5. Final Summary

Implemented in `src/summarize_results.py`.

This stage combines the major outputs into a single Markdown summary describing:

- Number of datasets analyzed
- Shared-column relationships
- Relationship counts retained
- Modeling dataset, target, and feature set

## Main Pipeline Outputs

Running `python main.py` creates an `output/` folder with these subfolders:

```text
output/
├── descriptions/
│   ├── dataset_background.md
│   ├── dataset_description.json
│   └── dataset_relationships.json
├── analysis/
│   ├── analysis_results.json
│   └── trend_report.md
├── graphs/
│   └── *.png
├── models/
│   ├── likelihood_report.md
│   └── likelihood_results.json
├── relationships/
│   ├── discovered_relationships.json
│   └── relationship_report.md
└── summaries/
    └── final_summary.md
```

## Research Questions Workflow

The project also contains a more targeted analysis script in `src/questions.py`.

This workflow is aimed at five research questions:

1. SDOH burden and emergency department utilization
2. High-utilizer phenotyping
3. Geography and SDOH overlap
4. Diagnosis groups and social needs
5. Temporal SDOH screening trends

Based on the code and method notes, this workflow produces outputs in `research_output/` such as:

- CSV summary tables
- Publication-style PNG charts
- Potential interactive geographic output if mapping dependencies are available

Because this script uses dataset-specific columns and file names, it is best suited for the DataFest healthcare dataset rather than arbitrary CSV collections.

## Research Questions Overview

### RQ1: SDOH Burden and ED Utilization

Compares ED visit rates between patients screened versus not screened for specific SDOH domains and applies Fisher's exact test to identify statistically notable differences.

### RQ2: High-Utilizer Phenotyping

Defines high utilizers as patients in the top decile of encounter counts, then compares their SDOH screening profiles and common diagnosis groups.

### RQ3: Geography x SDOH Overlap

Aggregates encounters, screening, and ED usage by geography and identifies potentially underscreened census tracts.

### RQ4: Diagnosis Group x Social Needs

Analyzes co-occurrence patterns between diagnosis groups and SDOH domains, including heatmap-style output.

### RQ5: Temporal SDOH Screening Trends

Tracks screening volume and screening rates over time, including year-over-year growth patterns.

## Configuration

Key configuration lives in `src/config.py`.

Important configurable values include:

- Data and output folder locations
- Sample row counts for profiling
- Current year used in age derivation
- Default target and feature candidate columns
- Relationship thresholds for missingness, cardinality, minimum rows, and effect size

If you want to tune sensitivity or reporting behavior, `config.py` is the first file to review.

## Typical Workflow

1. Create and activate the virtual environment.
2. Install dependencies.
3. Place CSV files inside `data/`.
4. Run `python main.py`.
5. Review the generated Markdown reports and JSON outputs in `output/`.
6. If working with the DataFest healthcare dataset, run the research-question script and inspect `research_output/`.

## Strengths of the Current Codebase

- Clear staged pipeline with separated responsibilities
- Reproducible local outputs
- Automatic relationship discovery with sensible filtering
- Simple baseline predictive modeling
- Research-question scripts tailored to healthcare and SDOH analysis

## Current Assumptions and Limitations

- The main pipeline expects flat CSV files directly in `data/`, not nested subfolders.
- The logistic modeling step only works when a binary-like target can be inferred.
- The model is intentionally lightweight and should be treated as a baseline, not a production prediction system.
- Automatically discovered relationships are associations, not causal findings.
- Some scripts are generic, while `src/questions.py` is tightly coupled to specific healthcare dataset schemas.
- `folium` is referenced as optional for mapping but is not included in `requirements.txt`.
- The repository currently does not include sample data, so first-time users must supply their own datasets.

## Useful Files to Read First

- `main.py`: top-level entry point for the general pipeline
- `src/config.py`: paths, thresholds, and defaults
- `src/questions.py`: research-question analysis workflow
- `methods_used.txt`: concise methods summary
- `SETUP.md`: environment setup instructions

## Future Improvements

- Add a committed sample dataset or synthetic demo data
- Add a dedicated CLI for selecting workflows and output locations
- Add tests for relationship discovery and modeling utilities
- Add notebook examples for common analysis scenarios
- Add schema validation for required research-question inputs
- Add model explainability outputs such as coefficients or feature importance summaries

## License / Usage

No license file is currently present in the repository. If this project will be shared externally, add a license and any data-use restrictions before distribution.
