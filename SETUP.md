# Setup Instructions

## Quick Start (Windows)

From repo root, run:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

This creates .venv and installs all dependencies with Python 3.11.

## Creating a Python Virtual Environment

Follow these steps to set up the Python environment for this project:

Important: Use Python 3.11 for this repo. Python 3.14 may fail for some scientific packages.

### Windows

```bash
# Verify available Python versions
py -0p

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
python -m pip install -r requirements.txt
```

## Required Libraries

The project uses the following libraries for data analysis:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **scipy** - Scientific computing
- **scikit-learn** - Machine learning library
- **jupyterlab** - Interactive notebooks and data apps
- **ipykernel** - Python kernel support for notebooks
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations
- **openpyxl** - Excel file support
- **requests** - HTTP library for API calls

## Verifying Installation

After activating your virtual environment and installing dependencies, you can verify the installation:

```bash
python -c "import pandas, numpy, matplotlib, scipy, sklearn, seaborn, plotly, openpyxl, requests; print('All libraries installed successfully!')"
```

## Running Jupyter Notebooks

To start a Jupyter notebook server:

```bash
jupyter lab
```

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```
