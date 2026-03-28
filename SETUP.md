# Setup Instructions

## Creating a Python Virtual Environment

Follow these steps to set up the Python environment for this project:

### Windows

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Required Libraries

The project uses the following libraries for data analysis:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **scipy** - Scientific computing
- **scikit-learn** - Machine learning library
- **jupyter** - Interactive notebooks
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations
- **openpyxl** - Excel file support
- **requests** - HTTP library for API calls

## Verifying Installation

After activating your virtual environment and installing dependencies, you can verify the installation:

```bash
python -c "import pandas; import numpy; import matplotlib; print('All libraries installed successfully!')"
```

## Running Jupyter Notebooks

To start a Jupyter notebook server:

```bash
jupyter notebook
```

## Deactivating Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```
