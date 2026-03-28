param(
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment with Python $PythonVersion..."
py -$PythonVersion -m venv .venv

Write-Host "Activating environment and installing dependencies..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Done. Activate with: .venv\\Scripts\\activate"
