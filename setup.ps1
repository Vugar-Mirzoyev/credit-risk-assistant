<#
.SYNOPSIS
    Scaffold the credit-risk-assistant repository on Windows.

.DESCRIPTION
    PowerShell equivalent of setup.sh. Creates the directory tree, .gitignore,
    requirements.txt, config/config.yaml and a README stub.

    Safe to run inside an existing folder: files that already exist are left
    alone unless -Force is passed.

.EXAMPLE
    .\setup.ps1
    Scaffolds the current directory.

.EXAMPLE
    .\setup.ps1 -Force
    Overwrites existing config files with the canonical versions.
#>

[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Write-ProjectFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Content
    )

    if ((Test-Path $Path) -and -not $Force) {
        Write-Host "  skip   $Path (exists)" -ForegroundColor DarkGray
        return
    }

    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    # UTF-8 without BOM: a BOM breaks YAML parsers and shell scripts.
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        (Join-Path (Get-Location) $Path), $Content, $encoding
    )
    Write-Host "  write  $Path" -ForegroundColor Green
}

Write-Host "==> creating directories" -ForegroundColor Cyan

$directories = @(
    'config',
    'data\raw', 'data\interim', 'data\processed',
    'src', 'src\data', 'src\features', 'src\models', 'src\evaluation',
    'notebooks', 'scripts', 'tests',
    'reports', 'reports\figures'
)

foreach ($directory in $directories) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

# Package markers
foreach ($package in @('src', 'src\data', 'src\features', 'src\models', 'src\evaluation')) {
    $init = Join-Path $package '__init__.py'
    if (-not (Test-Path $init)) { New-Item -ItemType File -Path $init | Out-Null }
}

# Keep empty data directories tracked without their contents
foreach ($keep in @('data\raw', 'data\interim', 'data\processed', 'reports\figures')) {
    $gitkeep = Join-Path $keep '.gitkeep'
    if (-not (Test-Path $gitkeep)) { New-Item -ItemType File -Path $gitkeep | Out-Null }
}

Write-Host "==> writing project files" -ForegroundColor Cyan

# --------------------------------------------------------------------------- #
Write-ProjectFile -Path '.gitignore' -Content @'
# Data - never commit. Raw LendingClub is ~1.6 GB.
data/raw/*
data/interim/*
data/processed/*
!data/**/.gitkeep

# Model artefacts
*.joblib
*.pkl
*.onnx

# Vector stores (layer 2)
.chroma/
*.faiss

# Secrets
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ipynb_checkpoints/
.coverage
htmlcov/

# OS / editor
.DS_Store
.idea/
.vscode/
Thumbs.db
'@

# --------------------------------------------------------------------------- #
Write-ProjectFile -Path 'requirements.txt' -Content @'
# Core
pandas>=2.1
numpy>=1.26
pyarrow>=14.0
pyyaml>=6.0

# Modelling
scikit-learn>=1.4
lightgbm>=4.1
statsmodels>=0.14

# Explainability
shap>=0.44

# Plotting
matplotlib>=3.8
seaborn>=0.13

# Notebooks
jupyter>=1.0

# Testing
pytest>=8.0
'@

# --------------------------------------------------------------------------- #
Write-ProjectFile -Path 'config\config.yaml' -Content @'
# Single source of truth. Nothing that affects a number lives outside this file.

data:
  raw_path: data/raw/accepted_2007_to_2018Q4.csv
  interim_path: data/interim/modelling_base.parquet
  status_column: loan_status
  date_column: issue_d

target:
  # Resolved outcomes only. Unresolved loans are neither good nor bad and are
  # excluded rather than silently counted as good.
  bad_statuses:
    - Charged Off
    - Default
  good_statuses:
    - Fully Paid
  # Historic "Does not meet the credit policy" loans are a different population.
  include_policy_failures: false

split:
  # Out-of-time. Set after inspecting the yearly volume table from step 6.
  cutoff_date: null
  min_train_months: 24

woe:
  max_bins: 10
  min_bin_fraction: 0.05
  min_iv: 0.02

model:
  random_state: 42
'@

# --------------------------------------------------------------------------- #
Write-ProjectFile -Path 'README.md' -Content @'
# Credit Risk Assistant

A two-layer system for consumer credit risk:

1. **Scoring** - a probability-of-default model on LendingClub loan data, with
   leakage auditing, out-of-time validation and population stability monitoring.
2. **Knowledge** - a retrieval-augmented assistant over credit policy documents.

Status: layer 1 in progress.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place `accepted_2007_to_2018Q4.csv` in `data/raw/`.

## Reproduce

```powershell
python scripts/profile_data.py
```
'@

# --------------------------------------------------------------------------- #
Write-Host "==> git" -ForegroundColor Cyan

if (-not (Test-Path '.git')) {
    git init --quiet
    Write-Host "  initialised repository" -ForegroundColor Green
} else {
    Write-Host "  skip   git init (already a repository)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "==> done" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next:"
Write-Host "  pip install -r requirements.txt"
Write-Host "  # then drop load.py, target.py and profile_data.py into place"
