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
python -m pip install -r requirements.txt
```

Place `accepted_2007_to_2018Q4.csv` in `data/raw/`.

## Reproduce

```powershell
python scripts/profile_data.py
python scripts/train_models.py
```

The first writes `reports/01_profile.md` and the column audit worksheet; the
second writes `reports/model_validation.md`.

## Tests

```powershell
pytest
```