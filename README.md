# Credit Risk Assistant

A probability-of-default model on 2.26 million LendingClub loans, built so that
every number in it can be defended: a leakage audit of all 151 columns, an
out-of-time split, a scorecard whose score decomposes into points per
attribute, and a boosted challenger to price what that interpretability costs.

The methodology is the deliverable, not the leaderboard number. A model with a
moderate Gini and a rigorous validation trail is more useful than a stronger
model with no audit, because only the first one can be deployed and defended.

Layer 2, a retrieval-augmented assistant over credit policy documents, is
scoped and not yet built.

## Results

Out of time, on loans originated from 2016Q4 onward (282,970 loans, none of
them seen during fitting):

| Model | Features | Gini | AUC | KS | Level error | Calib. slope |
|---|---:|---:|---:|---:|---:|---:|
| Scorecard (champion) | 23 | 0.3577 | 0.6789 | 0.2568 | -0.0005 | 0.836 |
| Scorecard, with the lender's own grade | 24 | 0.3929 | 0.6965 | 0.2841 | +0.0063 | 0.887 |
| Gradient boosting (challenger) | 97 | 0.4329 | 0.7164 | 0.3137 | -0.0191 | 0.856 |
| **The same scorecard, given post-origination columns** | 22 | **0.9982** | **0.9991** | 0.9772 | -0.0015 | 1.048 |

The last row is the point of the whole exercise. It is fitted deliberately, it
is never deployed, and it is reported because a reviewer shown only 0.3577 has
no way to judge whether that number is good. The gap between 0.9991 and 0.6789
is the size of the mistake the column audit prevents.

Full detail in [`reports/model_validation.md`](reports/model_validation.md);
the data profile and the column audit are in
[`reports/01_profile.md`](reports/01_profile.md).

## The three mistakes this is built to avoid

**Counting unfinished loans.** 912,569 of the 2,260,668 loans in the extract -
40.4% - were still running when the data was cut. They are neither good nor
bad yet. Treating them as good moves the observed bad rate from 19.96% to
11.90%, and every metric computed afterwards inherits that optimism while
nothing looks wrong. They are excluded, and the counts are published.

**Using the answer to predict the answer.** `total_pymnt`, `recoveries`,
`last_fico_range_*`, `hardship_*` and `settlement_*` are generated after
origination. All 151 columns were reviewed one by one and labelled with whether
the value existed when the application was decided, each with a written reason,
in [`config/column_availability.csv`](config/column_availability.csv). A
blacklist fails silently on the column nobody thought of; a whitelist fails
loudly, and `src/data/leakage.py` refuses an audit that is incomplete,
unjustified or self-contradictory.

**Learning from the future.** The split is by origination date, not at random:
fitted on loans issued before 2016-10-01, tested on those issued on or after
it. A loan originated exactly on the cutoff belongs to the holdout, because it
was not available to learn from at that moment. The
challenger's early stopping uses the last six months of the *training* window,
so the holdout influences nothing at all - not even the number of trees.

## What is in here

```
config/    config.yaml, and the 151-row column audit
src/       data (load, target, leakage, split), features (build, woe),
           models (scorecard, challenger), evaluation (metrics, stability,
           explain), pipeline
scripts/   profile_data.py, train_models.py
reports/   01_profile.md, model_validation.md, column_profile.csv, figures
tests/     one file per module, 173 tests
docs/      DESIGN.md - the five decisions, and the reasoning behind them
```

Start with [`docs/DESIGN.md`](docs/DESIGN.md). It states the five decisions
that shape everything else and the limitations they do not cover.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place `accepted_2007_to_2018Q4.csv` in `data/raw/`. It is about 1.6 GB and is
not tracked; the loader streams it in chunks and never holds more than about
140 MB at once.

## Reproduce

```powershell
python scripts/profile_data.py
python scripts/train_models.py
```

The first writes the data profile and the per-column statistics the audit is
made against; the second fits six models and writes the validation report.
Every figure in this README comes from one of those two commands.

## Tests

```powershell
pytest
```

Requirements are pinned to the versions the results were produced on.
