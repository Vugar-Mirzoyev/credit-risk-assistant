# Design Notes

Engineering decisions behind this repository, with the reasoning that produced
them. Written so that any of these choices can be challenged and defended.

---

## 1. Scope

A two-layer credit risk system built on public LendingClub loan data.

| Layer | Name | Purpose |
|---|---|---|
| 1 | **Scoring** | Probability-of-default model with leakage auditing, out-of-time validation, and population stability monitoring |
| 2 | **Knowledge** | Retrieval-augmented assistant over credit policy documents |

Validation and documentation are not a third layer. They are written *about*
the system, not built into it: the model validation report and the retrieval
quality measurement are outputs of layers 1 and 2, and they land in `reports/`.
Counting reporting as architecture would inflate the design, and a careful
reader would notice.

The methodology is the deliverable, not the leaderboard number. A model with a
moderate Gini and a rigorous validation trail is more useful than a stronger
model with no audit, because only the first one can be deployed and defended.

Layer 1 is completed and closed before Layer 2 begins. Each layer stands on its
own as finished work.

---

## 2. Design decisions

Five decisions define this project. Each one is reversible only by an explicit
change to this document.

### D1 — The target excludes unresolved loans

```
bad  (1)  = Charged Off, Default
good (0)  = Fully Paid
excluded  = Current, Issued, In Grace Period, Late (16-30), Late (31-120)
```

A loan still running has not yet revealed whether it is good or bad. Treating
it as good depresses the observed bad rate and inflates every metric computed
downstream. This is the single most common error in public analyses of this
dataset, and it is invisible unless the status distribution is inspected.

Measured on the full extract: 912,569 loans are unresolved, 40.4% of the file.
Counting them as good moves the observed bad rate from 19.96% to 11.90%, an
understatement of eight points before a single feature has been built.

Loans carrying the prefix `Does not meet the credit policy. Status:` were
originated under a different underwriting regime. They are a separate
population and are excluded by default rather than pooled in as extra volume.

Implementation: `src/data/target.py`. The function returns a `TargetReport`
alongside the filtered frame, so the row counts behind the decision are
recorded rather than lost.

### D2 — Features are whitelisted, not blacklisted

Every candidate feature must answer yes to one question:

> **Was this information available at the moment the application was decided?**

Fields such as `total_pymnt`, `recoveries`, `last_pymnt_amnt`, `out_prncp`,
`total_rec_prncp`, `last_fico_range_*`, `settlement_*` and `hardship_*` are
generated after origination. They encode the outcome. Measured on this extract,
the same scorecard given those columns reaches an out-of-time AUC of 0.9991
against 0.6789 for the clean one, and the boosted model reaches 0.9999. Both
are worthless, because none of that information exists when a real decision has
to be made.

A blacklist fails silently on the columns nobody thought of. A whitelist fails
loudly. The audit is therefore performed column by column against the observed
data, and its result is committed as a reviewable artefact,
`config/column_availability.csv`. Every row carries a written reason, because a
label nobody can justify is not an audit.

The audit records two things per column, kept separate on purpose.
**Availability** is a fact about the lending process: `app` or `post`, nothing
else. **Role** is what the column is permitted to be used for: `feature`,
`text`, `identifier`, `split_key`, `target` or `excluded`. Availability alone
does not settle usability. `zip_code` was available and is excluded on
redlining grounds; `id` was available and is an identifier; `issue_d` was
available and is the split key, and as a feature it would let the model learn
the calendar rather than the borrower. Forcing those into a single column would
put a false answer in one of them.

`src/data/leakage.py` enforces one invariant above the rest: **a column marked
`post` can never carry the role `feature`.** It also refuses an audit with a
missing label, a missing reason, a duplicate row, or a column the extract has
gained or lost since the audit was written.

The comparison between the leaky model and the clean model is a **deliverable,
not a debugging step**. Both AUC figures appear in the validation report,
because the gap between them is the most informative number in the project.

### D3 — The split is out-of-time, never random

```
train : issue_d <  cutoff
test  : issue_d >= cutoff
```

A random split lets the model learn from loans originated after the ones it is
tested on. That is information a lender does not have. Out-of-time evaluation
mirrors how the model would actually be used: fit on history, applied to the
future.

The cutoff is set in `config/config.yaml` after inspecting the yearly volume
table produced by `scripts/profile_data.py`, targeting roughly 20-25% of
resolved loans in the holdout window.

No calendar year lands in that band on this extract: 2017 onward is 16.8% of
resolved loans and 2016 onward is 38.6%. The cutoff is therefore quarterly.
`2016-10-01` puts 282,970 resolved loans in the holdout, 21.0% of the resolved
population.

### D4 — Champion and challenger are both reported

- **Champion:** logistic regression scorecard over WOE-transformed features.
  Every coefficient is inspectable and the score decomposes into contributions
  that can be shown to an applicant or a reviewer.
- **Challenger:** gradient boosting, LightGBM. Stronger discrimination, opaque,
  requires SHAP to be explained at all. LightGBM over the alternatives because
  it supports monotonic constraints, which allows the challenger to be held to
  the same directional logic the scorecard enforces by construction.

The goal is not to crown a winner. It is to **quantify what interpretability
costs** in Gini points. In a regulated lending context, a model that cannot be
explained often cannot be put into production regardless of its accuracy, so
that trade-off is the real result.

### D5 — Accuracy is not used

The resolved population carries a bad rate of 19.96%, which is 268,599 bad
loans against 1,345,350 resolved ones. Predicting "good" for every applicant
therefore scores 80.04%. Accuracy is banned from this repository.

Discrimination is measured with **Gini, KS and AUC**. Calibration is measured
separately, because a model can rank well and still produce probabilities that
are wrong in level, which breaks any downstream pricing or expected-loss
calculation.

Stability is measured with **PSI across quarters** rather than assumed.

---

## 3. Repository layout

```
credit-risk-assistant/
├── README.md
├── requirements.txt          # pinned to the versions results were produced on
├── pyproject.toml            # pytest configuration
├── setup.ps1                 # Windows scaffold
├── config/
│   ├── config.yaml           # every tunable number lives here
│   └── column_availability.csv  # D2 audit, maintained by hand
├── data/                     # not tracked
│   ├── raw/                  # accepted_2007_to_2018Q4.csv (~1.6 GB)
│   ├── interim/
│   └── processed/
├── docs/
│   └── DESIGN.md             # this file
├── src/
│   ├── config.py             # config.yaml -> typed dataclass
│   ├── pipeline.py           # audit -> load -> target -> derive -> split -> bin
│   ├── data/
│   │   ├── load.py           # memory-controlled loading
│   │   ├── target.py         # D1
│   │   ├── leakage.py        # D2
│   │   └── split.py          # D3
│   ├── features/
│   │   ├── woe.py            # WOE binning and information value
│   │   └── build.py
│   ├── models/
│   │   ├── scorecard.py      # D4 champion
│   │   └── challenger.py     # D4 challenger
│   └── evaluation/
│       ├── metrics.py        # D5: Gini, KS, AUC, calibration
│       ├── stability.py      # PSI, CSI
│       └── explain.py        # SHAP
├── scripts/
│   ├── profile_data.py       # produces reports/01_profile.md
│   └── train_models.py       # produces reports/model_validation.md
├── tests/                    # one file per module
│   ├── test_load.py
│   ├── test_target.py
│   ├── test_leakage.py
│   ├── test_split.py
│   ├── test_build.py
│   ├── test_woe.py
│   ├── test_scorecard.py
│   ├── test_challenger.py
│   └── test_metrics.py
└── reports/
    ├── 01_profile.md
    ├── column_profile.csv
    ├── figures/
    └── model_validation.md
```

Layer 2 adds `src/rag/` to the same repository. One system, not two projects.

---

## 4. Conventions

- `from __future__ import annotations` at the top of every module
- Full type hints on public functions
- `@dataclass` for configuration and result objects, not bare dictionaries
- **No magic numbers outside `config/config.yaml`.** Any number that moves a
  metric belongs in configuration.
- Docstrings explain why a choice was made. The code already says what it does.
- Every reported figure is reproducible from a command written in the README
- Errors are raised with actionable messages and never swallowed
- Tests are required for `target.py`, `woe.py` and `split.py`, the three places
  where a silent bug corrupts every result downstream
- Parquet, not CSV, for intermediate data

---

## 5. Definition of done

Layer 1 is complete when a reviewer can:

1. Read why unresolved loans were excluded, with the row counts
2. See which columns were rejected as leakage, and why
3. See both the leaky AUC and the clean AUC
4. Confirm the split is out-of-time
5. Compare champion and challenger on Gini, KS and calibration
6. See PSI by quarter
7. Reproduce everything from documented commands
8. Run the test suite and see it pass

If any of the eight is missing, the layer is not done, regardless of how the
model scores.

---

## 6. Known limitations

- LendingClub is an accepted-loans dataset. Rejected applications are absent,
  so the model is fitted on a population that already passed a screen. Reject
  inference is out of scope and this is stated rather than ignored.
- The dataset ends in 2018. Nothing here claims current-market validity.
- Bad rate and macroeconomic conditions shift across the sample period, which
  is exactly why stability is measured rather than assumed.
