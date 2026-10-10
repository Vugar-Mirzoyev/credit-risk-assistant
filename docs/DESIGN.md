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

## 2b. Layer 2 decisions

Six more, in the same form. Layer 1's habit carries over unchanged: measure
rather than assert, fail loudly, and publish the number that makes the other
numbers judgeable.

### D6 — The corpus is real regulation, and it is committed

`data/policies/` holds three parts of 12 CFR, fetched from the eCFR and kept in
the repository:

| Part | What it governs | Why it is here |
|---|---|---|
| 1002 | Regulation B, ECOA | Defines an "empirically derived, demonstrably and statistically sound" credit scoring system, the bases a decision may not rest on, and what an adverse action notice must contain |
| 1022 | Regulation V, FCRA | Risk-based pricing notices and credit score disclosure, around the bureau attributes that make up most of the feature set |
| 1026 | Regulation Z, TILA | Disclosure of rate, term and instalment - three of the model's own inputs |

Writing the documents *and* the questions would make any result unreadable: a
reviewer cannot tell a good retriever from a well-chosen corpus. Real
regulation is checkable against a public source.

Regulation Z is much the largest and much the least relevant to a credit
decision, and that is deliberate. A corpus in which everything is relevant
makes retrieval look good for the wrong reason.

The XML is committed rather than fetched on demand. eCFR content changes, so a
corpus rebuilt on every run is not reproducible; the issue date is recorded in
`data/policies/MANIFEST.json` and the download is scripted. US federal
regulations are public domain. This also fixes something Layer 1 could not: the
retrieval evaluation runs from a clone with no download at all.

### D7 — Retrieval is compared against a keyword baseline

Three retrievers, reported together: **BM25**, **dense embeddings**, and a
**hybrid** of the two.

An embedding retriever reported on its own is unfalsifiable. Regulatory text is
full of exact terms of art - "adverse action", "empirically derived", "annual
percentage rate" - and the expectation written here before anything was
measured was that a keyword index would often be the stronger one on that kind
of language. It was not: measured on the evaluation set, dense retrieval
reaches recall@5 of 0.756 against BM25's 0.659, and it wins on the lexical
questions too, which is where the keyword index was most expected to hold.
Hybrid fusion is best on paraphrased questions alone.

The prediction being wrong is the reason the comparison is in the design at
all. The same reasoning makes D4 report champion and challenger rather than a
winner: a number with nothing beside it cannot be judged, including by the
person who produced it.

### D8 — The evaluation set is hand-built, and its failures are listed

Questions are written by hand against the corpus, each paired with the section
that answers it. Retrieval is scored by whether the right **section** was
returned, not whether plausible text was.

Averages hide the cases that matter, so every missed question is listed
individually in the report, the same way every excluded column carries a
written reason in the D2 audit.

### D9 — Chunking is measured, not chosen

Chunk size and strategy are parameters that move the result, so they are swept
against the evaluation set and the table is published. Structure-aware
chunking has a natural unit here: eCFR sections carry their own numbers, which
is what lets an answer cite `12 CFR 1002.9` instead of gesturing at a
paragraph.

### D10 — Embeddings run locally

Sentence-transformers, no API key. A reviewer who clones the repository must be
able to reproduce every retrieval figure from a documented command. A hosted
embedding would score better and would break the property Layer 1 spent its
whole effort establishing.

### D11 — Generation is optional and measured separately

Most of what goes wrong in retrieval-augmented generation goes wrong in
retrieval, and retrieval is measurable without a language model. The generation
step therefore sits behind a configured key: the prompts and the evaluation
harness are committed and readable, faithfulness and answer relevance are
measured when a key is present, and every retrieval number stands without one.

Stating that boundary is the point. A repository whose headline numbers cannot
be reproduced without a paid credential should say so rather than imply
otherwise.

---

## 2c. Serving decisions

Layer 1 ends with a validated scorecard and a report about it. A model a
lender could use also needs a way to be asked for a score, an answer to "why"
attached to every score, and a record of which run produced it. Four more
decisions, in the same form. None of them changes a reported figure, and none
is a third layer: they package Layer 1 for use.

### D12 — The champion is served from an exported file, through the training code

`scripts/train_models.py` writes the champion to `models/champion.json`: every
bin with its edges or categories, its WOE, its training counts and its points,
plus the coefficients, the intercept and the scaling. The service loads that
file and rebuilds the same `FeatureBinning` objects `WoeBinner` fitted, so a
value is binned at serving time by the code that binned it in training.
Training-serving skew - two implementations of one transform drifting apart -
needs two implementations, and there is one.

JSON rather than a pickle. A pickle executes code when it is loaded and cannot
be reviewed; this file can be read and diffed like the points table it is, and
the service never imports scikit-learn, because a logistic regression over WOE
is a dot product once its coefficients are written down. The version is a hash
of the model content and nothing else, so a refit that changes nothing keeps
its version and any change to a bin produces a new one. A file whose content
no longer matches its version is refused.

The export is checked before it is written. The whole holdout is scored again
through the service's code, from raw columns with the derived features
recomputed, and compared with the fitted scorecard's own predictions. The
script refuses to write the file if any probability differs by more than
1e-9. Measured on this extract, over 282,970 loans, the largest difference
is 3.1e-16: the service and the fitted scorecard are the same model.

The file is committed. It holds aggregate counts per bin and no loan-level
record, and committing it is what lets the service, its container and the CI
build run from a clone without the extract.

### D13 — Every score carries its reasons, chosen the way Regulation B describes

Regulation B requires a lender to state the principal reasons for an adverse
action, and its official interpretation (Supplement I, comment 9(b)(2)-5)
names two acceptable ways to select them from a scoring system: the factors on
which the applicant fell furthest below the average of applicants at or just
above the minimum passing score, or below the average of all applicants.

The service uses the second. The first needs a passing score, which is a
cutoff, and a cutoff is a decision about risk appetite and pricing that this
project has no basis to make. The service therefore returns a probability, a
score and the reasons, and never approve or decline. Reasons are returned for
every application: they are what a notice would cite if a lender, applying
its own cutoff, declined it.

The all-applicants averages are computed at export from the training counts
each bin already carries. At most four reasons are returned - comment
9(b)(2)-1: "disclosure of more than four reasons is not likely to be helpful
to the applicant" - each named by a plain-language factor from
`config/reason_texts.yaml`. The service refuses to start if a scored feature
has no entry, because a feature that can cost an applicant points has to be
nameable in a notice (comment 9(b)(2)-2).

This is where the two layers meet. Layer 2 retrieves the commentary; the
service implements it.

### D14 — The input contract comes from the champion

The request schema is generated from the exported file at start-up. A
hand-written schema is a second copy of the model's inputs, and the day a
retrained champion gains or loses a feature the two disagree without anything
failing.

Two rules follow from training rather than from preference. A field may be
omitted only if training contained missing values for every feature it feeds,
because only then is there evidence for what missing means. A category is
accepted only if training saw it. In a batch, `WoeBinner` gives an unseen
value the neutral WOE and counts it in the transform report; one application
has no report to count it in, so it is refused with a 422 naming the field and
the reason.

Derived features need a third rule. Their inputs are always required, because
a derived feature's missing bin records arithmetic that came out undefined in
training, not an input that was absent: the champion's loan-to-income has a
missing bin because the extract contains incomes of zero. An application with
a zero income lands in that bin, exactly where training put the same case; an
application with no loan amount is refused, because scoring it there would
treat a missing field as a zero income. Where a derived feature has no missing
bin, an undefined value is refused like any other.

Logs carry the request id, status, latency and model version, and never an
applicant's fields. A scoring log is the first place personal data leaks from,
and nothing in operating the service needs the inputs written down.

### D15 — Every training run is tracked, and the champion is registered

Each run of `train_models.py` is recorded in MLflow, with a SQLite store under
`mlruns/` because the model registry needs a database behind it. Recorded:
every configuration value that moves a number, as parameters; every metric of
all six models, the champion's worst quarterly PSI and the parity error, as
metrics; the validation report, the column audit, the configuration and the
champion file, as artefacts - the audit trail attached to the run it
explains. The champion is registered as a model version that scores through
`src/serving`, so the registry cannot hold a second scoring implementation
either. The commit the run was made from is recorded, and so is whether the
working tree had uncommitted changes.

Tracking is a record, not a source of numbers. Every reported figure still
comes from a documented command, and `tracking.enabled: false` leaves the
script doing exactly what it did before tracking existed.

Continuous integration runs the test suite on every push, then builds the
container, waits for it to report healthy, scores an example application over
HTTP and checks that an incomplete one is refused. Neither job needs the
extract.

---

## 3. Repository layout

```
credit-risk-assistant/
├── README.md
├── requirements.txt          # pinned to the versions results were produced on
├── requirements-serve.txt    # D12: what the container installs, nothing more
├── pyproject.toml            # pytest configuration
├── Dockerfile                # the scoring service
├── .github/workflows/ci.yml  # tests, then the container scored over HTTP
├── config/
│   ├── config.yaml           # every tunable number lives here
│   ├── column_availability.csv  # D2 audit, maintained by hand
│   └── reason_texts.yaml     # D13: the factor named for each scored feature
├── models/
│   └── champion.json         # D12: committed, written by train_models.py
├── examples/
│   └── application.json      # one application, for the README and CI
├── mlruns/                   # D15, not tracked: run store and model registry
├── data/                     # not tracked
│   ├── raw/                  # accepted_2007_to_2018Q4.csv (~1.6 GB)
│   ├── interim/
│   └── processed/
├── docs/
│   └── DESIGN.md             # this file
├── data/policies/            # 12 CFR 1002, 1022, 1026 - committed, public domain
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
│   ├── evaluation/
│   │   ├── metrics.py        # D5: Gini, KS, AUC, calibration
│   │   ├── stability.py      # PSI, CSI
│   │   └── explain.py        # SHAP
│   ├── rag/                  # Layer 2
│   │   ├── corpus.py         # D6: eCFR XML -> citable sections
│   │   ├── chunk.py          # D9
│   │   ├── embed.py          # D10
│   │   ├── retrieve.py       # D7: BM25, dense, hybrid
│   │   ├── evaluate.py       # D8: recall@k, MRR, RAGAS
│   │   └── answer.py         # D11
│   ├── serving/
│   │   ├── artefact.py       # D12: export, load, version by content
│   │   ├── score.py          # D13, D14: score, reasons, input contract
│   │   └── api.py            # FastAPI: /score, /model, /health
│   └── tracking.py           # D15: MLflow run tracking and registry
├── scripts/
│   ├── profile_data.py       # produces reports/01_profile.md
│   ├── train_models.py       # produces reports/model_validation.md
│   ├── fetch_policies.py     # downloads the D6 corpus
│   └── evaluate_retrieval.py # produces reports/03_retrieval.md
├── tests/                    # one file per module
│   ├── test_load.py
│   ├── test_target.py
│   ├── test_leakage.py
│   ├── test_split.py
│   ├── test_build.py
│   ├── test_woe.py
│   ├── test_scorecard.py
│   ├── test_challenger.py
│   ├── test_metrics.py
│   ├── test_artefact.py
│   ├── test_score.py
│   ├── test_api.py
│   └── test_tracking.py
└── reports/
    ├── 01_profile.md
    ├── column_profile.csv
    ├── figures/
    └── model_validation.md
```

Layer 2 adds `src/rag/` to the same repository. One system, not two projects:
Layer 1 builds a scorecard whose points table exists so an applicant can be
told why they were declined, and Layer 2 reads the regulation that requires
it.

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
- Parquet, not CSV, if intermediate data is ever written. Layer 1 writes none:
  the whole path runs from the raw extract in about four minutes, and a cached
  intermediate would be one more thing that can silently go stale

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

Layer 2 is complete when a reviewer can:

1. See where the corpus came from and which version of it, without asking
2. Read the questions the retriever is scored on, and check that each one is
   answerable from the section it names
3. See a keyword baseline next to the embedding retriever, not the embedding
   retriever alone
4. See the chunking parameters swept rather than asserted
5. Read every question the best retriever still gets wrong
6. Reproduce all of it from a documented command, with no API key and no
   download

The sixth is the one Layer 1 could not offer: the extract it depends on is
1.6 GB and cannot be committed. The policy corpus is 4.6 MB of public-domain
regulation, so Layer 2 has no excuse.

Serving (D12-D15) is complete when a reviewer can:

1. Start the service from a clone with one documented command, or the
   container with two, without the extract
2. Score an application and see the probability, the score, the points per
   attribute and the reasons
3. See the parity between the service and the fitted scorecard measured over
   the whole holdout, not assumed
4. Read the method behind the reasons and the regulation it comes from
5. Be refused, with the field and the reason, when an application cannot be
   scored honestly
6. Find the run, the configuration and the code version behind any champion
   version
7. See the tests and the container pass in CI, without the extract

---

## 6. Known limitations

- LendingClub is an accepted-loans dataset. Rejected applications are absent,
  so the model is fitted on a population that already passed a screen. Reject
  inference is out of scope and this is stated rather than ignored.
- The dataset ends in 2018. Nothing here claims current-market validity.
- Bad rate and macroeconomic conditions shift across the sample period, which
  is exactly why stability is measured rather than assumed.
- A family of bureau fields - `open_il_*`, `open_rv_*`, `inq_fi`, `max_bal_bc`,
  `total_cu_tl`, `inq_last_12m` and their neighbours - is **100% null from 2007
  to 2014, 95% null in 2015 and 0% null from 2016 onward**. LendingClub began
  reporting them partway through the sample. They therefore barely exist in the
  training window and always exist in the holdout, and the characteristic
  stability index flags every one of them. The shift is real but it is a
  reporting change rather than a change in borrowers, and it is the reason the
  stability report distinguishes an index driven by empty buckets from one
  driven by the distributions. WOE keeps them usable by giving missingness its
  own bin; a model that leant on them heavily would still be resting on a field
  that did not exist for most of its training data.
- The service has no authentication, rate limiting or TLS. It is a scoring
  component meant to sit behind a gateway that provides them, not a public
  endpoint.
- Stability is measured at training time, PSI by quarter on the holdout, and
  not on live traffic. The service deliberately logs no inputs (D14), so
  monitoring live drift would need a separate store of scores, and
  characteristic drift a consented store of inputs. Neither exists here.
- The reasons follow the all-applicants method in Regulation B's commentary.
  Whether a given lender's notices meet the regulation is a compliance
  determination about that lender, not a property of this code.
- LendingClub carries no protected attributes, so disparate impact cannot be
  measured on this extract. What can be done is done in the audit: `zip_code`
  is excluded on redlining grounds (D2), and none of the champion's features
  is itself a basis Regulation B prohibits. Whether any of them acts as a
  proxy for one cannot be tested without the attributes it would proxy for.
