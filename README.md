# Credit Risk Assistant

A probability-of-default model on 2.26 million LendingClub loans, built so that
every number in it can be defended: a leakage audit of all 151 columns, an
out-of-time split, a scorecard whose score decomposes into points per
attribute, and a boosted challenger to price what that interpretability costs.

The methodology is the deliverable, not the leaderboard number. A model with a
moderate Gini and a rigorous validation trail is more useful than a stronger
model with no audit, because only the first one can be deployed and defended.

Layer 2 reads the regulation that requires the first layer to be explainable:
retrieval over 12 CFR 1002, 1022 and 1026, scored against a hand-written
evaluation set with a keyword baseline beside it.

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

## Portfolio scenarios

The same holdout as a book: 282,970 loans, $4,060.0m lent. Each scenario
multiplies every loan's odds of default by one factor, so the ranking and Gini
stay as they are and only the level moves:

| Scenario | Basis | Mean PD | Expected defaulted amount | vs base |
|---|---|---:|---:|---:|
| base | champion as fitted | 21.77% | $958.6m | - |
| worst_vintage | mean PD set to the worst training year (2016, before the cutoff) | 23.14% | $1,016.6m | +6.0% |
| severe | odds of default x1.5, an assumption | 28.58% | $1,246.0m | +30.0% |

Approving the lowest-PD half instead of everyone lowers the observed bad rate
from 21.82% to 13.09% and declines 70.0% of the bad loans; the report lists six
approval rates and recommends none, because a cutoff is a risk-appetite
decision (D13). The amount is PD x loan amount, an upper bound on loss, since
repayments and recoveries are post-origination fields. Full tables in
[`reports/04_portfolio_scenarios.md`](reports/04_portfolio_scenarios.md),
method in D16.

## Retrieval

1,114 sections of US consumer lending regulation, 41 hand-written questions,
scored on whether the section that answers each one comes back in the top five:

| Retriever | recall@5 | MRR | lexical | paraphrase |
|---|---:|---:|---:|---:|
| BM25 (keyword baseline) | 0.659 | 0.533 | 0.750 | 0.600 |
| **Dense embeddings** | **0.756** | **0.653** | **0.875** | 0.680 |
| Hybrid (rank fusion) | 0.732 | 0.616 | 0.750 | **0.720** |

The baseline is the point. An embedding retriever reported on its own cannot be
judged, and the expectation written into the design beforehand - that keyword
search would win on regulatory terms of art - turned out to be wrong. Chunking
was swept rather than chosen, and prepending the section heading to every chunk
is worth 7 points of recall, which is why it was measured instead of assumed.

Every question the best retriever still misses is listed by name in
[`reports/03_retrieval.md`](reports/03_retrieval.md).

The corpus is committed, so this runs from a clone with no download and no API
key.

## Serving

The champion is exported to [`models/champion.json`](models/champion.json) and
served over HTTP. One application goes in; out come the probability of
default, the score, the points each attribute contributed, and the principal
reasons - the factors on which the applicant fell furthest below the average
applicant, one of the two selection methods Regulation B's own commentary names
(Supplement I, comment 9(b)(2)-5). Layer 2 retrieves that commentary; the
service implements it.

```powershell
python -m uvicorn src.serving.api:create_app --factory --port 8000
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/score -ContentType "application/json" -InFile examples/application.json
```

For [`examples/application.json`](examples/application.json), a constructed
applicant rather than a real one: probability of default 0.345, score 505.6,
and these reasons:

| Reason | Applicant's bin | Points | Average | Shortfall |
|---|---|---:|---:|---:|
| Home ownership status | RENT, ANY | -4.16 | 0.19 | 4.34 |
| Number of accounts opened in the past 24 months | [6, 7) | -2.66 | 0.40 | 3.06 |
| Balance relative to credit limit across all accounts | [64, 72) | -2.89 | 0.07 | 2.96 |
| Debt-to-income ratio | [22.56, 25.61) | -2.08 | 0.30 | 2.38 |

Three properties are checked rather than claimed:

- **The service is the validated model.** Before the file is written, the whole
  holdout is scored again through the service's own code, from raw columns, and
  compared with the fitted scorecard. Over 282,970 loans the largest difference
  in probability is 3.1e-16; above 1e-9 the file is not written at all.
- **The request schema is the model's.** It is generated from the champion:
  categorical fields accept only the categories training saw - exactly as the
  extract spells them, including the leading space in `" 36 months"` - and a
  field may be omitted only where training contained missing values for it.
  Anything else is refused with a 422 that names the field, not scored as a
  guess.
- **Logs carry no applicant data.** Request id, status, latency and model
  version, nothing an applicant typed.

The service returns no approve or decline. That needs a cutoff, and a cutoff is
a lender's decision about risk appetite, not something this data can settle.

The container holds six pinned libraries and the committed champion - no
scikit-learn, no LightGBM, no torch - and runs as a non-root user:

```powershell
docker build -t credit-risk-scoring .
docker run --rm -p 8000:8000 credit-risk-scoring
```

Every run of `train_models.py` is recorded in MLflow - the configuration, the
metrics of all six models, the validation report and the column audit - and
the champion is registered as a model version that scores through the same
code as the service:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

CI runs the tests on every push, then builds the container, waits for it to
report healthy and scores the example over HTTP. Neither job needs the extract.

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
config/    config.yaml, the 151-row column audit, 41 retrieval questions,
           the reason text for every scored feature
data/      policies/ - three regulations, committed, public domain
models/    champion.json - the exported champion the service loads
src/       data (load, target, leakage, split), features (build, woe),
           models (scorecard, challenger), evaluation (metrics, stability,
           explain, scenarios), rag (corpus, chunk, retrieve, embed, evaluate),
           serving (artefact, score, api), tracking, pipeline
scripts/   profile_data.py, train_models.py, fetch_policies.py,
           evaluate_retrieval.py, portfolio_scenarios.py
reports/   01_profile.md, model_validation.md, 03_retrieval.md,
           04_portfolio_scenarios.md, figures
examples/  application.json - one constructed application
tests/     one file per module, 316 tests
docs/      DESIGN.md - the sixteen decisions, and the reasoning behind them
Dockerfile, requirements-serve.txt, .github/workflows/ci.yml
```

Start with [`docs/DESIGN.md`](docs/DESIGN.md). It states the sixteen
decisions that shape everything else - five for the model, six for retrieval,
four for serving, one for the portfolio - and the limitations they do not
cover.

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
python scripts/evaluate_retrieval.py
python scripts/portfolio_scenarios.py
```

The first writes the data profile and the per-column statistics the audit is
made against; the second fits six models, writes the validation report,
exports the champion after checking it scores as fitted, and records the run;
the third sweeps the chunking strategies and scores the three retrievers; the
fourth scores the holdout through the exported champion and writes the
portfolio scenarios. Every figure in this README comes from one of those four
commands, except the serving example, which comes from the service started on
the champion the second one writes.

Only the first, second and fourth need the LendingClub extract. The retrieval evaluation runs
on the committed corpus.

## Tests

```powershell
pytest
```

Requirements are pinned to the versions the results were produced on.
