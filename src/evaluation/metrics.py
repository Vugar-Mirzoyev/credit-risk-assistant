"""D5: discrimination and calibration, measured separately.

Two different questions, and a model can pass one while failing the other.

*Discrimination* asks whether the model puts the bad loans above the good ones.
AUC, Gini and KS all answer it, and they are reported together because they are
read differently: Gini is the number a credit risk audience expects, AUC is the
number everyone else expects, and KS says where in the score distribution the
separation actually happens.

*Calibration* asks whether the probabilities are right in level, not only in
order. A model can rank perfectly and still say 8% when the truth is 20%. Rank
is enough for an accept/reject cut-off; it is not enough for pricing, for
expected loss, or for any capital calculation, all of which multiply the
probability by an amount of money. The calibration slope and intercept come
from regressing the outcome on the logit of the prediction: a slope of 1 and an
intercept of 0 mean the probabilities can be used as probabilities.

**Accuracy is not here, and its absence is deliberate.** With the resolved bad
rate at 19.96%, predicting "good" for every applicant scores 80.04% and
identifies no bad loans at all. Any metric that rewards that is worse than
useless in this setting, so it is not implemented rather than implemented and
discouraged.

Which split these are computed on is the caller's business, but every result
carries its row count and observed bad rate so a report cannot quietly present
training figures as though they were out of time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

#: Probabilities are clipped this far from 0 and 1 before taking a logit.
LOGIT_EPSILON = 1e-6


@dataclass(frozen=True)
class Discrimination:
    """Does the model order the population correctly?"""

    rows: int
    bad_rate: float
    auc: float
    gini: float
    ks: float
    ks_score: float

    def summary(self) -> str:
        return (
            f"rows                    {self.rows:,}\n"
            f"observed bad rate       {self.bad_rate:.2%}\n"
            f"AUC                     {self.auc:.4f}\n"
            f"Gini                    {self.gini:.4f}\n"
            f"KS                      {self.ks:.4f} "
            f"(at predicted PD {self.ks_score:.4f})"
        )


@dataclass(frozen=True)
class CalibrationBucket:
    bucket: int
    count: int
    predicted_mean: float
    observed_rate: float

    @property
    def error(self) -> float:
        return self.predicted_mean - self.observed_rate


@dataclass(frozen=True)
class Calibration:
    """Are the probabilities right in level, not only in order?"""

    rows: int
    predicted_mean: float
    observed_rate: float
    brier: float
    expected_calibration_error: float
    max_calibration_error: float
    intercept: float
    slope: float
    buckets: tuple[CalibrationBucket, ...]

    @property
    def calibration_in_the_large(self) -> float:
        """Overall level error. Zero means the average prediction is right."""
        return self.predicted_mean - self.observed_rate

    def summary(self) -> str:
        return (
            f"rows                    {self.rows:,}\n"
            f"mean predicted PD       {self.predicted_mean:.4f}\n"
            f"observed bad rate       {self.observed_rate:.4f}\n"
            f"level error             {self.calibration_in_the_large:+.4f}\n"
            f"Brier score             {self.brier:.4f}\n"
            f"expected calib. error   {self.expected_calibration_error:.4f}\n"
            f"max calibration error   {self.max_calibration_error:.4f}\n"
            f"calibration slope       {self.slope:.4f}  (1.0 is correct)\n"
            f"calibration intercept   {self.intercept:+.4f}  (0.0 is correct)"
        )

    def to_markdown(self) -> str:
        lines = [
            "| Bucket | Loans | Mean predicted PD | Observed bad rate | Error |",
            "|---:|---:|---:|---:|---:|",
        ]
        for item in self.buckets:
            lines.append(
                f"| {item.bucket} | {item.count:,} | {item.predicted_mean:.4f} | "
                f"{item.observed_rate:.4f} | {item.error:+.4f} |"
            )
        return "\n".join(lines)


def discrimination(outcome, predicted) -> Discrimination:
    """AUC, Gini and KS on one set of predictions."""
    y, p = _validate(outcome, predicted)
    auc = float(roc_auc_score(y, p))
    false_positive, true_positive, thresholds = roc_curve(y, p)
    separation = true_positive - false_positive
    best = int(np.argmax(separation))
    return Discrimination(
        rows=len(y),
        bad_rate=float(y.mean()),
        auc=auc,
        gini=2.0 * auc - 1.0,
        ks=float(separation[best]),
        ks_score=float(np.clip(thresholds[best], 0.0, 1.0)),
    )


def calibration(outcome, predicted, *, buckets: int = 10) -> Calibration:
    """Level accuracy of the probabilities, bucket by bucket and overall."""
    y, p = _validate(outcome, predicted)
    grouped = _quantile_buckets(y, p, buckets)

    weights = np.array([b.count for b in grouped], dtype="float64")
    errors = np.array([abs(b.error) for b in grouped], dtype="float64")
    intercept, slope = _cox_calibration(y, p)

    return Calibration(
        rows=len(y),
        predicted_mean=float(p.mean()),
        observed_rate=float(y.mean()),
        brier=float(np.mean((p - y) ** 2)),
        expected_calibration_error=float((weights * errors).sum() / weights.sum()),
        max_calibration_error=float(errors.max()) if len(errors) else float("nan"),
        intercept=intercept,
        slope=slope,
        buckets=grouped,
    )


def gains_table(outcome, predicted, *, buckets: int = 10) -> pd.DataFrame:
    """Rank ordering by risk band, riskiest first.

    The table a reviewer reads before any single number: it shows whether the
    bad rate falls monotonically from the riskiest band to the safest, which is
    the property a cut-off policy depends on and which a good AUC does not
    guarantee on its own.
    """
    y, p = _validate(outcome, predicted)
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    edges = np.array_split(np.arange(len(y_sorted)), buckets)

    total_bad = float(y_sorted.sum())
    total_good = float(len(y_sorted) - total_bad)
    rows = []
    cumulative_bad = 0.0
    cumulative_good = 0.0
    for index, positions in enumerate(edges, start=1):
        if len(positions) == 0:
            continue
        chunk = y_sorted[positions]
        bad = float(chunk.sum())
        good = float(len(chunk) - bad)
        cumulative_bad += bad
        cumulative_good += good
        rows.append(
            {
                "bucket": index,
                "loans": len(chunk),
                "min_predicted_pd": float(p_sorted[positions].min()),
                "max_predicted_pd": float(p_sorted[positions].max()),
                "bad": int(bad),
                "bad_rate": bad / len(chunk),
                "cumulative_bad_share": cumulative_bad / total_bad if total_bad else float("nan"),
                "cumulative_good_share": cumulative_good / total_good if total_good else float("nan"),
                "lift": (bad / len(chunk)) / (total_bad / len(y_sorted)),
            }
        )
    return pd.DataFrame(rows)


def _quantile_buckets(
    y: np.ndarray, p: np.ndarray, buckets: int
) -> tuple[CalibrationBucket, ...]:
    """Equal-count buckets of predicted probability.

    Equal-width buckets would leave most of them empty: predicted PDs cluster,
    and a bucket holding four loans says nothing about calibration.
    """
    codes = pd.qcut(pd.Series(p), q=buckets, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bucket": codes, "y": y, "p": p})
    grouped = frame.groupby("bucket", observed=True).agg(
        count=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")
    )
    return tuple(
        CalibrationBucket(
            bucket=int(index) + 1,
            count=int(row.count),
            predicted_mean=float(row.predicted),
            observed_rate=float(row.observed),
        )
        for index, row in enumerate(grouped.itertuples())
    )


def _cox_calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Regress the outcome on the logit of the prediction.

    Slope 1 and intercept 0 mean the predictions can be used as probabilities.
    A slope below 1 means they are too extreme at both ends; an intercept away
    from 0 means the whole level is shifted.
    """
    clipped = np.clip(p, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, max_iter=1000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def _validate(outcome, predicted) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(outcome).ravel()
    p = np.asarray(predicted, dtype="float64").ravel()

    if len(y) != len(p):
        raise ValueError(
            f"{len(y)} outcomes against {len(p)} predictions. They must be "
            "aligned row for row."
        )
    if len(y) == 0:
        raise ValueError("Nothing to measure: the arrays are empty.")

    unexpected = sorted(set(np.unique(y)) - {0, 1})
    if unexpected:
        raise ValueError(
            f"The outcome contains {unexpected} as well as 0 and 1. These "
            "metrics are defined on the binary target built in D1."
        )
    if y.min() == y.max():
        raise ValueError(
            "The outcome is a single class, so ranking cannot be assessed. "
            "Check the window being evaluated: a slice with no bad loans in it "
            "produces no measurable discrimination."
        )
    if np.isnan(p).any():
        raise ValueError("Predictions contain nulls; every row must be scored.")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(
            f"Predictions run from {p.min():.4g} to {p.max():.4g}. These are "
            "probabilities of bad, not scores; pass predict_proba output rather "
            "than points."
        )
    return y.astype("int64"), p
