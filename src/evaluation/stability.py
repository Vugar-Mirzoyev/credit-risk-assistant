"""D5: population stability, measured rather than assumed.

A model is fitted on one population and applied to another. The question is
not whether the population moved - it always does - but whether it moved enough
that the model is no longer describing it.

The population stability index compares two distributions bucket by bucket::

    PSI = sum over buckets of (actual share - expected share)
                              * ln(actual share / expected share)

Read conventionally: below 0.10 the population is stable, 0.10 to 0.25 is a
shift worth watching, above 0.25 is a shift worth stopping for. The same
arithmetic applied to one input column instead of the score is the
characteristic stability index, and it answers the follow-up question: which
feature moved?

**The buckets are fixed on the expected distribution and then applied to the
actual one.** Recomputing quantiles on each window would compare every
distribution against itself and return approximately zero every time, which is
a monitoring system that can never fire.

**Empty buckets are floored, not merged.** This is the opposite of the rule in
``woe.py``, and deliberately so. There, bins are ours to choose and merging
keeps every bin interpretable. Here the buckets have to be identical on both
sides or the two distributions are not being compared at all, so a bucket the
actual window happens not to reach is floored to a small share. The floor
bounds that bucket's contribution instead of letting a logarithm of zero take
the whole index to infinity, and the floor is reported so the reader knows it
was applied.

Why this module exists at all: the champion's calibration slope on the holdout
is 0.84 rather than 1.0, and its bad rate rises from 19.47% in training to
21.82% out of time. Something moved. PSI says how much, and CSI says where.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Share substituted for a bucket the actual window never reaches. Without it
#: the logarithm is undefined and one empty bucket destroys the whole index.
SHARE_FLOOR = 1e-6

MISSING_LABEL = "Missing"

#: The conventional reading of a PSI value.
STABLE_BELOW = 0.10
UNSTABLE_ABOVE = 0.25


@dataclass(frozen=True)
class StabilityBucket:
    label: str
    expected: int
    actual: int
    expected_share: float
    actual_share: float
    contribution: float
    floored: bool = False


@dataclass(frozen=True)
class Stability:
    """One comparison of an expected distribution against an actual one."""

    psi: float
    expected_rows: int
    actual_rows: int
    buckets: tuple[StabilityBucket, ...]

    @property
    def band(self) -> str:
        return psi_band(self.psi)

    @property
    def floored_buckets(self) -> int:
        return sum(1 for b in self.buckets if b.floored)

    def summary(self) -> str:
        line = (
            f"PSI                     {self.psi:.4f}  ({self.band})\n"
            f"expected rows           {self.expected_rows:,}\n"
            f"actual rows             {self.actual_rows:,}\n"
            f"buckets                 {len(self.buckets)}"
        )
        if self.floored_buckets:
            line += f"\nfloored buckets         {self.floored_buckets}"
        return line

    def to_markdown(self) -> str:
        lines = [
            "| Bucket | Expected | Actual | Expected % | Actual % | Contribution |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in self.buckets:
            flag = " *" if item.floored else ""
            lines.append(
                f"| {item.label}{flag} | {item.expected:,} | {item.actual:,} | "
                f"{item.expected_share:.2%} | {item.actual_share:.2%} | "
                f"{item.contribution:.4f} |"
            )
        if self.floored_buckets:
            lines += ["", f"`*` share floored at {SHARE_FLOOR:g}."]
        return "\n".join(lines)


def psi_band(value: float) -> str:
    if value < STABLE_BELOW:
        return "stable"
    if value < UNSTABLE_ABOVE:
        return "shifted, watch"
    return "shifted, investigate"


def numeric_stability(
    expected, actual, *, buckets: int = 10
) -> Stability:
    """PSI between two numeric distributions, bucketed on the expected one."""
    reference = pd.Series(np.asarray(expected, dtype="float64"))
    current = pd.Series(np.asarray(actual, dtype="float64"))
    _assert_not_empty(reference, current)

    edges = _quantile_edges(reference.dropna().to_numpy(), buckets)
    labels = [_edge_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    expected_counts = _bucket_counts(reference, edges, len(labels))
    actual_counts = _bucket_counts(current, edges, len(labels))
    labels = labels + [MISSING_LABEL]

    return _assemble(labels, expected_counts, actual_counts, len(reference), len(current))


def categorical_stability(expected, actual) -> Stability:
    """PSI over categories, with anything new in the actual window included."""
    reference = pd.Series(np.asarray(expected, dtype=object))
    current = pd.Series(np.asarray(actual, dtype=object))
    _assert_not_empty(reference, current)

    reference_counts = reference.dropna().astype("str").value_counts()
    current_counts = current.dropna().astype("str").value_counts()
    labels = sorted(set(reference_counts.index) | set(current_counts.index))

    expected_counts = [int(reference_counts.get(label, 0)) for label in labels]
    actual_counts = [int(current_counts.get(label, 0)) for label in labels]
    expected_counts.append(int(reference.isna().sum()))
    actual_counts.append(int(current.isna().sum()))

    return _assemble(
        labels + [MISSING_LABEL],
        expected_counts,
        actual_counts,
        len(reference),
        len(current),
    )


def stability_by_quarter(
    expected,
    actual,
    quarters,
    *,
    buckets: int = 10,
) -> pd.DataFrame:
    """PSI of each holdout quarter against the training distribution.

    Quarter by quarter rather than as one holdout figure, because a population
    that drifts steadily and one that jumps once look identical in aggregate
    and call for different responses.
    """
    current = pd.Series(np.asarray(actual, dtype="float64"))
    period = pd.PeriodIndex(pd.to_datetime(pd.Series(quarters)), freq="Q")
    if len(period) != len(current):
        raise ValueError(
            f"{len(period)} quarters against {len(current)} values. They must "
            "be aligned row for row."
        )

    rows = []
    for quarter in period.unique().sort_values():
        window = current[period == quarter]
        result = numeric_stability(expected, window, buckets=buckets)
        rows.append(
            {
                "quarter": str(quarter),
                "rows": len(window),
                "psi": round(result.psi, 4),
                "band": result.band,
            }
        )
    return pd.DataFrame(rows)


def characteristic_stability(
    expected_frame: pd.DataFrame,
    actual_frame: pd.DataFrame,
    features: list[str],
    *,
    buckets: int = 10,
) -> pd.DataFrame:
    """CSI per feature: which inputs moved, sorted by how far.

    A stable score can hide two features moving in opposite directions, so the
    score index alone is not enough to decide whether a model still fits its
    population.
    """
    rows = []
    for feature in features:
        reference = expected_frame[feature]
        current = actual_frame[feature]
        if pd.api.types.is_numeric_dtype(reference):
            result = numeric_stability(reference, current, buckets=buckets)
            kind = "numeric"
        else:
            result = categorical_stability(reference, current)
            kind = "categorical"
        rows.append(
            {
                "feature": feature,
                "kind": kind,
                "csi": round(result.psi, 4),
                "band": result.band,
                "buckets": len(result.buckets),
            }
        )
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(
        drop=True
    )


def _assemble(
    labels: list[str],
    expected_counts: list[int],
    actual_counts: list[int],
    expected_rows: int,
    actual_rows: int,
) -> Stability:
    buckets: list[StabilityBucket] = []
    total = 0.0
    for label, expected, actual in zip(labels, expected_counts, actual_counts):
        if expected == 0 and actual == 0:
            # A bucket neither distribution reaches says nothing about either.
            continue
        expected_share = expected / expected_rows if expected_rows else 0.0
        actual_share = actual / actual_rows if actual_rows else 0.0
        floored = expected_share < SHARE_FLOOR or actual_share < SHARE_FLOOR
        safe_expected = max(expected_share, SHARE_FLOOR)
        safe_actual = max(actual_share, SHARE_FLOOR)
        contribution = (safe_actual - safe_expected) * math.log(
            safe_actual / safe_expected
        )
        total += contribution
        buckets.append(
            StabilityBucket(
                label=label,
                expected=expected,
                actual=actual,
                expected_share=expected_share,
                actual_share=actual_share,
                contribution=contribution,
                floored=floored,
            )
        )
    return Stability(
        psi=total,
        expected_rows=expected_rows,
        actual_rows=actual_rows,
        buckets=tuple(buckets),
    )


def _bucket_counts(values: pd.Series, edges: np.ndarray, count: int) -> list[int]:
    """Counts per fixed bucket, with missing values as a bucket of their own.

    A change in how often a field is populated is drift, and one of the easier
    kinds to miss: dropping the nulls would hide it completely.
    """
    present = values.dropna().to_numpy()
    codes = np.clip(np.searchsorted(edges, present, side="right") - 1, 0, count - 1)
    counts = np.bincount(codes, minlength=count).tolist()
    counts.append(int(values.isna().sum()))
    return counts


def _quantile_edges(values: np.ndarray, buckets: int) -> np.ndarray:
    if values.size == 0:
        return np.array([-math.inf, math.inf])
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, buckets + 1))
    interior = np.unique(quantiles)[1:-1]
    return np.concatenate(([-math.inf], interior, [math.inf]))


def _edge_label(lower: float, upper: float) -> str:
    left = "-inf" if lower == -math.inf else f"{lower:,.4g}"
    right = "inf" if upper == math.inf else f"{upper:,.4g}"
    return f"[{left}, {right})"


def _assert_not_empty(reference: pd.Series, current: pd.Series) -> None:
    if len(reference) == 0 or len(current) == 0:
        raise ValueError(
            f"Nothing to compare: {len(reference)} expected rows and "
            f"{len(current)} actual rows. A window with no loans in it has no "
            "stability to measure."
        )
