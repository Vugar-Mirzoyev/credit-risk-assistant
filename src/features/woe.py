"""Weight of evidence binning and information value.

WOE turns any attribute, numeric or categorical, into one number on a log-odds
scale::

    WOE(bin) = ln( share of goods in bin / share of bads in bin )

with goods being ``target == 0``. Higher WOE means safer. A logistic regression
fitted on WOE columns has one coefficient per feature, every coefficient is
inspectable, and the score decomposes into points per attribute, which is what
makes the champion in D4 explainable to an applicant or a reviewer.

Information value summarises a whole feature::

    IV = sum over bins of (share of goods - share of bads) * WOE

**Everything here is fitted on the training split alone.** Bin edges are
learned parameters. Computing them on the full frame and splitting afterwards
leaks the test window's distribution into training, and the resulting Gini is
optimistic in a way no later check reveals, because the test rows really were
held out - only the boundaries were not.

Four decisions shape the implementation.

*Missing is a bin, and nothing is imputed.* In credit data missingness carries
information: ``mths_since_last_delinq`` is null precisely for borrowers who
have never been delinquent, the safest group rather than an unknown one.
Imputing the median would move them from the end of a risk ordering to its
middle. It also means there is no imputation constant, so there is one fewer
quantity that could be learned from the wrong rows.

*Bins are merged, never epsilon-corrected.* A bin with no bads gives an
infinite WOE. The usual patch adds half an observation to every cell; merging
the bin into its neighbour instead keeps every bin something a person can read
and invents no observations that were not there.

*Monotonicity is enforced by merging.* A risk ordering that goes up, down and
up again across a numeric range is usually a sampling artefact, and a reviewer
will ask about it. Adjacent violators are merged until the ordering is
monotone in one direction or the other; the direction is not prescribed.

*Categoricals use the same merge, on a different ordering.* Numeric bins are
ordered by value; categories are ordered by their event rate. Merging adjacent
groups then puts rare categories together with categories of similar risk,
rather than in one "other" bucket that mixes the safest with the riskiest.
Under that ordering WOE is monotone by construction, so the monotonicity step
has nothing left to do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import WoeConfig

#: Prefix on transformed columns. The values are no longer the original
#: quantity, and a coefficient labelled `woe_dti` cannot be mistaken for one
#: fitted on raw debt-to-income.
WOE_PREFIX = "woe_"

#: Assigned to a category that the training split never contained. Neutral is
#: the honest position for an attribute value carrying no evidence either way,
#: and inferring one from the test target would be the leak this project is
#: about.
UNSEEN_WOE = 0.0

MISSING_LABEL = "Missing"


@dataclass(frozen=True)
class Bin:
    """One bin, with the counts it was computed from."""

    label: str
    count: int
    good: int
    bad: int
    woe: float
    iv: float
    includes_missing: bool = False

    @property
    def event_rate(self) -> float:
        return self.bad / self.count if self.count else float("nan")


@dataclass(frozen=True)
class FeatureBinning:
    """A fitted feature: how to bin it, and the WOE of each bin."""

    column: str
    kind: str  # "numeric" or "categorical"
    bins: tuple[Bin, ...]
    iv: float
    edges: tuple[float, ...] | None = None
    category_map: dict[str, int] = field(default_factory=dict)
    missing_index: int | None = None

    def transform(self, values: pd.Series) -> tuple[np.ndarray, int]:
        """Map values to WOE. Returns the WOE array and the unseen-value count."""
        woes = np.array([b.woe for b in self.bins], dtype="float64")
        out = np.full(len(values), UNSEEN_WOE, dtype="float64")
        missing = values.isna().to_numpy()

        if self.kind == "numeric":
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
            codes = np.searchsorted(np.asarray(self.edges), numeric, side="right") - 1
            codes = np.clip(codes, 0, len(self.edges) - 2)
            out[~missing] = woes[codes[~missing]]
            unseen = 0
        else:
            keys = values.astype("str")
            codes = keys.map(self.category_map)
            known = codes.notna().to_numpy() & ~missing
            out[known] = woes[codes[known].to_numpy(dtype="int64")]
            unseen = int((~known & ~missing).sum())

        if self.missing_index is not None:
            out[missing] = woes[self.missing_index]
        return out, unseen


@dataclass
class FitReport:
    """Every feature considered, with the reason it was kept or dropped."""

    rows: int
    good: int
    bad: int
    features: tuple[FeatureBinning, ...]
    selected: tuple[str, ...]
    dropped_low_iv: tuple[str, ...]
    suspicious: tuple[str, ...]
    min_iv: float
    suspicious_iv: float

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column": binning.column,
                    "kind": binning.kind,
                    "bins": len(binning.bins),
                    "iv": round(binning.iv, 6),
                    "band": iv_band(binning.iv, self.suspicious_iv),
                    "selected": binning.column in set(self.selected),
                }
                for binning in sorted(
                    self.features, key=lambda b: b.iv, reverse=True
                )
            ]
        )

    def summary(self) -> str:
        return (
            f"fitting rows            {self.rows:,}\n"
            f"  good                  {self.good:,}\n"
            f"  bad                   {self.bad:,}\n"
            f"features considered     {len(self.features)}\n"
            f"selected                {len(self.selected)}\n"
            f"dropped, IV < {self.min_iv:<9.2f}{len(self.dropped_low_iv)}\n"
            f"suspicious, IV > {self.suspicious_iv:<6.2f}{len(self.suspicious)}"
            + (
                "\n  " + ", ".join(self.suspicious)
                if self.suspicious
                else ""
            )
        )


@dataclass
class TransformReport:
    rows: int
    columns: int
    unseen_by_column: dict[str, int]

    @property
    def unseen_total(self) -> int:
        return sum(self.unseen_by_column.values())

    def summary(self) -> str:
        lines = [
            f"rows transformed        {self.rows:,}",
            f"woe columns             {self.columns}",
            f"unseen category values  {self.unseen_total:,}",
        ]
        for column, count in sorted(
            self.unseen_by_column.items(), key=lambda item: -item[1]
        ):
            if count:
                lines.append(f"  {column:<22}{count:,}")
        return "\n".join(lines)


def iv_band(value: float, suspicious_iv: float) -> str:
    """The conventional reading, with the top band named as a warning."""
    if value >= suspicious_iv:
        return "suspicious"
    if value >= 0.3:
        return "strong"
    if value >= 0.1:
        return "medium"
    if value >= 0.02:
        return "weak"
    return "unpredictive"


class WoeBinner:
    """Fit WOE bins on the training split and apply them anywhere."""

    def __init__(self, config: WoeConfig) -> None:
        self.config = config
        self._fitted: dict[str, FeatureBinning] = {}
        self.report: FitReport | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "WoeBinner":
        target = _validate_target(target, len(frame))
        good_total = int((target == 0).sum())
        bad_total = int((target == 1).sum())
        if good_total == 0 or bad_total == 0:
            raise ValueError(
                f"The fitting rows contain {good_total} good and {bad_total} bad "
                "loans. WOE is a ratio of the two distributions and is "
                "undefined when either is empty."
            )

        binnings: list[FeatureBinning] = []
        for column in frame.columns:
            values = frame[column]
            kind = _kind_of(values)
            if kind == "numeric":
                binning = _fit_numeric(
                    column, values, target, good_total, bad_total, self.config
                )
            else:
                binning = _fit_categorical(
                    column, values, target, good_total, bad_total, self.config
                )
            binnings.append(binning)

        selected = tuple(b.column for b in binnings if b.iv >= self.config.min_iv)
        self._fitted = {b.column: b for b in binnings if b.column in set(selected)}
        self.report = FitReport(
            rows=len(frame),
            good=good_total,
            bad=bad_total,
            features=tuple(binnings),
            selected=selected,
            dropped_low_iv=tuple(
                b.column for b in binnings if b.iv < self.config.min_iv
            ),
            suspicious=tuple(
                b.column for b in binnings if b.iv >= self.config.suspicious_iv
            ),
            min_iv=self.config.min_iv,
            suspicious_iv=self.config.suspicious_iv,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, TransformReport]:
        if not self._fitted:
            raise RuntimeError(
                "The binner has not been fitted, or every feature fell below "
                "min_iv. Call fit on the training split first and check the "
                "fit report before transforming."
            )
        missing = [c for c in self._fitted if c not in frame.columns]
        if missing:
            raise KeyError(
                f"Columns fitted but absent from the frame being transformed: "
                f"{sorted(missing)}. Train and test must carry the same "
                "features or the model is scoring on something else."
            )

        data: dict[str, np.ndarray] = {}
        unseen: dict[str, int] = {}
        for column, binning in self._fitted.items():
            values, count = binning.transform(frame[column])
            data[WOE_PREFIX + column] = values
            unseen[column] = count

        out = pd.DataFrame(data, index=frame.index)
        return out, TransformReport(
            rows=len(frame), columns=out.shape[1], unseen_by_column=unseen
        )

    def binning(self, column: str) -> FeatureBinning:
        return self._fitted[column]

    @property
    def selected_columns(self) -> tuple[str, ...]:
        return tuple(self._fitted)


# ---------------------------------------------------------------- fitting


@dataclass
class _Group:
    """A candidate bin during merging. Counts only; identity lives alongside."""

    good: int
    bad: int
    members: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return self.good + self.bad


def _fit_numeric(
    column: str,
    values: pd.Series,
    target: pd.Series,
    good_total: int,
    bad_total: int,
    config: WoeConfig,
) -> FeatureBinning:
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.notna().to_numpy()
    observed = numeric.to_numpy(dtype="float64")[present]
    outcome = target.to_numpy()[present]

    missing_group = _group_from(target.to_numpy()[~present])

    if observed.size == 0:
        bins = _finalise([], missing_group, good_total, bad_total, labels=[])
        return FeatureBinning(column, "numeric", bins, _total_iv(bins), edges=(-math.inf, math.inf), missing_index=0 if missing_group else None)

    edges = _quantile_edges(observed, config.pre_bins)
    codes = np.clip(np.searchsorted(edges, observed, side="right") - 1, 0, len(edges) - 2)
    groups = [
        _group_from(outcome[codes == index]) for index in range(len(edges) - 1)
    ]

    budget = config.max_bins - int(
        _missing_stands_alone(missing_group, observed.size, config.min_bin_fraction)
    )
    groups, edges = _merge_numeric(
        groups, list(edges), good_total, bad_total, config, max(budget, 1)
    )
    labels = [_numeric_label(edges[i], edges[i + 1]) for i in range(len(groups))]
    bins = _finalise(
        groups, missing_group, good_total, bad_total, labels,
        config.min_bin_fraction,
    )
    missing_index = _missing_index(bins)
    return FeatureBinning(
        column,
        "numeric",
        bins,
        _total_iv(bins),
        edges=tuple(edges),
        missing_index=missing_index,
    )


def _fit_categorical(
    column: str,
    values: pd.Series,
    target: pd.Series,
    good_total: int,
    bad_total: int,
    config: WoeConfig,
) -> FeatureBinning:
    keys = values.astype("str")
    present = values.notna().to_numpy()
    missing_group = _group_from(target.to_numpy()[~present])

    frame = pd.DataFrame(
        {"key": keys.to_numpy()[present], "y": target.to_numpy()[present]}
    )
    if frame.empty:
        bins = _finalise([], missing_group, good_total, bad_total, labels=[])
        return FeatureBinning(column, "categorical", bins, _total_iv(bins), missing_index=0 if missing_group else None)

    # Named `n` and `bad` rather than `size` and `sum`: those two collide with
    # Series attributes and the attribute wins silently.
    per_category = frame.groupby("key")["y"].agg(n="size", bad="sum")
    per_category["rate"] = per_category["bad"] / per_category["n"]
    # Ordering by event rate is what makes "merge adjacent" group rare
    # categories with categories of similar risk rather than with each other.
    per_category = per_category.sort_values("rate")

    groups = [
        _Group(
            good=int(row.n - row.bad),
            bad=int(row.bad),
            members=[str(row.Index)],
        )
        for row in per_category.itertuples()
    ]
    budget = config.max_bins - int(
        _missing_stands_alone(
            missing_group, int(present.sum()), config.min_category_fraction
        )
    )
    groups = _merge_categorical(
        groups, good_total, bad_total, config, max(budget, 1)
    )

    labels = [_categorical_label(group.members) for group in groups]
    bins = _finalise(
        groups, missing_group, good_total, bad_total, labels,
        config.min_category_fraction,
    )

    category_map: dict[str, int] = {}
    for index, group in enumerate(groups):
        for member in group.members:
            category_map[member] = index

    return FeatureBinning(
        column,
        "categorical",
        bins,
        _total_iv(bins),
        category_map=category_map,
        missing_index=_missing_index(bins),
    )


def _missing_stands_alone(
    missing: _Group | None, value_rows: int, min_fraction: float
) -> bool:
    """Will the missing group become a bin of its own?

    Decidable before any merging, because it depends only on the missing
    group's own size and class balance. It has to be decided early: a missing
    bin appended after the cap has been applied would put the feature one bin
    over max_bins, quietly breaking the contract the configuration states.
    """
    if missing is None or missing.count == 0:
        return False
    if missing.good == 0 or missing.bad == 0:
        return False
    return missing.count >= min_fraction * (value_rows + missing.count)


def _merge_numeric(
    groups: list[_Group],
    edges: list[float],
    good_total: int,
    bad_total: int,
    config: WoeConfig,
    value_budget: int,
) -> tuple[list[_Group], list[float]]:
    minimum = config.min_bin_fraction * sum(g.count for g in groups)

    def merge(index: int) -> None:
        groups[index] = _Group(
            good=groups[index].good + groups[index + 1].good,
            bad=groups[index].bad + groups[index + 1].bad,
        )
        del groups[index + 1]
        del edges[index + 1]

    _merge_until_large_enough(groups, minimum, merge)
    _merge_away_empty_classes(groups, merge)
    if config.enforce_monotonic:
        _merge_until_monotone(groups, good_total, bad_total, merge)
    _merge_until_at_most(groups, value_budget, good_total, bad_total, merge)
    return groups, edges


def _merge_categorical(
    groups: list[_Group],
    good_total: int,
    bad_total: int,
    config: WoeConfig,
    value_budget: int,
) -> list[_Group]:
    minimum = config.min_category_fraction * sum(g.count for g in groups)

    def merge(index: int) -> None:
        groups[index] = _Group(
            good=groups[index].good + groups[index + 1].good,
            bad=groups[index].bad + groups[index + 1].bad,
            members=groups[index].members + groups[index + 1].members,
        )
        del groups[index + 1]

    _merge_until_large_enough(groups, minimum, merge)
    _merge_away_empty_classes(groups, merge)
    _merge_until_at_most(groups, value_budget, good_total, bad_total, merge)
    return groups


def _merge_until_large_enough(groups, minimum, merge) -> None:
    while len(groups) > 1:
        smallest = min(range(len(groups)), key=lambda i: groups[i].count)
        if groups[smallest].count >= minimum:
            return
        merge(_neighbour_of(groups, smallest))


def _merge_away_empty_classes(groups, merge) -> None:
    """A bin with no goods or no bads gives an infinite WOE. Merge, do not patch."""
    while len(groups) > 1:
        offender = next(
            (i for i, g in enumerate(groups) if g.good == 0 or g.bad == 0), None
        )
        if offender is None:
            return
        merge(_neighbour_of(groups, offender))


def _merge_until_monotone(groups, good_total, bad_total, merge) -> None:
    while len(groups) > 2:
        woes = [_woe(g, good_total, bad_total) for g in groups]
        if _is_monotone(woes):
            return
        # Merge the adjacent pair whose ordering is most badly reversed.
        worst = max(
            range(len(woes) - 1),
            key=lambda i: abs(woes[i + 1] - woes[i]) if _reverses(woes, i) else -1.0,
        )
        if not _reverses(woes, worst):
            return
        merge(worst)


def _merge_until_at_most(groups, max_bins, good_total, bad_total, merge) -> None:
    while len(groups) > max_bins:
        woes = [_woe(g, good_total, bad_total) for g in groups]
        closest = min(
            range(len(woes) - 1), key=lambda i: abs(woes[i + 1] - woes[i])
        )
        merge(closest)


def _reverses(woes: Sequence[float], index: int) -> bool:
    """Does the step at `index` go against the overall direction?"""
    increasing = woes[-1] >= woes[0]
    step = woes[index + 1] - woes[index]
    return step < 0 if increasing else step > 0


def _is_monotone(woes: Sequence[float]) -> bool:
    if len(woes) < 2:
        return True
    ups = all(b >= a for a, b in zip(woes, woes[1:]))
    downs = all(b <= a for a, b in zip(woes, woes[1:]))
    return ups or downs


def _neighbour_of(groups: Sequence[_Group], index: int) -> int:
    """Index to pass to merge so that `index` is absorbed by its smaller neighbour."""
    if index == 0:
        return 0
    if index == len(groups) - 1:
        return index - 1
    return index - 1 if groups[index - 1].count <= groups[index + 1].count else index


def _finalise(
    groups: Sequence[_Group],
    missing: _Group | None,
    good_total: int,
    bad_total: int,
    labels: Sequence[str],
    min_fraction: float | None = None,
) -> tuple[Bin, ...]:
    """Attach the missing bin, absorbing it if it is too small to stand alone."""
    bins = [
        _bin(label, group, good_total, bad_total)
        for label, group in zip(labels, groups)
    ]
    if missing is None or missing.count == 0:
        return tuple(bins)

    too_small = False
    if min_fraction is not None and bins:
        total = sum(g.count for g in groups) + missing.count
        too_small = missing.count < min_fraction * total
    unstable = missing.good == 0 or missing.bad == 0

    if bins and (too_small or unstable):
        # Not enough evidence to give missingness its own WOE. Fold it into the
        # bin it most resembles rather than inventing a number for it.
        rate = missing.bad / missing.count
        nearest = min(range(len(bins)), key=lambda i: abs(bins[i].event_rate - rate))
        merged = _Group(
            good=groups[nearest].good + missing.good,
            bad=groups[nearest].bad + missing.bad,
        )
        bins[nearest] = _bin(
            f"{labels[nearest]} or {MISSING_LABEL}",
            merged,
            good_total,
            bad_total,
            includes_missing=True,
        )
        return tuple(bins)

    bins.append(_bin(MISSING_LABEL, missing, good_total, bad_total, includes_missing=True))
    return tuple(bins)


def _bin(
    label: str,
    group: _Group,
    good_total: int,
    bad_total: int,
    includes_missing: bool = False,
) -> Bin:
    woe = _woe(group, good_total, bad_total)
    dist_good = group.good / good_total if good_total else 0.0
    dist_bad = group.bad / bad_total if bad_total else 0.0
    return Bin(
        label=label,
        count=group.count,
        good=group.good,
        bad=group.bad,
        woe=woe,
        iv=(dist_good - dist_bad) * woe,
        includes_missing=includes_missing,
    )


def _woe(group: _Group, good_total: int, bad_total: int) -> float:
    if group.good == 0 or group.bad == 0 or good_total == 0 or bad_total == 0:
        return 0.0
    return math.log((group.good / good_total) / (group.bad / bad_total))


def _total_iv(bins: Iterable[Bin]) -> float:
    return float(sum(b.iv for b in bins))


def _missing_index(bins: Sequence[Bin]) -> int | None:
    for index, item in enumerate(bins):
        if item.includes_missing:
            return index
    return None


def _group_from(outcome: np.ndarray) -> _Group | None:
    if outcome.size == 0:
        return None
    bad = int(outcome.sum())
    return _Group(good=int(outcome.size - bad), bad=bad)


def _quantile_edges(values: np.ndarray, pre_bins: int) -> np.ndarray:
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, pre_bins + 1))
    interior = np.unique(quantiles)[1:-1]
    return np.concatenate(([-math.inf], interior, [math.inf]))


def _numeric_label(lower: float, upper: float) -> str:
    left = "-inf" if lower == -math.inf else f"{lower:,.4g}"
    right = "inf" if upper == math.inf else f"{upper:,.4g}"
    return f"[{left}, {right})"


def _categorical_label(members: Sequence[str], limit: int = 4) -> str:
    if len(members) <= limit:
        return ", ".join(members)
    return ", ".join(members[:limit]) + f", +{len(members) - limit} more"


def _kind_of(values: pd.Series) -> str:
    """Numeric or categorical, from the dtype the loader declared.

    This is not inference about content. `load.dtype_map` states the type of
    every column explicitly, so by the time a frame reaches here the question
    has already been answered.
    """
    return "numeric" if pd.api.types.is_numeric_dtype(values) else "categorical"


def _validate_target(target: pd.Series, expected_rows: int) -> pd.Series:
    if len(target) != expected_rows:
        raise ValueError(
            f"The target has {len(target)} rows and the frame has "
            f"{expected_rows}. They must be aligned row for row."
        )
    values = pd.Series(np.asarray(target)).astype("int64")
    unexpected = sorted(set(values.unique()) - {0, 1})
    if unexpected:
        raise ValueError(
            f"The target contains {unexpected} as well as 0 and 1. WOE is "
            "defined on a binary outcome; see D1 for how the target is built."
        )
    return values
