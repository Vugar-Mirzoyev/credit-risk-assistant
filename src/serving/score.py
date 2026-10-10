"""Scoring an application, and the reasons that go with the score.

Two paths share one core. `score_frame` scores any number of rows and exists
so `train_models.py` can push the whole holdout through the serving path and
prove it reproduces the fitted scorecard to floating-point precision.
`score_application` scores one applicant for the API and is stricter, because
a service has no report to put a warning in.

D13 - every score carries its reasons. Regulation B requires a lender who
declines an application to state the principal reasons, and its commentary
(9(b)(2)-5) names two acceptable ways to choose them from a scoring system.
The one used here is the second: the features on which the applicant scored
furthest below the average of all applicants. The first needs the average of
applicants at the minimum passing score, and a passing score is a cutoff -
a business decision about risk appetite that this project does not make. The
service therefore returns a probability, a score and the reasons, and no
approve or decline. Reasons are returned for every application: they are what
a notice would cite if the lender, applying its own cutoff, declined it.

D14 - the input contract comes from the artefact. A missing value is accepted
only where the training data contained missing values for that feature, and a
category is accepted only if training saw it. In a batch, `WoeBinner` gives
an unseen value the neutral WOE and counts it in the transform report. One
application has no report: a neutral score there would be a silent guess, so
it is refused with the reason instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.features.build import DERIVATIONS, Derivation
from src.features.woe import UNSEEN_CODE
from src.serving.artefact import Champion, ScoredFeature

_DERIVATIONS: dict[str, Derivation] = {d.name: d for d in DERIVATIONS}

#: Derivation sources that are not themselves numeric. Anything not listed
#: here and not a categorical scorecard feature is a number.
_SOURCE_KINDS: dict[str, str] = {
    "emp_length": "text",
    "earliest_cr_line": "text",
    "sec_app_earliest_cr_line": "text",
    "issue_d": "date",
}


class UnscorableApplication(ValueError):
    """The application cannot be placed in a bin the champion was fitted on."""

    def __init__(self, problems: Mapping[str, str]) -> None:
        self.problems = dict(problems)
        super().__init__(
            "; ".join(f"{name}: {why}" for name, why in self.problems.items())
        )


@dataclass(frozen=True)
class InputSpec:
    """One field an application must, or may, carry."""

    name: str
    kind: str  # numeric | categorical | text | date
    required: bool
    categories: tuple[str, ...] = ()
    feeds: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameScores:
    probability_of_default: np.ndarray
    score: np.ndarray
    #: rows x features, the bin each row fell in, `UNSEEN_CODE` where none.
    codes: np.ndarray


@dataclass(frozen=True)
class Attribute:
    feature: str
    bin: str
    points: float


@dataclass(frozen=True)
class Reason:
    feature: str
    description: str
    bin: str
    points: float
    reference_points: float

    @property
    def shortfall(self) -> float:
        return self.reference_points - self.points


@dataclass(frozen=True)
class ScoreResult:
    model_version: str
    probability_of_default: float
    score: float
    reasons: tuple[Reason, ...]
    attributes: tuple[Attribute, ...]


def input_specs(champion: Champion) -> tuple[InputSpec, ...]:
    """The fields an application carries, derived from the champion itself.

    A field is required when some feature that consumes it has no missing bin:
    training never saw that feature missing, so there is no evidence for what
    missing would mean.

    The inputs of a derived feature are always required. Its missing bin, if
    it has one, records arithmetic that came out undefined in training - an
    income of zero makes loan-to-income undefined - and not an input that was
    absent. Scoring an absent loan amount in that bin would be treating it as
    a zero income.
    """
    required: dict[str, bool] = {}
    feeds: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    categories: dict[str, tuple[str, ...]] = {}

    for feature in champion.features:
        binning = feature.binning
        sources = _sources_of(feature.name)
        derived = feature.name in _DERIVATIONS
        for source in sources:
            feeds.setdefault(source, []).append(feature.name)
            required[source] = required.get(source, False) or derived or (
                binning.missing_index is None
            )
            if source == feature.name and binning.kind == "categorical":
                kinds[source] = "categorical"
                categories[source] = tuple(binning.category_map)
            else:
                kinds.setdefault(source, _SOURCE_KINDS.get(source, "numeric"))

    return tuple(
        InputSpec(
            name=name,
            kind=kinds[name],
            required=required[name],
            categories=categories.get(name, ()),
            feeds=tuple(feeds[name]),
        )
        for name in feeds
    )


def score_frame(champion: Champion, frame: pd.DataFrame) -> FrameScores:
    """Score every row, exactly as the fitted scorecard would.

    Derived features are always recomputed from their sources, even when the
    frame already carries them, so the derivation is part of what the parity
    check in `train_models.py` proves.
    """
    columns = _feature_columns(champion, frame)
    codes = np.column_stack(
        [f.binning.codes(columns[f.name]) for f in champion.features]
    ) if champion.features else np.empty((len(frame), 0), dtype="int64")

    logit = np.full(len(frame), champion.intercept, dtype="float64")
    score = np.full(len(frame), champion.base_points, dtype="float64")
    for index, feature in enumerate(champion.features):
        found = codes[:, index] != UNSEEN_CODE
        woe = np.zeros(len(frame), dtype="float64")
        points = np.zeros(len(frame), dtype="float64")
        woe[found] = np.array([b.woe for b in feature.binning.bins])[codes[found, index]]
        points[found] = np.array(feature.points)[codes[found, index]]
        logit += feature.coefficient * woe
        score += points

    return FrameScores(
        probability_of_default=1.0 / (1.0 + np.exp(-logit)),
        score=score,
        codes=codes,
    )


def score_application(
    champion: Champion,
    application: Mapping[str, Any],
    *,
    reason_texts: Mapping[str, str],
    max_reasons: int,
) -> ScoreResult:
    """Score one applicant and select the principal reasons (D13, D14)."""
    specs = {spec.name: spec for spec in input_specs(champion)}
    problems: dict[str, str] = {}
    for name, spec in specs.items():
        value = application.get(name)
        if value is None and spec.required:
            problems[name] = (
                "required: the training data never had it missing, so a "
                "missing value has no evidence behind it"
            )
        elif spec.kind == "categorical" and value is not None and (
            str(value) not in spec.categories
        ):
            problems[name] = (
                f"{value!r} was never seen in training; expected one of "
                f"{list(spec.categories)}"
            )
    unexpected = sorted(set(application) - set(specs))
    for name in unexpected:
        problems[name] = "not an input of this model"
    if problems:
        raise UnscorableApplication(problems)

    frame = pd.DataFrame(
        {name: [_coerce(application.get(name), spec)] for name, spec in specs.items()}
    )
    scores = score_frame(champion, frame)
    codes = scores.codes[0]

    # A derived feature can still come out with no bin - an income of zero
    # makes loan_to_income undefined - and that is only visible after binning.
    for index, feature in enumerate(champion.features):
        if codes[index] == UNSEEN_CODE:
            problems[feature.name] = (
                "the inputs it is computed from give a value the model has no "
                f"bin for (sources: {', '.join(_sources_of(feature.name))})"
            )
    if problems:
        raise UnscorableApplication(problems)

    attributes = tuple(
        Attribute(
            feature=f.name,
            bin=f.binning.bins[codes[i]].label,
            points=f.points[codes[i]],
        )
        for i, f in enumerate(champion.features)
    )
    return ScoreResult(
        model_version=champion.version,
        probability_of_default=float(scores.probability_of_default[0]),
        score=float(scores.score[0]),
        reasons=select_reasons(
            champion.features, attributes, reason_texts, max_reasons
        ),
        attributes=attributes,
    )


def select_reasons(
    features: Sequence[ScoredFeature],
    attributes: Sequence[Attribute],
    reason_texts: Mapping[str, str],
    max_reasons: int,
) -> tuple[Reason, ...]:
    """Features scored furthest below the population average, largest first.

    A feature where the applicant scored at or above the average is never a
    reason: it did not count against them. Ties keep scorecard order, so the
    same application always produces the same list.
    """
    candidates = [
        Reason(
            feature=feature.name,
            description=reason_texts[feature.name],
            bin=attribute.bin,
            points=attribute.points,
            reference_points=feature.reference_points,
        )
        for feature, attribute in zip(features, attributes)
    ]
    below = [r for r in candidates if r.shortfall > 0]
    below.sort(key=lambda r: -r.shortfall)
    return tuple(below[:max_reasons])


def check_reason_texts(champion: Champion, reason_texts: Mapping[str, str]) -> None:
    """Every scored feature needs a reason a person can read (Reg B 9(b)(2)-2)."""
    missing = [f.name for f in champion.features if not reason_texts.get(f.name)]
    if missing:
        raise KeyError(
            f"No reason text for {missing} in config/reason_texts.yaml. A "
            "feature the champion scores can be a principal reason for a "
            "decline, and the notice must be able to name it."
        )


def _sources_of(feature: str) -> tuple[str, ...]:
    derivation = _DERIVATIONS.get(feature)
    return derivation.sources if derivation is not None else (feature,)


def _feature_columns(champion: Champion, frame: pd.DataFrame) -> dict[str, pd.Series]:
    columns: dict[str, pd.Series] = {}
    for feature in champion.features:
        derivation = _DERIVATIONS.get(feature.name)
        if derivation is None:
            columns[feature.name] = frame[feature.name]
            continue
        absent = [s for s in derivation.sources if s not in frame.columns]
        if absent:
            raise KeyError(
                f"{feature.name!r} is derived from {list(derivation.sources)} "
                f"and the frame lacks {absent}."
            )
        columns[feature.name] = pd.Series(
            np.asarray(derivation.compute(frame), dtype="float64"), index=frame.index
        )
    return columns


def _coerce(value: Any, spec: InputSpec) -> Any:
    if value is None:
        return np.nan if spec.kind == "numeric" else None
    if spec.kind == "numeric":
        return float(value)
    if spec.kind == "date":
        return pd.Timestamp(value)
    return str(value)
