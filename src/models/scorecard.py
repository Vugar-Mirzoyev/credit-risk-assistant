"""D4 champion: a logistic regression over WOE features, expressed as points.

The champion is not chosen because it wins. It is chosen because every part of
it can be shown to someone: one coefficient per feature, and a score that
decomposes into points per attribute, so an applicant can be told which of
their characteristics cost them and a reviewer can check that each one moves
the score in the direction the data says it should.

Three things protect that property, and none of them are about accuracy.

*Collinear features are pruned.* Correlated inputs make coefficients unstable
and sometimes flip their sign. Measured on this extract, ``installment`` and
``loan_amnt`` correlate at 0.86 on WOE and ``num_rev_tl_bal_gt_0`` and
``num_actv_rev_tl`` at 0.99: the same fact counted twice. Features are taken in
descending IV and any feature correlating above the configured ceiling with one
already kept is dropped, along with a record of what it was dropped against.

*Wrong-signed coefficients are removed and the model refitted.* WOE is built so
that higher means safer, and the model predicts the probability of bad, so
every coefficient must come out negative. A positive one says the multivariate
fit contradicts the feature's own univariate relationship. That is what
collinearity looks like from the inside, and it would mean telling an applicant
their score fell because their income was high. The offending feature is
dropped and the fit repeated until the signs agree.

*Nothing is rebalanced and nothing is penalised.* ``class_weight`` and
resampling both shift the intercept and break calibration, and D5 measures
calibration separately because an expected-loss calculation needs
probabilities right in level, not only in rank. A bad rate near 20% is not
imbalanced enough to need either. Regularisation is left off as well: with a
million rows and thirty features it buys nothing, and a shrunk coefficient no
longer reads as a clean odds ratio.

Two variants are fitted rather than one. ``sub_grade``, ``grade`` and
``int_rate`` are LendingClub's own risk assessment, correlated with each other
at 0.94 to 0.97 and carrying by far the highest IV in the extract. A model
built on them largely re-learns theirs. The champion proper is fitted without
them; the variant with them is reported alongside, so the gap measures what the
incumbent's risk opinion is worth in Gini rather than hiding inside a single
headline number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import ModelConfig, ScorecardConfig
from src.features.woe import WOE_PREFIX, WoeBinner

#: LendingClub's own grade and its derivatives. Excluded from the champion.
GRADE_FAMILY: tuple[str, ...] = ("sub_grade", "grade", "int_rate")

#: WOE is ln(share of goods / share of bads), so higher is safer, and the model
#: predicts the probability of bad. Every coefficient must therefore be
#: negative. This is a property of the construction, not an empirical guess.
EXPECTED_SIGN = -1.0


@dataclass(frozen=True)
class ScorecardScaling:
    """Points scaling: `base_score` at `base_odds`, doubling every `pdo`."""

    base_score: float
    base_odds: float
    pdo: float

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)


@dataclass(frozen=True)
class AttributePoints:
    """One row of the scorecard: a bin of a feature, and what it is worth."""

    feature: str
    bin_label: str
    count: int
    event_rate: float
    woe: float
    coefficient: float
    points: float


@dataclass(frozen=True)
class Scorecard:
    """A fitted champion, and the points it decomposes into."""

    name: str
    features: tuple[str, ...]
    coefficients: dict[str, float]
    intercept: float
    scaling: ScorecardScaling
    attributes: tuple[AttributePoints, ...]

    @property
    def base_points(self) -> float:
        """Where every applicant starts, before any attribute is counted."""
        return self.scaling.offset - self.scaling.factor * self.intercept

    def _matrix(self, woe_frame: pd.DataFrame) -> np.ndarray:
        missing = [f for f in self.features if WOE_PREFIX + f not in woe_frame.columns]
        if missing:
            raise KeyError(
                f"WOE columns absent for {missing}. The frame being scored must "
                "come from the same fitted binner as the frame used to fit."
            )
        return woe_frame[[WOE_PREFIX + f for f in self.features]].to_numpy(
            dtype="float64"
        )

    def predict_proba(self, woe_frame: pd.DataFrame) -> np.ndarray:
        """Probability of bad. Calibrated on the fitting population."""
        logit = self.intercept + self._matrix(woe_frame) @ np.array(
            [self.coefficients[f] for f in self.features]
        )
        return 1.0 / (1.0 + np.exp(-logit))

    def score(self, woe_frame: pd.DataFrame) -> np.ndarray:
        """Points. Higher is safer, by construction of the scaling."""
        contributions = -self.scaling.factor * (
            self._matrix(woe_frame)
            * np.array([self.coefficients[f] for f in self.features])
        )
        return self.base_points + contributions.sum(axis=1)

    def points_table(self) -> pd.DataFrame:
        """The artefact. This is what is handed to a reviewer or an applicant."""
        return pd.DataFrame(
            [
                {
                    "feature": a.feature,
                    "bin": a.bin_label,
                    "count": a.count,
                    "event_rate": round(a.event_rate, 6),
                    "woe": round(a.woe, 6),
                    "coefficient": round(a.coefficient, 6),
                    "points": round(a.points, 2),
                }
                for a in self.attributes
            ]
        )


@dataclass
class ScorecardReport:
    """Why the champion contains what it contains."""

    name: str
    rows: int
    considered: tuple[str, ...]
    dropped_correlated: tuple[tuple[str, str, float], ...] = field(
        default_factory=tuple
    )
    dropped_wrong_sign: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    refits: int = 1
    selected: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = [
            f"variant                 {self.name}",
            f"fitting rows            {self.rows:,}",
            f"features considered     {len(self.considered)}",
            f"dropped, correlated     {len(self.dropped_correlated)}",
            f"dropped, wrong sign     {len(self.dropped_wrong_sign)}",
            f"refits                  {self.refits}",
            f"features in scorecard   {len(self.selected)}",
        ]
        for dropped, against, correlation in self.dropped_correlated:
            lines.append(f"  {dropped:<30} vs {against:<26} r={correlation:.3f}")
        for feature, coefficient in self.dropped_wrong_sign:
            lines.append(f"  {feature:<30} coefficient {coefficient:+.4f}")
        return "\n".join(lines)


def without_grade_family(features: Sequence[str]) -> tuple[str, ...]:
    """The champion's feature set: everything but the incumbent's own grade."""
    return tuple(f for f in features if f not in GRADE_FAMILY)


def prune_correlated(
    woe_frame: pd.DataFrame,
    features: Sequence[str],
    iv_by_feature: dict[str, float],
    max_correlation: float,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, float], ...]]:
    """Keep the higher-IV feature of any correlated pair, and say what went."""
    ordered = sorted(features, key=lambda f: -iv_by_feature.get(f, 0.0))
    columns = [WOE_PREFIX + f for f in ordered]
    correlation = woe_frame[columns].corr().abs()

    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for feature in ordered:
        column = WOE_PREFIX + feature
        against = next(
            (
                k
                for k in kept
                if correlation.loc[column, WOE_PREFIX + k] > max_correlation
            ),
            None,
        )
        if against is None:
            kept.append(feature)
        else:
            dropped.append(
                (feature, against, float(correlation.loc[column, WOE_PREFIX + against]))
            )
    return tuple(kept), tuple(dropped)


def fit_scorecard(
    woe_frame: pd.DataFrame,
    target: pd.Series,
    binner: WoeBinner,
    config: ModelConfig,
    *,
    features: Sequence[str] | None = None,
    name: str = "champion",
) -> tuple[Scorecard, ScorecardReport]:
    """Fit the champion on the training split.

    `woe_frame` must be the output of `binner.transform`, and `target` must be
    aligned to it row for row.
    """
    candidates = list(features if features is not None else binner.selected_columns)
    if not candidates:
        raise ValueError(
            "No features to fit on. Check the WOE fit report: every feature may "
            "have fallen below min_iv."
        )

    iv_by_feature = {b.column: b.iv for b in (binner.report.features if binner.report else ())}
    kept, dropped_correlated = prune_correlated(
        woe_frame, candidates, iv_by_feature, config.max_correlation
    )

    outcome = np.asarray(target, dtype="int64")
    wrong_sign: list[tuple[str, float]] = []
    refits = 0

    while True:
        if not kept:
            raise ValueError(
                "Every feature was removed for carrying a coefficient of the "
                "wrong sign. That is not a scorecard, it is a signal that the "
                "WOE transform or the target is inverted somewhere upstream."
            )
        refits += 1
        model = _fit(woe_frame, kept, outcome, config)
        coefficients = dict(zip(kept, model.coef_[0]))
        offenders = {f: c for f, c in coefficients.items() if c * EXPECTED_SIGN < 0}
        if not offenders:
            break
        # Remove the single worst offender and refit, rather than all at once:
        # dropping one often repairs the signs of the others.
        worst = max(offenders, key=lambda f: abs(offenders[f]))
        wrong_sign.append((worst, float(offenders[worst])))
        kept = [f for f in kept if f != worst]

    scaling = ScorecardScaling(
        base_score=config.scorecard.base_score,
        base_odds=config.scorecard.base_odds,
        pdo=config.scorecard.pdo,
    )
    scorecard = Scorecard(
        name=name,
        features=tuple(kept),
        coefficients={f: float(c) for f, c in coefficients.items()},
        intercept=float(model.intercept_[0]),
        scaling=scaling,
        attributes=_attributes(kept, coefficients, binner, scaling),
    )
    report = ScorecardReport(
        name=name,
        rows=len(woe_frame),
        considered=tuple(candidates),
        dropped_correlated=dropped_correlated,
        dropped_wrong_sign=tuple(wrong_sign),
        refits=refits,
        selected=tuple(kept),
    )
    return scorecard, report


def _fit(
    woe_frame: pd.DataFrame,
    features: Sequence[str],
    outcome: np.ndarray,
    config: ModelConfig,
) -> LogisticRegression:
    matrix = woe_frame[[WOE_PREFIX + f for f in features]].to_numpy(dtype="float64")
    # C=inf is unpenalised maximum likelihood. `penalty=None` says the same
    # thing but is deprecated from scikit-learn 1.8 and removed in 1.10.
    model = LogisticRegression(
        C=np.inf,
        max_iter=config.max_iter,
        random_state=config.random_state,
        class_weight=None,
    )
    model.fit(matrix, outcome)
    return model


def _attributes(
    features: Sequence[str],
    coefficients: dict[str, float],
    binner: WoeBinner,
    scaling: ScorecardScaling,
) -> tuple[AttributePoints, ...]:
    rows: list[AttributePoints] = []
    for feature in features:
        coefficient = coefficients[feature]
        for item in binner.binning(feature).bins:
            rows.append(
                AttributePoints(
                    feature=feature,
                    bin_label=item.label,
                    count=item.count,
                    event_rate=item.event_rate,
                    woe=item.woe,
                    coefficient=coefficient,
                    points=-scaling.factor * coefficient * item.woe,
                )
            )
    return tuple(rows)
