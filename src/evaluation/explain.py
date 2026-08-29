"""SHAP for the challenger, because D4 says it cannot be explained without it.

The champion needs nothing like this. Its score is a base plus one term per
attribute, and the points table already says exactly what each attribute
contributed and why. The challenger has no such decomposition: a thousand trees
voting produce a number, and no coefficient exists to point at.

SHAP supplies the missing decomposition. For each row it attributes the model's
log-odds output to the features, additively::

    base value + sum of the row's SHAP values = the model's raw output

That identity is what makes it an explanation rather than a plausible-looking
ranking, so this module checks it and reports the largest discrepancy it found
rather than assuming it holds.

Two things are worth knowing about what comes out.

*Gain and SHAP answer different questions.* The challenger's report already
carries gain-based importance, which says how much a feature reduced the loss
when the model split on it. Mean absolute SHAP says how far a feature actually
moves predictions. They usually agree on the top few and disagree lower down,
and where they disagree the SHAP ordering is the one that describes the model's
behaviour.

*The holdout is sampled, not explained whole.* SHAP costs a pass over every
tree for every row, and the sample size is carried on every result so a figure
derived from twenty thousand rows is never read as a population figure.

The model explained here is truncated to the tree count early stopping settled
on. Without that, SHAP would describe a slightly larger model than the one
whose predictions were scored, and the additivity check would quietly fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.config import ExplainConfig
from src.models.challenger import Challenger

#: Largest tolerated gap between base value + SHAP and the model's raw output.
ADDITIVITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class FeatureAttribution:
    feature: str
    mean_abs_shap: float
    mean_shap: float
    share: float


@dataclass(frozen=True)
class GlobalExplanation:
    """What drives the model, measured on a sample of the holdout."""

    rows_sampled: int
    rows_available: int
    base_value: float
    attributions: tuple[FeatureAttribution, ...]
    max_additivity_error: float

    @property
    def additive(self) -> bool:
        return self.max_additivity_error <= ADDITIVITY_TOLERANCE

    def table(self, top: int | None = None) -> pd.DataFrame:
        rows = self.attributions if top is None else self.attributions[:top]
        return pd.DataFrame(
            [
                {
                    "feature": a.feature,
                    "mean_abs_shap": round(a.mean_abs_shap, 6),
                    "mean_shap": round(a.mean_shap, 6),
                    "share": round(a.share, 4),
                }
                for a in rows
            ]
        )

    def summary(self) -> str:
        return (
            f"rows sampled            {self.rows_sampled:,} of "
            f"{self.rows_available:,}\n"
            f"base value (log-odds)   {self.base_value:+.4f}\n"
            f"features attributed     {len(self.attributions)}\n"
            f"max additivity error    {self.max_additivity_error:.2e}"
            + ("" if self.additive else "  ** additivity failed **")
        )


@dataclass(frozen=True)
class RowExplanation:
    """Why this one applicant received this one prediction."""

    base_value: float
    raw_output: float
    probability: float
    contributions: tuple[tuple[str, object, float], ...] = field(
        default_factory=tuple
    )

    def table(self, top: int | None = None) -> pd.DataFrame:
        rows = self.contributions if top is None else self.contributions[:top]
        return pd.DataFrame(
            [
                {"feature": name, "value": value, "shap": round(contribution, 6)}
                for name, value, contribution in rows
            ]
        )

    def summary(self) -> str:
        return (
            f"base value (log-odds)   {self.base_value:+.4f}\n"
            f"model raw output        {self.raw_output:+.4f}\n"
            f"probability of bad      {self.probability:.4f}\n"
            f"contributions sum to    "
            f"{self.base_value + sum(c for _, _, c in self.contributions):+.4f}"
        )


def _truncated_booster(challenger: Challenger) -> lgb.Booster:
    """The booster with only the trees early stopping kept.

    LightGBM leaves the extra rounds it tried in the model, and `predict` skips
    them by passing `num_iteration`. SHAP has no such argument, so the model is
    truncated here instead: explaining a larger model than the one that was
    scored would break the additivity identity for reasons nothing in the
    output would reveal.
    """
    booster = challenger.model.booster_
    return lgb.Booster(
        model_str=booster.model_to_string(num_iteration=challenger.best_iteration)
    )


def _shap_matrix(explainer: shap.TreeExplainer, matrix: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        # Older SHAP returns one array per class for binary problems.
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _base_value(explainer: shap.TreeExplainer) -> float:
    expected = explainer.expected_value
    if np.ndim(expected) == 0:
        return float(expected)
    return float(np.asarray(expected).ravel()[-1])


def explain_global(
    challenger: Challenger,
    frame: pd.DataFrame,
    *,
    config: ExplainConfig,
    random_state: int = 42,
) -> GlobalExplanation:
    """Mean absolute SHAP per feature, on a sample of the given frame."""
    missing = [f for f in challenger.features if f not in frame.columns]
    if missing:
        raise KeyError(
            f"Columns absent from the frame being explained: {sorted(missing)}."
        )
    if len(frame) == 0:
        raise ValueError("Nothing to explain: the frame is empty.")

    sample = frame
    if config.sample_rows and len(frame) > config.sample_rows:
        sample = frame.sample(config.sample_rows, random_state=random_state)
    matrix = sample[list(challenger.features)]

    booster = _truncated_booster(challenger)
    explainer = shap.TreeExplainer(booster)
    values = _shap_matrix(explainer, matrix)
    base = _base_value(explainer)

    raw = booster.predict(matrix, raw_score=True)
    error = float(np.abs(base + values.sum(axis=1) - raw).max())

    mean_abs = np.abs(values).mean(axis=0)
    total = mean_abs.sum()
    order = np.argsort(-mean_abs)
    attributions = tuple(
        FeatureAttribution(
            feature=challenger.features[index],
            mean_abs_shap=float(mean_abs[index]),
            mean_shap=float(values[:, index].mean()),
            share=float(mean_abs[index] / total) if total else 0.0,
        )
        for index in order
    )
    return GlobalExplanation(
        rows_sampled=len(sample),
        rows_available=len(frame),
        base_value=base,
        attributions=attributions,
        max_additivity_error=error,
    )


def explain_row(
    challenger: Challenger, frame: pd.DataFrame, position: int = 0
) -> RowExplanation:
    """The decomposition for a single applicant, largest contribution first."""
    if not 0 <= position < len(frame):
        raise IndexError(
            f"Row {position} is outside a frame of {len(frame)} rows."
        )
    row = frame.iloc[[position]][list(challenger.features)]

    booster = _truncated_booster(challenger)
    explainer = shap.TreeExplainer(booster)
    values = _shap_matrix(explainer, row)[0]
    base = _base_value(explainer)
    raw = float(booster.predict(row, raw_score=True)[0])

    order = np.argsort(-np.abs(values))
    contributions = tuple(
        (
            challenger.features[index],
            row.iloc[0, index],
            float(values[index]),
        )
        for index in order
    )
    return RowExplanation(
        base_value=base,
        raw_output=raw,
        probability=1.0 / (1.0 + np.exp(-raw)),
        contributions=contributions,
    )


def compare_with_scorecard(
    explanation: GlobalExplanation, points_table: pd.DataFrame
) -> pd.DataFrame:
    """Rank the same features by each model's own notion of influence.

    The champion's influence is the spread of points a feature can move the
    score by; the challenger's is mean absolute SHAP. The units have nothing to
    do with each other, so only the ordering is comparable - and the ordering is
    the interesting part, because it shows whether the two models are reading
    the population the same way.
    """
    spread = (
        points_table.groupby("feature")["points"]
        .agg(lambda column: column.max() - column.min())
        .rename("points_spread")
        .reset_index()
    )
    spread["champion_rank"] = spread["points_spread"].rank(ascending=False).astype(int)

    shap_table = explanation.table()
    shap_table["challenger_rank"] = (
        shap_table["mean_abs_shap"].rank(ascending=False).astype(int)
    )

    merged = spread.merge(
        shap_table[["feature", "mean_abs_shap", "challenger_rank"]],
        on="feature",
        how="outer",
    )
    merged["rank_gap"] = (
        merged["champion_rank"] - merged["challenger_rank"]
    ).abs()
    return merged.sort_values("challenger_rank").reset_index(drop=True)
