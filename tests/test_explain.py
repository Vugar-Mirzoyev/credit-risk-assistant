"""Tests for SHAP attribution.

The property that makes SHAP an explanation rather than a ranking is
additivity: base value plus the row's contributions must equal the model's own
output. If that fails, every number in the table is decoration, so it is
checked directly and on a model whose trees are deliberately truncated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ChallengerConfig, ExplainConfig, ModelConfig, ScorecardConfig
from src.evaluation.explain import (
    ADDITIVITY_TOLERANCE,
    compare_with_scorecard,
    explain_global,
    explain_row,
)
from src.models.challenger import fit_challenger

CHALLENGER = ChallengerConfig(
    validation_months=6,
    learning_rate=0.1,
    num_leaves=7,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    n_estimators=40,
    early_stopping_rounds=10,
    deterministic=True,
)

MODEL = ModelConfig(
    random_state=42,
    max_correlation=0.75,
    max_iter=200,
    scorecard=ScorecardConfig(base_score=600.0, base_odds=50.0, pdo=20.0),
)

EXPLAIN = ExplainConfig(sample_rows=500, top_features=10)

FEATURES = ["driver", "noise", "c"]


def _fitted(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2013-01-01", "2015-12-01", freq="MS")
    frame = pd.DataFrame(
        {
            "issue_d": rng.choice(months, n),
            "driver": rng.uniform(0, 1, n),
            "noise": rng.normal(0, 1, n),
            "c": pd.Categorical(rng.choice(["a", "b"], n)),
        }
    )
    risk = 0.75 - 0.6 * frame["driver"]
    target = pd.Series((rng.uniform(0, 1, n) < risk).astype("int8"), index=frame.index)
    challenger, _ = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )
    return challenger, frame


def test_contributions_add_up_to_the_models_own_output():
    challenger, frame = _fitted()

    explanation = explain_global(challenger, frame, config=EXPLAIN)

    assert explanation.additive
    assert explanation.max_additivity_error <= ADDITIVITY_TOLERANCE


def test_a_single_row_reconstructs_its_own_prediction():
    challenger, frame = _fitted()

    row = explain_row(challenger, frame, position=3)
    rebuilt = row.base_value + sum(c for _, _, c in row.contributions)

    assert rebuilt == pytest.approx(row.raw_output, abs=ADDITIVITY_TOLERANCE)
    assert row.probability == pytest.approx(
        1.0 / (1.0 + np.exp(-row.raw_output))
    )


def test_the_row_explanation_agrees_with_the_scored_probability():
    # The truncated booster must be the same model predict_proba uses, or the
    # explanation describes something the applicant was not scored by.
    challenger, frame = _fitted()

    row = explain_row(challenger, frame, position=7)
    scored = challenger.predict_proba(frame.iloc[[7]])[0]

    assert row.probability == pytest.approx(scored, abs=1e-6)


def test_the_feature_that_drives_the_target_ranks_first():
    challenger, frame = _fitted()

    explanation = explain_global(challenger, frame, config=EXPLAIN)

    assert explanation.attributions[0].feature == "driver"
    assert explanation.attributions[0].share > 0.5


def test_shares_sum_to_one():
    challenger, frame = _fitted()

    explanation = explain_global(challenger, frame, config=EXPLAIN)

    assert sum(a.share for a in explanation.attributions) == pytest.approx(1.0)


def test_the_sample_size_is_carried_not_hidden():
    challenger, frame = _fitted(n=3000)

    explanation = explain_global(challenger, frame, config=EXPLAIN)

    assert explanation.rows_sampled == EXPLAIN.sample_rows
    assert explanation.rows_available == 3000
    assert "of 3,000" in explanation.summary()


def test_a_frame_smaller_than_the_sample_is_explained_whole():
    challenger, frame = _fitted(n=3000)
    small = frame.iloc[:200]

    explanation = explain_global(challenger, small, config=EXPLAIN)

    assert explanation.rows_sampled == 200


def test_sampling_is_reproducible():
    challenger, frame = _fitted()

    first = explain_global(challenger, frame, config=EXPLAIN, random_state=7)
    second = explain_global(challenger, frame, config=EXPLAIN, random_state=7)

    assert [a.feature for a in first.attributions] == [
        a.feature for a in second.attributions
    ]
    assert first.attributions[0].mean_abs_shap == pytest.approx(
        second.attributions[0].mean_abs_shap
    )


def test_contributions_are_ordered_by_size():
    challenger, frame = _fitted()

    row = explain_row(challenger, frame)
    magnitudes = [abs(c) for _, _, c in row.contributions]

    assert magnitudes == sorted(magnitudes, reverse=True)


def test_comparing_the_two_models_lines_up_the_orderings():
    challenger, frame = _fitted()
    explanation = explain_global(challenger, frame, config=EXPLAIN)
    points = pd.DataFrame(
        {
            "feature": ["driver", "driver", "noise", "noise"],
            "points": [20.0, -15.0, 3.0, -2.0],
        }
    )

    table = compare_with_scorecard(explanation, points)

    assert set(table["feature"]) >= {"driver", "noise"}
    driver = table[table.feature == "driver"].iloc[0]
    assert driver["champion_rank"] == 1
    assert driver["challenger_rank"] == 1
    assert driver["rank_gap"] == 0


def test_explaining_a_frame_without_the_fitted_columns_raises():
    challenger, frame = _fitted()

    with pytest.raises(KeyError, match="absent from the frame"):
        explain_global(challenger, frame.drop(columns=["noise"]), config=EXPLAIN)


def test_an_empty_frame_raises():
    challenger, frame = _fitted()

    with pytest.raises(ValueError, match="frame is empty"):
        explain_global(challenger, frame.iloc[:0], config=EXPLAIN)


def test_a_row_outside_the_frame_raises():
    challenger, frame = _fitted()

    with pytest.raises(IndexError, match="outside a frame"):
        explain_row(challenger, frame, position=len(frame))
