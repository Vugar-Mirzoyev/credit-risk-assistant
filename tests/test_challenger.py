"""Tests for the challenger.

The property worth protecting is that the holdout never influences the fit.
Everything else here is checking that the model is reproducible and that its
report describes what actually happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ChallengerConfig, ModelConfig, ScorecardConfig
from src.models.challenger import (
    fit_challenger,
    split_for_early_stopping,
)

CHALLENGER = ChallengerConfig(
    validation_months=6,
    learning_rate=0.1,
    num_leaves=7,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    n_estimators=60,
    early_stopping_rounds=10,
    deterministic=True,
)

MODEL = ModelConfig(
    random_state=42,
    max_correlation=0.75,
    max_iter=200,
    scorecard=ScorecardConfig(base_score=600.0, base_odds=50.0, pdo=20.0),
)


def _training_frame(n=4000, seed=0):
    """Three years of monthly originations with a learnable signal."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2013-01-01", "2015-12-01", freq="MS")
    frame = pd.DataFrame(
        {
            "issue_d": rng.choice(months, n),
            "x": rng.uniform(0, 1, n),
            "z": rng.normal(0, 1, n),
            "c": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    risk = 0.65 - 0.45 * frame["x"] + 0.08 * frame["z"]
    target = pd.Series((rng.uniform(0, 1, n) < risk).astype("int8"), index=frame.index)
    return frame, target


FEATURES = ["x", "z", "c"]


def test_the_early_stopping_window_is_the_last_months_of_training():
    frame, _ = _training_frame()

    fitting, validation, boundary = split_for_early_stopping(frame, "issue_d", 6)

    assert boundary == pd.Timestamp("2015-07-01")
    assert frame.loc[fitting, "issue_d"].max() < boundary
    assert frame.loc[validation, "issue_d"].min() >= boundary
    assert len(fitting) + len(validation) == len(frame)


def test_the_holdout_is_never_seen_by_the_fit():
    # The rows the challenger touches must all come from before the split
    # cutoff. If early stopping ever reached past it, the reported Gini would
    # not be out of time.
    frame, target = _training_frame()
    cutoff = pd.Timestamp("2016-01-01")

    _, report = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )

    assert report.fit_last < cutoff
    assert report.validation_last < cutoff
    assert report.fit_rows + report.validation_rows == len(frame)


def test_a_window_that_would_swallow_the_training_data_raises():
    frame, target = _training_frame()
    config = ChallengerConfig(**{**vars(CHALLENGER), "validation_months": 600})

    with pytest.raises(ValueError, match="too short"):
        fit_challenger(
            frame,
            target,
            features=FEATURES,
            date_column="issue_d",
            config=config,
            model_config=MODEL,
        )


def test_a_non_positive_window_raises():
    frame, _ = _training_frame()

    with pytest.raises(ValueError, match="validation_months"):
        split_for_early_stopping(frame, "issue_d", 0)


def test_a_missing_date_column_raises():
    frame, _ = _training_frame()

    with pytest.raises(KeyError, match="issue_d"):
        split_for_early_stopping(frame.drop(columns=["issue_d"]), "issue_d", 6)


def test_categorical_columns_are_recognised_from_their_dtype():
    frame, target = _training_frame()

    challenger, report = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )

    assert challenger.categorical_features == ("c",)
    assert report.categorical_features == ("c",)


def test_predictions_are_probabilities_and_carry_signal():
    frame, target = _training_frame()

    challenger, _ = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )
    probability = challenger.predict_proba(frame)

    assert probability.shape == (len(frame),)
    assert ((probability >= 0) & (probability <= 1)).all()
    # x drives the risk downward, so the correlation must be negative.
    assert np.corrcoef(probability, frame["x"])[0, 1] < -0.5


def test_the_fit_is_reproducible():
    # Two runs with the same seed must agree exactly, or no reported figure is
    # reproducible from a documented command.
    frame, target = _training_frame()
    kwargs = dict(
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )

    first, _ = fit_challenger(frame, target, **kwargs)
    second, _ = fit_challenger(frame, target, **kwargs)

    np.testing.assert_array_equal(
        first.predict_proba(frame), second.predict_proba(frame)
    )


def test_the_report_records_where_it_stopped():
    frame, target = _training_frame()

    _, report = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )

    assert report.trees_offered == CHALLENGER.n_estimators
    assert 1 <= report.best_iteration <= CHALLENGER.n_estimators
    assert report.stopped_early == (report.best_iteration < report.trees_offered)
    assert "trees used" in report.summary()


def test_importance_covers_every_feature_and_sums_to_one():
    frame, target = _training_frame()

    _, report = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )

    assert set(report.importance["feature"]) == set(FEATURES)
    assert report.importance["share"].sum() == pytest.approx(1.0)


def test_scoring_a_frame_without_the_fitted_columns_raises():
    frame, target = _training_frame()
    challenger, _ = fit_challenger(
        frame,
        target,
        features=FEATURES,
        date_column="issue_d",
        config=CHALLENGER,
        model_config=MODEL,
    )

    with pytest.raises(KeyError, match="absent from the frame"):
        challenger.predict_proba(frame.drop(columns=["z"]))


def test_fitting_with_no_features_raises():
    frame, target = _training_frame()

    with pytest.raises(ValueError, match="No features"):
        fit_challenger(
            frame,
            target,
            features=[],
            date_column="issue_d",
            config=CHALLENGER,
            model_config=MODEL,
        )
