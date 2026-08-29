"""Tests for the champion.

What is protected here is not accuracy, it is the properties that make the
model presentable: every coefficient in the direction the data says, a score
that decomposes into points, and points on a scale that means what it claims.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig, ScorecardConfig, WoeConfig
from src.features.woe import WOE_PREFIX, WoeBinner
from src.models.scorecard import (
    GRADE_FAMILY,
    ScorecardScaling,
    fit_scorecard,
    prune_correlated,
    without_grade_family,
)

MODEL = ModelConfig(
    random_state=42,
    max_correlation=0.75,
    max_iter=200,
    scorecard=ScorecardConfig(base_score=600.0, base_odds=50.0, pdo=20.0),
)

WOE = WoeConfig(
    pre_bins=20,
    max_bins=10,
    min_bin_fraction=0.05,
    min_category_fraction=0.01,
    enforce_monotonic=True,
    min_iv=0.0,
    suspicious_iv=0.5,
)


def _fitted(n=6000, seed=0, extra=None):
    """Two informative features and a target that depends on both."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)}
    )
    if extra is not None:
        frame = frame.assign(**extra(frame, rng))
    risk = 0.7 - 0.4 * frame["a"] - 0.3 * frame["b"]
    target = pd.Series((rng.uniform(0, 1, n) < risk).astype("int8"))
    binner = WoeBinner(WOE).fit(frame, target)
    woe_frame, _ = binner.transform(frame)
    return frame, target, binner, woe_frame


def test_every_coefficient_comes_out_negative():
    # WOE is higher for safer bins and the model predicts bad, so a positive
    # coefficient would mean the score falls as the evidence improves.
    _, target, binner, woe_frame = _fitted()

    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    assert scorecard.features
    assert all(c < 0 for c in scorecard.coefficients.values())


def test_a_wrong_signed_feature_is_removed_and_the_model_refitted():
    # A near-duplicate that survives correlation pruning can still flip a sign.
    # Whatever the cause, the scorecard must not ship with one.
    def add_mirror(frame, rng):
        return {"a_noisy": frame["a"] + rng.normal(0, 0.30, len(frame))}

    _, target, binner, woe_frame = _fitted(extra=add_mirror, seed=5)
    scorecard, report = fit_scorecard(woe_frame, target, binner, MODEL)

    assert all(c < 0 for c in scorecard.coefficients.values())
    for feature, _ in report.dropped_wrong_sign:
        assert feature not in scorecard.features
    assert report.refits == len(report.dropped_wrong_sign) + 1


def test_correlated_features_are_pruned_keeping_the_stronger():
    woe_frame = pd.DataFrame(
        {
            WOE_PREFIX + "strong": [0.0, 1.0, 2.0, 3.0, 4.0],
            WOE_PREFIX + "copy": [0.0, 1.0, 2.0, 3.0, 4.01],
            WOE_PREFIX + "other": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    ivs = {"strong": 0.3, "copy": 0.29, "other": 0.1}

    kept, dropped = prune_correlated(
        woe_frame, ["strong", "copy", "other"], ivs, 0.75
    )

    assert "strong" in kept
    assert "copy" not in kept
    assert dropped[0][0] == "copy"
    assert dropped[0][1] == "strong"
    assert dropped[0][2] > 0.75


def test_the_score_decomposes_into_the_points_it_reports():
    # The promise D4 makes: the score is the base plus one contribution per
    # attribute. If this fails, the points table is decoration.
    _, target, binner, woe_frame = _fitted()
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    scores = scorecard.score(woe_frame)
    factor = scorecard.scaling.factor
    rebuilt = scorecard.base_points + sum(
        -factor
        * scorecard.coefficients[f]
        * woe_frame[WOE_PREFIX + f].to_numpy(dtype="float64")
        for f in scorecard.features
    )

    np.testing.assert_allclose(scores, rebuilt)


def test_points_and_probability_agree_through_the_scaling():
    _, target, binner, woe_frame = _fitted()
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    probability = scorecard.predict_proba(woe_frame)
    odds_good_to_bad = (1 - probability) / probability
    expected = scorecard.scaling.offset + scorecard.scaling.factor * np.log(
        odds_good_to_bad
    )

    np.testing.assert_allclose(scorecard.score(woe_frame), expected, rtol=1e-9)


def test_the_scale_means_what_it_says():
    scaling = ScorecardScaling(base_score=600.0, base_odds=50.0, pdo=20.0)

    # base_score sits at base_odds, and every pdo points doubles the odds.
    assert scaling.offset + scaling.factor * math.log(50.0) == pytest.approx(600.0)
    assert scaling.factor * math.log(2.0) == pytest.approx(20.0)


def test_a_higher_score_means_a_lower_probability_of_bad():
    _, target, binner, woe_frame = _fitted()
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    scores = scorecard.score(woe_frame)
    probability = scorecard.predict_proba(woe_frame)

    assert np.corrcoef(scores, probability)[0, 1] < -0.99


def test_the_points_table_covers_every_bin_of_every_feature():
    _, target, binner, woe_frame = _fitted()
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    table = scorecard.points_table()
    expected = sum(len(binner.binning(f).bins) for f in scorecard.features)

    assert len(table) == expected
    assert set(table["feature"]) == set(scorecard.features)
    assert table["points"].notna().all()


def test_the_grade_family_is_removable_as_a_set():
    features = ("sub_grade", "grade", "int_rate", "dti", "loan_amnt")

    assert without_grade_family(features) == ("dti", "loan_amnt")
    assert set(GRADE_FAMILY) == {"sub_grade", "grade", "int_rate"}


def test_the_report_names_what_was_dropped_and_why():
    def add_mirror(frame, rng):
        return {"a_copy": frame["a"] + rng.normal(0, 0.01, len(frame))}

    _, target, binner, woe_frame = _fitted(extra=add_mirror)
    _, report = fit_scorecard(woe_frame, target, binner, MODEL, name="variant")

    assert report.name == "variant"
    assert report.rows == len(woe_frame)
    dropped = {d[0] for d in report.dropped_correlated}
    assert "a_copy" in dropped or "a" in dropped
    assert "r=" in report.summary()


def test_scoring_a_frame_from_a_different_binner_raises():
    _, target, binner, woe_frame = _fitted()
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    with pytest.raises(KeyError, match="same fitted binner"):
        scorecard.predict_proba(pd.DataFrame({"woe_unrelated": [0.1, 0.2]}))


def test_fitting_with_no_features_raises():
    _, target, binner, woe_frame = _fitted()

    with pytest.raises(ValueError, match="No features to fit on"):
        fit_scorecard(woe_frame, target, binner, MODEL, features=[])


def test_the_model_is_not_rebalanced():
    # class_weight would move the intercept and break calibration, which D5
    # measures separately. The mean predicted probability should sit on the
    # observed bad rate.
    _, target, binner, woe_frame = _fitted()
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)

    assert scorecard.predict_proba(woe_frame).mean() == pytest.approx(
        target.mean(), abs=0.005
    )
