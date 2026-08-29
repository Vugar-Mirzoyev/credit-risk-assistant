"""Tests for D5.

Discrimination and calibration are checked against their definitions rather
than against remembered values, and the two are checked to be independent: a
model that ranks perfectly but is badly calibrated has to be caught by the
second set of numbers and not by the first.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_curve

from src.evaluation import metrics
from src.evaluation.metrics import calibration, discrimination, gains_table


def _calibrated(n=20000, seed=0):
    """Predictions that are correct in level by construction."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.uniform(0, 1, n) < p).astype("int64")
    return y, p


def test_gini_is_twice_auc_minus_one():
    y, p = _calibrated()

    result = discrimination(y, p)

    assert result.gini == pytest.approx(2 * result.auc - 1)


def test_perfect_ranking_scores_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    result = discrimination(y, p)

    assert result.auc == pytest.approx(1.0)
    assert result.gini == pytest.approx(1.0)
    assert result.ks == pytest.approx(1.0)


def test_a_coin_flip_scores_about_nothing():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40000)
    p = rng.uniform(0, 1, 40000)

    result = discrimination(y, p)

    assert result.auc == pytest.approx(0.5, abs=0.02)
    assert abs(result.gini) < 0.04


def test_ks_is_the_widest_gap_between_the_two_distributions():
    y, p = _calibrated()
    false_positive, true_positive, _ = roc_curve(y, p)

    result = discrimination(y, p)

    assert result.ks == pytest.approx((true_positive - false_positive).max())


def test_discrimination_records_the_population_it_measured():
    y, p = _calibrated()

    result = discrimination(y, p)

    assert result.rows == len(y)
    assert result.bad_rate == pytest.approx(y.mean())


def test_well_calibrated_predictions_have_slope_one_and_no_level_error():
    y, p = _calibrated(n=60000)

    result = calibration(y, p)

    assert result.slope == pytest.approx(1.0, abs=0.06)
    assert result.intercept == pytest.approx(0.0, abs=0.06)
    assert abs(result.calibration_in_the_large) < 0.01
    assert result.expected_calibration_error < 0.01


def test_a_shifted_level_is_caught_even_though_the_ranking_is_untouched():
    # Halving every probability leaves the ordering identical, so AUC, Gini and
    # KS do not move at all. Only the calibration numbers can see it, which is
    # the entire reason D5 measures them separately.
    y, p = _calibrated(n=60000)
    shifted = p / 2.0

    assert discrimination(y, shifted).auc == pytest.approx(
        discrimination(y, p).auc
    )

    result = calibration(y, shifted)
    assert result.calibration_in_the_large < -0.05
    assert result.expected_calibration_error > 0.05


def test_overconfident_predictions_show_a_slope_below_one():
    # Stretching the logits leaves the ranking alone and makes the extremes too
    # extreme, which is what a slope under 1 means.
    y, p = _calibrated(n=60000)
    logit = np.log(p / (1 - p))
    stretched = 1.0 / (1.0 + np.exp(-2.0 * logit))

    result = calibration(y, stretched)

    assert result.slope < 0.8


def test_the_brier_score_matches_its_definition():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.3, 0.6])

    result = calibration(y, p, buckets=2)

    assert result.brier == pytest.approx(np.mean((p - y) ** 2))


def test_calibration_buckets_cover_every_row():
    y, p = _calibrated()

    result = calibration(y, p, buckets=10)

    assert sum(b.count for b in result.buckets) == len(y)
    assert len(result.buckets) == 10
    assert "Bucket" in result.to_markdown()


def test_the_gains_table_orders_from_riskiest_to_safest():
    y, p = _calibrated(n=40000)

    table = gains_table(y, p, buckets=10)

    assert table["bucket"].tolist() == list(range(1, 11))
    assert table["loans"].sum() == len(y)
    # Bad rate must fall from the riskiest band to the safest.
    assert table["bad_rate"].iloc[0] > table["bad_rate"].iloc[-1]
    assert table["cumulative_bad_share"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_good_share"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > 1.0


def test_accuracy_is_not_offered():
    # D5 bans it: at a 19.96% bad rate, predicting good for everyone scores
    # 80.04% and finds no bad loans. It is absent rather than present and
    # discouraged, so it cannot end up in a report by accident.
    assert not hasattr(metrics, "accuracy")
    assert "accuracy" not in dir(metrics)


def test_a_single_class_cannot_be_ranked():
    with pytest.raises(ValueError, match="single class"):
        discrimination(np.zeros(10, dtype="int64"), np.linspace(0.1, 0.9, 10))


def test_a_non_binary_outcome_raises():
    with pytest.raises(ValueError, match="as well as 0 and 1"):
        discrimination(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_misaligned_arrays_raise():
    with pytest.raises(ValueError, match="aligned row for row"):
        discrimination(np.array([0, 1]), np.array([0.1, 0.5, 0.9]))


def test_scores_passed_instead_of_probabilities_raise():
    # Points run in the hundreds. Passing them here would silently produce
    # nonsense calibration, so it is refused with the reason.
    with pytest.raises(ValueError, match="probabilities of bad"):
        calibration(np.array([0, 1, 0, 1]), np.array([520.0, 610.0, 480.0, 590.0]))


def test_unscored_rows_raise():
    with pytest.raises(ValueError, match="contain nulls"):
        discrimination(np.array([0, 1, 0]), np.array([0.1, np.nan, 0.9]))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        discrimination(np.array([]), np.array([]))
