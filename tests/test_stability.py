"""Tests for PSI and CSI.

The failure that matters here is a monitoring index that cannot fire: bucket
the actual window on its own quantiles and every comparison returns zero, which
looks like a stable population and is really a broken instrument.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stability import (
    categorical_stability,
    characteristic_stability,
    numeric_stability,
    psi_band,
    stability_by_quarter,
)


def _normal(n=20000, loc=0.0, scale=1.0, seed=0):
    return np.random.default_rng(seed).normal(loc, scale, n)


def test_a_distribution_against_itself_is_zero():
    values = _normal()

    assert numeric_stability(values, values).psi == pytest.approx(0.0, abs=1e-12)


def test_the_same_population_sampled_twice_is_stable():
    assert numeric_stability(_normal(seed=1), _normal(seed=2)).psi < 0.01


def test_a_shifted_population_is_detected():
    # A full standard deviation of drift must not read as stable.
    result = numeric_stability(_normal(seed=1), _normal(loc=1.0, seed=2))

    assert result.psi > 0.25
    assert result.band == "shifted, investigate"


def test_the_index_grows_with_the_size_of_the_shift():
    reference = _normal(seed=1)
    small = numeric_stability(reference, _normal(loc=0.2, seed=2)).psi
    large = numeric_stability(reference, _normal(loc=1.0, seed=2)).psi

    assert small < large


def test_buckets_come_from_the_expected_distribution_not_the_actual():
    # The bug this guards against: bucketing each window on its own quantiles
    # compares every distribution with itself and returns zero forever.
    reference = _normal(seed=1)
    shifted = _normal(loc=2.0, scale=0.5, seed=2)

    result = numeric_stability(reference, shifted)
    shares = [b.actual_share for b in result.buckets]

    assert result.psi > 0.25
    # If the buckets had been recomputed on the actual window, its mass would
    # be spread evenly across them.
    assert max(shares) > 0.5


def test_contributions_sum_to_the_index():
    result = numeric_stability(_normal(seed=1), _normal(loc=0.5, seed=2))

    assert result.psi == pytest.approx(sum(b.contribution for b in result.buckets))


def test_psi_matches_its_definition_on_counts_done_by_hand():
    # Two buckets, 50/50 expected against 25/75 actual.
    expected = np.array([0.0] * 100 + [1.0] * 100)
    actual = np.array([0.0] * 50 + [1.0] * 150)

    result = numeric_stability(expected, actual, buckets=2)
    by_hand = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(
        0.75 / 0.5
    )

    assert result.psi == pytest.approx(by_hand)


def test_an_empty_bucket_is_floored_rather_than_infinite():
    # The actual window never reaches the top of the expected range. Merging is
    # not available here: the buckets have to be identical on both sides or the
    # comparison is not a comparison.
    expected = np.linspace(0, 100, 10000)
    actual = np.linspace(0, 40, 10000)

    result = numeric_stability(expected, actual)

    assert math.isfinite(result.psi)
    assert result.floored_buckets > 0
    assert "floored" in result.summary()


def test_a_change_in_the_missing_rate_is_drift():
    # Dropping the nulls would hide it completely.
    expected = np.concatenate([_normal(9000), np.full(1000, np.nan)])
    actual = np.concatenate([_normal(5000, seed=3), np.full(5000, np.nan)])

    result = numeric_stability(expected, actual)
    missing = [b for b in result.buckets if b.label == "Missing"]

    assert len(missing) == 1
    assert missing[0].expected_share == pytest.approx(0.1)
    assert missing[0].actual_share == pytest.approx(0.5)
    assert result.psi > 0.25


def test_categorical_stability_sees_a_changed_mix():
    expected = ["a"] * 700 + ["b"] * 300
    actual = ["a"] * 300 + ["b"] * 700

    result = categorical_stability(expected, actual)

    assert result.psi > 0.25
    assert {b.label for b in result.buckets} >= {"a", "b"}


def test_a_category_that_only_appears_later_is_counted():
    expected = ["a"] * 500 + ["b"] * 500
    actual = ["a"] * 400 + ["b"] * 400 + ["brand_new"] * 200

    result = categorical_stability(expected, actual)
    new = [b for b in result.buckets if b.label == "brand_new"]

    assert len(new) == 1
    assert new[0].expected == 0
    assert new[0].actual == 200
    assert math.isfinite(result.psi)


def test_quarterly_stability_reports_one_row_per_quarter():
    reference = _normal(seed=1)
    dates = pd.to_datetime(
        ["2017-01-15"] * 500 + ["2017-04-15"] * 500 + ["2017-07-15"] * 500
    )
    actual = np.concatenate(
        [_normal(500, seed=2), _normal(500, loc=0.5, seed=3), _normal(500, loc=1.5, seed=4)]
    )

    table = stability_by_quarter(reference, actual, dates)

    assert table["quarter"].tolist() == ["2017Q1", "2017Q2", "2017Q3"]
    assert table["rows"].tolist() == [500, 500, 500]
    # Drift grows through the year, so the index has to as well.
    assert table["psi"].is_monotonic_increasing


def test_quarterly_stability_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned row for row"):
        stability_by_quarter(
            _normal(100), _normal(100), pd.to_datetime(["2017-01-01"] * 50)
        )


def test_characteristic_stability_ranks_the_movers_first():
    rng = np.random.default_rng(7)
    train = pd.DataFrame(
        {
            "steady": rng.normal(0, 1, 5000),
            "drifting": rng.normal(0, 1, 5000),
            "category": pd.Categorical(rng.choice(["x", "y"], 5000)),
        }
    )
    test = pd.DataFrame(
        {
            "steady": rng.normal(0, 1, 5000),
            "drifting": rng.normal(1.5, 1, 5000),
            "category": pd.Categorical(rng.choice(["x", "y"], 5000)),
        }
    )

    table = characteristic_stability(train, test, list(train.columns))

    assert table["feature"].iloc[0] == "drifting"
    assert table.loc[table.feature == "steady", "band"].iloc[0] == "stable"
    assert set(table["kind"]) == {"numeric", "categorical"}


def test_the_bands_follow_the_conventional_reading():
    assert psi_band(0.05) == "stable"
    assert psi_band(0.15) == "shifted, watch"
    assert psi_band(0.40) == "shifted, investigate"


def test_an_empty_window_raises():
    with pytest.raises(ValueError, match="no loans in it"):
        numeric_stability(_normal(100), np.array([]))


def test_the_floor_is_reported_not_hidden():
    expected = np.linspace(0, 100, 10000)
    actual = np.linspace(0, 40, 10000)

    result = numeric_stability(expected, actual)

    assert "floored at half an observation" in result.to_markdown()
    assert "of the index" in result.summary()


def test_an_index_made_of_empty_buckets_says_so():
    # A feature that is almost entirely missing in one window and populated in
    # the other produces a huge index whose digits belong to the floor, not to
    # the data. The band is worth reading; the value is not, and the result has
    # to say which is which.
    expected = np.concatenate([np.full(9000, np.nan), np.linspace(0, 1, 1000)])
    actual = np.linspace(0, 1, 10000)

    result = numeric_stability(expected, actual)

    assert result.band == "shifted, investigate"
    assert result.floor_driven
    assert "read the band" in result.summary()


def test_an_ordinary_shift_is_not_floor_driven():
    result = numeric_stability(_normal(seed=1), _normal(loc=1.0, seed=2))

    assert result.band == "shifted, investigate"
    assert not result.floor_driven


def test_the_floor_scales_with_the_window():
    # Half an observation in a small window is a larger share than in a big
    # one, so the same qualitative gap must not produce the same number by
    # accident.
    small = numeric_stability(np.linspace(0, 100, 1000), np.linspace(0, 40, 1000))
    large = numeric_stability(
        np.linspace(0, 100, 100000), np.linspace(0, 40, 100000)
    )

    assert large.psi > small.psi
