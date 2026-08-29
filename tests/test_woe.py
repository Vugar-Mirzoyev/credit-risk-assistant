"""Tests for the WOE engine.

DESIGN.md names this one of the three modules where a silent bug corrupts every
result downstream, and the failure mode is quiet: bins that look reasonable but
were fitted on the wrong rows, or an infinite WOE quietly turned into a finite
one. The arithmetic is therefore checked against the definition by hand, not
only for plausibility.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import WoeConfig
from src.features import woe as woe_module
from src.features.woe import WOE_PREFIX, WoeBinner, iv_band

CONFIG = WoeConfig(
    pre_bins=20,
    max_bins=10,
    min_bin_fraction=0.05,
    min_category_fraction=0.01,
    enforce_monotonic=True,
    min_iv=0.02,
    suspicious_iv=0.5,
)


def _loose(**overrides) -> WoeConfig:
    fields = {
        "pre_bins": CONFIG.pre_bins,
        "max_bins": CONFIG.max_bins,
        "min_bin_fraction": CONFIG.min_bin_fraction,
        "min_category_fraction": CONFIG.min_category_fraction,
        "enforce_monotonic": CONFIG.enforce_monotonic,
        "min_iv": CONFIG.min_iv,
        "suspicious_iv": CONFIG.suspicious_iv,
    }
    fields.update(overrides)
    return WoeConfig(**fields)


def _ordered_frame(n=2000, seed=0):
    """A feature where risk falls as the value rises."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < (0.6 - 0.5 * x)).astype("int8")
    return pd.DataFrame({"x": x}), pd.Series(y)


def test_woe_matches_the_definition():
    # Two bins, counts chosen so the arithmetic can be done by hand.
    x = pd.Series([0.0] * 100 + [1.0] * 100)
    y = pd.Series([1] * 40 + [0] * 60 + [1] * 10 + [0] * 90, dtype="int8")

    binner = WoeBinner(_loose(pre_bins=2, min_iv=0.0)).fit(pd.DataFrame({"x": x}), y)
    bins = binner.binning("x").bins

    good_total, bad_total = 150, 50
    low = next(b for b in bins if b.bad == 40)
    expected = math.log((low.good / good_total) / (low.bad / bad_total))
    assert low.woe == pytest.approx(expected)
    assert low.iv == pytest.approx(
        (low.good / good_total - low.bad / bad_total) * expected
    )


def test_information_value_is_the_sum_over_bins():
    frame, target = _ordered_frame()
    binner = WoeBinner(_loose(min_iv=0.0)).fit(frame, target)
    binning = binner.binning("x")

    assert binning.iv == pytest.approx(sum(b.iv for b in binning.bins))
    assert binning.iv > 0


def test_higher_woe_means_safer():
    # Risk falls as x rises, so WOE must rise with x.
    frame, target = _ordered_frame()
    binner = WoeBinner(_loose(min_iv=0.0)).fit(frame, target)
    bins = [b for b in binner.binning("x").bins if not b.includes_missing]

    rates = [b.event_rate for b in bins]
    woes = [b.woe for b in bins]
    assert rates[0] > rates[-1]
    assert woes[0] < woes[-1]


def test_woe_is_monotone_when_enforced():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 4000)
    # A deliberately non-monotone relationship: risk is highest in the middle.
    y = (rng.uniform(0, 1, 4000) < (0.2 + 0.5 * np.exp(-((x - 0.5) ** 2) / 0.02))).astype("int8")
    frame, target = pd.DataFrame({"x": x}), pd.Series(y)

    binning = WoeBinner(_loose(min_iv=0.0)).fit(frame, target).binning("x")
    woes = [b.woe for b in binning.bins if not b.includes_missing]

    assert woe_module._is_monotone(woes), woes


def test_monotonicity_can_be_switched_off():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < (0.2 + 0.5 * np.exp(-((x - 0.5) ** 2) / 0.02))).astype("int8")
    frame, target = pd.DataFrame({"x": x}), pd.Series(y)

    free = WoeBinner(_loose(min_iv=0.0, enforce_monotonic=False)).fit(frame, target)
    forced = WoeBinner(_loose(min_iv=0.0, enforce_monotonic=True)).fit(frame, target)

    # Letting the shape stand keeps more of the signal; forcing it costs IV.
    assert free.binning("x").iv > forced.binning("x").iv


def test_missing_values_get_their_own_bin():
    # Missingness here is much safer than either observed level, exactly the
    # pattern that imputing to the median would destroy.
    x = pd.Series([0.1] * 400 + [0.9] * 400 + [None] * 200)
    y = pd.Series(
        [1] * 200 + [0] * 200  # x = 0.1, half bad
        + [1] * 100 + [0] * 300  # x = 0.9, a quarter bad
        + [1] * 10 + [0] * 190,  # missing, 5% bad
        dtype="int8",
    )

    binning = WoeBinner(_loose(min_iv=0.0)).fit(pd.DataFrame({"x": x}), y).binning("x")
    missing = [b for b in binning.bins if b.includes_missing]

    assert len(missing) == 1
    assert missing[0].count == 200
    assert missing[0].label == "Missing"  # standing alone, not folded in
    assert missing[0].woe > 0  # safer than average


def test_a_missing_group_with_no_bads_is_folded_rather_than_given_infinity():
    # A bin with no bads has an undefined WOE. The rule is the same as for
    # value bins: merge it into the bin it most resembles, do not add half an
    # observation to every cell to make the arithmetic work.
    x = pd.Series([0.1] * 400 + [0.9] * 400 + [None] * 200)
    y = pd.Series(
        [1] * 200 + [0] * 200 + [1] * 100 + [0] * 300 + [0] * 200, dtype="int8"
    )

    binning = WoeBinner(_loose(min_iv=0.0)).fit(pd.DataFrame({"x": x}), y).binning("x")
    missing = [b for b in binning.bins if b.includes_missing]

    assert len(missing) == 1
    assert "Missing" in missing[0].label and missing[0].label != "Missing"
    assert math.isfinite(missing[0].woe)


def test_a_tiny_missing_group_is_folded_in_rather_than_given_its_own_woe():
    x = pd.Series([0.1] * 500 + [0.9] * 499 + [None])
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 1000).astype("int8"))

    binning = WoeBinner(_loose(min_iv=0.0)).fit(pd.DataFrame({"x": x}), y).binning("x")

    assert binning.missing_index is not None
    folded = binning.bins[binning.missing_index]
    assert folded.count > 1  # absorbed, not standing alone on one observation


def test_no_bin_carries_an_infinite_woe():
    # A slice with no bads would give +inf; it must be merged away instead.
    x = pd.Series(list(range(1000)))
    y = pd.Series([0] * 900 + [1] * 100, dtype="int8")

    binning = WoeBinner(_loose(min_iv=0.0)).fit(pd.DataFrame({"x": x}), y).binning("x")

    assert all(math.isfinite(b.woe) for b in binning.bins)
    assert all(b.good > 0 and b.bad > 0 for b in binning.bins if not b.includes_missing)


def test_bins_respect_the_minimum_size_and_the_cap():
    frame, target = _ordered_frame(n=5000)
    config = _loose(min_iv=0.0, min_bin_fraction=0.1, max_bins=4)

    binning = WoeBinner(config).fit(frame, target).binning("x")
    value_bins = [b for b in binning.bins if not b.includes_missing]

    assert len(value_bins) <= 4
    assert all(b.count >= 0.1 * len(frame) for b in value_bins)


def test_max_bins_counts_the_missing_bin_too():
    # The cap is a contract stated in configuration. A missing bin appended
    # after the cap was applied would put the feature one bin over it, which is
    # exactly the kind of quiet violation nobody notices in a report.
    rng = np.random.default_rng(11)
    x = pd.Series(rng.uniform(0, 1, 5000))
    x[rng.uniform(0, 1, 5000) < 0.25] = None
    y = pd.Series((rng.uniform(0, 1, 5000) < 0.3).astype("int8"))

    for cap in (3, 5, 10):
        binning = (
            WoeBinner(_loose(min_iv=0.0, max_bins=cap))
            .fit(pd.DataFrame({"x": x}), y)
            .binning("x")
        )
        assert len(binning.bins) <= cap, (cap, len(binning.bins))


def test_categories_are_grouped_by_similar_risk():
    # Two safe rare categories and two risky rare ones. Merging by adjacency in
    # event-rate order must not put them in the same bucket.
    keys = ["safe_big"] * 400 + ["risky_big"] * 400
    keys += ["safe_rare"] * 10 + ["risky_rare"] * 10
    outcome = [0] * 380 + [1] * 20 + [1] * 300 + [0] * 100 + [0] * 10 + [1] * 10
    frame = pd.DataFrame({"c": pd.Series(keys, dtype="category")})
    target = pd.Series(outcome, dtype="int8")

    binning = WoeBinner(_loose(min_iv=0.0)).fit(frame, target).binning("c")

    assert binning.category_map["safe_rare"] != binning.category_map["risky_rare"]
    assert binning.category_map["safe_rare"] == binning.category_map["safe_big"]


def test_an_unseen_category_gets_neutral_woe_and_is_counted():
    train_keys = ["a"] * 300 + ["b"] * 300
    frame = pd.DataFrame({"c": pd.Series(train_keys, dtype="category")})
    target = pd.Series([0] * 250 + [1] * 50 + [0] * 150 + [1] * 150, dtype="int8")
    binner = WoeBinner(_loose(min_iv=0.0)).fit(frame, target)

    unseen = pd.DataFrame({"c": pd.Series(["a", "zzz", "b"], dtype="object")})
    out, report = binner.transform(unseen)

    assert out[WOE_PREFIX + "c"].iloc[1] == 0.0
    assert report.unseen_by_column["c"] == 1
    assert report.unseen_total == 1


def test_transform_applies_the_training_bins_not_the_frames_own():
    # The point of fitting on train alone: the same value must receive the same
    # WOE no matter what else is in the frame it arrives in.
    frame, target = _ordered_frame(n=3000)
    binner = WoeBinner(_loose(min_iv=0.0)).fit(frame, target)

    everything, _ = binner.transform(frame)
    slice_only, _ = binner.transform(frame.iloc[:10])

    np.testing.assert_allclose(
        everything[WOE_PREFIX + "x"].to_numpy()[:10],
        slice_only[WOE_PREFIX + "x"].to_numpy(),
    )


def test_features_below_min_iv_are_dropped_and_reported():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {"signal": rng.uniform(0, 1, 4000), "noise": rng.uniform(0, 1, 4000)}
    )
    target = pd.Series(
        (rng.uniform(0, 1, 4000) < (0.7 - 0.6 * frame["signal"])).astype("int8")
    )

    binner = WoeBinner(CONFIG).fit(frame, target)

    assert "signal" in binner.selected_columns
    assert "noise" in binner.report.dropped_low_iv
    assert "noise" not in binner.selected_columns


def test_a_suspiciously_strong_feature_is_flagged_but_not_dropped():
    # An almost perfect predictor is what leakage looks like. It must be
    # reported, because after an availability audit it means the audit missed
    # something - and it must not be silently removed either.
    x = pd.Series([0.0] * 500 + [1.0] * 500)
    y = pd.Series([1] * 480 + [0] * 20 + [0] * 480 + [1] * 20, dtype="int8")

    binner = WoeBinner(CONFIG).fit(pd.DataFrame({"x": x}), y)

    assert binner.binning("x").iv >= CONFIG.suspicious_iv
    assert "x" in binner.report.suspicious
    assert "x" in binner.selected_columns


def test_iv_bands_name_the_top_one_as_a_warning():
    assert iv_band(0.01, 0.5) == "unpredictive"
    assert iv_band(0.05, 0.5) == "weak"
    assert iv_band(0.2, 0.5) == "medium"
    assert iv_band(0.4, 0.5) == "strong"
    assert iv_band(0.9, 0.5) == "suspicious"


def test_a_constant_feature_carries_no_information():
    frame = pd.DataFrame({"c": [1.0] * 1000})
    rng = np.random.default_rng(7)
    target = pd.Series(rng.integers(0, 2, 1000).astype("int8"))

    binner = WoeBinner(CONFIG).fit(frame, target)

    assert binner.report.features[0].iv == pytest.approx(0.0)
    assert "c" in binner.report.dropped_low_iv


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        WoeBinner(CONFIG).transform(pd.DataFrame({"x": [1.0]}))


def test_transforming_a_frame_missing_a_fitted_column_raises():
    frame, target = _ordered_frame()
    binner = WoeBinner(_loose(min_iv=0.0)).fit(frame, target)

    with pytest.raises(KeyError, match="absent from the frame"):
        binner.transform(pd.DataFrame({"other": [1.0, 2.0]}))


def test_a_non_binary_target_raises():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="as well as 0 and 1"):
        WoeBinner(CONFIG).fit(frame, pd.Series([0, 1, 2]))


def test_a_misaligned_target_raises():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="aligned row for row"):
        WoeBinner(CONFIG).fit(frame, pd.Series([0, 1]))


def test_a_single_class_cannot_define_woe():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="undefined when either is empty"):
        WoeBinner(CONFIG).fit(frame, pd.Series([0, 0, 0]))


def test_index_is_preserved_through_transform():
    frame, target = _ordered_frame(n=100)
    frame.index = range(1000, 1100)
    binner = WoeBinner(_loose(min_iv=0.0)).fit(frame, target)

    out, _ = binner.transform(frame)

    assert out.index.tolist() == list(range(1000, 1100))
