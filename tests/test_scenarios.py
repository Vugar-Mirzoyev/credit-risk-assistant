"""Tests for D16.

The stress has to move the level of PD and nothing else, the solver has to hit
the anchor it is given, and the approval table has to read the real outcomes
of the loans it approves. Each is checked against its definition.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import Config
from src.evaluation.scenarios import (
    Scenario,
    approval_table,
    scenario_table,
    solve_multiplier,
    stress,
    worst_vintage,
)


def _portfolio(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.uniform(0, 1, n) < p).astype("int64")
    amount = rng.uniform(1000, 40000, n)
    return p, y, amount


BASE = Scenario("base", 1.0, "as fitted")
DOUBLE = Scenario("double", 2.0, "odds x2")


def test_stress_multiplies_the_odds():
    p = np.array([0.1, 0.2, 0.5, 0.9])

    stressed = stress(p, 1.5)

    np.testing.assert_allclose(stressed / (1 - stressed), 1.5 * p / (1 - p))


def test_stress_by_one_changes_nothing():
    p, _, _ = _portfolio()

    np.testing.assert_allclose(stress(p, 1.0), p)


def test_stress_keeps_the_ranking():
    p, y, _ = _portfolio()

    stressed = stress(p, 3.0)

    assert (np.argsort(stressed, kind="mergesort") == np.argsort(p, kind="mergesort")).all()
    assert roc_auc_score(y, stressed) == pytest.approx(roc_auc_score(y, p))


def test_stress_stays_a_probability_at_the_edges():
    stressed = stress(np.array([0.0, 1.0, 0.999999]), 1000.0)

    assert stressed[0] == 0.0
    assert stressed[1] == 1.0
    assert stressed[2] <= 1.0


@pytest.mark.parametrize("bad", [np.array([-0.1]), np.array([1.2]), np.array([np.nan])])
def test_stress_refuses_a_value_that_is_not_a_probability(bad):
    with pytest.raises(ValueError, match="must lie in"):
        stress(bad, 1.5)


def test_solve_multiplier_hits_the_target():
    p, _, _ = _portfolio()

    multiplier = solve_multiplier(p, 0.40)

    assert stress(p, multiplier).mean() == pytest.approx(0.40, abs=1e-9)
    assert multiplier > 1


def test_solve_multiplier_returns_one_for_the_current_mean():
    p, _, _ = _portfolio()

    assert solve_multiplier(p, float(p.mean())) == pytest.approx(1.0, rel=1e-9)


def test_solve_multiplier_can_lower_the_level():
    p, _, _ = _portfolio()

    assert solve_multiplier(p, 0.10) < 1


@pytest.mark.parametrize("target", [0.0, 1.0, -0.2, 1.5])
def test_solve_multiplier_refuses_an_impossible_target(target):
    p, _, _ = _portfolio()

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        solve_multiplier(p, target)


def test_worst_vintage_ignores_a_year_below_the_volume_floor():
    dates = pd.Series(
        pd.to_datetime(["2007-03-01"] * 10 + ["2012-05-01"] * 100 + ["2015-02-01"] * 100)
    )
    outcome = pd.Series([1] * 9 + [0] + [1] * 30 + [0] * 70 + [1] * 20 + [0] * 80)

    anchor = worst_vintage(dates, outcome, min_loans=50)

    assert anchor.year == 2012
    assert anchor.loans == 100
    assert anchor.bad_rate == pytest.approx(0.30)


def test_worst_vintage_refuses_when_no_year_is_large_enough():
    dates = pd.Series(pd.to_datetime(["2015-01-01"] * 5))

    with pytest.raises(ValueError, match="No year has at least"):
        worst_vintage(dates, pd.Series([0, 1, 0, 0, 1]), min_loans=10)


def test_scenario_table_sums_what_it_reports():
    p, _, amount = _portfolio()

    table = scenario_table(p, amount, (BASE, DOUBLE)).set_index("scenario")

    assert table.loc["base", "expected_bads"] == pytest.approx(p.sum())
    assert table.loc["base", "expected_defaulted_amount"] == pytest.approx((p * amount).sum())
    assert table.loc["base", "change_vs_first"] == 0.0
    assert table.loc["double", "mean_pd"] > table.loc["base", "mean_pd"]
    assert table.loc["double", "change_vs_first"] > 0


def test_scenario_table_refuses_a_negative_exposure():
    p, _, amount = _portfolio(n=10)
    amount[3] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        scenario_table(p, amount, (BASE,))


def test_scenario_refuses_a_non_positive_multiplier():
    with pytest.raises(ValueError, match="positive finite"):
        Scenario("broken", 0.0, "none")


def test_approving_everyone_gives_the_portfolio_bad_rate():
    p, y, amount = _portfolio()

    row = approval_table(p, y, amount, (1.0,), (BASE,)).iloc[0]

    assert row["approved"] == len(p)
    assert row["observed_bad_rate"] == pytest.approx(y.mean())
    assert row["bads_declined_share"] == pytest.approx(0.0)
    assert row["expected_bad_rate_base"] == pytest.approx(p.mean())


def test_approval_takes_the_lowest_pd_first():
    p = np.array([0.5, 0.1, 0.3, 0.2])
    y = np.array([1, 0, 1, 0])
    amount = np.ones(4)

    row = approval_table(p, y, amount, (0.5,), (BASE,)).iloc[0]

    assert row["approved"] == 2
    assert row["highest_pd_approved"] == pytest.approx(0.2)
    assert row["observed_bad_rate"] == 0.0
    assert row["bads_declined_share"] == 1.0


def test_approval_rounds_the_approved_count_up():
    p, y, amount = _portfolio(n=7)

    table = approval_table(p, y, amount, (0.5,), (BASE,))

    assert table["approved"].iloc[0] == math.ceil(0.5 * 7)


def test_a_tighter_book_has_a_lower_bad_rate_when_the_model_ranks():
    p, y, amount = _portfolio()

    table = approval_table(p, y, amount, (0.5, 0.7, 0.9, 1.0), (BASE, DOUBLE))

    assert table["observed_bad_rate"].is_monotonic_increasing
    assert table["bads_declined_share"].is_monotonic_decreasing
    assert (table["expected_bad_rate_double"] > table["expected_bad_rate_base"]).all()


@pytest.mark.parametrize("rate", [0.0, 1.2, -0.5])
def test_approval_refuses_a_rate_outside_the_unit_interval(rate):
    p, y, amount = _portfolio(n=10)

    with pytest.raises(ValueError, match="Approval rates"):
        approval_table(p, y, amount, (rate,), (BASE,))


def test_approval_refuses_an_outcome_that_is_not_binary():
    p, _, amount = _portfolio(n=4)

    with pytest.raises(ValueError, match="0 \\(good\\) or 1 \\(bad\\)"):
        approval_table(p, np.array([0, 1, 2, 0]), amount, (1.0,), (BASE,))


def test_configuration_carries_the_scenarios():
    config = Config.load().scenarios

    assert config.severe_odds_multiplier > 1
    assert config.min_vintage_loans > 0
    assert all(0 < rate <= 1 for rate in config.approval_rates)
    assert config.report_path.name.endswith(".md")
