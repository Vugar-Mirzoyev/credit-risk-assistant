"""D16: portfolio scenarios on the out-of-time holdout.

The validation report answers whether the champion ranks and whether its
probabilities are right in level. A risk function asks two further questions
of the same model, and this module answers them without fitting anything new:

1. **What happens to the portfolio if defaults rise?** Each scenario multiplies
   every loan's odds of default by one factor. A constant odds multiplier is a
   constant shift of the logit, the same thing as moving the scorecard's
   intercept, so the ranking the model produced is untouched: Gini, KS and the
   points table are identical under every scenario, and only the level moves.
   Scaling the probability itself would not have that property, and a large
   enough factor would push it above 1.

2. **What does a given approval rate buy?** Applicants are approved lowest PD
   first, and for each approval rate the holdout's real outcomes give the bad
   rate of the approved book and the share of all bad loans that were
   declined. The rates are simulated, not chosen: a cutoff is a risk-appetite
   decision this project has no basis to make (D13).

Exposure is the loan amount at origination, and the money figure is called
the *expected defaulted amount*, not a loss. A loan repays principal before it
defaults and something is recovered after, and both are recorded only after
origination (D2). Loss given default is out of scope, and the figure is an
upper bound on the loss rather than an estimate of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Search range for a solved odds multiplier, as powers of ten either side of 1.
#: A target that needs more than a thousandfold change in odds is not a
#: scenario anyone would defend, and is reported as an error instead.
_SEARCH_DECADES = 3
_SOLVE_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200


@dataclass(frozen=True)
class Scenario:
    """One stressed state of the portfolio."""

    name: str
    odds_multiplier: float
    #: Where the multiplier came from, written into the report next to it.
    basis: str

    def __post_init__(self) -> None:
        if not self.odds_multiplier > 0 or not math.isfinite(self.odds_multiplier):
            raise ValueError(
                f"Scenario {self.name!r} has odds multiplier "
                f"{self.odds_multiplier}; it must be a positive finite number."
            )


@dataclass(frozen=True)
class VintageAnchor:
    """The worst calendar year of the training window, by observed bad rate."""

    year: int
    loans: int
    bad_rate: float


def stress(probability: np.ndarray, odds_multiplier: float) -> np.ndarray:
    """Multiply every loan's odds of default by one factor.

    Computed as p*m / (1 - p + p*m), which is the same as adding log(m) to the
    logit but stays exact at p = 0 and p = 1 instead of passing through
    infinity.
    """
    p = np.asarray(probability, dtype="float64")
    if np.any((p < 0) | (p > 1)) or np.any(~np.isfinite(p)):
        raise ValueError("Probabilities of default must lie in [0, 1].")
    if not odds_multiplier > 0:
        raise ValueError(f"Odds multiplier must be positive, got {odds_multiplier}.")
    stressed = p * odds_multiplier
    return stressed / (1.0 - p + stressed)


def solve_multiplier(probability: np.ndarray, target_mean: float) -> float:
    """The odds multiplier that moves the portfolio's mean PD to `target_mean`.

    The mean of the stressed probabilities rises strictly with the multiplier,
    so bisection on its logarithm finds the one value that hits the target.
    """
    if not 0.0 < target_mean < 1.0:
        raise ValueError(f"Target mean PD must lie strictly between 0 and 1, got {target_mean}.")
    p = np.asarray(probability, dtype="float64")
    low, high = -_SEARCH_DECADES * math.log(10), _SEARCH_DECADES * math.log(10)
    if not stress(p, math.exp(low)).mean() <= target_mean <= stress(p, math.exp(high)).mean():
        raise ValueError(
            f"A mean PD of {target_mean:.4f} cannot be reached by changing the odds "
            f"of this portfolio by less than a factor of 10^{_SEARCH_DECADES}."
        )
    for _ in range(_MAX_ITERATIONS):
        middle = (low + high) / 2.0
        if stress(p, math.exp(middle)).mean() < target_mean:
            low = middle
        else:
            high = middle
        if high - low < _SOLVE_TOLERANCE:
            break
    return math.exp((low + high) / 2.0)


def worst_vintage(
    dates: pd.Series, outcome: pd.Series, *, min_loans: int
) -> VintageAnchor:
    """The highest yearly bad rate among years with at least `min_loans` loans.

    The volume floor matters: 2007 carries a few thousand loans and a high bad
    rate, and anchoring a portfolio stress on a year that small would let a
    sampling accident set the scenario.
    """
    if len(dates) != len(outcome):
        raise ValueError("dates and outcome must have the same length.")
    years = pd.to_datetime(dates).dt.year
    table = (
        pd.DataFrame({"year": years.to_numpy(), "bad": np.asarray(outcome)})
        .groupby("year")["bad"]
        .agg(loans="size", bad_rate="mean")
    )
    eligible = table[table["loans"] >= min_loans]
    if eligible.empty:
        raise ValueError(
            f"No year has at least {min_loans:,} loans; the largest has "
            f"{int(table['loans'].max()):,}. Lower scenarios.min_vintage_loans "
            "in config/config.yaml only if that year is still a fair anchor."
        )
    year = int(eligible["bad_rate"].idxmax())
    row = eligible.loc[year]
    return VintageAnchor(year=year, loans=int(row["loans"]), bad_rate=float(row["bad_rate"]))


def scenario_table(
    probability: np.ndarray, exposure: np.ndarray, scenarios: tuple[Scenario, ...]
) -> pd.DataFrame:
    """Mean PD, expected bad loans and expected defaulted amount per scenario.

    The first scenario is the reference the others are compared with.
    """
    p, amount = _checked(probability, exposure)
    if not scenarios:
        raise ValueError("At least one scenario is needed.")
    rows = []
    for scenario in scenarios:
        stressed = stress(p, scenario.odds_multiplier)
        rows.append(
            {
                "scenario": scenario.name,
                "basis": scenario.basis,
                "odds_multiplier": scenario.odds_multiplier,
                "mean_pd": float(stressed.mean()),
                "expected_bads": float(stressed.sum()),
                "expected_defaulted_amount": float((stressed * amount).sum()),
            }
        )
    table = pd.DataFrame(rows)
    reference = table["expected_defaulted_amount"].iloc[0]
    table["change_vs_first"] = table["expected_defaulted_amount"] / reference - 1.0
    return table


def approval_table(
    probability: np.ndarray,
    outcome: np.ndarray,
    exposure: np.ndarray,
    approval_rates: tuple[float, ...],
    scenarios: tuple[Scenario, ...],
) -> pd.DataFrame:
    """What each approval rate buys, on real outcomes and under each scenario.

    Applicants are approved lowest PD first. A stressed PD ranks the loans in
    the same order, so one ordering serves every scenario and the approved book
    at a given rate is the same loans in all of them.
    """
    p, amount = _checked(probability, exposure)
    y = np.asarray(outcome, dtype="float64")
    if len(y) != len(p):
        raise ValueError("outcome must have one value per loan.")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("outcome must be 0 (good) or 1 (bad).")
    for rate in approval_rates:
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"Approval rates must lie in (0, 1], got {rate}.")

    order = np.argsort(p, kind="mergesort")
    total_bads = y.sum()
    rows = []
    for rate in approval_rates:
        approved = order[: math.ceil(rate * len(p))]
        row = {
            "approval_rate": rate,
            "approved": len(approved),
            "highest_pd_approved": float(p[approved].max()),
            "observed_bad_rate": float(y[approved].mean()),
            "bads_declined_share": float(1.0 - y[approved].sum() / total_bads)
            if total_bads
            else 0.0,
        }
        for scenario in scenarios:
            stressed = stress(p[approved], scenario.odds_multiplier)
            row[f"expected_bad_rate_{scenario.name}"] = float(stressed.mean())
            row[f"expected_defaulted_amount_{scenario.name}"] = float(
                (stressed * amount[approved]).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _checked(probability: np.ndarray, exposure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probability, dtype="float64")
    amount = np.asarray(exposure, dtype="float64")
    if len(p) == 0:
        raise ValueError("The portfolio is empty.")
    if len(amount) != len(p):
        raise ValueError("exposure must have one value per loan.")
    if np.any(~np.isfinite(amount)) or np.any(amount < 0):
        raise ValueError("exposure must be finite and non-negative for every loan.")
    return p, amount
