"""A small fitted champion, shared by the serving tests.

Synthetic, so the suite still runs from a clone with no download. It is built
to exercise every path the real champion takes through the service: a
derived feature (`loan_to_income`, from two inputs), a numeric feature with a
missing bin (`dti`), one with no missing bin (`annual_inc`), and a categorical
(`home_ownership`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig, ScorecardConfig, WoeConfig
from src.features.build import DERIVATIONS
from src.features.woe import WoeBinner
from src.models.scorecard import Scorecard, fit_scorecard
from src.serving.artefact import Champion, champion_from_dict, export_champion

MODEL = ModelConfig(
    random_state=42,
    max_correlation=0.75,
    max_iter=200,
    scorecard=ScorecardConfig(base_score=600.0, base_odds=50.0, pdo=20.0),
)

WOE = WoeConfig(
    pre_bins=20,
    max_bins=6,
    min_bin_fraction=0.05,
    min_category_fraction=0.01,
    enforce_monotonic=True,
    min_iv=0.0,
    suspicious_iv=0.5,
)

FEATURES = ["annual_inc", "loan_to_income", "dti", "home_ownership"]

REASON_TEXTS = {
    "annual_inc": "Annual income",
    "loan_to_income": "Loan amount relative to income",
    "dti": "Debt-to-income ratio",
    "home_ownership": "Home ownership status",
}


@dataclass(frozen=True)
class Fitted:
    raw: pd.DataFrame
    woe_frame: pd.DataFrame
    scorecard: Scorecard
    payload: dict
    champion: Champion


def _raw(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    income = rng.uniform(20_000, 150_000, n)
    loan = rng.uniform(1_000, 35_000, n)
    dti = rng.uniform(0, 40, n)
    dti_missing = rng.uniform(0, 1, n) < 0.15
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.45, 0.4, 0.15])

    risk = (
        0.08
        + 0.9 * (loan / income)
        + 0.004 * np.where(dti_missing, 30.0, dti)
        - 0.0000008 * income
        + np.select([home == "RENT", home == "OWN"], [0.08, 0.04], 0.0)
    )
    target = pd.Series((rng.uniform(0, 1, n) < np.clip(risk, 0.02, 0.9)).astype("int8"))
    raw = pd.DataFrame(
        {
            "annual_inc": income,
            "loan_amnt": loan,
            "dti": np.where(dti_missing, np.nan, dti),
            "home_ownership": pd.Series(home, dtype="object"),
        }
    )
    return raw, target


def fit(n: int = 8000, seed: int = 3) -> Fitted:
    raw, target = _raw(n, seed)
    built = raw.copy()
    for derivation in DERIVATIONS:
        if derivation.name == "loan_to_income":
            built["loan_to_income"] = derivation.compute(built)
    binner = WoeBinner(WOE).fit(built[FEATURES], target)
    woe_frame, _ = binner.transform(built[FEATURES])
    scorecard, _ = fit_scorecard(woe_frame, target, binner, MODEL)
    payload = export_champion(scorecard, binner, metadata={"source": "synthetic"})
    return Fitted(
        raw=raw,
        woe_frame=woe_frame,
        scorecard=scorecard,
        payload=payload,
        champion=champion_from_dict(payload),
    )


@pytest.fixture(scope="session")
def fitted() -> Fitted:
    return fit()


@pytest.fixture(scope="session")
def reason_texts() -> dict[str, str]:
    return dict(REASON_TEXTS)
