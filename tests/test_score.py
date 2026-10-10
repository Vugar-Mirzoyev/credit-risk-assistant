"""Tests for scoring through the serving path (D12-D14).

The first test is the one the rest depend on: the service, given raw inputs,
must produce exactly what the fitted scorecard produces from its WOE frame.
Everything after it is about the cases a batch never has to face - one
applicant, with a field missing or a value training never saw.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.serving.artefact import champion_from_dict, model_version
from src.serving.score import (
    Attribute,
    UnscorableApplication,
    check_reason_texts,
    input_specs,
    score_application,
    score_frame,
    select_reasons,
)


def _application(fitted, row: int = 0) -> dict:
    record = fitted.raw.iloc[row].to_dict()
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}


def test_the_serving_path_reproduces_the_fitted_scorecard(fitted):
    # Raw inputs in, derived feature recomputed, bins looked up, points added:
    # the result must match the scorecard to floating-point precision.
    scores = score_frame(fitted.champion, fitted.raw)

    expected_pd = fitted.scorecard.predict_proba(fitted.woe_frame)
    expected_score = fitted.scorecard.score(fitted.woe_frame)
    assert np.max(np.abs(scores.probability_of_default - expected_pd)) < 1e-12
    assert np.max(np.abs(scores.score - expected_score)) < 1e-9


def test_one_application_scores_as_its_row_does(fitted, reason_texts):
    frame_scores = score_frame(fitted.champion, fitted.raw)

    for row in (0, 1, 2):
        result = score_application(
            fitted.champion, _application(fitted, row),
            reason_texts=reason_texts, max_reasons=4,
        )
        assert result.probability_of_default == pytest.approx(
            frame_scores.probability_of_default[row], abs=1e-12
        )
        assert result.score == pytest.approx(frame_scores.score[row], abs=1e-9)
        assert result.model_version == fitted.champion.version


def test_the_score_decomposes_into_its_attributes(fitted, reason_texts):
    result = score_application(
        fitted.champion, _application(fitted), reason_texts=reason_texts, max_reasons=4
    )

    total = fitted.champion.base_points + sum(a.points for a in result.attributes)
    assert result.score == pytest.approx(total, abs=1e-9)
    assert [a.feature for a in result.attributes] == [
        f.name for f in fitted.champion.features
    ]


def test_reasons_are_the_largest_shortfalls_below_average(fitted, reason_texts):
    for row in range(40):
        result = score_application(
            fitted.champion, _application(fitted, row),
            reason_texts=reason_texts, max_reasons=2,
        )
        shortfalls = [r.shortfall for r in result.reasons]
        assert len(result.reasons) <= 2
        assert all(s > 0 for s in shortfalls)
        assert shortfalls == sorted(shortfalls, reverse=True)

        # Nothing left out may fall further below average than what was kept.
        chosen = {r.feature for r in result.reasons}
        for feature, attribute in zip(fitted.champion.features, result.attributes):
            if feature.name not in chosen and result.reasons:
                gap = feature.reference_points - attribute.points
                assert gap <= shortfalls[-1] + 1e-12


def test_an_applicant_above_average_everywhere_has_no_reasons(fitted, reason_texts):
    best = [
        Attribute(feature=f.name, bin="best", points=max(f.points))
        for f in fitted.champion.features
    ]
    assert select_reasons(fitted.champion.features, best, reason_texts, 4) == ()


def test_reasons_carry_the_configured_description(fitted, reason_texts):
    application = _application(fitted)
    application["annual_inc"] = 21_000.0  # lowest income band
    result = score_application(
        fitted.champion, application, reason_texts=reason_texts, max_reasons=4
    )

    by_feature = {r.feature: r for r in result.reasons}
    assert by_feature["annual_inc"].description == "Annual income"


def test_a_missing_required_field_is_refused(fitted, reason_texts):
    application = _application(fitted)
    application["annual_inc"] = None

    with pytest.raises(UnscorableApplication) as raised:
        score_application(
            fitted.champion, application, reason_texts=reason_texts, max_reasons=4
        )
    assert "annual_inc" in raised.value.problems


def test_a_missing_optional_field_lands_in_the_missing_bin(fitted, reason_texts):
    application = _application(fitted)
    application["dti"] = None

    result = score_application(
        fitted.champion, application, reason_texts=reason_texts, max_reasons=4
    )
    dti = next(a for a in result.attributes if a.feature == "dti")
    assert "Missing" in dti.bin


def test_a_category_training_never_saw_is_refused(fitted, reason_texts):
    # In a batch this would score as neutral WOE and be counted. For one
    # applicant there is no report to count it in, so it is refused.
    application = _application(fitted)
    application["home_ownership"] = "rent"

    with pytest.raises(UnscorableApplication) as raised:
        score_application(
            fitted.champion, application, reason_texts=reason_texts, max_reasons=4
        )
    assert "never seen in training" in raised.value.problems["home_ownership"]


def test_an_undefined_derived_feature_is_refused(fitted, reason_texts):
    # An income of zero is a valid number and a valid income bin, but it makes
    # loan_to_income undefined, which only shows up after derivation.
    application = _application(fitted)
    application["annual_inc"] = 0.0

    with pytest.raises(UnscorableApplication) as raised:
        score_application(
            fitted.champion, application, reason_texts=reason_texts, max_reasons=4
        )
    assert "loan_amnt" in raised.value.problems["loan_to_income"]


def test_an_undefined_derived_feature_uses_its_missing_bin_when_training_had_one(
    fitted, reason_texts
):
    # The real champion's loan_to_income has a missing bin, because training
    # contained zero incomes. An application with a zero income then lands
    # exactly where training put the same case.
    payload = copy.deepcopy(fitted.payload)
    feature = next(f for f in payload["model"]["features"] if f["name"] == "loan_to_income")
    feature["missing_index"] = len(feature["bins"]) - 1
    feature["bins"][-1]["includes_missing"] = True
    feature["bins"][-1]["label"] += " or Missing"
    payload["version"] = model_version(payload["model"])
    champion = champion_from_dict(payload)

    application = _application(fitted)
    application["annual_inc"] = 0.0
    result = score_application(
        champion, application, reason_texts=reason_texts, max_reasons=4
    )
    lti = next(a for a in result.attributes if a.feature == "loan_to_income")
    assert lti.bin.endswith("or Missing")

    # The missing bin stands for undefined arithmetic, not an absent input, so
    # the loan amount itself is still required.
    specs = {s.name: s for s in input_specs(champion)}
    assert specs["loan_amnt"].required


def test_a_field_the_model_does_not_use_is_refused(fitted, reason_texts):
    application = _application(fitted)
    application["zip_code"] = "100xx"

    with pytest.raises(UnscorableApplication) as raised:
        score_application(
            fitted.champion, application, reason_texts=reason_texts, max_reasons=4
        )
    assert "zip_code" in raised.value.problems


def test_the_input_contract_comes_from_the_champion(fitted):
    specs = {s.name: s for s in input_specs(fitted.champion)}

    # loan_amnt is not a feature, but loan_to_income cannot be computed without it.
    assert specs["loan_amnt"].feeds == ("loan_to_income",)
    assert specs["loan_amnt"].required
    assert specs["annual_inc"].required
    # dti has a missing bin, so training says what missing means.
    assert not specs["dti"].required
    assert specs["home_ownership"].kind == "categorical"
    assert set(specs["home_ownership"].categories) == {"RENT", "MORTGAGE", "OWN"}


def test_every_scored_feature_needs_a_reason_text(fitted, reason_texts):
    incomplete = {k: v for k, v in reason_texts.items() if k != "dti"}

    with pytest.raises(KeyError, match="dti"):
        check_reason_texts(fitted.champion, incomplete)
