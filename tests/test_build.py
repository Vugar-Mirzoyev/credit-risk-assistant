"""Tests for feature derivation.

The rule worth protecting here is that arithmetic cannot launder a
post-origination column into a feature. Everything else is checking that the
derived quantities are the ones a lender would recognise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.leakage import ColumnAudit
from src.features.build import (
    DERIVATIONS,
    Derivation,
    build_features,
    model_columns,
)

AUDIT = (
    ColumnAudit("earliest_cr_line", "app", "feature", "bureau at application"),
    ColumnAudit("sec_app_earliest_cr_line", "app", "feature", "co-applicant"),
    ColumnAudit("emp_length", "app", "feature", "stated at application"),
    ColumnAudit("loan_amnt", "app", "feature", "requested"),
    ColumnAudit("annual_inc", "app", "feature", "stated"),
    ColumnAudit("installment", "app", "feature", "part of the offer"),
    ColumnAudit("issue_d", "app", "split_key", "the decision moment"),
    ColumnAudit("total_pymnt", "post", "excluded", "accumulates after origination"),
    ColumnAudit("loan_status", "post", "target", "the outcome"),
)


def _frame(**overrides):
    data = {
        "earliest_cr_line": ["Jan-2000", "Jun-2010", None],
        "sec_app_earliest_cr_line": ["Jan-1995", None, None],
        "emp_length": ["10+ years", "< 1 year", "3 years"],
        "loan_amnt": [10000.0, 5000.0, 20000.0],
        "annual_inc": [50000.0, 0.0, 100000.0],
        "installment": [300.0, 150.0, 600.0],
        "issue_d": pd.to_datetime(["2015-01-01", "2015-01-01", "2015-01-01"]),
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_credit_history_is_months_between_the_two_dates():
    out, _ = build_features(_frame(), AUDIT)

    # Jan-2000 to Jan-2015 is fifteen years.
    assert out["credit_history_months"].iloc[0] == 180
    # Jun-2010 to Jan-2015 is four years and seven months.
    assert out["credit_history_months"].iloc[1] == 55
    assert np.isnan(out["credit_history_months"].iloc[2])


def test_a_credit_line_opened_after_origination_is_unknown_not_negative():
    # A negative history is a broken record, and a negative duration fed into
    # a monotone binning would anchor the bottom bin with nonsense.
    frame = _frame(earliest_cr_line=["Jan-2020", "Jun-2010", None])

    out, _ = build_features(frame, AUDIT)

    assert np.isnan(out["credit_history_months"].iloc[0])


def test_employment_length_becomes_an_ordered_number():
    out, _ = build_features(_frame(), AUDIT)

    assert out["emp_length_years"].tolist() == [10.0, 1.0, 3.0]


def test_employment_length_keeps_missing_as_missing():
    frame = _frame(emp_length=["10+ years", None, "3 years"])

    out, _ = build_features(frame, AUDIT)

    assert np.isnan(out["emp_length_years"].iloc[1])


def test_income_ratios_use_the_ratio_not_either_figure():
    out, _ = build_features(_frame(), AUDIT)

    assert out["loan_to_income"].iloc[0] == pytest.approx(10000 / 50000)
    assert out["payment_to_income"].iloc[0] == pytest.approx(300 * 12 / 50000)


def test_a_zero_income_gives_an_unknown_ratio_not_an_infinite_one():
    out, _ = build_features(_frame(), AUDIT)

    assert np.isnan(out["loan_to_income"].iloc[1])
    assert np.isnan(out["payment_to_income"].iloc[1])
    assert np.isfinite(out["loan_to_income"].iloc[[0, 2]]).all()


def test_replaced_raw_columns_are_dropped():
    out, report = build_features(_frame(), AUDIT)

    for column in ("earliest_cr_line", "sec_app_earliest_cr_line", "emp_length"):
        assert column not in out.columns
        assert column in report.replaced


def test_sources_that_remain_useful_are_kept():
    out, _ = build_features(_frame(), AUDIT)

    # loan_amnt and annual_inc are features in their own right as well as
    # inputs to a ratio, so deriving from them must not remove them.
    for column in ("loan_amnt", "annual_inc", "installment"):
        assert column in out.columns


def test_deriving_from_a_post_origination_column_is_refused():
    smuggled = Derivation(
        name="repayment_ratio",
        sources=("total_pymnt", "loan_amnt"),
        replaces=(),
        reason="looks innocent, is not",
        compute=lambda f: f["total_pymnt"] / f["loan_amnt"],
    )

    with pytest.raises(ValueError, match="post-origination"):
        build_features(_frame(), AUDIT, derivations=(smuggled,))


def test_deriving_from_an_unaudited_column_is_refused():
    unknown = Derivation(
        name="mystery",
        sources=("never_reviewed",),
        replaces=(),
        reason="",
        compute=lambda f: f["never_reviewed"],
    )

    with pytest.raises(KeyError, match="audit does not mention"):
        build_features(_frame(), AUDIT, derivations=(unknown,))


def test_a_derivation_whose_inputs_are_absent_is_skipped_and_reported():
    frame = _frame().drop(columns=["sec_app_earliest_cr_line"])

    out, report = build_features(frame, AUDIT)

    assert "sec_app_credit_history_months" not in out.columns
    assert "sec_app_credit_history_months" in report.skipped
    assert "credit_history_months" in out.columns


def test_a_skipped_derivation_does_not_drop_the_column_it_would_have_replaced():
    frame = _frame().drop(columns=["issue_d"])

    out, report = build_features(frame, AUDIT)

    # Without issue_d neither duration can be computed, so the raw date
    # columns must survive rather than being removed for nothing.
    assert "earliest_cr_line" in out.columns
    assert "credit_history_months" in report.skipped


def test_the_report_records_how_much_is_unknown():
    out, report = build_features(_frame(), AUDIT)

    assert report.rows == 3
    assert report.null_share["credit_history_months"] == pytest.approx(1 / 3)
    assert report.null_share["sec_app_credit_history_months"] == pytest.approx(2 / 3)


def test_nothing_is_fitted_so_a_slice_derives_the_same_values():
    # These are row-wise transforms with no learned parameters, which is why
    # they are safe to run before the split. If this ever fails, something in
    # here started depending on the population.
    frame = _frame()

    whole, _ = build_features(frame, AUDIT)
    part, _ = build_features(frame.iloc[:1], AUDIT)

    assert whole["loan_to_income"].iloc[0] == part["loan_to_income"].iloc[0]
    assert whole["credit_history_months"].iloc[0] == part["credit_history_months"].iloc[0]


def test_the_source_frame_is_not_modified():
    frame = _frame()
    before = list(frame.columns)

    build_features(frame, AUDIT)

    assert list(frame.columns) == before


def test_model_columns_swaps_the_raw_for_the_derived():
    columns = model_columns(AUDIT, DERIVATIONS)

    assert "earliest_cr_line" not in columns
    assert "credit_history_months" in columns
    assert "emp_length" not in columns
    assert "emp_length_years" in columns
    assert "loan_amnt" in columns  # kept: an input that is also a feature
