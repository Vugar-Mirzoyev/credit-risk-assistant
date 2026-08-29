"""Tests for D3.

DESIGN.md names split as one of the three modules where a silent bug corrupts
every result downstream. A split that quietly is not out of time produces a
Gini that looks fine and means nothing, so these tests assert the boundary
itself and not only the row counts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import split_out_of_time

CUTOFF = "2016-10-01"


def _frame(dates, targets=None):
    data = {"issue_d": pd.to_datetime(pd.Series(dates))}
    data["loan_amnt"] = range(len(data["issue_d"]))
    if targets is not None:
        data["target"] = targets
    return pd.DataFrame(data)


def test_rows_before_the_cutoff_train_and_the_rest_test():
    frame = _frame(["2015-01-01", "2016-09-01", "2016-11-01", "2017-05-01"])

    train, test, report = split_out_of_time(
        frame, cutoff=CUTOFF, date_column="issue_d"
    )

    assert len(train) == 2
    assert len(test) == 2
    assert report.rows_out == 4


def test_the_boundary_belongs_to_the_test_side():
    # A loan originated exactly on the cutoff was not available to learn from
    # at that moment, so it must not be in train.
    frame = _frame(["2016-09-30", "2016-10-01"])

    train, test, _ = split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")

    assert train["issue_d"].tolist() == [pd.Timestamp("2016-09-30")]
    assert test["issue_d"].tolist() == [pd.Timestamp("2016-10-01")]


def test_train_never_reaches_the_cutoff():
    frame = _frame(pd.date_range("2014-01-01", "2018-01-01", freq="ME"))

    _, _, report = split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")

    assert report.train_last < report.cutoff
    assert report.test_first >= report.cutoff


def test_a_cutoff_past_the_data_leaves_the_test_side_empty_and_raises():
    frame = _frame(["2012-01-01", "2013-01-01"])

    with pytest.raises(ValueError, match="test side empty"):
        split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")


def test_a_cutoff_before_the_data_leaves_the_train_side_empty_and_raises():
    frame = _frame(["2017-01-01", "2018-01-01"])

    with pytest.raises(ValueError, match="train side empty"):
        split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")


def test_too_little_history_raises():
    # Twelve months of training data against a 24-month requirement.
    frame = _frame(list(pd.date_range("2015-10-01", "2016-09-01", freq="MS")) + ["2017-01-01"])

    with pytest.raises(ValueError, match="short of"):
        split_out_of_time(
            frame, cutoff=CUTOFF, date_column="issue_d", min_train_months=24
        )


def test_enough_history_passes():
    frame = _frame(list(pd.date_range("2014-01-01", "2016-09-01", freq="MS")) + ["2017-01-01"])

    _, _, report = split_out_of_time(
        frame, cutoff=CUTOFF, date_column="issue_d", min_train_months=24
    )

    assert report.train_months == 33


def test_undated_rows_are_counted_not_silently_dropped():
    frame = _frame(["2015-01-01", None, "2017-01-01"])

    train, test, report = split_out_of_time(
        frame, cutoff=CUTOFF, date_column="issue_d"
    )

    assert len(train) == 1
    assert len(test) == 1
    assert report.missing_date_excluded == 1
    assert report.rows_in == 3
    assert report.rows_out == 2


def test_bad_rates_are_reported_on_both_sides():
    frame = _frame(
        ["2015-01-01", "2015-02-01", "2017-01-01", "2017-02-01"],
        targets=[0, 0, 1, 0],
    )

    _, _, report = split_out_of_time(
        frame, cutoff=CUTOFF, date_column="issue_d", target_column="target"
    )

    assert report.train_bad_rate == pytest.approx(0.0)
    assert report.test_bad_rate == pytest.approx(0.5)


def test_bad_rate_is_absent_rather_than_invented_without_a_target():
    frame = _frame(["2015-01-01", "2017-01-01"])

    _, _, report = split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")

    assert report.train_bad_rate is None
    assert report.test_bad_rate is None
    assert "not measured" in report.summary()


def test_holdout_share_is_measured_against_dated_rows():
    frame = _frame(["2015-01-01"] * 8 + ["2017-01-01"] * 2)

    _, _, report = split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")

    assert report.holdout_share == pytest.approx(0.2)


def test_a_missing_date_column_names_the_problem():
    frame = pd.DataFrame({"loan_amnt": [1, 2]})

    with pytest.raises(KeyError, match="issue_d"):
        split_out_of_time(frame, cutoff=CUTOFF, date_column="issue_d")


def test_the_split_preserves_every_column_and_the_index():
    frame = _frame(["2015-01-01", "2017-01-01"], targets=[0, 1])
    frame.index = [41, 77]

    train, test, _ = split_out_of_time(
        frame, cutoff=CUTOFF, date_column="issue_d", target_column="target"
    )

    assert list(train.columns) == list(frame.columns)
    assert train.index.tolist() == [41]
    assert test.index.tolist() == [77]
