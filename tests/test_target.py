"""Tests for D1.

DESIGN.md names this one of the three modules where a silent bug corrupts every
result downstream. The failure mode is not an exception, it is a bad rate that
is quietly wrong, so these tests assert on the counts and not just on shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import TargetConfig
from src.data.target import (
    POLICY_FAILURE_PREFIX,
    TARGET_COLUMN,
    TargetReport,
    build_target,
)

CONFIG = TargetConfig(
    bad_statuses=("Charged Off", "Default"),
    good_statuses=("Fully Paid",),
    unresolved_statuses=("Current", "Issued", "In Grace Period", "Late (16-30 days)"),
    include_policy_failures=False,
)


def _frame(statuses, **extra):
    data = {"loan_status": statuses, "loan_amnt": list(range(len(statuses)))}
    data.update(extra)
    return pd.DataFrame(data)


def test_only_resolved_loans_survive():
    frame = _frame(
        [
            "Fully Paid",
            "Charged Off",
            "Current",
            "In Grace Period",
            "Default",
            "Late (16-30 days)",
        ]
    )

    out, report = build_target(frame, CONFIG, status_column="loan_status")

    assert len(out) == 3
    assert report.good == 1
    assert report.bad == 2
    assert report.unresolved_excluded == 3
    assert report.rows_out == 3


def test_labels_follow_d1():
    frame = _frame(["Fully Paid", "Charged Off", "Default"])

    out, _ = build_target(frame, CONFIG, status_column="loan_status")

    assert out[TARGET_COLUMN].tolist() == [0, 1, 1]


def test_status_column_is_dropped_so_the_outcome_cannot_travel_with_the_features():
    frame = _frame(["Fully Paid", "Charged Off"])

    out, _ = build_target(frame, CONFIG, status_column="loan_status")

    assert "loan_status" not in out.columns
    assert "loan_amnt" in out.columns


def test_policy_failures_are_excluded_by_default():
    frame = _frame(
        [
            "Fully Paid",
            f"{POLICY_FAILURE_PREFIX}Fully Paid",
            f"{POLICY_FAILURE_PREFIX}Charged Off",
        ]
    )

    out, report = build_target(frame, CONFIG, status_column="loan_status")

    assert len(out) == 1
    assert report.policy_excluded == 2
    assert report.good == 1
    assert report.bad == 0


def test_policy_failures_are_judged_on_the_underlying_outcome_when_pooled_in():
    config = TargetConfig(
        bad_statuses=CONFIG.bad_statuses,
        good_statuses=CONFIG.good_statuses,
        unresolved_statuses=CONFIG.unresolved_statuses,
        include_policy_failures=True,
    )
    frame = _frame(
        [
            "Fully Paid",
            f"{POLICY_FAILURE_PREFIX}Fully Paid",
            f"{POLICY_FAILURE_PREFIX}Charged Off",
        ]
    )

    out, report = build_target(frame, config, status_column="loan_status")

    assert len(out) == 3
    assert report.policy_excluded == 0
    assert report.good == 2
    assert report.bad == 1


def test_an_unrecognised_status_raises_instead_of_being_dropped():
    frame = _frame(["Fully Paid", "Charged Off", "Restructured Under New Programme"])

    with pytest.raises(ValueError, match="Unrecognised loan status"):
        build_target(frame, CONFIG, status_column="loan_status")


def test_a_missing_status_column_names_the_problem():
    frame = pd.DataFrame({"loan_amnt": [1, 2]})

    with pytest.raises(KeyError, match="loan_status"):
        build_target(frame, CONFIG, status_column="loan_status")


def test_null_statuses_are_counted_not_silently_dropped():
    frame = _frame(["Fully Paid", None, "Charged Off"])

    out, report = build_target(frame, CONFIG, status_column="loan_status")

    assert len(out) == 2
    assert report.missing_status_excluded == 1
    assert report.rows_in == 3


def test_categorical_status_is_handled():
    # The loader hands back loan_status as a category, not a plain string.
    frame = _frame(["Fully Paid", "Charged Off", "Current"])
    frame["loan_status"] = frame["loan_status"].astype("category")

    out, report = build_target(frame, CONFIG, status_column="loan_status")

    assert len(out) == 2
    assert report.unresolved_excluded == 1


def test_bad_rate_and_the_shortcut_it_replaces():
    # Nine resolved loans, one bad, plus ten unresolved. The real bad rate is
    # 1/9; counting the unresolved as good would report 1/19.
    frame = _frame(["Charged Off"] + ["Fully Paid"] * 8 + ["Current"] * 10)

    _, report = build_target(frame, CONFIG, status_column="loan_status")

    assert report.bad_rate == pytest.approx(1 / 9)
    assert report.naive_bad_rate == pytest.approx(1 / 19)
    assert report.unresolved_share == pytest.approx(10 / 19)


def test_the_audit_must_reconcile():
    # Every row lands in exactly one bucket. A report that does not add up is
    # refused at construction, because the exclusion counts are the evidence
    # for D1 and cannot be approximately right.
    with pytest.raises(ValueError, match="does not reconcile"):
        TargetReport(
            rows_in=100,
            status_counts={},
            good=10,
            bad=5,
            unresolved_excluded=10,
            policy_excluded=0,
            missing_status_excluded=0,
            config=CONFIG,
        )


def test_index_is_not_used_to_align_the_target():
    # A frame arriving with a non-default index must still be labelled row by
    # row. Silent index alignment here would scramble the target.
    frame = _frame(["Fully Paid", "Current", "Charged Off"])
    frame.index = [17, 4, 92]

    out, _ = build_target(frame, CONFIG, status_column="loan_status")

    assert out[TARGET_COLUMN].tolist() == [0, 1]
    assert out.index.tolist() == [17, 92]


def test_markdown_audit_labels_each_status():
    frame = _frame(
        ["Fully Paid", "Charged Off", "Current", f"{POLICY_FAILURE_PREFIX}Fully Paid"]
    )

    _, report = build_target(frame, CONFIG, status_column="loan_status")
    table = report.to_markdown()

    assert "| Fully Paid | 1 | good (0) |" in table
    assert "| Charged Off | 1 | bad (1) |" in table
    assert "| Current | 1 | excluded, unresolved |" in table
    assert "excluded, separate population" in table
