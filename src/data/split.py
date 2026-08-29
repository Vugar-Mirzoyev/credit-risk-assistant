"""D3: the out-of-time split.

::

    train : issue_d <  cutoff
    test  : issue_d >= cutoff

A random split lets the model learn from loans originated after the ones it is
tested on. That is information a lender does not have. Out-of-time evaluation
mirrors how the model would actually be used: fit on history, applied to the
future.

The cutoff is `2016-10-01`, chosen from the quarterly table in
`reports/01_profile.md` rather than picked: no calendar year lands in the
20-25% holdout band D3 asks for, so it sits inside 2016.

**Everything fitted must be fitted on train alone.** That includes things that
do not feel like fitting: WOE bin boundaries, imputation values, category
groupings, scaling constants. Learning bin edges on the full frame and then
splitting leaks the distribution of the test window into training, and the
resulting Gini is optimistic in a way no later check will reveal. This module
therefore runs before feature construction, not after it, and the report it
returns is the evidence of that ordering.

The report also carries the bad rate on each side. They differ - the test
window is later and its population is not the training population - which is
the concrete reason D5 measures PSI by quarter instead of assuming stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class SplitReport:
    """What the split did, so the ordering can be shown rather than claimed."""

    cutoff: pd.Timestamp
    rows_in: int
    train_rows: int
    test_rows: int
    missing_date_excluded: int
    train_first: pd.Timestamp | None
    train_last: pd.Timestamp | None
    test_first: pd.Timestamp | None
    test_last: pd.Timestamp | None
    train_bad_rate: float | None
    test_bad_rate: float | None

    @property
    def rows_out(self) -> int:
        return self.train_rows + self.test_rows

    @property
    def holdout_share(self) -> float:
        return self.test_rows / self.rows_out if self.rows_out else float("nan")

    @property
    def train_months(self) -> int:
        """Calendar months spanned by the training window, inclusive."""
        if self.train_first is None or self.train_last is None:
            return 0
        return (
            (self.train_last.year - self.train_first.year) * 12
            + (self.train_last.month - self.train_first.month)
            + 1
        )

    def summary(self) -> str:
        return (
            f"cutoff                  {self.cutoff.date()}\n"
            f"rows in                 {self.rows_in:,}\n"
            f"  missing date          {self.missing_date_excluded:,}\n"
            f"train                   {self.train_rows:,} "
            f"({_span(self.train_first, self.train_last)}, "
            f"{self.train_months} months)\n"
            f"test                    {self.test_rows:,} "
            f"({_span(self.test_first, self.test_last)})\n"
            f"holdout share           {self.holdout_share:.2%}\n"
            f"bad rate train          {_rate(self.train_bad_rate)}\n"
            f"bad rate test           {_rate(self.test_bad_rate)}"
        )


def split_out_of_time(
    frame: pd.DataFrame,
    *,
    cutoff: date | datetime | str | pd.Timestamp,
    date_column: str,
    target_column: str | None = None,
    min_train_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    """Split on origination date. Returns ``(train, test, report)``.

    The boundary belongs to the test side: a loan originated exactly on the
    cutoff was not available to learn from at that moment.
    """
    if date_column not in frame.columns:
        raise KeyError(
            f"No {date_column!r} column in the frame. The out-of-time split "
            "cannot be made without the origination date; request it from the "
            "loader alongside the features."
        )

    boundary = pd.Timestamp(cutoff)
    dates = pd.to_datetime(frame[date_column])

    known = dates.notna()
    missing = int((~known).sum())

    is_train = known & (dates < boundary)
    is_test = known & (dates >= boundary)

    train = frame.loc[is_train]
    test = frame.loc[is_test]

    report = SplitReport(
        cutoff=boundary,
        rows_in=len(frame),
        train_rows=len(train),
        test_rows=len(test),
        missing_date_excluded=missing,
        train_first=_first(dates[is_train]),
        train_last=_last(dates[is_train]),
        test_first=_first(dates[is_test]),
        test_last=_last(dates[is_test]),
        train_bad_rate=_bad_rate(train, target_column),
        test_bad_rate=_bad_rate(test, target_column),
    )

    _assert_usable(report, min_train_months)
    return train, test, report


def _assert_usable(report: SplitReport, min_train_months: int | None) -> None:
    """Refuse a split that cannot support an honest evaluation."""
    accounted = report.train_rows + report.test_rows + report.missing_date_excluded
    if accounted != report.rows_in:
        raise ValueError(
            f"Split does not reconcile: {accounted} rows accounted for against "
            f"{report.rows_in} rows in. Every row must land on exactly one "
            "side or be counted as undated."
        )

    if report.train_rows == 0 or report.test_rows == 0:
        empty = "train" if report.train_rows == 0 else "test"
        raise ValueError(
            f"The cutoff {report.cutoff.date()} leaves the {empty} side empty. "
            "It falls outside the range of the data; read a candidate off the "
            "quarterly table in reports/01_profile.md."
        )

    # The split has to actually be out of time. If this ever fails, the
    # boundary logic is wrong and every metric downstream is measuring nothing.
    if report.train_last is not None and report.train_last >= report.cutoff:
        raise ValueError(
            f"Training data reaches {report.train_last.date()}, at or past the "
            f"cutoff {report.cutoff.date()}. The split is not out of time."
        )
    if report.test_first is not None and report.test_first < report.cutoff:
        raise ValueError(
            f"Test data starts {report.test_first.date()}, before the cutoff "
            f"{report.cutoff.date()}. The split is not out of time."
        )

    if min_train_months is not None and report.train_months < min_train_months:
        raise ValueError(
            f"The training window spans {report.train_months} months, short of "
            f"the {min_train_months} required by min_train_months. A model "
            "fitted on less history than that has not seen a full credit "
            "cycle; either move the cutoff later or lower the requirement "
            "deliberately in config/config.yaml."
        )


def _bad_rate(frame: pd.DataFrame, target_column: str | None) -> float | None:
    if target_column is None or target_column not in frame.columns or frame.empty:
        return None
    return float(frame[target_column].mean())


def _first(dates: pd.Series) -> pd.Timestamp | None:
    return None if dates.empty else dates.min()


def _last(dates: pd.Series) -> pd.Timestamp | None:
    return None if dates.empty else dates.max()


def _span(first: pd.Timestamp | None, last: pd.Timestamp | None) -> str:
    if first is None or last is None:
        return "empty"
    return f"{first.date()} to {last.date()}"


def _rate(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2%}"
