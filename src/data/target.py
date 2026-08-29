"""D1: the target, and the audit that justifies it.

A loan that is still running has not yet revealed whether it is good or bad.
The common shortcut is to treat every non-default loan as good, which on this
extract moves the observed bad rate from 19.97% to 11.90% — eight points of
optimism baked in before a single feature exists. Every metric computed
afterwards inherits it, and nothing about the model looks wrong.

So unresolved loans are excluded, and the counts behind that exclusion are
returned in a ``TargetReport`` rather than discarded. The report also computes
the number the shortcut would have produced, because the argument for D1 is
only convincing with both figures side by side.

Two rules keep the exclusion honest.

A status appearing in none of the configured good, bad or unresolved lists
raises. The alternative — quietly dropping whatever is unrecognised — is the
same silent-shrinkage failure this module exists to prevent, and it would be
invisible in every downstream number. New outcomes must be a decision.

The status column is dropped from the returned frame. Keeping the raw outcome
string beside the derived target invites exactly the leakage accident D2 is
about; the report retains every count, so nothing is lost by removing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from src.config import TargetConfig

#: Loans originated under the earlier underwriting regime carry this prefix on
#: their status. They are a different population, not extra volume.
POLICY_FAILURE_PREFIX = "Does not meet the credit policy. Status:"

TARGET_COLUMN = "target"

BAD = 1
GOOD = 0


@dataclass
class TargetReport:
    """The row counts behind D1, kept so the decision can be defended."""

    rows_in: int
    status_counts: Mapping[str, int]
    good: int
    bad: int
    unresolved_excluded: int
    policy_excluded: int
    missing_status_excluded: int
    config: TargetConfig
    _reconciled: bool = field(default=False, repr=False)

    @property
    def include_policy_failures(self) -> bool:
        return self.config.include_policy_failures

    def __post_init__(self) -> None:
        accounted = (
            self.good
            + self.bad
            + self.unresolved_excluded
            + self.policy_excluded
            + self.missing_status_excluded
        )
        if accounted != self.rows_in:
            raise ValueError(
                f"Target audit does not reconcile: {accounted} rows accounted "
                f"for against {self.rows_in} rows in. Every row must land in "
                "exactly one bucket or the exclusion counts cannot be trusted."
            )
        self._reconciled = True

    @property
    def rows_out(self) -> int:
        """Resolved loans, the population the model is fitted and tested on."""
        return self.good + self.bad

    @property
    def bad_rate(self) -> float:
        """The real bad rate: bad over resolved."""
        return self.bad / self.rows_out if self.rows_out else float("nan")

    @property
    def naive_bad_rate(self) -> float:
        """What counting unresolved loans as good would have produced.

        Not used anywhere in modelling. It exists so the cost of the shortcut
        is a measured number in the validation report rather than an assertion.
        """
        denominator = self.rows_out + self.unresolved_excluded
        return self.bad / denominator if denominator else float("nan")

    @property
    def unresolved_share(self) -> float:
        return self.unresolved_excluded / self.rows_in if self.rows_in else float("nan")

    def summary(self) -> str:
        return (
            f"rows in                 {self.rows_in:,}\n"
            f"  good (0)              {self.good:,}\n"
            f"  bad (1)               {self.bad:,}\n"
            f"  unresolved, excluded  {self.unresolved_excluded:,} "
            f"({self.unresolved_share:.1%} of rows in)\n"
            f"  policy failures       {self.policy_excluded:,}"
            f"{'' if not self.include_policy_failures else ' (pooled in, not excluded)'}\n"
            f"  missing status        {self.missing_status_excluded:,}\n"
            f"rows out                {self.rows_out:,}\n"
            f"bad rate                {self.bad_rate:.2%}\n"
            f"bad rate if unresolved  {self.naive_bad_rate:.2%}\n"
            f"  were counted good     "
            f"(understates by {self.bad_rate - self.naive_bad_rate:.2%} points)"
        )

    def to_markdown(self) -> str:
        """Table for `reports/`, so the audit is reviewable without running code."""
        lines = [
            "| Status | Rows | Treatment |",
            "|---|---:|---|",
        ]
        for status, count in sorted(
            self.status_counts.items(), key=lambda item: -item[1]
        ):
            lines.append(f"| {status} | {count:,} | {_treatment_label(status, self)} |")
        lines.extend(
            [
                "",
                f"- Rows in: **{self.rows_in:,}**",
                f"- Rows out (resolved): **{self.rows_out:,}**",
                f"- Bad rate: **{self.bad_rate:.2%}**",
                f"- Bad rate if unresolved loans were counted as good: "
                f"**{self.naive_bad_rate:.2%}** "
                f"({self.bad_rate - self.naive_bad_rate:.2%} points lower)",
            ]
        )
        return "\n".join(lines)


def build_target(
    frame: pd.DataFrame,
    config: TargetConfig,
    *,
    status_column: str,
) -> tuple[pd.DataFrame, TargetReport]:
    """Apply D1: keep resolved loans, label them, and report what was removed.

    Returns the frame restricted to resolved loans with a ``target`` column
    added and the status column removed, together with the audit.
    """
    if status_column not in frame.columns:
        raise KeyError(
            f"No {status_column!r} column in the frame. The target cannot be "
            "derived without it; add it to the columns requested from the "
            "loader."
        )

    status = frame[status_column].astype("str")
    missing = frame[status_column].isna()
    status = status.mask(missing)
    status_counts = _count_statuses(status)

    is_policy_failure = status.str.startswith(POLICY_FAILURE_PREFIX, na=False)
    if config.include_policy_failures:
        # Pooled in: strip the prefix and judge them on the underlying outcome.
        effective = status.mask(
            is_policy_failure, status.str.removeprefix(POLICY_FAILURE_PREFIX)
        )
        policy_excluded = 0
    else:
        effective = status.mask(is_policy_failure)
        policy_excluded = int(is_policy_failure.sum())

    _assert_every_status_is_accounted_for(effective, config)

    is_bad = effective.isin(config.bad_statuses)
    is_good = effective.isin(config.good_statuses)
    resolved = is_bad | is_good

    report = TargetReport(
        rows_in=len(frame),
        status_counts=status_counts,
        good=int(is_good.sum()),
        bad=int(is_bad.sum()),
        unresolved_excluded=int(effective.isin(config.unresolved_statuses).sum()),
        policy_excluded=policy_excluded,
        missing_status_excluded=int(missing.sum()),
        config=config,
    )

    out = frame.loc[resolved].drop(columns=[status_column])
    out[TARGET_COLUMN] = is_bad.loc[resolved].astype("int8").to_numpy()
    return out, report


def _count_statuses(status: pd.Series) -> dict[str, int]:
    counts = status.value_counts(dropna=True)
    return {str(key): int(value) for key, value in counts.items()}


def _assert_every_status_is_accounted_for(
    effective: pd.Series, config: TargetConfig
) -> None:
    """A status in none of the three lists is a decision, not a silent drop."""
    known = (
        set(config.good_statuses)
        | set(config.bad_statuses)
        | set(config.unresolved_statuses)
    )
    seen = {str(value) for value in effective.dropna().unique()}
    unknown = sorted(seen - known)
    if unknown:
        raise ValueError(
            f"Unrecognised loan status: {unknown}. Every status must be "
            "classified as good, bad or unresolved in config/config.yaml "
            "before it can be modelled on. Excluding it silently would shrink "
            "the population without appearing in any downstream number."
        )


def _treatment_label(status: str, report: TargetReport) -> str:
    """How D1 treats one observed status, for the reviewable audit table."""
    config = report.config
    if status.startswith(POLICY_FAILURE_PREFIX):
        if not config.include_policy_failures:
            return "excluded, separate population"
        status = status.removeprefix(POLICY_FAILURE_PREFIX)
    if status in config.bad_statuses:
        return "bad (1)"
    if status in config.good_statuses:
        return "good (0)"
    if status in config.unresolved_statuses:
        return "excluded, unresolved"
    return "unclassified"
