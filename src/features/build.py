"""Derived features, and the rule that keeps derivation honest.

Some quantities a lender actually underwrites on are not columns in the
extract. Length of credit history is a date subtraction. Payment-to-income is a
ratio. Employment length arrives as an ordered phrase typed as an unordered
category. Handing those to the binner in raw form throws away exactly the
structure that makes them useful: ``earliest_cr_line`` as 754 unordered
categories is 754 pieces of noise, and it produced 4,426 test rows carrying a
month the training split had never seen.

**A derived feature inherits the availability of its inputs.** Every derivation
declares its sources and this module refuses to run one whose sources are not
all marked ``app`` in the D2 audit. Without that rule, arithmetic becomes the
back door: nothing stops someone writing ``ratio = total_pymnt / loan_amnt``
and calling it a feature, and the whitelist would never see it. Role is not
checked, only availability - ``issue_d`` carries the role ``split_key`` and is
still a legitimate input to a duration, because the date of the decision is
known at the moment the decision is made.

Unlike WOE, nothing here is fitted. Every derivation is row-wise arithmetic
with no parameters learned from any population, so these can be applied to the
whole frame before the split without leaking anything. That is a property of
this module, not a general licence: the moment a transform needs a mean, a
quantile or a category list, it belongs after the split and inside something
that is fitted on train alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.data.leakage import ColumnAudit

#: `Mon-YYYY`, the format every date string in this extract uses.
MONTH_FORMAT = "%b-%Y"


@dataclass(frozen=True)
class Derivation:
    """One derived column, with the sources it may be held to."""

    name: str
    sources: tuple[str, ...]
    replaces: tuple[str, ...]
    reason: str
    compute: Callable[[pd.DataFrame], pd.Series]


def _months_between(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Whole months from `start` to `end`, negative spans treated as unknown."""
    opened = pd.to_datetime(frame[start].astype("str"), format=MONTH_FORMAT, errors="coerce")
    decided = pd.to_datetime(frame[end])
    months = (decided.dt.year - opened.dt.year) * 12 + (
        decided.dt.month - opened.dt.month
    )
    # A credit line opened after the loan was issued is not a short history, it
    # is a broken record. Blank it rather than feeding a negative duration into
    # a monotone binning.
    return months.where(months >= 0).astype("float64")


def _credit_history_months(frame: pd.DataFrame) -> pd.Series:
    return _months_between(frame, "earliest_cr_line", "issue_d")


def _sec_app_credit_history_months(frame: pd.DataFrame) -> pd.Series:
    return _months_between(frame, "sec_app_earliest_cr_line", "issue_d")


def _emp_length_years(frame: pd.DataFrame) -> pd.Series:
    """`10+ years` to 10, `< 1 year` to 0. An ordinal restored to its order."""
    text = frame["emp_length"].astype("str")
    years = text.str.extract(r"(\d+)", expand=False).astype("float64")
    return years.where(frame["emp_length"].notna())


def _positive_income(frame: pd.DataFrame) -> pd.Series:
    """Annual income, with non-positive values blanked.

    The extract contains incomes of zero. A ratio against zero is not a very
    large number, it is an undefined one, and the missing bin is where
    undefined belongs.
    """
    income = frame["annual_inc"].astype("float64")
    return income.where(income > 0)


def _loan_to_income(frame: pd.DataFrame) -> pd.Series:
    return frame["loan_amnt"].astype("float64") / _positive_income(frame)


def _payment_to_income(frame: pd.DataFrame) -> pd.Series:
    annual_payment = frame["installment"].astype("float64") * 12.0
    return annual_payment / _positive_income(frame)


#: The derivations, in the order they are applied.
DERIVATIONS: tuple[Derivation, ...] = (
    Derivation(
        name="credit_history_months",
        sources=("earliest_cr_line", "issue_d"),
        replaces=("earliest_cr_line",),
        reason=(
            "Length of credit history at application. The raw column is a "
            "Mon-YYYY string typed as a category, so a binner reads 754 "
            "unordered labels instead of one ordered quantity, and test months "
            "absent from training have no value at all."
        ),
        compute=_credit_history_months,
    ),
    Derivation(
        name="sec_app_credit_history_months",
        sources=("sec_app_earliest_cr_line", "issue_d"),
        replaces=("sec_app_earliest_cr_line",),
        reason="As credit_history_months, for the co-applicant.",
        compute=_sec_app_credit_history_months,
    ),
    Derivation(
        name="emp_length_years",
        sources=("emp_length",),
        replaces=("emp_length",),
        reason=(
            "Employment length is ordinal but arrives as a phrase. As a "
            "category its order is lost and monotonicity cannot be enforced."
        ),
        compute=_emp_length_years,
    ),
    Derivation(
        name="loan_to_income",
        sources=("loan_amnt", "annual_inc"),
        replaces=(),
        reason=(
            "How large the loan is against what the borrower earns. A lender "
            "underwrites on the ratio, not on either figure alone."
        ),
        compute=_loan_to_income,
    ),
    Derivation(
        name="payment_to_income",
        sources=("installment", "annual_inc"),
        replaces=(),
        reason=(
            "Annualised instalment against income: what share of earnings the "
            "loan consumes. Closer to affordability than the loan size is."
        ),
        compute=_payment_to_income,
    ),
)


@dataclass
class BuildReport:
    """What was derived, what it replaced, and how much of it is unknown."""

    rows: int
    derived: tuple[str, ...] = field(default_factory=tuple)
    replaced: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)
    null_share: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"rows                    {self.rows:,}",
            f"derived                 {len(self.derived)}",
            f"raw columns replaced    {len(self.replaced)}",
        ]
        if self.skipped:
            lines.append(f"skipped, sources absent {len(self.skipped)}")
            lines.append("  " + ", ".join(self.skipped))
        for name in self.derived:
            lines.append(f"  {name:<30}{self.null_share.get(name, 0.0):6.2%} null")
        return "\n".join(lines)


def build_features(
    frame: pd.DataFrame,
    audit: Sequence[ColumnAudit],
    *,
    derivations: Sequence[Derivation] = DERIVATIONS,
) -> tuple[pd.DataFrame, BuildReport]:
    """Add derived columns and drop the raw ones they supersede.

    A derivation whose sources are not all present in the frame is skipped and
    reported, so a caller working with a subset of columns is not forced to
    carry every input. A derivation whose sources are not all available at
    decision time is an error, not a skip.
    """
    _assert_sources_are_available(derivations, audit)

    out = frame.copy()
    report = BuildReport(rows=len(frame))
    derived: list[str] = []
    replaced: list[str] = []
    skipped: list[str] = []

    for derivation in derivations:
        if any(source not in out.columns for source in derivation.sources):
            skipped.append(derivation.name)
            continue
        values = derivation.compute(out)
        out[derivation.name] = np.asarray(values, dtype="float64")
        derived.append(derivation.name)
        report.null_share[derivation.name] = float(out[derivation.name].isna().mean())

    for derivation in derivations:
        if derivation.name not in derived:
            continue
        for column in derivation.replaces:
            if column in out.columns:
                out = out.drop(columns=[column])
                replaced.append(column)

    report.derived = tuple(derived)
    report.replaced = tuple(replaced)
    report.skipped = tuple(skipped)
    return out, report


def _assert_sources_are_available(
    derivations: Sequence[Derivation], audit: Sequence[ColumnAudit]
) -> None:
    """Arithmetic must not launder a post-origination column into a feature."""
    availability = {entry.column: entry.availability for entry in audit}
    for derivation in derivations:
        unknown = [s for s in derivation.sources if s not in availability]
        if unknown:
            raise KeyError(
                f"{derivation.name!r} derives from {unknown}, which the column "
                "audit does not mention. Every input has to be reviewed before "
                "anything built from it can be used."
            )
        late = [s for s in derivation.sources if availability[s] != "app"]
        if late:
            raise ValueError(
                f"{derivation.name!r} derives from {late}, which the audit "
                "marks as post-origination. A derived feature inherits the "
                "availability of its inputs, so this would smuggle information "
                "that did not exist when the application was decided into the "
                "model under a new name."
            )


def derived_column_names(
    derivations: Sequence[Derivation] = DERIVATIONS,
) -> tuple[str, ...]:
    return tuple(d.name for d in derivations)


def replaced_column_names(
    derivations: Sequence[Derivation] = DERIVATIONS,
) -> tuple[str, ...]:
    return tuple(c for d in derivations for c in d.replaces)


def model_columns(
    audit: Sequence[ColumnAudit],
    derivations: Sequence[Derivation] = DERIVATIONS,
) -> tuple[str, ...]:
    """The whitelist after derivation: features kept, plus what was derived."""
    from src.data.leakage import feature_columns

    replaced = set(replaced_column_names(derivations))
    kept = [c for c in feature_columns(audit) if c not in replaced]
    return tuple(kept) + derived_column_names(derivations)
