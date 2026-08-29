"""D2: the whitelist, and the machinery that makes it fail loudly.

Every candidate feature has to answer one question::

    Was this information available at the moment the application was decided?

Fields such as ``total_pymnt``, ``recoveries``, ``last_fico_range_*``,
``hardship_*`` and ``settlement_*`` are generated after origination. They encode
the outcome. A model trained on them reaches an AUC near 0.98 and is worthless,
because none of that information exists when a real decision has to be made.

A blacklist fails silently on the columns nobody thought of. A whitelist fails
loudly. So the audit lives in ``config/column_availability.csv``, one row per
observed column, and this module refuses to hand out a feature set unless that
file is complete and internally consistent.

**Two dimensions, deliberately separate.**

``availability`` is a fact about the lending process: ``app`` if the value
existed when the application was decided, ``post`` if it came later. Nothing
else belongs in it.

``role`` is what the column is permitted to be used for. It exists because
availability alone does not settle the question. ``zip_code`` is available and
still excluded, on redlining grounds rather than temporal ones. ``id`` is
available and is an identifier. ``issue_d`` is available and is the split key -
as a feature it would let the model learn the calendar rather than the
borrower. Collapsing the two dimensions would force a lie into one of them.

The invariant this module enforces is the one that matters: **a column marked
``post`` can never carry the role ``feature``.** Everything else is bookkeeping.

``leaky_columns`` exists because D2 makes the leaky-versus-clean comparison a
deliverable rather than a debugging step. It returns the clean feature set plus
exactly the post-origination columns, so the gap between the two AUCs measures
leakage and nothing else.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

#: The value existed when the application was decided, or it did not.
AVAILABILITY_VALUES = frozenset({"app", "post"})

#: What the column may be used for.
ROLE_VALUES = frozenset(
    {
        "feature",  # offered to the model
        "text",     # available, but free text needing extraction first
        "identifier",  # row identity, never a feature
        "split_key",  # the out-of-time axis, D3
        "target",   # the outcome, D1
        "excluded",  # available or not, but not to be used - reason says why
    }
)

REQUIRED_FIELDS = ("column", "availability", "role", "reason")


@dataclass(frozen=True)
class ColumnAudit:
    """One reviewed column."""

    column: str
    availability: str
    role: str
    reason: str

    @property
    def is_feature(self) -> bool:
        return self.role == "feature"

    @property
    def is_post_origination(self) -> bool:
        return self.availability == "post"


@dataclass
class LeakageReport:
    """What the audit permits, and what it holds back."""

    by_availability: dict[str, int]
    by_role: dict[str, int]
    features: tuple[str, ...]
    text: tuple[str, ...]
    identifiers: tuple[str, ...]
    excluded_post: tuple[str, ...]
    excluded_available: tuple[str, ...]
    target: str
    split_key: str

    @property
    def total(self) -> int:
        return sum(self.by_availability.values())

    def summary(self) -> str:
        return (
            f"columns audited         {self.total}\n"
            f"  available at decision {self.by_availability.get('app', 0)}\n"
            f"  post-origination      {self.by_availability.get('post', 0)}\n"
            f"features offered        {len(self.features)}\n"
            f"free text, held back    {len(self.text)}\n"
            f"identifiers             {len(self.identifiers)}\n"
            f"excluded as leakage     {len(self.excluded_post)}\n"
            f"excluded on other       {len(self.excluded_available)} "
            f"grounds\n"
            f"target                  {self.target}\n"
            f"split key               {self.split_key}"
        )


def load_audit(path: Path | str) -> tuple[ColumnAudit, ...]:
    """Read the audit and refuse it if it is not usable as a whitelist.

    Every check here exists because the failure it catches would otherwise be
    invisible: a typo in a label silently drops a column, a missing reason
    turns the audit into an unexplained list, and a ``post`` column marked
    ``feature`` reintroduces exactly the leakage the file exists to prevent.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No column audit at {path}. D2 requires every column to be "
            "reviewed before any of them can be modelled on. Run "
            "scripts/profile_data.py to produce the column list, then label it."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path} has no rows. An empty whitelist is not a whitelist.")

    missing_fields = [f for f in REQUIRED_FIELDS if f not in (rows[0] or {})]
    if missing_fields:
        raise ValueError(
            f"{path} is missing required fields {missing_fields}. Expected "
            f"{list(REQUIRED_FIELDS)}: a label nobody can justify is not an audit."
        )

    audit = tuple(
        ColumnAudit(
            column=(row["column"] or "").strip(),
            availability=(row["availability"] or "").strip(),
            role=(row["role"] or "").strip(),
            reason=(row["reason"] or "").strip(),
        )
        for row in rows
    )
    _validate(audit, path)
    return audit


def _validate(audit: Sequence[ColumnAudit], path: Path) -> None:
    duplicates = [c for c, n in Counter(a.column for a in audit).items() if n > 1]
    if duplicates:
        raise ValueError(
            f"{path} lists these columns more than once: {sorted(duplicates)}. "
            "Two rows for one column means two answers to a question that has "
            "one."
        )

    for entry in audit:
        if entry.availability not in AVAILABILITY_VALUES:
            raise ValueError(
                f"{entry.column!r} in {path} has availability "
                f"{entry.availability!r}; expected one of "
                f"{sorted(AVAILABILITY_VALUES)}. A '?' left in the file means "
                "the audit is unfinished, and an unfinished audit cannot gate "
                "anything."
            )
        if entry.role not in ROLE_VALUES:
            raise ValueError(
                f"{entry.column!r} in {path} has role {entry.role!r}; expected "
                f"one of {sorted(ROLE_VALUES)}."
            )
        if not entry.reason:
            raise ValueError(
                f"{entry.column!r} in {path} carries no reason. Every label has "
                "to be defensible out loud, so the justification is required, "
                "not optional."
            )
        if entry.is_post_origination and entry.is_feature:
            raise ValueError(
                f"{entry.column!r} is marked post-origination and also a "
                "feature. That combination is the exact failure D2 exists to "
                "prevent: it was not knowable when the decision was made, so "
                "it cannot be an input to the decision."
            )

    _exactly_one(audit, "target", path)
    _exactly_one(audit, "split_key", path)


def _exactly_one(audit: Sequence[ColumnAudit], role: str, path: Path) -> None:
    named = [a.column for a in audit if a.role == role]
    if len(named) != 1:
        raise ValueError(
            f"{path} names {len(named)} columns with role {role!r} ({named}); "
            "exactly one is required."
        )


def assert_covers(audit: Sequence[ColumnAudit], observed: Iterable[str]) -> None:
    """The audit and the data must describe the same set of columns.

    This is the loud failure. If the extract gains a column, it arrives
    unaudited and must not silently become a feature; if it loses one, the
    audit is describing something that is no longer there.
    """
    observed = list(observed)
    audited = {a.column for a in audit}
    unaudited = sorted(set(observed) - audited)
    absent = sorted(audited - set(observed))
    if unaudited or absent:
        parts = []
        if unaudited:
            parts.append(
                f"present in the data but not audited: {unaudited}. Label them "
                "before using anything from this extract"
            )
        if absent:
            parts.append(f"audited but absent from the data: {absent}")
        raise ValueError(
            "The column audit and the extract have drifted apart - " + "; ".join(parts)
        )


def feature_columns(audit: Sequence[ColumnAudit]) -> tuple[str, ...]:
    """The whitelist: everything the model is allowed to see."""
    return tuple(a.column for a in audit if a.is_feature)


def text_columns(audit: Sequence[ColumnAudit]) -> tuple[str, ...]:
    """Available free text, held back until something can extract from it."""
    return tuple(a.column for a in audit if a.role == "text")


def identifier_columns(audit: Sequence[ColumnAudit]) -> tuple[str, ...]:
    return tuple(a.column for a in audit if a.role == "identifier")


def target_column(audit: Sequence[ColumnAudit]) -> str:
    return next(a.column for a in audit if a.role == "target")


def split_key(audit: Sequence[ColumnAudit]) -> str:
    return next(a.column for a in audit if a.role == "split_key")


def leaky_columns(audit: Sequence[ColumnAudit]) -> tuple[str, ...]:
    """The clean feature set plus exactly the post-origination columns.

    Not for production. This is the deliberately contaminated feature set used
    to produce the leaky AUC that D2 requires alongside the clean one. It adds
    only post-origination columns, so the gap between the two models measures
    leakage rather than a mixture of leakage and the other exclusions.

    The target is never included; predicting the outcome from the outcome
    measures nothing.
    """
    clean = feature_columns(audit)
    post = tuple(
        a.column for a in audit if a.is_post_origination and a.role != "target"
    )
    return clean + post


def build_report(audit: Sequence[ColumnAudit]) -> LeakageReport:
    return LeakageReport(
        by_availability=dict(Counter(a.availability for a in audit)),
        by_role=dict(Counter(a.role for a in audit)),
        features=feature_columns(audit),
        text=text_columns(audit),
        identifiers=identifier_columns(audit),
        excluded_post=tuple(
            a.column
            for a in audit
            if a.role == "excluded" and a.is_post_origination
        ),
        excluded_available=tuple(
            a.column
            for a in audit
            if a.role == "excluded" and not a.is_post_origination
        ),
        target=target_column(audit),
        split_key=split_key(audit),
    )
