"""Memory-controlled loading of the raw LendingClub extract.

The raw file is roughly 1.6 GB across 151 columns. Read with inferred dtypes it
becomes float64 numerics and object-backed strings, which does not fit
comfortably in memory and costs an extra pass over the file to infer. Every
entry point here therefore bounds either the rows (chunked) or the columns
(``usecols``), and nothing in this module ever materialises the whole file.

Two properties of this file would otherwise corrupt results silently.

First, it is not one export. It is sixteen LendingClub exports concatenated,
and each kept its own two-line footer::

    Total amount funded in policy code 1: 1465324575,,,,,...
    Total amount funded in policy code 2: 521953170,,,,,...

with one further marker, ``Loans that do not meet the credit policy``, dividing
off the population D1 excludes. Measured over the whole file that is 33
non-loan lines, sitting *inside* the data rather than after it, each carrying
nulls in all 150 remaining fields. Read naively they become rows whose ``id``
is a sentence, which forces ``id`` to string and inflates every null rate in
the column profile. They are detected by an unparsable ``id``, dropped, and
*counted* in the ``LoadReport`` rather than absorbed. ``skipfooter`` is no use
against them: the lines are interior, not trailing, and it would force the
Python parser engine, which is unusable at this file size.

Second, numeric columns are read as float32 to halve their footprint. float32
represents integers exactly only below 2**24, which is 16,777,216. Rather than
assume the money fields stay under that ceiling, it was measured: of the 112
numeric columns exactly one, ``annual_inc``, crosses it, reaching 110,000,000.
That column is read as float64 and the rest stay float32. The check that
produced the list runs on every chunk, so the list cannot go quietly stale — an
extract whose values move raises instead of rounding.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

FLOAT_DTYPE = "float32"

#: float32 represents every integer below this exactly; above it, silently not.
FLOAT32_EXACT_INTEGER_LIMIT = 2**24

#: Row identity, and the trailer detector: a real loan has a numeric id.
ID_COLUMN = "id"

# Column classification. Anything not named below is read as float32, so a new
# string column added upstream fails loudly on the next run instead of quietly
# becoming an object column. That is the intended behaviour, and it is the same
# whitelist-not-blacklist reasoning as D2.

_ID_COLUMNS = frozenset({"id", "member_id"})

#: Free text. Profiled structurally so the leakage audit can label it, never
#: kept for modelling. ``desc`` carries embedded commas and newlines and is the
#: single largest memory cost in the file.
_TEXT_COLUMNS = frozenset({"emp_title", "title", "desc", "url"})

#: ``Mon-YYYY`` strings. Only the configured date column is parsed to datetime
#: here, because the out-of-time split and the quarterly PSI both depend on it.
#: The rest stay categorical and are parsed in feature building if needed.
_DATE_COLUMNS = frozenset(
    {
        "issue_d",
        "earliest_cr_line",
        "last_pymnt_d",
        "next_pymnt_d",
        "last_credit_pull_d",
        "sec_app_earliest_cr_line",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "debt_settlement_flag_date",
        "settlement_date",
    }
)

#: Numeric columns whose observed magnitude reaches the float32 exact-integer
#: ceiling and so must be float64. Measured over the full extract, not assumed:
#: ``annual_inc`` reaches 110,000,000 against a ceiling of 16,777,216, and it
#: is the only one of the 112 numeric columns that does.
#: ``_assert_float32_lossless`` re-derives this on every chunk, so the set
#: cannot silently fall out of date.
_FLOAT64_COLUMNS = frozenset({"annual_inc"})

#: Repeated low-cardinality strings; categorical is a large win over str.
_CATEGORICAL_COLUMNS = frozenset(
    {
        "term",
        "grade",
        "sub_grade",
        "emp_length",
        "home_ownership",
        "verification_status",
        "loan_status",
        "pymnt_plan",
        "purpose",
        "zip_code",
        "addr_state",
        "initial_list_status",
        "application_type",
        "verification_status_joint",
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "hardship_loan_status",
        "disbursement_method",
        "debt_settlement_flag",
        "settlement_status",
    }
)


@dataclass
class LoadReport:
    """What the load actually did, so the row counts are not lost.

    Same reasoning as the target audit in D1: a number that is not written down
    cannot be defended later.
    """

    path: Path
    rows_kept: int = 0
    unparsable_rows_dropped: int = 0
    chunks_read: int = 0
    peak_chunk_memory_mb: float = 0.0
    columns_read: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"source                  {self.path.name}\n"
            f"columns read            {len(self.columns_read)}\n"
            f"chunks                  {self.chunks_read}\n"
            f"rows kept               {self.rows_kept:,}\n"
            f"unparsable rows dropped {self.unparsable_rows_dropped}\n"
            f"peak chunk memory       {self.peak_chunk_memory_mb:.1f} MB"
        )


def read_header(path: Path | str) -> tuple[str, ...]:
    """Column names only. Reads no data rows."""
    return tuple(pd.read_csv(path, nrows=0).columns)


def dtype_map(columns: Iterable[str]) -> dict[str, str]:
    """The intended dtype per column. No inference anywhere in this module."""
    mapping: dict[str, str] = {}
    for column in columns:
        if column in _ID_COLUMNS or column in _TEXT_COLUMNS:
            mapping[column] = "str"
        elif column in _DATE_COLUMNS or column in _CATEGORICAL_COLUMNS:
            mapping[column] = "category"
        elif column in _FLOAT64_COLUMNS:
            mapping[column] = "float64"
        else:
            mapping[column] = FLOAT_DTYPE
    return mapping


def _read_dtypes(columns: Iterable[str]) -> dict[str, str]:
    """What ``read_csv`` is actually handed, which is not the intended map.

    Categorical columns are read as ``str`` and converted afterwards. Declaring
    ``category`` directly to the C parser fails on this file: the parser infers
    the category set per internal block, a block in which the column is
    entirely null gets float64 categories, and concatenating the blocks then
    raises ``dtype of categories must be the same``. Several columns here are
    null for years at a time (the ``hardship_*`` and ``sec_app_*`` families
    only start being populated late in the sample), so this is not a corner
    case, it is the normal path. Reading ``str`` and converting inside the
    chunk avoids the whole class of problem and costs only transient memory
    that never outlives one chunk.
    """
    return {
        column: ("str" if dtype == "category" else dtype)
        for column, dtype in dtype_map(columns).items()
    }


class ChunkStream:
    """Iterate the raw file in bounded chunks, accumulating a ``LoadReport``.

    This is the profiling path: it must see all 151 columns so the leakage
    audit can label every one of them, but it must never hold them all at
    once::

        stream = ChunkStream(path, chunksize=200_000, date_column="issue_d")
        for chunk in stream:
            accumulate(chunk)
        print(stream.report.summary())

    The report is only complete once the stream has been fully consumed.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        chunksize: int,
        date_column: str,
        columns: Sequence[str] | None = None,
        max_unparsable_rows: int = 100,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Raw extract not found at {self.path}. Place "
                "accepted_2007_to_2018Q4.csv in data/raw/ (see README)."
            )
        self.chunksize = chunksize
        self.date_column = date_column
        self.max_unparsable_rows = max_unparsable_rows

        header = read_header(self.path)
        self._requested = list(columns) if columns else None
        self._usecols = (
            _validate_columns(self._requested, header) if self._requested else None
        )
        self._drop_id_after = bool(self._usecols) and ID_COLUMN not in (
            self._requested or ()
        )
        self._columns = self._usecols or list(header)
        self.report = LoadReport(path=self.path)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        self.report = LoadReport(path=self.path)
        reader = pd.read_csv(
            self.path,
            usecols=self._usecols,
            dtype=_read_dtypes(self._columns),
            chunksize=self.chunksize,
            engine="c",
        )
        try:
            for chunk in reader:
                cleaned, dropped = _clean_chunk(chunk, self.date_column)
                self.report.chunks_read += 1
                self.report.rows_kept += len(cleaned)
                self.report.unparsable_rows_dropped += dropped
                self.report.peak_chunk_memory_mb = max(
                    self.report.peak_chunk_memory_mb, _memory_mb(cleaned)
                )
                self._assert_trailer_count_sane()
                if self._drop_id_after:
                    cleaned = cleaned.drop(columns=[ID_COLUMN])
                self.report.columns_read = tuple(cleaned.columns)
                yield cleaned
        except ValueError as error:
            raise _explain_dtype_failure(error) from error

    def _assert_trailer_count_sane(self) -> None:
        dropped = self.report.unparsable_rows_dropped
        if dropped <= self.max_unparsable_rows:
            return
        raise ValueError(
            f"{dropped} rows have a non-numeric {ID_COLUMN!r} after "
            f"{self.report.chunks_read} chunks, above the configured ceiling "
            f"of {self.max_unparsable_rows}. This file is expected to contain "
            "exactly two such trailer lines; a larger number means the parse "
            "is misaligned, most likely a quoting problem in a free-text "
            "column. Stop and inspect rather than model on a silently reduced "
            "population."
        )


def load_frame(
    path: Path | str,
    *,
    columns: Sequence[str],
    chunksize: int,
    date_column: str,
    max_unparsable_rows: int = 100,
) -> tuple[pd.DataFrame, LoadReport]:
    """Read a restricted set of columns into one frame.

    The modelling path. Memory is bounded by the caller's column list, which
    after the leakage audit is a small fraction of the 151 available.
    """
    stream = ChunkStream(
        path,
        chunksize=chunksize,
        date_column=date_column,
        columns=columns,
        max_unparsable_rows=max_unparsable_rows,
    )
    chunks = list(stream)
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return frame, stream.report


def sample_frame(path: Path | str, *, nrows: int, date_column: str) -> pd.DataFrame:
    """A bounded sample with the declared dtypes. For inspection, not for fitting."""
    header = read_header(path)
    frame = pd.read_csv(path, nrows=nrows, dtype=_read_dtypes(header), engine="c")
    cleaned, _ = _clean_chunk(frame, date_column)
    return cleaned


def _validate_columns(columns: Sequence[str], header: Sequence[str]) -> list[str]:
    """Fail on a requested column that does not exist, rather than returning less."""
    missing = [column for column in columns if column not in header]
    if missing:
        raise KeyError(
            f"Requested columns absent from the {len(header)}-column header: "
            f"{missing}. Check the name against reports/column_profile.csv."
        )
    requested = list(columns)
    if ID_COLUMN not in requested:
        # Needed to detect the trailer lines; dropped again before yielding.
        requested.insert(0, ID_COLUMN)
    return requested


def _clean_chunk(chunk: pd.DataFrame, date_column: str) -> tuple[pd.DataFrame, int]:
    """Drop trailer rows, parse the split date, apply categories, check float32.

    Order matters: the trailer rows go first so they cannot influence anything
    downstream, and the date column is parsed before the categorical cast so it
    is never briefly turned into a category it would immediately leave.
    """
    numeric_id = pd.to_numeric(chunk[ID_COLUMN], errors="coerce")
    keep = numeric_id.notna()
    dropped = int((~keep).sum())
    if dropped:
        chunk = chunk.loc[keep]
    # No defensive copy. The chunk comes straight from the parser and is owned
    # exclusively by this function, and under copy-on-write assigning a column
    # cannot reach back into anything else. A copy here would double the peak
    # footprint of a module whose entire purpose is to bound it.

    if date_column in chunk.columns:
        chunk[date_column] = pd.to_datetime(
            chunk[date_column], format="%b-%Y", errors="coerce"
        )

    intended = dtype_map(chunk.columns)
    for column, dtype in intended.items():
        if dtype == "category" and column != date_column:
            chunk[column] = chunk[column].astype("category")

    _assert_float32_lossless(chunk)
    return chunk, dropped


def _assert_float32_lossless(chunk: pd.DataFrame) -> None:
    """float32 is exact below 2**24. Verify instead of assuming."""
    numeric = chunk.select_dtypes(include=[FLOAT_DTYPE])
    if numeric.empty:
        return
    largest = numeric.abs().max()
    offenders = largest[largest >= FLOAT32_EXACT_INTEGER_LIMIT]
    if offenders.empty:
        return
    worst = str(offenders.idxmax())
    raise ValueError(
        f"Column {worst!r} reaches {offenders.max():,.0f}, at or above the "
        f"float32 exact-integer limit of {FLOAT32_EXACT_INTEGER_LIMIT:,}. "
        "Values in this column are no longer exact. Move it to float64 in "
        "load.dtype_map before using it in any reported figure."
    )


def _memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(deep=True).sum()) / 1024**2


def _explain_dtype_failure(error: ValueError) -> ValueError:
    """Turn a bare parser error into an instruction."""
    message = str(error)
    if "could not convert" not in message and "Unable to parse" not in message:
        return error
    return ValueError(
        f"{message}\n\nA column declared numeric contains text. Every column "
        "not listed in load._TEXT_COLUMNS, _DATE_COLUMNS, _CATEGORICAL_COLUMNS "
        "or _ID_COLUMNS is read as float32 by design, so an unclassified "
        "string column fails here rather than silently becoming an object "
        "column. Add the column named above to the right set."
    )
