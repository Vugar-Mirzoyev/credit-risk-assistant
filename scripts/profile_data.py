"""Profile the raw extract: what is in it, and what it can support.

Produces three artefacts, all reproducible from one command:

``reports/column_profile.csv``
    One row per column, with the shape facts a reviewer needs to decide
    whether the field existed when the application was decided. The last
    column, ``availability``, is deliberately blank. It is the D2 audit
    worksheet, and it is filled in by hand, because "was this known at
    decision time?" is a question about the lending process and not one the
    data can answer about itself.

``reports/01_profile.md``
    The narrative: load audit, D1 target audit, and the yearly table D3 needs
    to place the out-of-time cutoff.

``reports/figures/01_volume_by_quarter.png``
    Resolved volume and bad rate per quarter. The cutoff has to sit somewhere
    with enough volume on both sides, and the bad rate visibly moves across
    the sample, which is the reason D5 measures stability rather than assuming
    it.

Two passes over the file rather than one. The first streams all 151 columns
through ``ChunkStream`` to accumulate per-column statistics; the second reads
only the status and date columns and hands them to the tested ``build_target``.
Collapsing them into one pass would mean rebuilding the target audit out of
accumulated counts, which is a change to a tested module to save ten seconds.

Run from anywhere::

    python scripts/profile_data.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# `python scripts/profile_data.py` puts scripts/ on the import path, not the
# repository root, so `import src` would fail. This keeps the command in the
# README working without requiring the project to be installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config
from src.data.load import ChunkStream, LoadReport, load_frame
from src.data.target import TARGET_COLUMN, TargetReport, build_target


@dataclass
class ColumnStats:
    """Per-column accumulator. Everything here is additive across chunks."""

    name: str
    dtype: str = ""
    rows: int = 0
    non_null: int = 0
    distinct: set = field(default_factory=set)
    distinct_capped: bool = False
    minimum: float | pd.Timestamp | None = None
    maximum: float | pd.Timestamp | None = None
    total: float = 0.0
    examples: list[str] = field(default_factory=list)

    @property
    def null_pct(self) -> float:
        return 100.0 * (self.rows - self.non_null) / self.rows if self.rows else 0.0

    @property
    def distinct_label(self) -> str:
        return f">={len(self.distinct):,}" if self.distinct_capped else f"{len(self.distinct):,}"

    @property
    def mean(self) -> float | None:
        if self.non_null == 0 or not _is_numeric(self.dtype):
            return None
        return self.total / self.non_null

    def update(self, series: pd.Series, max_distinct: int) -> None:
        self.dtype = str(series.dtype)
        self.rows += len(series)
        values = series.dropna()
        self.non_null += len(values)
        if values.empty:
            return

        if _is_numeric(self.dtype):
            self.minimum = _least(self.minimum, float(values.min()))
            self.maximum = _most(self.maximum, float(values.max()))
            self.total += float(values.astype("float64").sum())
        elif _is_datetime(self.dtype):
            self.minimum = _least(self.minimum, values.min())
            self.maximum = _most(self.maximum, values.max())

        if not self.distinct_capped:
            for value in values.unique():
                self.distinct.add(value)
                if len(self.distinct) >= max_distinct:
                    self.distinct_capped = True
                    break

        while len(self.examples) < 3:
            candidates = [
                _short(value)
                for value in values.unique()[:5]
                if _short(value) not in self.examples
            ]
            if not candidates:
                break
            self.examples.append(candidates[0])


def profile_columns(config: Config) -> tuple[pd.DataFrame, LoadReport]:
    """One streamed pass over every column."""
    stream = ChunkStream(
        config.data.raw_path,
        chunksize=config.data.chunksize,
        date_column=config.data.date_column,
        max_unparsable_rows=config.data.max_unparsable_rows,
    )
    stats: dict[str, ColumnStats] = {}
    for index, chunk in enumerate(stream, start=1):
        for column in chunk.columns:
            stats.setdefault(column, ColumnStats(name=column)).update(
                chunk[column], config.profile.max_distinct_tracked
            )
        print(f"  chunk {index:>2} | {stream.report.rows_kept:>10,} rows", flush=True)

    frame = pd.DataFrame(
        [
            {
                "column": item.name,
                "dtype": item.dtype,
                "non_null": item.non_null,
                "null_pct": round(item.null_pct, 2),
                "distinct": item.distinct_label,
                "min": _render(item.minimum),
                "max": _render(item.maximum),
                "mean": _render(item.mean),
                "examples": " | ".join(item.examples),
            }
            for item in stats.values()
        ]
    )
    return frame, stream.report


def load_availability(path: Path) -> pd.DataFrame:
    """Read the D2 audit. This script never writes it.

    The labelling answers a question about the lending process, not about the
    data, so it is maintained by hand. Keeping it in `config/` rather than in
    the generated `reports/` means rerunning the profile cannot wipe it, which
    it would if the labels lived in the file this script rewrites.
    """
    if not path.is_file():
        return pd.DataFrame(columns=["column", "availability", "reason"])
    audit = pd.read_csv(path)
    missing = {"column", "availability", "reason"} - set(audit.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns {sorted(missing)}. The audit "
            "must carry a label and the reasoning behind it, because a label "
            "nobody can justify is not an audit."
        )
    return audit


def profile_target(config: Config) -> tuple[pd.DataFrame, TargetReport]:
    """Second pass: the resolved population, dated, for D1 and D3."""
    frame, _ = load_frame(
        config.data.raw_path,
        columns=[config.data.status_column, config.data.date_column],
        chunksize=config.data.chunksize,
        date_column=config.data.date_column,
        max_unparsable_rows=config.data.max_unparsable_rows,
    )
    resolved, report = build_target(
        frame, config.target, status_column=config.data.status_column
    )
    return resolved, report


def yearly_table(resolved: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Resolved volume and bad rate per year, with the holdout share D3 needs.

    ``share_from_end`` is the fraction of resolved loans originated in this
    year or later. D3 wants a cutoff putting roughly 20-25% of the population
    in the holdout window, so the cutoff year is read straight off this column.
    """
    year = resolved[date_column].dt.year
    table = (
        resolved.assign(year=year)
        .groupby("year")[TARGET_COLUMN]
        .agg(loans="size", bad="sum")
        .reset_index()
    )
    table["bad_rate"] = table["bad"] / table["loans"]
    table["share"] = table["loans"] / table["loans"].sum()
    table["share_from_end"] = table["share"][::-1].cumsum()[::-1]
    return table


def quarterly_table(resolved: pd.DataFrame, date_column: str) -> pd.DataFrame:
    quarter = resolved[date_column].dt.to_period("Q")
    table = (
        resolved.assign(quarter=quarter)
        .groupby("quarter")[TARGET_COLUMN]
        .agg(loans="size", bad="sum")
        .reset_index()
    )
    table["bad_rate"] = table["bad"] / table["loans"]
    table["share"] = table["loans"] / table["loans"].sum()
    table["share_from_end"] = table["share"][::-1].cumsum()[::-1]
    # Accumulated over the whole table, before any filtering: the holdout for a
    # cutoff runs to the end of the sample, not to the end of a display window.
    table["loans_from_end"] = table["loans"][::-1].cumsum()[::-1]
    return table


def cutoff_candidates(
    quarterly: pd.DataFrame, low: float = 0.15, high: float = 0.35
) -> pd.DataFrame:
    """Quarters whose holdout share brackets the band D3 asks for.

    No calendar year lands in the 20-25% window on this extract: 2017 and later
    is 16.8% of resolved loans and 2016 and later is 38.6%. The cutoff therefore
    has to sit inside 2016 or 2017, which is only visible quarterly.
    """
    return quarterly[quarterly["share_from_end"].between(low, high)].copy()


def write_figure(quarterly: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = quarterly["quarter"].astype("str")
    figure, volume_axis = plt.subplots(figsize=(12, 5))
    volume_axis.bar(labels, quarterly["loans"], color="#b0c4d8")
    volume_axis.set_ylabel("resolved loans")
    volume_axis.set_xlabel("issue quarter")

    rate_axis = volume_axis.twinx()
    rate_axis.plot(labels, 100 * quarterly["bad_rate"], color="#a03030", linewidth=2)
    rate_axis.set_ylabel("bad rate (%)")

    step = max(1, len(labels) // 20)
    volume_axis.set_xticks(range(0, len(labels), step))
    volume_axis.set_xticklabels(labels[::step], rotation=90, fontsize=8)
    volume_axis.set_title("Resolved volume and bad rate by issue quarter")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_markdown(
    config: Config,
    load_report: LoadReport,
    target_report: TargetReport,
    yearly: pd.DataFrame,
    candidates: pd.DataFrame,
    columns: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 01 - Data profile",
        "",
        "Generated by `python scripts/profile_data.py`. Every figure below is",
        "reproduced by rerunning that command against the raw extract.",
        "",
        "## Load",
        "",
        "```",
        load_report.summary(),
        "```",
        "",
        "The dropped rows are not loans. The extract is sixteen LendingClub",
        "exports concatenated, each of which kept its own two-line",
        '"Total amount funded in policy code N" footer, plus one',
        '"Loans that do not meet the credit policy" divider. They sit inside the',
        "data rather than after it, and every one carries nulls in all 150",
        "remaining fields.",
        "",
        "## Target (D1)",
        "",
        "```",
        target_report.summary(),
        "```",
        "",
        target_report.to_markdown(),
        "",
        "## Volume by year (D3)",
        "",
        "`share_from_end` is the fraction of resolved loans issued in that year",
        "or later. D3 asks for a cutoff leaving roughly 20-25% of the resolved",
        "population in the holdout window, so the candidate cutoff is the year",
        "whose `share_from_end` lands in that band.",
        "",
        _yearly_markdown(yearly),
        "",
        "No calendar year lands in the band. Everything from 2017 onward is",
        f"{_pct(yearly, 2017)} of resolved loans and everything from 2016 onward is",
        f"{_pct(yearly, 2016)}, so the cutoff has to sit inside a year and the",
        "choice is only visible quarter by quarter:",
        "",
        _cutoff_markdown(candidates),
        "",
        "`cutoff_date` in `config/config.yaml` stays null until one of these is",
        "chosen. Reading the table: setting the cutoff to a quarter puts that",
        "quarter and everything after it in the holdout.",
        "",
        "![Resolved volume and bad rate by issue quarter](figures/01_volume_by_quarter.png)",
        "",
        "The bad rate moves across the sample. That is why D5 measures stability",
        "with PSI by quarter rather than assuming the population holds still.",
        "",
        "## Columns (D2 audit)",
        "",
        f"All {len(columns)} columns are listed in",
        "[`column_profile.csv`](column_profile.csv). The `availability` label",
        "comes from `config/column_availability.csv`, which is maintained by",
        "hand and never written by this script: `app` if the value was known",
        "when the application was decided, `post` if it was generated",
        "afterwards, `?` if it still needs a decision. A whitelist built this",
        "way fails loudly on the columns nobody thought about; a blacklist",
        "fails silently.",
        "",
        _availability_markdown(columns),
        "",
        "Nothing downstream may use a column until it carries a label.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _availability_markdown(columns: pd.DataFrame) -> str:
    """Coverage of the audit, and everything still open."""
    labels = columns["availability"].fillna("").astype("str")
    counts = labels[labels != ""].value_counts()
    lines = ["| Label | Columns |", "|---|---:|"]
    for label in ("app", "post", "?"):
        lines.append(f"| `{label}` | {int(counts.get(label, 0))} |")
    unlabelled = columns.loc[labels == "", "column"].tolist()
    lines.append(f"| *unlabelled* | {len(unlabelled)} |")

    if unlabelled:
        lines += ["", "Still unlabelled: " + ", ".join(f"`{c}`" for c in unlabelled)]

    open_rows = columns[labels == "?"]
    if not open_rows.empty:
        lines += [
            "",
            "Columns still marked `?`, each with the reason it is open:",
            "",
            "| Column | Why it is open |",
            "|---|---|",
        ]
        for row in open_rows.itertuples():
            reason = str(getattr(row, "reason", "") or "")
            lines.append(f"| `{row.column}` | {reason} |")
    return "\n".join(lines)


def _pct(yearly: pd.DataFrame, year: int) -> str:
    row = yearly.loc[yearly["year"] == year, "share_from_end"]
    return f"{row.iloc[0]:.1%}" if len(row) else "n/a"


def _cutoff_markdown(candidates: pd.DataFrame) -> str:
    lines = [
        "| Cutoff quarter | Holdout loans | Holdout share | Bad rate in quarter |",
        "|---|---:|---:|---:|",
    ]
    for row in candidates.itertuples():
        lines.append(
            f"| {row.quarter} | {row.loans_from_end:,} | "
            f"{row.share_from_end:.2%} | {row.bad_rate:.2%} |"
        )
    return "\n".join(lines)


def _yearly_markdown(yearly: pd.DataFrame) -> str:
    lines = [
        "| Year | Resolved loans | Bad | Bad rate | Share | Share from end |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in yearly.itertuples():
        lines.append(
            f"| {row.year} | {row.loans:,} | {row.bad:,} | {row.bad_rate:.2%} | "
            f"{row.share:.2%} | {row.share_from_end:.2%} |"
        )
    return "\n".join(lines)


def _is_numeric(dtype: str) -> bool:
    return dtype.startswith("float") or dtype.startswith("int")


def _is_datetime(dtype: str) -> bool:
    return dtype.startswith("datetime")


def _least(current, candidate):
    return candidate if current is None else min(current, candidate)


def _most(current, candidate):
    return candidate if current is None else max(current, candidate)


def _short(value) -> str:
    text = str(value).strip().replace("\n", " ")
    return text[:40] + "..." if len(text) > 40 else text


def _render(value) -> str:
    """Readable, not scientific. This file is read by a person, in a spreadsheet."""
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        if abs(value) >= 0.001:
            return f"{value:,.3f}"
        return f"{value:.3e}"
    return str(value)


def main() -> None:
    config = Config.load()
    print(f"Profiling {config.data.raw_path.name}")

    print("pass 1/2: streaming all columns")
    columns, load_report = profile_columns(config)

    audit = load_availability(config.profile.availability_path)
    unknown = sorted(set(audit["column"]) - set(columns["column"]))
    if unknown:
        raise ValueError(
            f"{config.profile.availability_path} labels columns that are not "
            f"in the extract: {unknown}. A label pointing at nothing means the "
            "audit and the data have drifted apart."
        )
    columns = columns.merge(audit, on="column", how="left")
    columns["availability"] = columns["availability"].fillna("")
    columns["reason"] = columns["reason"].fillna("")

    config.profile.column_csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns.to_csv(config.profile.column_csv_path, index=False, encoding="utf-8")
    print(f"  wrote {config.profile.column_csv_path}")

    print("pass 2/2: target and dates")
    resolved, target_report = profile_target(config)
    yearly = yearly_table(resolved, config.data.date_column)
    quarterly = quarterly_table(resolved, config.data.date_column)
    candidates = cutoff_candidates(quarterly)

    write_figure(quarterly, config.profile.figure_path)
    print(f"  wrote {config.profile.figure_path}")

    write_markdown(
        config,
        load_report,
        target_report,
        yearly,
        candidates,
        columns,
        config.profile.report_path,
    )
    print(f"  wrote {config.profile.report_path}")

    print()
    print(target_report.summary())


if __name__ == "__main__":
    main()
