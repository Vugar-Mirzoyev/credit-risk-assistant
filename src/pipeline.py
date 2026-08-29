"""The Layer 1 data path, assembled once.

Audit, load, target, derive, split, bin. Every module does one thing and this
is where they are wired together, in the order the design requires:

1. the D2 audit is read first, and it decides which columns are even loaded
2. the loader reads only those, chunked
3. D1 removes the unresolved loans and labels the rest
4. derivation runs before the split, because nothing in it is fitted
5. D3 splits out of time
6. WOE bins are fitted **on the training split alone**

Step 6 is the reason this file exists rather than the steps being repeated in
each script that needs them. Bin edges are learned parameters, and a script
that fits them before splitting would produce numbers that look right and are
not. Assembling the order once means it cannot be got wrong differently in two
places.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import Config
from src.data import leakage
from src.data.leakage import ColumnAudit
from src.data.load import LoadReport, load_frame, read_header
from src.data.split import SplitReport, split_out_of_time
from src.data.target import TARGET_COLUMN, TargetReport, build_target
from src.features.build import DERIVATIONS, BuildReport, build_features, model_columns
from src.features.woe import FitReport, WoeBinner


@dataclass(frozen=True)
class Dataset:
    """Everything downstream needs, prepared in the right order."""

    audit: tuple[ColumnAudit, ...]
    train: pd.DataFrame
    test: pd.DataFrame
    features: tuple[str, ...]
    binner: WoeBinner
    woe_train: pd.DataFrame
    woe_test: pd.DataFrame
    date_column: str
    reports: dict[str, object] = field(default_factory=dict)

    @property
    def y_train(self) -> pd.Series:
        return self.train[TARGET_COLUMN]

    @property
    def y_test(self) -> pd.Series:
        return self.test[TARGET_COLUMN]

    def summary(self) -> str:
        return "\n\n".join(
            f"--- {name} ---\n{report.summary()}"
            for name, report in self.reports.items()
            if hasattr(report, "summary")
        )


def prepare(config: Config, *, verbose: bool = True) -> Dataset:
    """Run the whole data path and return the pieces the models need."""
    return _prepare(config, leaky=False, verbose=verbose)


def prepare_leaky(config: Config, *, verbose: bool = True) -> Dataset:
    """The same path, deliberately given the post-origination columns.

    **Nothing fitted on this may ever be deployed.** It exists because D2 makes
    the leaky-versus-clean comparison a deliverable rather than a debugging
    step: a reviewer shown only the clean Gini has no way to judge whether it
    is good, and the gap is the most informative number in the project.

    It is a separate function rather than a flag on `prepare` on purpose. A
    boolean argument would put leakage one keyword away from the normal path,
    and the whole point of D2 is that this cannot happen by accident.
    """
    return _prepare(config, leaky=True, verbose=verbose)


def _prepare(config: Config, *, leaky: bool, verbose: bool) -> Dataset:

    def say(message: str) -> None:
        if verbose:
            print(message, flush=True)

    audit = leakage.load_audit(config.leakage.audit_path)
    # Check the audit against the extract before reading a single row: a column
    # the audit has never seen must not reach a model, and finding that out
    # after a minute of loading helps nobody.
    leakage.assert_covers(audit, read_header(config.data.raw_path))
    date_column = leakage.split_key(audit)
    status_column = leakage.target_column(audit)
    whitelisted = list(
        leakage.leaky_columns(audit) if leaky else leakage.feature_columns(audit)
    )

    label = "leaky" if leaky else "whitelisted"
    say(f"loading {len(whitelisted)} {label} columns")
    frame, load_report = load_frame(
        config.data.raw_path,
        columns=whitelisted + [status_column, date_column],
        chunksize=config.data.chunksize,
        date_column=config.data.date_column,
        max_unparsable_rows=config.data.max_unparsable_rows,
    )
    say("applying the target (D1)")
    resolved, target_report = build_target(
        frame, config.target, status_column=status_column
    )
    del frame

    say("deriving features")
    built, build_report = build_features(resolved, audit)
    del resolved

    say("splitting out of time (D3)")
    train, test, split_report = split_out_of_time(
        built,
        cutoff=config.split.cutoff_date,
        date_column=date_column,
        target_column=TARGET_COLUMN,
        min_train_months=config.split.min_train_months,
    )
    del built

    modelled = model_columns(audit, DERIVATIONS)
    if leaky:
        modelled = tuple(modelled) + tuple(
            c for c in whitelisted if c not in modelled
        )
    features = tuple(
        column for column in dict.fromkeys(modelled) if column in train.columns
    )

    say(f"fitting WOE bins on the training split, {len(features)} features")
    binner = WoeBinner(config.woe).fit(train[list(features)], train[TARGET_COLUMN])
    woe_train, transform_train = binner.transform(train[list(features)])
    woe_test, transform_test = binner.transform(test[list(features)])

    return Dataset(
        audit=audit,
        train=train,
        test=test,
        features=features,
        binner=binner,
        woe_train=woe_train,
        woe_test=woe_test,
        date_column=date_column,
        reports={
            "load": load_report,
            "target": target_report,
            "build": build_report,
            "split": split_report,
            "woe fit": binner.report,
            "woe transform, test": transform_test,
        },
    )


__all__ = [
    "Dataset",
    "prepare",
    "TARGET_COLUMN",
    "LoadReport",
    "TargetReport",
    "BuildReport",
    "SplitReport",
    "FitReport",
]
