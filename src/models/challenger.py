"""D4 challenger: gradient boosting, to price what interpretability costs.

The challenger is not here to win. It is here to give the champion's Gini
something to be measured against, so the sentence "the scorecard is weaker but
it can be deployed" carries a number instead of a shrug.

Its job is therefore to be as good as it honestly can be. It sees the raw
whitelisted columns rather than the WOE ones, because binning throws away the
within-bin variation a tree could use, and it sees every whitelisted feature
rather than the ones that passed the scorecard's IV filter, because IV
selection is a scorecard convention and not something a boosted model needs.
Both of those widen the comparison beyond model class alone, which is why a
second variant fitted on the champion's exact WOE columns is worth reporting
next to it: that one isolates the model class and nothing else.

**Early stopping never touches the holdout.** Choosing the number of trees
against the test window would let it influence the fit, and the reported Gini
would no longer be out of time in any meaningful sense - the rows were held
out, but the stopping point was not. The last months of the *training* window
are carved off as an early-stopping set instead, which repeats D3's reasoning
one level in and mirrors how the model would really be tuned.

**No monotonic constraints.** LightGBM supports them, and a risk team actually
deploying a boosted model would very likely use them: they are the gradient
boosting analogue of the scorecard's sign discipline. They are left off here on
purpose. D4 describes the challenger as opaque and needing SHAP to be explained
at all, and constraining it would make it partly explainable, blurring the
trade-off the whole comparison exists to measure. That is a deliberate scope
decision, not an oversight.

**No hyperparameter search.** The parameters are fixed, conservative and in
configuration, with early stopping choosing the number of trees. The
methodology is the deliverable, not the leaderboard number, and a search would
buy a decimal place at the cost of a great deal of compute and code. It is the
obvious extension, and it is stated rather than quietly omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ChallengerConfig, ModelConfig


@dataclass(frozen=True)
class Challenger:
    """A fitted boosted model, with the columns it expects."""

    name: str
    features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    model: lgb.LGBMClassifier
    best_iteration: int

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of bad, using the tree count early stopping settled on."""
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise KeyError(
                f"Columns absent from the frame being scored: {sorted(missing)}. "
                "The challenger expects the same raw columns it was fitted on, "
                "with the same categorical dtypes."
            )
        matrix = frame[list(self.features)]
        return self.model.predict_proba(
            matrix, num_iteration=self.best_iteration
        )[:, 1]


@dataclass
class ChallengerReport:
    """How the model was fitted, and where it stopped."""

    name: str
    fit_rows: int
    validation_rows: int
    fit_first: pd.Timestamp
    fit_last: pd.Timestamp
    validation_first: pd.Timestamp
    validation_last: pd.Timestamp
    trees_offered: int
    best_iteration: int
    features: tuple[str, ...] = field(default_factory=tuple)
    categorical_features: tuple[str, ...] = field(default_factory=tuple)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def stopped_early(self) -> bool:
        return self.best_iteration < self.trees_offered

    def summary(self) -> str:
        return (
            f"variant                 {self.name}\n"
            f"features                {len(self.features)} "
            f"({len(self.categorical_features)} categorical)\n"
            f"fit rows                {self.fit_rows:,} "
            f"({self.fit_first.date()} to {self.fit_last.date()})\n"
            f"early-stopping rows     {self.validation_rows:,} "
            f"({self.validation_first.date()} to {self.validation_last.date()})\n"
            f"trees offered           {self.trees_offered:,}\n"
            f"trees used              {self.best_iteration:,}"
            + ("" if self.stopped_early else "  (ran to the limit)")
        )


def split_for_early_stopping(
    frame: pd.DataFrame, date_column: str, validation_months: int
) -> tuple[pd.Index, pd.Index, pd.Timestamp]:
    """Carve the last `validation_months` of the training window off the end.

    Out of time again, one level in. A random validation split would let the
    model stop at a point chosen with knowledge of months it is also fitted on,
    which is a milder version of the mistake D3 exists to prevent.
    """
    if validation_months <= 0:
        raise ValueError(
            f"validation_months is {validation_months}. Early stopping needs a "
            "window that is not the holdout; set a positive number of months "
            "in config/config.yaml."
        )
    if date_column not in frame.columns:
        raise KeyError(
            f"No {date_column!r} column. The early-stopping window is defined "
            "by date, so the fitting frame has to carry it."
        )

    dates = pd.to_datetime(frame[date_column])
    boundary = (
        dates.max().to_period("M") - (validation_months - 1)
    ).to_timestamp()

    fitting = frame.index[dates < boundary]
    validation = frame.index[dates >= boundary]
    if len(fitting) == 0 or len(validation) == 0:
        raise ValueError(
            f"A {validation_months}-month early-stopping window at {boundary.date()} "
            f"leaves {len(fitting):,} fitting rows and {len(validation):,} "
            "validation rows. The training window is too short to give up that "
            "much; shorten validation_months or move the split cutoff later."
        )
    return fitting, validation, boundary


def fit_challenger(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    features: list[str],
    date_column: str,
    config: ChallengerConfig,
    model_config: ModelConfig,
    name: str = "challenger",
) -> tuple[Challenger, ChallengerReport]:
    """Fit on the training split, stopping against its last months."""
    if not features:
        raise ValueError("No features to fit on.")
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"Columns absent from the fitting frame: {sorted(missing)}.")

    fitting, validation, _ = split_for_early_stopping(
        frame, date_column, config.validation_months
    )
    categorical = tuple(
        f for f in features if isinstance(frame[f].dtype, pd.CategoricalDtype)
    )

    x_fit = frame.loc[fitting, features]
    x_validation = frame.loc[validation, features]
    y_fit = target.loc[fitting]
    y_validation = target.loc[validation]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        random_state=model_config.random_state,
        # Without these LightGBM's result depends on the thread count, and
        # "reproducible from a documented command" would not survive being run
        # on a different machine.
        deterministic=config.deterministic,
        force_row_wise=config.deterministic,
        verbose=-1,
    )
    model.fit(
        x_fit,
        y_fit,
        eval_X=x_validation,
        eval_y=y_validation,
        eval_metric="auc",
        categorical_feature=list(categorical),
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    best = int(model.best_iteration_ or config.n_estimators)
    dates = pd.to_datetime(frame[date_column])
    report = ChallengerReport(
        name=name,
        fit_rows=len(fitting),
        validation_rows=len(validation),
        fit_first=dates.loc[fitting].min(),
        fit_last=dates.loc[fitting].max(),
        validation_first=dates.loc[validation].min(),
        validation_last=dates.loc[validation].max(),
        trees_offered=config.n_estimators,
        best_iteration=best,
        features=tuple(features),
        categorical_features=categorical,
        importance=_importance(model, features),
    )
    return (
        Challenger(
            name=name,
            features=tuple(features),
            categorical_features=categorical,
            model=model,
            best_iteration=best,
        ),
        report,
    )


def _importance(model: lgb.LGBMClassifier, features: list[str]) -> pd.DataFrame:
    """Gain-based importance: a first look, not an explanation.

    Gain says how much each feature reduced the loss when it was split on. It
    does not say which way a feature pushes any individual decision, which is
    the whole reason `explain.py` exists.
    """
    gain = model.booster_.feature_importance(importance_type="gain")
    table = pd.DataFrame({"feature": features, "gain": gain})
    total = table["gain"].sum()
    table["share"] = table["gain"] / total if total else 0.0
    return table.sort_values("gain", ascending=False).reset_index(drop=True)
