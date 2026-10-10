"""D15: every training run is recorded, and the champion is registered.

`scripts/train_models.py` already writes a validation report, and a report
answers "what does this model do". It does not answer "which run produced the
file the service is loading, from which code, with which configuration, and
what did the previous run say". A tracking store answers that, and a registry
gives each champion a version number a deployment can name.

MLflow, with a SQLite store under `mlruns/`. A plain file store would do for
tracking, but the model registry needs a database behind it, and SQLite is a
database with nothing to run.

What is recorded:

- every configuration value that moves a number, as parameters
- every metric of all six models, plus the champion's worst quarterly PSI and
  the serving-parity error, as metrics
- the validation report, the column audit and the configuration, as artefacts:
  the audit trail a reviewer would ask for, attached to the run it explains
- the champion, as a registered model that scores through `src/serving`

Tracking is a record, not a source of numbers. Every figure in the README
still comes from a documented command, and turning tracking off in the
configuration changes nothing else the script does.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlflow
import pandas as pd

from src.config import PROJECT_ROOT, TrackingConfig

#: Libraries the registered model needs at load time. Pinned to what is
#: installed, so the registry records the environment the run actually used.
_SERVING_LIBRARIES = ("numpy", "pandas", "pyyaml")


class ChampionModel(mlflow.pyfunc.PythonModel):
    """The champion as an MLflow model, scoring through the service's own code.

    A second scoring implementation for the registry would be the training-
    serving skew D12 exists to prevent, so this wraps `src/serving` rather than
    re-deriving anything.
    """

    def load_context(self, context: Any) -> None:
        from src.serving.artefact import load_champion

        self.champion = load_champion(Path(context.artifacts["champion"]))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict | None = None
    ) -> pd.DataFrame:
        from src.serving.score import score_frame

        scores = score_frame(self.champion, model_input)
        return pd.DataFrame(
            {
                "probability_of_default": scores.probability_of_default,
                "score": scores.score,
            },
            index=model_input.index,
        )


@dataclass(frozen=True)
class TrackedRun:
    run_id: str
    experiment_id: str
    registered_version: str | None


def tracking_uri(config: TrackingConfig) -> str:
    config.store_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{config.store_path.as_posix()}"


def code_version(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """The commit the run was made from, and whether the tree had changes.

    Recorded as `unknown` rather than raised when git is unavailable: a run
    from an exported copy is still a valid run, and saying so is the honest
    record. A dirty tree is recorded, not refused, for the same reason.
    """
    try:
        commit = _git(root, "rev-parse", "HEAD")
        dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": "unknown"}
    return {"commit": commit, "dirty": str(dirty).lower()}


def flatten(prefix: str, value: Any) -> dict[str, str]:
    """Nested configuration as `section.key` parameters."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        out: dict[str, str] = {}
        for key, item in value.items():
            out.update(flatten(f"{prefix}.{key}" if prefix else str(key), item))
        return out
    return {prefix: str(value)}


def metrics_from_table(table: pd.DataFrame) -> dict[str, float]:
    """One metric per model and column of a validation table."""
    out: dict[str, float] = {}
    for row in table.to_dict(orient="records"):
        slug = _slug(row["model"])
        for column, value in row.items():
            if column in ("model", "kind"):
                continue
            out[f"{slug}.{column}"] = float(value)
    return out


def track_training_run(
    config: TrackingConfig,
    *,
    run_name: str,
    params: Mapping[str, str],
    metrics: Mapping[str, float],
    artefacts: Sequence[Path],
    champion_path: Path,
    tags: Mapping[str, str] | None = None,
) -> TrackedRun:
    """Record one run and register the champion it produced."""
    mlflow.set_tracking_uri(tracking_uri(config))
    experiment = mlflow.get_experiment_by_name(config.experiment)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(
            config.experiment,
            artifact_location=config.artifact_path.resolve().as_uri(),
        )
    )

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(dict(tags or {}))
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        for path in artefacts:
            mlflow.log_artifact(str(path), artifact_path="audit")
        info = mlflow.pyfunc.log_model(
            name="champion",
            python_model=ChampionModel(),
            artifacts={"champion": str(champion_path)},
            code_paths=[str(PROJECT_ROOT / "src")],
            pip_requirements=[f"{lib}=={version(lib)}" for lib in _SERVING_LIBRARIES],
            registered_model_name=config.registered_model,
        )

    registered = getattr(info, "registered_model_version", None)
    return TrackedRun(
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        registered_version=None if registered is None else str(registered),
    )


def _slug(name: str) -> str:
    return "_".join(name.replace(",", " ").split())


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
