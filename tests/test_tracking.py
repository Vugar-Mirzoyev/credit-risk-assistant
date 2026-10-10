"""Tests for run tracking and the model registry (D15).

The registered model is checked by loading it back and scoring with it: a
registry entry that cannot reproduce the service's numbers is a record of the
wrong thing.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import TrackingConfig
from src.serving.artefact import write_champion
from src.serving.score import score_frame
from src.tracking import (
    code_version,
    flatten,
    metrics_from_table,
    track_training_run,
    tracking_uri,
)


@pytest.fixture()
def tracking(tmp_path) -> TrackingConfig:
    return TrackingConfig(
        enabled=True,
        store_path=tmp_path / "mlruns" / "mlflow.db",
        artifact_path=tmp_path / "mlruns" / "artifacts",
        experiment="test-experiment",
        registered_model="test-champion",
    )


def test_a_run_is_recorded_and_its_champion_registered(fitted, tracking, tmp_path):
    champion_path = tmp_path / "champion.json"
    write_champion(fitted.payload, champion_path)
    report = tmp_path / "model_validation.md"
    report.write_text("# Model validation\n", encoding="utf-8")

    run = track_training_run(
        tracking,
        run_name="test",
        params={"split.cutoff_date": "2016-10-01"},
        metrics={"champion_no_grade.test_gini": 0.3577},
        artefacts=[report],
        champion_path=champion_path,
        tags={"champion_version": fitted.champion.version},
    )

    client = MlflowClient(tracking_uri=tracking_uri(tracking))
    recorded = client.get_run(run.run_id)
    assert recorded.data.params["split.cutoff_date"] == "2016-10-01"
    assert recorded.data.metrics["champion_no_grade.test_gini"] == pytest.approx(0.3577)
    assert recorded.data.tags["champion_version"] == fitted.champion.version
    assert [a.path for a in client.list_artifacts(run.run_id, "audit")] == [
        "audit/model_validation.md"
    ]
    assert run.registered_version == "1"


def test_the_registered_model_scores_like_the_service(fitted, tracking, tmp_path):
    champion_path = tmp_path / "champion.json"
    write_champion(fitted.payload, champion_path)
    run = track_training_run(
        tracking, run_name="test", params={}, metrics={}, artefacts=[],
        champion_path=champion_path,
    )

    mlflow.set_tracking_uri(tracking_uri(tracking))
    model = mlflow.pyfunc.load_model(
        f"models:/{tracking.registered_model}/{run.registered_version}"
    )
    sample = fitted.raw.head(200)
    predicted = model.predict(sample)

    expected = score_frame(fitted.champion, sample)
    assert np.allclose(predicted["probability_of_default"], expected.probability_of_default)
    assert np.allclose(predicted["score"], expected.score)


def test_a_second_run_registers_a_second_version(fitted, tracking, tmp_path):
    champion_path = tmp_path / "champion.json"
    write_champion(fitted.payload, champion_path)
    versions = [
        track_training_run(
            tracking, run_name=f"run {i}", params={}, metrics={}, artefacts=[],
            champion_path=champion_path,
        ).registered_version
        for i in range(2)
    ]
    assert versions == ["1", "2"]


def test_metrics_are_named_by_model_and_column():
    table = pd.DataFrame(
        [{"model": "champion, no grade", "kind": "scorecard", "features": 23, "test_gini": 0.3577}]
    )
    assert metrics_from_table(table) == {
        "champion_no_grade.features": 23.0,
        "champion_no_grade.test_gini": 0.3577,
    }


def test_nested_configuration_flattens_to_dotted_parameters(tracking):
    flat = flatten("tracking", tracking)
    assert flat["tracking.experiment"] == "test-experiment"
    assert flatten("", {"a": {"b": 1}}) == {"a.b": "1"}


def test_code_version_is_recorded_even_without_git(tmp_path):
    recorded = code_version(tmp_path)
    assert set(recorded) == {"commit", "dirty"}
