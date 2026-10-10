"""Tests for the HTTP interface.

The scoring itself is covered in test_score.py. What is protected here is the
contract: the schema the service publishes is the champion's own, a bad
request is refused with a reason rather than scored, and the logs never
carry what an applicant typed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app


@pytest.fixture(scope="module")
def client(fitted, reason_texts):
    app = create_app(fitted.champion, reason_texts, max_reasons=4)
    return TestClient(app)


def _application(fitted, row: int = 0) -> dict:
    record = fitted.raw.iloc[row].to_dict()
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}


def test_health_reports_the_model_version(client, fitted):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": fitted.champion.version}


def test_a_valid_application_is_scored(client, fitted):
    response = client.post("/score", json=_application(fitted))

    assert response.status_code == 200
    body = response.json()
    assert 0.0 < body["probability_of_default"] < 1.0
    assert len(body["attributes"]) == len(fitted.champion.features)
    assert len(body["reasons"]) <= 4
    assert response.headers["x-model-version"] == fitted.champion.version


def test_the_request_id_is_echoed(client, fitted):
    response = client.post(
        "/score", json=_application(fitted), headers={"x-request-id": "abc123"}
    )
    assert response.headers["x-request-id"] == "abc123"


def test_an_unknown_category_is_refused_before_scoring(client, fitted):
    application = _application(fitted)
    application["home_ownership"] = "rent"

    response = client.post("/score", json=application)
    assert response.status_code == 422


def test_a_missing_required_field_is_refused(client, fitted):
    application = _application(fitted)
    del application["annual_inc"]

    response = client.post("/score", json=application)
    assert response.status_code == 422


def test_an_optional_field_may_be_omitted(client, fitted):
    application = _application(fitted)
    del application["dti"]

    response = client.post("/score", json=application)
    assert response.status_code == 200


def test_an_extra_field_is_refused(client, fitted):
    application = _application(fitted)
    application["zip_code"] = "100xx"

    response = client.post("/score", json=application)
    assert response.status_code == 422


def test_an_undefined_derived_feature_names_its_sources(client, fitted):
    application = _application(fitted)
    application["annual_inc"] = 0.0

    response = client.post("/score", json=application)
    assert response.status_code == 422
    assert "loan_to_income" in response.json()["detail"]


def test_the_published_schema_is_the_champions(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Application"]

    assert set(schema["required"]) == {"annual_inc", "loan_amnt", "home_ownership"}
    assert set(schema["properties"]["home_ownership"]["enum"]) == {"RENT", "MORTGAGE", "OWN"}


def test_the_model_endpoint_states_the_reason_method(client, fitted):
    body = client.get("/model").json()

    assert body["version"] == fitted.champion.version
    assert "9(b)(2)-5" in body["reason_method"]
    assert [f["name"] for f in body["features"]] == [
        f.name for f in fitted.champion.features
    ]
    assert all(f["description"] for f in body["features"])


def test_logs_never_carry_applicant_fields(client, fitted, caplog):
    application = _application(fitted)
    application["annual_inc"] = 87_654.0

    with caplog.at_level(logging.INFO, logger="credit_risk.serving"):
        client.post("/score", json=application)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "latency_ms" in logged
    assert "87654" not in logged and "87,654" not in logged


def test_the_service_refuses_to_start_without_reason_texts(fitted):
    with pytest.raises(KeyError, match="reason"):
        create_app(fitted.champion, {"annual_inc": "Annual income"}, max_reasons=4)
