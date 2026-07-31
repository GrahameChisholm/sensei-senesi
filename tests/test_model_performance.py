"""Tests for api/model_performance.py and the /model-performance endpoint (A4)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.model_performance import (
    ModelPerformanceData,
    get_default_model_performance,
    load_stored_report,
)

_SAMPLE_HEADLINE = {
    "overall_mae": 1.5672,
    "overall_rmse": 2.4405,
    "gate": {"passed": False},
}


def test_load_stored_report_returns_none_headline_when_file_is_missing(tmp_path):
    report = load_stored_report(tmp_path / "does-not-exist.json")

    assert report.headline is None
    assert report.has_live_accuracy is False


def test_load_stored_report_reads_real_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_SAMPLE_HEADLINE))

    report = load_stored_report(path)

    assert report.headline == _SAMPLE_HEADLINE
    assert report.has_live_accuracy is False


def test_get_default_model_performance_takes_no_arguments():
    # The whole point of this wrapper: report_path must never be a caller-controllable
    # parameter (see this module's own docstring) -- calling with no arguments must work.
    result = get_default_model_performance()
    assert isinstance(result, ModelPerformanceData)


def test_model_performance_endpoint_has_no_client_controllable_path_parameter():
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
    params = openapi["paths"]["/model-performance"]["get"].get("parameters", [])
    assert params == []


@pytest.fixture
def client():
    def fake_report():
        return ModelPerformanceData(headline=_SAMPLE_HEADLINE, has_live_accuracy=False)

    app.dependency_overrides[get_default_model_performance] = fake_report
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_get_model_performance_endpoint_returns_the_stored_report(client):
    response = client.get("/model-performance")

    assert response.status_code == 200
    body = response.json()
    assert body["headline"] == _SAMPLE_HEADLINE
    assert body["has_live_accuracy"] is False


def test_get_model_performance_endpoint_returns_null_headline_when_nothing_stored():
    app.dependency_overrides[get_default_model_performance] = lambda: ModelPerformanceData(
        headline=None, has_live_accuracy=False
    )
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/model-performance")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["headline"] is None
