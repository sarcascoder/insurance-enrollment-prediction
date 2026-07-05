"""Tests for the FastAPI prediction service.

We inject a lightweight model trained on the sample data rather than depending
on the full ``python -m src.train`` artifact, so the API contract is tested in
isolation and the suite stays fast.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import api, config
from src.preprocess import build_preprocessor


@pytest.fixture
def client(sample_df, monkeypatch):
    """A TestClient whose model is a tiny pipeline fit on the sample data."""
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(sample_df[config.FEATURE_COLUMNS], sample_df[config.TARGET])
    # Inject directly so the endpoint doesn't load from disk.
    monkeypatch.setattr(api, "_model", model)
    return TestClient(api.app)


VALID_PAYLOAD = {
    "age": 42,
    "gender": "Female",
    "marital_status": "Married",
    "salary": 85000.0,
    "employment_type": "Full-time",
    "region": "West",
    "has_dependents": "Yes",
    "tenure_years": 6.5,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["enrolled"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_out_of_range_age(client):
    bad = {**VALID_PAYLOAD, "age": 5}  # below the ge=16 constraint
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_unknown_category(client):
    bad = {**VALID_PAYLOAD, "region": "Atlantis"}  # not a valid Region enum
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post("/predict/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD])
    assert resp.status_code == 200
    assert len(resp.json()) == 2
