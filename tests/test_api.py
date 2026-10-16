"""API tests against a tiny model trained on random data, so they run without Elliptic."""

import json
import sqlite3

import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from overseerml.api import FEATURES, TOP_K, create_app


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURES)))
    y = (X[:, 0] > 0).astype(int)
    xgb.XGBClassifier(n_estimators=5).fit(X, y).save_model(tmp / "model.json")
    meta = {"version": "test-v0", "features": FEATURES, "threshold": 0.5}
    (tmp / "meta.json").write_text(json.dumps(meta))
    client = TestClient(create_app(tmp, tmp / "requests.db"))
    return client, tmp / "requests.db", X


def body(row):
    return dict(zip(FEATURES, map(float, row), strict=True))


def test_health(setup):
    client, _, _ = setup
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-v0"}


def test_predict_scores_in_range_and_logs(setup):
    client, db, X = setup
    for row in X[:20]:
        out = client.post("/predict", json=body(row)).json()
        assert 0.0 <= out["risk_score"] <= 1.0
        assert out["alert"] == (out["risk_score"] >= 0.5)
        assert len(out["top_factors"]) == TOP_K
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT model_version FROM requests").fetchall()
    assert len(rows) >= 20 and {r[0] for r in rows} == {"test-v0"}


def test_shap_explains_the_score(setup):
    # Feature 0 is the only signal, so it must be the top factor.
    client, _, X = setup
    out = client.post("/predict", json=body(X[0])).json()
    assert out["top_factors"][0]["feature"] == FEATURES[0]


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.pop("local_1"),  # missing feature
        lambda b: b.update(extra=1.0),  # unknown feature
        lambda b: b.update(local_1="abc"),  # not a number
    ],
)
def test_rejects_bad_input(setup, change):
    client, _, X = setup
    b = body(X[0])
    change(b)
    assert client.post("/predict", json=b).status_code == 422


def test_rejects_non_finite(setup):
    client, _, X = setup
    raw = json.dumps(body(X[0])).replace(str(float(X[0, 0])), "NaN", 1)
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422
