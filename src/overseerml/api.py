"""FastAPI service: score one transaction and explain the score.

Run with `uvicorn overseerml.api:create_app --factory`. The model folder and
request log come from OVERSEER_MODEL_DIR and OVERSEER_LOG_DB.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, create_model

# Same names as cryptoguard.data.FEATURE_COLUMNS, spelled out so serving does not
# depend on the training code; create_app checks they match the artifact.
FEATURES = [f"local_{i}" for i in range(1, 94)] + [f"agg_{i}" for i in range(1, 73)]
TOP_K = 5

# All 165 features required, no extras, no NaN/inf: the model never saw missing
# values in training, so a partial row would get a silently unreliable score.
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
    **{name: (float, ...) for name in FEATURES},
)


class Factor(BaseModel):
    feature: str
    value: float
    shap: float  # contribution in log-odds; positive pushes toward illicit


class Prediction(BaseModel):
    request_id: str
    model_version: str
    risk_score: float
    alert: bool
    threshold: float
    top_factors: list[Factor]


SCHEMA = """CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, ts REAL, model_version TEXT,
    risk_score REAL, alert INTEGER, latency_ms REAL)"""


def create_app(model_dir: Path | None = None, log_db: Path | None = None) -> FastAPI:
    model_dir = Path(model_dir or os.environ.get("OVERSEER_MODEL_DIR", "models/v1"))
    log_db = Path(log_db or os.environ.get("OVERSEER_LOG_DB", "logs/requests.db"))

    meta = json.loads((model_dir / "meta.json").read_text())
    if meta["features"] != FEATURES:
        raise ValueError(f"{model_dir} was trained on different features than the API accepts.")
    booster = xgb.Booster(model_file=model_dir / "model.json")
    version, threshold = meta["version"], meta["threshold"]

    log_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(log_db)) as db, db:
        db.execute(SCHEMA)

    app = FastAPI(title="OverseerML", version=version)

    @app.exception_handler(RequestValidationError)
    def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # The default 422 echoes the input back, which crashes on NaN (not valid JSON).
        errors = [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": version}

    @app.post("/predict")
    def predict(tx: Transaction) -> Prediction:
        start = time.perf_counter()
        x = np.array([[getattr(tx, f) for f in FEATURES]])
        dm = xgb.DMatrix(x, feature_names=FEATURES)
        score = float(booster.predict(dm)[0])
        # XGBoost's built-in exact TreeSHAP; the last column is the bias term.
        contribs = booster.predict(dm, pred_contribs=True)[0, :-1]
        top = np.argsort(-np.abs(contribs), kind="stable")[:TOP_K]

        result = Prediction(
            request_id=str(uuid.uuid4()),
            model_version=version,
            risk_score=score,
            alert=score >= threshold,
            threshold=threshold,
            top_factors=[Factor(feature=FEATURES[i], value=x[0, i], shap=contribs[i]) for i in top],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        with closing(sqlite3.connect(log_db)) as db, db:
            db.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)",
                (result.request_id, time.time(), version, score, result.alert, latency_ms),
            )
        return result

    return app
