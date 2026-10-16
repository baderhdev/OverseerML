# OverseerML

**OverseerML is not a separate fraud detector.** It takes the model from
[CryptoGuard](https://github.com/baderhdev/cryptoguard), an XGBoost classifier
that flags illicit Bitcoin transactions in the Elliptic dataset, and adds
what that model needs to run as a production system: serving, experiment
tracking, drift monitoring, and controlled retraining.

All modeling choices (features, time-aware split, SMOTE, the cost-based alert
threshold) come from CryptoGuard. OverseerML installs CryptoGuard pinned to a
fixed commit and calls its code instead of copying it.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Serving: FastAPI, input validation, SHAP factors, request log, Docker | done |
| 2 | Experiment tracking with MLflow | planned |
| 3 | Drift detection on a time-ordered replay of the test period | planned |
| 4 | Champion/challenger retraining | planned |
| 5 | Streamlit dashboard | planned |

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv sync
uv run overseer-train          # downloads Elliptic (~150 MB), writes models/v1/
docker build -t overseerml:v1 .
docker run -p 8000:8000 overseerml:v1
```

Interactive API docs are at http://localhost:8000/docs.

## Phase 1: serving

**Training artifact.** CryptoGuard trains and evaluates its models but never
saves them. `overseer-train` reruns CryptoGuard's own code and saves the model
in XGBoost's native JSON format, plus a `meta.json` holding the version, feature
list, alert threshold, test metrics, and the CryptoGuard commit it came from.
The retrained model reproduces CryptoGuard's reported results exactly:

| | CryptoGuard | `models/v1` |
|---|---:|---:|
| Test PR-AUC | 0.807 | 0.807 |
| Alerts at threshold 0.144 | 1,436 | 1,436 |
| Precision / recall | 0.573 / 0.760 | 0.573 / 0.760 |

CryptoGuard's model is SMOTE followed by XGBoost. SMOTE only runs during
training, so the saved artifact is just the XGBoost model and scores are
identical.

**`POST /predict`** takes all 165 Elliptic features by name and returns:

```json
{
  "request_id": "…",
  "model_version": "v1",
  "risk_score": 0.956,
  "alert": true,
  "threshold": 0.144,
  "top_factors": [{"feature": "local_53", "value": -0.487, "shap": 1.063}, "…"]
}
```

`alert` uses CryptoGuard's cost-based threshold, which treats a missed illicit
transaction as costing 20 times as much as a false alert.
`top_factors` are the five largest exact TreeSHAP contributions, in log-odds.
Elliptic's features are anonymized, so these explain what the model relied on.
They do not give a reason an analyst could check.

Input is strict. A missing feature, an unknown field, a non-number, or NaN or
infinity is rejected with a 422. The model never saw missing values in
training, so filling them in would give scores that look valid but aren't.

**`GET /health`** returns the status and the model version being served.

**Request log.** Every prediction is written to SQLite (`logs/requests.db`)
with its request ID, timestamp, model version, score, alert flag, and latency.
Raw features are not stored. To keep the log after the container stops, mount
the folder: `docker run -v ./logs:/app/logs …`.

## Development

```bash
uv run pytest      # API tests use a tiny synthetic model, no dataset needed
uv run ruff check .
```

## Data license

Elliptic is CC BY-NC-ND 4.0, which forbids redistribution, so `data/` is never
committed. The loader downloads it and verifies it against pinned checksums.
