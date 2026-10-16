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
| 2 | Experiment tracking with MLflow | done |
| 3 | Drift detection on a time-ordered replay of the test period | done |
| 4 | Champion/challenger retraining | planned |
| 5 | Streamlit dashboard | planned |

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv sync
uv run overseer-train          # downloads Elliptic (~150 MB), writes models/v1/, logs to MLflow
uv run mlflow ui               # browse runs at http://localhost:5000
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

## Phase 2: experiment tracking

Every `overseer-train` run is an MLflow run in the `overseerml` experiment,
stored locally in `mlflow.db` (run metadata) and `mlruns/` (files). No server
is needed. Each run records:

- **Parameters**: all XGBoost hyperparameters, resampling (SMOTE), the
  train and validation boundaries, the cost assumptions, and the CryptoGuard
  commit.
- **Metrics**: validation and test PR-AUC, the alert threshold, and the full
  test confusion matrix at that threshold (TP, FP, FN, TN) with precision,
  recall and F1.
- **Artifacts**: `model.json` and `meta.json`, the same files the API serves.

`meta.json` also stores the run ID, so any served model version can be traced
back to the exact run that produced it.

## Phase 3: drift detection

`uv run overseer-drift` replays time steps 30-49 as if each two-week step were
a batch of new transactions arriving, scores them with the served model, and
writes per-batch results to [`reports/drift/`](reports/drift/). Each batch gets
three signals:

- **Feature drift**: PSI of each of the 165 features against the training
  period (steps 1-29). A feature counts as drifted above PSI 0.25.
- **Score drift**: PSI of the model's risk scores against the training period.
- **Lagged performance**: PR-AUC as monitoring would actually see it. Labels
  arrive 2 steps (about 4 weeks) late, so at step *t* the metric pools the
  latest 3 steps whose labels have arrived.

The drift cutoffs come from the validation steps (30-34), where the model
still works (PR-AUC 0.95-1.00). A batch is flagged only if it drifts more than
any validation batch did. The performance alert fires when lagged PR-AUC
falls below 0.80.

### Result: input drift misses the dark-market shutdown

| Step | Illicit | Hindsight PR-AUC | Lagged PR-AUC | Features drifted | Drift flag | Performance alert |
|---|---:|---:|---:|---:|:-:|:-:|
| 37 | 40 | 0.89 | 0.99 | 59 | yes | |
| 42 | 239 | 0.90 | 0.89 | 32 | | |
| **43** (shutdown) | 24 | **0.06** | 0.90 | 32 | | |
| 44 | 24 | 0.04 | 0.89 | 33 | | |
| 45 | 5 | 0.01 | 0.87 | 36 | | |
| 46 | 2 | 0.26 | 0.75 | 58 | yes | **yes** |
| 47 | 22 | 0.07 | 0.03 | 35 | | **yes** |
| 48-49 | 36, 56 | 0.27-0.31 | 0.03-0.04 | 43 | | **yes** |

The model's hindsight PR-AUC collapses at step 43, but no input-drift signal
reacts. There are two reasons:

1. **Elliptic's features drift all the time.** About 32 of 165 features
   exceed PSI 0.25 in nearly every batch, including validation batches where
   PR-AUC is 0.99. Six neighbor-aggregate features sit near PSI 8 throughout.
   "Many features drifted" is the normal state of this data, so it can't
   signal a failure.
2. **The shutdown is concept drift, not input drift.** Illicit volume fell
   from 239 to 24 transactions, in batches of about 5,000. What changed is
   which transactions are illicit, not what transactions look like, so feature
   and score distributions barely move.

Only the label-based performance alert catches it, at step 46, three steps
after the shutdown. Labels lag by 2 steps, so step 45 is the earliest any
label-based alert could fire. The two drift flags (37 and 46) don't line up
with the failure, so they are reported for information only. The performance
alert is the retraining trigger for Phase 4.

**Caveat:** the 0.80 floor was chosen after seeing these results. The drift
cutoffs were not, because they come from the validation steps.

## Development

```bash
uv run pytest      # API tests use a tiny synthetic model, no dataset needed
uv run ruff check .
```

## Data license

Elliptic is CC BY-NC-ND 4.0, which forbids redistribution, so `data/` is never
committed. The loader downloads it and verifies it against pinned checksums.
