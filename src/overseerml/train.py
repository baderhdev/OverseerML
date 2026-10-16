"""Retrain CryptoGuard's headline model, save it as a servable artifact, and track the run.

CryptoGuard never saves its models, so this reruns its own code (data loader,
temporal split, model definition, threshold rule) and writes:

    <out>/model.json  XGBoost's native format: portable across versions, no pickle
    <out>/meta.json   version, features, alert threshold, and the metrics behind it

Each run is logged to MLflow (local SQLite store `mlflow.db`, view with
`uv run mlflow ui`): hyperparameters, PR-AUC, the test confusion matrix at the
alert threshold, and both files above.

The first model trained also becomes production (models/production.json, which
the API serves); after that only the champion/challenger gate in retrain.py
changes it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path

import mlflow
import pandas as pd
from cryptoguard import evaluate as ev
from cryptoguard.data import FEATURE_COLUMNS, load_elliptic
from cryptoguard.features import TRAIN_END, VAL_END, temporal_split
from cryptoguard.models import build_xgboost_smote
from cryptoguard.pipeline import COST_FN, COST_FP

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
MODELS = ROOT / "models"
PRODUCTION = MODELS / "production.json"


def set_production(version: str) -> None:
    PRODUCTION.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTION.write_text(json.dumps({"version": version}) + "\n", newline="\n")


def cryptoguard_commit() -> str:
    """Git commit of the installed CryptoGuard, so each artifact names the code that built it."""
    direct_url = json.loads(distribution("cryptoguard").read_text("direct_url.json") or "{}")
    return direct_url.get("vcs_info", {}).get("commit_id", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v1", help="artifact version name")
    parser.add_argument("--out", type=Path, help="output folder (default: models/<version>)")
    args = parser.parse_args()
    out = args.out or MODELS / args.version

    mlflow.set_experiment("overseerml")
    with mlflow.start_run(run_name=args.version) as run:
        meta = train_and_save(args.version, out, run.info.run_id, load_elliptic(DATA_DIR))
    if not PRODUCTION.exists():
        set_production(args.version)
    print(json.dumps(meta["metrics"], indent=2))
    print(f"Wrote {out} (MLflow run {run.info.run_id})")


def train_and_save(
    version: str,
    out: Path,
    run_id: str,
    df: pd.DataFrame,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
) -> dict:
    """Train on steps <= train_end, pick the threshold on steps up to val_end, and
    report "test" metrics on every later step in `df`."""
    train, val, test = temporal_split(df, train_end, val_end)
    # CryptoGuard compared SMOTE with class weights on validation and adopted
    # SMOTE (results.json "smote.adopted"), so that is the model rebuilt here.
    pipeline = build_xgboost_smote().fit(train.X, train.y)
    # SMOTE only runs during fit; at inference the pipeline is just the booster.
    booster = pipeline[-1]

    val_s = booster.predict_proba(val.X)[:, 1]
    test_s = booster.predict_proba(test.X)[:, 1]
    threshold = ev.cost_optimal_threshold(val.y, val_s, COST_FN, COST_FP)
    at_threshold = ev.threshold_metrics(test.y, test_s, threshold)
    # threshold_metrics leaves out true negatives; add them to complete the confusion matrix.
    at_threshold["tn"] = len(test.y) - at_threshold["tp"] - at_threshold["fp"] - at_threshold["fn"]

    test_end = int(test.time_step.max())

    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "model.json")
    meta = {
        "version": version,
        "mlflow_run_id": run_id,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cryptoguard_commit": cryptoguard_commit(),
        "features": FEATURE_COLUMNS,
        "threshold": threshold,
        "windows": {"train_end": train_end, "val_end": val_end, "test_end": test_end},
        "metrics": {
            "val_pr_auc": ev.pr_auc(val.y, val_s),
            "test_pr_auc": ev.pr_auc(test.y, test_s),
            **{f"test_{k}": v for k, v in at_threshold.items() if k != "threshold"},
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", newline="\n")

    mlflow.log_params(
        {
            "cryptoguard_commit": meta["cryptoguard_commit"],
            "resampling": "SMOTE",
            "train_end_step": train_end,
            "val_end_step": val_end,
            "test_end_step": test_end,
            "cost_fn": COST_FN,
            "cost_fp": COST_FP,
            **{k: v for k, v in booster.get_params().items() if v is not None},
        }
    )
    mlflow.log_metrics({**meta["metrics"], "threshold": threshold})
    mlflow.log_artifacts(str(out), artifact_path="model")
    return meta


if __name__ == "__main__":
    main()
