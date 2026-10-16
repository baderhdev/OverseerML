"""Retrain CryptoGuard's headline model and save it as a servable artifact.

CryptoGuard never saves its models, so this reruns its own code (data loader,
temporal split, model definition, threshold rule) and writes:

    <out>/model.json  XGBoost's native format: portable across versions, no pickle
    <out>/meta.json   version, features, alert threshold, and the metrics behind it
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path

from cryptoguard import evaluate as ev
from cryptoguard.data import FEATURE_COLUMNS, load_elliptic
from cryptoguard.features import temporal_split
from cryptoguard.models import build_xgboost_smote
from cryptoguard.pipeline import COST_FN, COST_FP

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"


def cryptoguard_commit() -> str:
    """Git commit of the installed CryptoGuard, so each artifact names the code that built it."""
    direct_url = json.loads(distribution("cryptoguard").read_text("direct_url.json") or "{}")
    return direct_url.get("vcs_info", {}).get("commit_id", "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v1", help="artifact version name")
    parser.add_argument("--out", type=Path, help="output folder (default: models/<version>)")
    args = parser.parse_args()
    out = args.out or ROOT / "models" / args.version

    train, val, test = temporal_split(load_elliptic(DATA_DIR))
    # CryptoGuard compared SMOTE with class weights on validation and adopted
    # SMOTE (results.json "smote.adopted"), so that is the model rebuilt here.
    pipeline = build_xgboost_smote().fit(train.X, train.y)
    # SMOTE only runs during fit; at inference the pipeline is just the booster.
    booster = pipeline[-1]

    val_s = booster.predict_proba(val.X)[:, 1]
    test_s = booster.predict_proba(test.X)[:, 1]
    threshold = ev.cost_optimal_threshold(val.y, val_s, COST_FN, COST_FP)
    at_threshold = ev.threshold_metrics(test.y, test_s, threshold)

    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / "model.json")
    meta = {
        "version": args.version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cryptoguard_commit": cryptoguard_commit(),
        "features": FEATURE_COLUMNS,
        "threshold": threshold,
        "metrics": {
            "val_pr_auc": ev.pr_auc(val.y, val_s),
            "test_pr_auc": ev.pr_auc(test.y, test_s),
            **{f"test_{k}": v for k, v in at_threshold.items() if k != "threshold"},
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", newline="\n")
    print(json.dumps(meta["metrics"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
