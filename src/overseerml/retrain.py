"""Champion/challenger retraining, replayed over time steps 30-49.

Each step is scored by whichever model is in production at that moment, and
those logged scores feed the same lagged PR-AUC monitor as drift.py. When it
falls below PERF_FLOOR at step t (labels known up to t - LABEL_LAG), a
challenger is trained on the newest data that is known by then:

    train      steps 1 .. t-6
    threshold  steps t-5 .. t-4
    compare    steps t-3 .. t-2   (both models, neither trained on it)

The challenger is promoted only if its PR-AUC on the comparison steps beats
the champion's by at least MARGIN. Every attempt is an MLflow run tagged with
the decision, and is written to reports/retraining/audit.jsonl. A promotion
moves models/production.json, which is what the API serves.

The replay starts from v1 as production, so rerunning it rewrites the audit
log and the pointer from scratch. Each decision is appended to the audit log
before the pointer moves, so a run that stops partway leaves them in agreement.
"""

from __future__ import annotations

import json
from functools import cache

import mlflow
import numpy as np
import pandas as pd
from cryptoguard.data import FEATURE_COLUMNS, load_elliptic
from cryptoguard.features import TRAIN_END
from xgboost import XGBClassifier

from overseerml.drift import LABEL_LAG, PERF_FLOOR, SCORE_EDGES, arrived, bin_shares, pr_auc
from overseerml.train import DATA_DIR, MODELS, ROOT, set_production, train_and_save

MARGIN = 0.02
COMPARE_STEPS = 2
THRESHOLD_STEPS = 2
OUT = ROOT / "reports" / "retraining"


@cache
def load(version: str) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODELS / version / "model.json")
    return model


@cache
def threshold(version: str) -> float:
    return json.loads((MODELS / version / "meta.json").read_text())["threshold"]


def score(version: str, rows: pd.DataFrame) -> np.ndarray:
    return load(version).predict_proba(rows[FEATURE_COLUMNS])[:, 1]


def windows(step: int) -> tuple[int, int, int]:
    """Last training step, last threshold step, and first comparison step for an
    attempt at `step`; comparison runs to the newest labeled step, step - LABEL_LAG."""
    compare_from = step - LABEL_LAG - COMPARE_STEPS + 1
    val_end = compare_from - 1
    return val_end - THRESHOLD_STEPS, val_end, compare_from


def decide(champion_pr_auc: float, challenger_pr_auc: float) -> tuple[bool, str]:
    gain = challenger_pr_auc - champion_pr_auc
    if np.isnan(gain):
        return False, "comparison steps lack both classes, so PR-AUC is undefined"
    if gain >= MARGIN:
        return True, f"challenger beats champion by {gain:+.3f} (margin {MARGIN})"
    return False, f"challenger gain {gain:+.3f} is below the {MARGIN} margin"


def attempt(df: pd.DataFrame, step: int, champion: str) -> dict:
    """Train a challenger with what is known at `step`, compare it, and log the decision."""
    known = df[df["time_step"] <= step - LABEL_LAG]
    train_end, val_end, compare_from = windows(step)
    version = f"retrain-s{step}"

    with mlflow.start_run(run_name=version) as run:
        meta = train_and_save(version, MODELS / version, run.info.run_id, known, train_end, val_end)
        compare = known[known["time_step"] >= compare_from]
        champion_pr_auc = pr_auc(compare, score(champion, compare))
        challenger_pr_auc = meta["metrics"]["test_pr_auc"]
        promoted, reason = decide(champion_pr_auc, challenger_pr_auc)
        mlflow.log_metric("champion_compare_pr_auc", champion_pr_auc)
        mlflow.set_tags(
            {
                "role": "challenger",
                "trigger_step": step,
                "champion": champion,
                "decision": "promoted" if promoted else "rejected",
                "reason": reason,
            }
        )

    return {
        "trigger_step": step,
        "champion": champion,
        "challenger": version,
        "mlflow_run_id": run.info.run_id,
        "train_steps": f"1-{train_end}",
        "threshold_steps": f"{train_end + 1}-{val_end}",
        "compare_steps": f"{compare_from}-{step - LABEL_LAG}",
        "compare_illicit": int((compare["label"] == 1).sum()),
        "champion_pr_auc": round(champion_pr_auc, 4),
        "challenger_pr_auc": round(challenger_pr_auc, 4),
        "promoted": promoted,
        "reason": reason,
    }


def main() -> None:
    df = load_elliptic(DATA_DIR)
    replay = df[df["time_step"] > TRAIN_END].copy()
    champion = "v1"
    OUT.mkdir(parents=True, exist_ok=True)
    audit_log = OUT / "audit.jsonl"
    audit_log.write_text("")
    set_production(champion)
    mlflow.set_experiment("overseerml")

    timeline, score_hist = [], {}
    for step in range(TRAIN_END + 1, int(df["time_step"].max()) + 1):
        batch = replay["time_step"] == step
        replay.loc[batch, "score"] = score(champion, replay[batch])
        replay.loc[batch, "served_by"] = champion
        served = replay[batch]
        score_hist[step] = bin_shares(served["score"], SCORE_EDGES)

        # Scores logged at the time, by whichever model was serving then.
        labels_in = replay[arrived(replay["time_step"], step)]
        lagged = pr_auc(labels_in, labels_in["score"].to_numpy())
        alert = lagged < PERF_FLOOR
        timeline.append(
            {
                "time_step": step,
                "champion": champion,
                "transactions": int(batch.sum()),
                "alerts": int((served["score"] >= threshold(champion)).sum()),
                "pr_auc_lagged": round(lagged, 4),
                # Known only in hindsight, once this step's labels arrive.
                "pr_auc_hindsight": round(pr_auc(served, served["score"].to_numpy()), 4),
                "perf_alert": alert,
            }
        )
        if alert:
            record = attempt(df, step, champion)
            print(json.dumps(record))
            # Written before the pointer moves, so the log always explains production.
            with audit_log.open("a", newline="\n") as f:
                f.write(json.dumps(record) + "\n")
            if record["promoted"]:
                champion = record["challenger"]
                set_production(champion)

    pd.DataFrame(timeline).to_csv(OUT / "timeline.csv", index=False, lineterminator="\n")
    bins = [f"{lo:.2f}-{lo + 0.05:.2f}" for lo in np.r_[0, SCORE_EDGES]]
    pd.DataFrame(score_hist, index=bins).T.rename_axis("time_step").round(4).to_csv(
        OUT / "score_hist.csv", lineterminator="\n"
    )
    print(pd.DataFrame(timeline).to_string(index=False))
    print(f"Production model: {champion}. Wrote {OUT}")


if __name__ == "__main__":
    main()
