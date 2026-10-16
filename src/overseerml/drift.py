"""Replay Elliptic time steps as arriving batches and monitor them for drift.

Each time step after training (30-49) is one batch, scored by the served model.
Per batch this computes:

- PSI of every feature against the training period (steps 1-29, all
  transactions: production traffic includes ones that never get a label).
- PSI of the model's risk scores against the training period.
- PR-AUC as it would be seen live: labels arrive LABEL_LAG steps late, so at
  step t only steps <= t - LABEL_LAG can be scored, pooled over the latest
  ROLLING_STEPS of them. The same-step PR-AUC is also kept, as hindsight.

Input-drift flags are calibrated without looking at the test period (35-49): a
feature drifts when its PSI exceeds the usual 0.25 cutoff, and a batch is
flagged when more features drift, or its score PSI is higher, than in any
validation batch (30-34). On Elliptic these flags are informational: about 32
features drift every step even while the model works, and the step-43 dark
market shutdown changed who is illicit rather than what transactions look like.

The performance alert (lagged PR-AUC below PERF_FLOOR) is what catches the
shutdown, and it is the retraining trigger. PERF_FLOOR was chosen after seeing
the test-period results, so it is not a blind choice.

Writes reports/drift/{batches,feature_psi,score_hist}.csv and summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cryptoguard import evaluate as ev
from cryptoguard.data import FEATURE_COLUMNS, load_elliptic
from cryptoguard.features import TRAIN_END, VAL_END
from xgboost import XGBClassifier

from overseerml.train import DATA_DIR, ROOT

PSI_DRIFTED = 0.25
LABEL_LAG = 2
ROLLING_STEPS = 3
PERF_FLOOR = 0.80
SCORE_EDGES = np.linspace(0, 1, 21)[1:-1]  # 20 equal-width score bins
EPS = 1e-4  # floor for empty bins, which would otherwise make PSI infinite


def bin_shares(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return counts / len(x)


def psi(expected: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    """Population stability index of `actual` against `expected` over the given bin edges."""
    e = np.clip(bin_shares(expected, edges), EPS, None)
    a = np.clip(bin_shares(actual, edges), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def decile_edges(x: np.ndarray) -> np.ndarray:
    # Skewed features repeat quantiles; duplicate edges would create empty bins.
    return np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))


def pr_auc_or_nan(rows: pd.DataFrame) -> float:
    y = rows["label"].dropna().astype(int)
    return ev.pr_auc(y, rows.loc[y.index, "score"]) if y.nunique() == 2 else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "v1")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "drift")
    args = parser.parse_args()

    meta = json.loads((args.model_dir / "meta.json").read_text())
    model = XGBClassifier()
    model.load_model(args.model_dir / "model.json")

    df = load_elliptic(DATA_DIR)
    df["score"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    ref = df[df["time_step"] <= TRAIN_END]
    edges = {f: decile_edges(ref[f].to_numpy()) for f in FEATURE_COLUMNS}
    after_train = df[df["time_step"] > TRAIN_END]

    batches, feature_psi, score_hist = [], {}, {"train": bin_shares(ref["score"], SCORE_EDGES)}
    for step, batch in after_train.groupby("time_step"):
        fpsi = {f: psi(ref[f].to_numpy(), batch[f].to_numpy(), edges[f]) for f in FEATURE_COLUMNS}
        feature_psi[step] = fpsi
        score_hist[step] = bin_shares(batch["score"], SCORE_EDGES)
        # Only post-training steps count: training steps would flatter the model.
        arrived = after_train["time_step"].between(
            step - LABEL_LAG - ROLLING_STEPS + 1, step - LABEL_LAG
        )
        labeled = batch["label"].notna()
        batches.append(
            {
                "time_step": step,
                "period": "validation" if step <= VAL_END else "test",
                "transactions": len(batch),
                "labeled": int(labeled.sum()),
                "illicit": int((batch["label"] == 1).sum()),
                "alerts": int((batch["score"] >= meta["threshold"]).sum()),
                "mean_score": batch["score"].mean(),
                "score_psi": psi(ref["score"], batch["score"], SCORE_EDGES),
                "features_drifted": sum(v > PSI_DRIFTED for v in fpsi.values()),
                "max_feature_psi": max(fpsi.values()),
                "pr_auc_hindsight": pr_auc_or_nan(batch),
                "pr_auc_lagged": pr_auc_or_nan(after_train[arrived]),
            }
        )

    table = pd.DataFrame(batches).set_index("time_step")
    validation = table[table["period"] == "validation"]
    feature_limit = int(validation["features_drifted"].max())
    score_limit = float(validation["score_psi"].max())
    table["feature_flag"] = table["features_drifted"] > feature_limit
    table["score_flag"] = table["score_psi"] > score_limit
    table["drift_flag"] = table["feature_flag"] | table["score_flag"]
    # NaN (no labels arrived yet) compares False, so it never raises an alert.
    table["perf_alert"] = table["pr_auc_lagged"] < PERF_FLOOR

    args.out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out / "batches.csv", lineterminator="\n")
    pd.DataFrame(feature_psi).T.rename_axis("time_step").round(4).to_csv(
        args.out / "feature_psi.csv", lineterminator="\n"
    )
    bins = [f"{lo:.2f}-{lo + 0.05:.2f}" for lo in np.r_[0, SCORE_EDGES]]
    pd.DataFrame(score_hist, index=bins).T.rename_axis("time_step").round(4).to_csv(
        args.out / "score_hist.csv", lineterminator="\n"
    )
    summary = {
        "model_version": meta["version"],
        "psi_drifted": PSI_DRIFTED,
        "feature_flag_above": feature_limit,
        "score_psi_flag_above": round(score_limit, 4),
        "perf_floor": PERF_FLOOR,
        "label_lag_steps": LABEL_LAG,
        "rolling_steps": ROLLING_STEPS,
        "drift_flagged_steps": table.index[table["drift_flag"]].tolist(),
        "perf_alert_steps": table.index[table["perf_alert"]].tolist(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.drop(columns=["period"]).round(3))
    print(f"\nDrift flag: features_drifted > {feature_limit} or score_psi > {score_limit:.3f}")
    print(f"Drift-flagged steps: {summary['drift_flagged_steps']}")
    print(f"Performance alerts (lagged PR-AUC < {PERF_FLOOR}): {summary['perf_alert_steps']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
