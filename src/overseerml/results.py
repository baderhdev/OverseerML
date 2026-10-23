"""Write RESULTS.md from the outputs of one `make reproduce` run.

Every number is computed here or read from this run's reports, and printed
with Python's repr, so nothing is rounded for presentation.

Windows: pre-drift is test steps 35-42, post-drift is 43-49 (the dark-market
shutdown is at step 43, as in CryptoGuard). A challenger is scored only on
post-drift steps after its threshold window, so it is never scored on data it
was trained or tuned on; v1 is scored on the same steps for the comparison.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime

import pandas as pd
from cryptoguard.data import load_elliptic
from cryptoguard.features import VAL_END
from cryptoguard.pipeline import SHUTDOWN_STEP

from overseerml.drift import PSI_DRIFTED, pr_auc
from overseerml.retrain import MARGIN, score
from overseerml.train import DATA_DIR, MODELS, ROOT

REPORTS = ROOT / "reports"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def steps_pr_auc(df: pd.DataFrame, version: str, steps: range) -> float:
    rows = df[df["time_step"].isin(steps)]
    return pr_auc(rows, score(version, rows))


def main() -> None:
    df = load_elliptic(DATA_DIR)
    last = int(df["time_step"].max())
    pre, post = range(VAL_END + 1, SHUTDOWN_STEP), range(SHUTDOWN_STEP, last + 1)
    batches = pd.read_csv(REPORTS / "drift" / "batches.csv", index_col="time_step")
    feature_psi = pd.read_csv(REPORTS / "drift" / "feature_psi.csv", index_col="time_step")
    drift_summary = json.loads((REPORTS / "drift" / "summary.json").read_text())
    timeline = pd.read_csv(REPORTS / "retraining" / "timeline.csv", index_col="time_step")
    audit = [
        json.loads(ln) for ln in (REPORTS / "retraining" / "audit.jsonl").read_text().splitlines()
    ]
    load = json.loads((REPORTS / "loadtest.json").read_text())
    production = json.loads((MODELS / "production.json").read_text())["version"]

    v1_pre, v1_post = steps_pr_auc(df, "v1", pre), steps_pr_auc(df, "v1", post)
    flagged = batches.index[batches["drift_flag"]]
    first_flag = int(flagged[0]) if len(flagged) else None
    alerts = timeline.index[timeline["perf_alert"]]
    first_alert = int(alerts[0]) if len(alerts) else None

    md = [
        "# Results",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} by `make reproduce` "
        f"from commit `{git('rev-parse', 'HEAD')}`"
        + (" with uncommitted changes" if git("status", "--porcelain", "--", "src") else "")
        + f", Python {platform.python_version()} on {platform.platform()}.",
        "",
        "All numbers come from this one run and are printed unrounded. Training is "
        "deterministic on one platform but not across platforms (see README, Phase 6), "
        "so a run on another OS gives slightly different values.",
        "",
        "## How to reproduce",
        "",
        "Requires Python 3.12, uv, Docker, and GNU make.",
        "",
        "```bash",
        "make reproduce",
        "```",
        "",
        "which runs, in order:",
        "",
        "```bash",
        "uv sync --locked",
        "uv run overseer-train                  # retrain v1 from CryptoGuard's code",
        "uv run overseer-drift                  # drift replay, steps 30-49",
        "uv run overseer-retrain                # champion/challenger replay",
        "docker build -t overseerml .           # image serves models/production.json",
        "uv run python -m overseerml.loadtest   # load test against the container",
        "uv run python -m overseerml.results    # writes this file",
        "```",
        "",
        "## 1. Champion (v1) PR-AUC before and after the drift",
        "",
        f"Pre-drift = steps {pre.start}-{pre.stop - 1}, post-drift = steps "
        f"{post.start}-{post.stop - 1} (dark-market shutdown at step {SHUTDOWN_STEP}). "
        "Labeled transactions of all steps in a window are pooled.",
        "",
        "| Window | v1 PR-AUC |",
        "|---|---|",
        f"| Pre-drift ({pre.start}-{pre.stop - 1}) | {v1_pre!r} |",
        f"| Post-drift ({post.start}-{post.stop - 1}) | {v1_post!r} |",
        f"| Drop | {v1_pre - v1_post!r} |",
        "",
        "## 2. When drift was first detected",
        "",
    ]
    if first_flag is None:
        md.append("The input-drift detector flagged no step.")
    else:
        row = batches.loc[first_flag]
        drifted = feature_psi.loc[first_flag]
        drifted = drifted[drifted > PSI_DRIFTED].sort_values(ascending=False)
        md += [
            f"**Input-drift detector (PSI): first flag at step {first_flag}**, "
            f"{SHUTDOWN_STEP - first_flag} steps *before* the shutdown, while v1's hindsight "
            f"PR-AUC that step was {row['pr_auc_hindsight']!r}. Elliptic has no dates, only "
            "two-week time steps.",
            "",
            f"A step is flagged when more than {drift_summary['feature_flag_above']} features "
            f"have PSI > {PSI_DRIFTED} against training steps 1-29, or its score PSI exceeds "
            f"{drift_summary['score_psi_flag_above']!r} (both the highest seen in validation "
            f"steps 30-34). Step {first_flag} had {int(row['features_drifted'])} drifted "
            f"features (score PSI {row['score_psi']!r}). All steps flagged: "
            f"{[int(s) for s in flagged]}. The detector did **not** flag the shutdown step "
            f"{SHUTDOWN_STEP}.",
            "",
        ]
        if first_alert is not None:
            md += [
                f"**Performance detector: first alert at step {first_alert}**, "
                f"{first_alert - SHUTDOWN_STEP} steps after the shutdown, when lagged "
                f"PR-AUC fell to {timeline.loc[first_alert, 'pr_auc_lagged']!r} (floor 0.8; "
                "labels arrive 2 steps late). This alert is what triggered retraining.",
                "",
            ]
        md += [
            f"<details><summary>The {len(drifted)} features with PSI &gt; {PSI_DRIFTED} at "
            f"step {first_flag}</summary>",
            "",
            "| Feature | PSI |",
            "|---|---|",
            *[f"| {f} | {v!r} |" for f, v in drifted.items()],
            "",
            "</details>",
        ]

    md += [
        "",
        "## 3. Retrained challengers on the post-drift window",
        "",
        "Each challenger is scored on the post-drift steps after its threshold window, and "
        "v1 on the same steps. Recovered share = (challenger - v1) / (v1 pre-drift - v1 on "
        "those steps). This is hindsight scoring: in the replay a challenger only serves "
        "steps after it is promoted.",
        "",
        "| Challenger | Trained on | Scored on | Challenger PR-AUC | v1 PR-AUC "
        "| Recovered share of the drop | Promoted |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in audit:
        version = record["challenger"]
        val_end = json.loads((MODELS / version / "meta.json").read_text())["windows"]["val_end"]
        steps = range(max(post.start, val_end + 1), post.stop)
        challenger, champion = steps_pr_auc(df, version, steps), steps_pr_auc(df, "v1", steps)
        md.append(
            f"| {version} | steps {record['train_steps']} | {steps.start}-{steps.stop - 1} | "
            f"{challenger!r} | {champion!r} | {(challenger - champion) / (v1_pre - champion)!r} "
            f"| {'yes' if record['promoted'] else 'no'} |"
        )

    md += [
        "",
        "## 4. Promotion decisions",
        "",
        f"A challenger is promoted only if its PR-AUC on the comparison steps (the two newest "
        f"labeled steps, which neither model trained on) beats the serving model's by at "
        f"least {MARGIN}.",
        "",
        "| Trigger step | Serving model | Challenger | Compared on | Illicit in comparison "
        "| Serving PR-AUC | Challenger PR-AUC | Challenger - serving | Promoted |",
        "|---|---|---|---|---|---|---|---|---|",
        *[
            f"| {r['trigger_step']} | {r['champion']} | {r['challenger']} | steps "
            f"{r['compare_steps']} | {r['compare_illicit']} | {r['champion_pr_auc']!r} | "
            f"{r['challenger_pr_auc']!r} | {r['challenger_pr_auc'] - r['champion_pr_auc']!r} "
            f"| {'yes' if r['promoted'] else 'no'} |"
            for r in audit
        ],
        "",
        f"Production model after the replay: `{production}`.",
        "",
        "## 5. API latency",
        "",
        f"`POST /predict` against the Docker image serving `{load['model_version']}`: "
        f"{load['warmup_requests']} sequential warm-up requests (excluded), then "
        f"{load['requests']} requests from {load['concurrency']} concurrent client threads, "
        "each a real transaction sampled from test steps 35-49. Client and server ran on "
        "the same machine.",
        "",
        "| Measure | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) |",
        "|---|---|---|---|---|---|",
        *[
            f"| {label} | {s['p50']!r} | {s['p95']!r} | {s['p99']!r} | {s['mean']!r} "
            f"| {s['max']!r} |"
            for label, s in (
                ("Client round trip", load["client_latency_ms"]),
                ("Server (scoring + SHAP)", load["server_latency_ms"]),
            )
        ],
        "",
        f"Throughput: {load['throughput_rps']!r} requests/s ({load['requests']} requests in "
        f"{load['wall_seconds']!r} s). Server-logged requests after warm-up: "
        f"{load['server_logged_requests']}.",
        "",
        f"Hardware: {load['host']['cpu']}, {load['host']['logical_cpus']} logical CPUs, "
        f"{load['host']['ram_bytes']} bytes RAM, {load['host']['os']}. Docker "
        f"{load['host']['docker']['ServerVersion']} ({load['host']['docker']['OperatingSystem']}) "
        f"with {load['host']['docker']['NCPU']} CPUs and {load['host']['docker']['MemTotal']} "
        "bytes of memory available to containers.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"Wrote {ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
