"""Monitoring dashboard: `uv run streamlit run src/overseerml/dashboard.py`.

Reads only the committed outputs of overseer-drift (reports/drift) and
overseer-retrain (reports/retraining), so it needs neither the dataset, the
API, nor the MLflow store.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPORTS = Path(__file__).resolve().parents[2] / "reports"
SHUTDOWN_STEP = 43

# Reference data-viz palette: categorical slots 1-3 (validated all-pairs), one
# sequential blue ramp, ink grays, and status colors that always carry a label.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# Mid-gray ink reads on both Streamlit themes (about 5:1 on dark, 3.5:1 on white).
INK, MUTED = "#8a8984", "#a3a29c"
WARNING, CRITICAL = "#fab219", "#d03b3b"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


AUDIT_COLUMNS = [
    "trigger_step",
    "champion",
    "challenger",
    "compare_steps",
    "compare_illicit",
    "champion_pr_auc",
    "challenger_pr_auc",
    "promoted",
    "reason",
]


@st.cache_data
def load() -> dict:
    audit_lines = (REPORTS / "retraining" / "audit.jsonl").read_text().splitlines()
    return {
        "timeline": pd.read_csv(REPORTS / "retraining" / "timeline.csv"),
        "served_hist": pd.read_csv(REPORTS / "retraining" / "score_hist.csv"),
        "drift": pd.read_csv(REPORTS / "drift" / "batches.csv"),
        "feature_psi": pd.read_csv(REPORTS / "drift" / "feature_psi.csv", index_col=0),
        "drift_summary": json.loads((REPORTS / "drift" / "summary.json").read_text()),
        # Explicit columns so a replay with no retraining attempts still loads.
        "audit": pd.DataFrame([json.loads(line) for line in audit_lines], columns=AUDIT_COLUMNS),
    }


def shutdown_rule() -> alt.LayerChart:
    rule = alt.Chart(pd.DataFrame({"time_step": [SHUTDOWN_STEP]}))
    return rule.mark_rule(color=INK, strokeDash=[2, 3]).encode(x="time_step:Q") + rule.mark_text(
        text="dark-market shutdown", align="left", dx=4, dy=-6, color=INK, baseline="bottom"
    ).encode(x="time_step:Q", y=alt.value(0))


def show(chart: alt.TopLevelMixin, data: pd.DataFrame) -> None:
    st.altair_chart(chart.properties(height=260), width="stretch")
    with st.expander("Data"):
        st.dataframe(data, hide_index=True)


st.set_page_config(page_title="OverseerML", layout="wide")
d = load()
timeline, audit, drift = d["timeline"], d["audit"], d["drift"]
served = list(dict.fromkeys(timeline["champion"]))
model_color = alt.Scale(domain=served, range=MODEL_COLORS[: len(served)])
production = audit["challenger"][audit["promoted"]].iloc[-1] if audit["promoted"].any() else "v1"
first, last = timeline["time_step"].min(), timeline["time_step"].max()
step_axis = alt.X(
    "time_step:Q",
    title="Time step (two weeks each)",
    scale=alt.Scale(domain=[first - 0.5, last + 0.5], nice=False),
    axis=alt.Axis(values=list(range(first, last + 1)), format="d"),
)

st.title("OverseerML")
st.caption(
    "Replay of Elliptic time steps 30-49 as arriving batches, monitored and retrained by "
    "OverseerML around [CryptoGuard](https://github.com/baderhdev/cryptoguard)'s model. "
    "Built from the files in `reports/`; the API does not need to be running."
)

kpis = st.columns(4)
kpis[0].metric("Production model", production)
kpis[1].metric("Promotions", f"{int(audit['promoted'].sum())} of {len(audit)} attempts")
kpis[2].metric("Performance alerts", f"{int(timeline['perf_alert'].sum())} steps")
kpis[3].metric("Input-drift flags", f"{int(drift['drift_flag'].sum())} steps")

st.header("Prediction volume")
st.write(
    "Transactions scored per step, colored by the model serving it, with the alerts it raised. "
    "After the step-46 promotion, alerts more than double: the retrained model's cost-based "
    "threshold flags far more transactions for analyst review."
)
base = alt.Chart(timeline)
volume = (
    base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=22).encode(
        x=step_axis,
        y=alt.Y("transactions:Q", title="Transactions"),
        color=alt.Color("champion:N", scale=model_color, title="Serving model"),
        tooltip=["time_step", "champion", "transactions", "alerts"],
    )
    + base.mark_line(color=INK, strokeWidth=2, point=alt.OverlayMarkDef(color=INK, size=40)).encode(
        x=step_axis, y="alerts:Q", tooltip=["time_step", "champion", "alerts"]
    )
    + base.transform_filter(alt.datum.time_step == first)
    .mark_text(text="alerts raised", align="left", dx=-6, dy=-14, color=INK)
    .encode(x=step_axis, y="alerts:Q")
    + shutdown_rule()
)
show(volume, timeline[["time_step", "champion", "transactions", "alerts"]])

st.header("Model performance")
st.write(
    "**Lagged** PR-AUC is what monitoring could see: labels arrive two steps late and are pooled "
    "over three steps. **Hindsight** is the same step's PR-AUC once its labels are in. The gray "
    "line is the original v1 model if it had never been retrained. A performance alert fires "
    "below the 0.80 floor and triggers retraining."
)
first_promotion = timeline.loc[timeline["champion"] != "v1", "time_step"].min()
perf = (
    timeline.melt(
        id_vars=["time_step", "champion"],
        value_vars=["pr_auc_lagged", "pr_auc_hindsight"],
        var_name="measure",
        value_name="pr_auc",
    )
    .replace({"pr_auc_lagged": "Lagged (served)", "pr_auc_hindsight": "Hindsight (served)"})
    .pipe(
        lambda f: pd.concat(
            [
                f,
                # Only where it differs from the served model, plus one step to join on.
                drift[drift["time_step"] >= first_promotion - 1]
                .assign(measure="Hindsight (v1, never retrained)", champion="v1")[
                    ["time_step", "champion", "measure", "pr_auc_hindsight"]
                ]
                .rename(columns={"pr_auc_hindsight": "pr_auc"}),
            ]
        )
    )
    .dropna()
)
measures = ["Hindsight (served)", "Lagged (served)", "Hindsight (v1, never retrained)"]
floor = alt.Chart(pd.DataFrame({"y": [0.80]}))
alerts = timeline[timeline["perf_alert"]]
performance = (
    alt.Chart(perf)
    .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40))
    .encode(
        x=step_axis,
        y=alt.Y("pr_auc:Q", title="PR-AUC", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "measure:N",
            scale=alt.Scale(domain=measures, range=[MODEL_COLORS[0], MODEL_COLORS[0], MUTED]),
            title=None,
            legend=alt.Legend(orient="bottom", symbolType="stroke", symbolStrokeWidth=2),
        ),
        # Same field and title as color, so Vega-Lite merges them into one legend.
        strokeDash=alt.StrokeDash(
            "measure:N",
            scale=alt.Scale(domain=measures, range=[[1, 0], [6, 4], [1, 0]]),
            title=None,
        ),
        tooltip=["time_step", "measure", "champion", alt.Tooltip("pr_auc:Q", format=".3f")],
    )
    + floor.mark_rule(color=CRITICAL, strokeDash=[4, 4]).encode(y="y:Q")
    + floor.mark_text(text="alert floor 0.80", align="left", x=4, dy=-6, color=CRITICAL).encode(
        y="y:Q"
    )
    + alt.Chart(alerts)
    .mark_point(shape="triangle-down", size=90, filled=True, color=CRITICAL)
    .encode(
        x=step_axis,
        y=alt.value(8),
        tooltip=[alt.Tooltip("time_step", title="Performance alert at step"), "champion"],
    )
    + alt.Chart(alerts.head(1))
    .mark_text(text="performance alert", align="right", dx=-10, color=CRITICAL)
    .encode(x=step_axis, y=alt.value(8))
    + shutdown_rule()
)
show(
    performance,
    perf.pivot_table(index="time_step", columns="measure", values="pr_auc").reset_index(),
)

st.header("Score distribution drift")
st.write(
    "Share of each step's transactions in each risk-score band, as scored by the model serving "
    "that step. Darker means more transactions. The color scale is square-root so the "
    "thin high-score bands stay visible next to the large near-zero band."
)
hist = d["served_hist"].melt(id_vars="time_step", var_name="score_band", value_name="share")
hist = hist.merge(timeline[["time_step", "champion"]], on="time_step")
bands = list(d["served_hist"].columns[1:])
score_drift = (
    alt.Chart(hist)
    .mark_rect()
    .encode(
        x=alt.X("time_step:O", title="Time step", scale=alt.Scale(paddingInner=0.08)),
        y=alt.Y(
            "score_band:O",
            title="Risk score",
            sort=list(reversed(bands)),
            scale=alt.Scale(paddingInner=0.08),
        ),
        color=alt.Color(
            "share:Q",
            scale=alt.Scale(type="sqrt", range=BLUE_RAMP),
            title="Share of step",
            legend=alt.Legend(format=".0%"),
        ),
        tooltip=["time_step", "champion", "score_band", alt.Tooltip("share:Q", format=".1%")],
    )
)
show(score_drift, d["served_hist"])

st.header("Feature drift (input monitoring)")
limit = d["drift_summary"]["feature_flag_above"]
st.write(
    f"Features with PSI above 0.25 against the training period. A step is flagged when more "
    f"than {limit} drift, the most seen in any validation step, when the model still worked. "
    "Input drift is constant in this data and did not react to the shutdown at step 43; the "
    "performance alert above is what caught it."
)
drift_chart = (
    alt.Chart(drift)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=22)
    .encode(
        x=step_axis,
        y=alt.Y("features_drifted:Q", title="Features with PSI > 0.25"),
        color=alt.condition(alt.datum.feature_flag, alt.value(WARNING), alt.value(BLUE_RAMP[2])),
        tooltip=["time_step", "features_drifted", "feature_flag", "score_psi", "period"],
    )
    + alt.Chart(drift[drift["feature_flag"]])
    .mark_text(text="⚠ flagged", dy=-8, color=INK)
    .encode(x=step_axis, y="features_drifted:Q")
    + alt.Chart(pd.DataFrame({"y": [limit]}))
    .mark_rule(color=INK, strokeDash=[4, 4])
    .encode(y="y:Q")
    + alt.Chart(pd.DataFrame({"y": [limit]}))
    .mark_text(text=f"flag above {limit}", align="left", x=4, dy=-6, color=INK)
    .encode(y="y:Q")
    + shutdown_rule()
)
show(drift_chart, drift[["time_step", "period", "features_drifted", "feature_flag", "score_psi"]])

psi = d["feature_psi"]
top = psi.max().nlargest(20).index
st.subheader("20 most drifted features")
st.write(
    "PSI per step, capped at 1 for color: six neighbor-aggregate features sit near 8 in every step."
)
heat = psi[top].reset_index().melt(id_vars="time_step", var_name="feature", value_name="psi")
show(
    alt.Chart(heat)
    .mark_rect()
    .encode(
        x=alt.X("time_step:O", title="Time step", scale=alt.Scale(paddingInner=0.08)),
        y=alt.Y("feature:N", sort=list(top), title=None, scale=alt.Scale(paddingInner=0.08)),
        color=alt.Color(
            "psi:Q", scale=alt.Scale(domain=[0, 1], clamp=True, range=BLUE_RAMP), title="PSI"
        ),
        tooltip=["time_step", "feature", alt.Tooltip("psi:Q", format=".2f")],
    )
    .properties(height=420),
    psi[top].reset_index(),
)

st.header("Promotion history")
st.write(
    "Every retraining attempt: the challenger is promoted only if it beats the serving model's "
    "PR-AUC by 0.02 on the two newest labeled steps, which neither model trained on."
)
decisions = audit["promoted"].map({True: "✔ promoted", False: "✖ rejected"})
st.dataframe(
    audit.assign(promoted=decisions).rename(columns={"promoted": "decision"}),
    hide_index=True,
    width="stretch",
)
