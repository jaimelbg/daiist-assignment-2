"""
Assignment 2: Gradio dashboard for the site-gate helmet check.

    uv run python main.py app

Loads the committed weights in weights/ and what train.py wrote to
artifacts/. Never trains anything. The test-split scores in
artifacts/test_scores.csv are what every threshold, cost and staffing figure
below is recomputed from, using the same functions train.py used.
"""

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

from ppe_decision import (
    CLASS_NAMES,
    Costs,
    Operations,
    confusion_counts,
    cost_per_case,
    outcome_rates,
    review_share,
    staffing,
    sweep,
    violation_score,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = sorted((ROOT / "weights").glob("*.pt"))[0]
RESULTS = json.loads((ARTIFACTS / "results.json").read_text())
TEST = pd.read_csv(ARTIFACTS / "test_scores.csv")
SCORES, LABELS = TEST["score"].to_numpy(), TEST["violation"].to_numpy(bool)
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
BAND = RESULTS["decision"]["band"]
CHOSEN = RESULTS["decision"]["single_threshold"]["threshold"]
MODEL = YOLO(str(WEIGHTS))
COLOURS = {"boots": (120, 120, 120), "gloves": (180, 105, 255), "helmet": (0, 200, 255), "human": (255, 140, 0), "vest": (0, 220, 0)}


def costs_from(base_rate: float, fn: float, fp: float) -> Costs:
    return Costs(**{**RESULTS["costs"], "base_rate": base_rate, "fn": fn, "fp": fp, "tp": fp})


def decide(score: float, t_pass: float, t_fail: float) -> str:
    if score >= t_fail:
        return "FAIL: stop the worker"
    if score < min(t_pass, t_fail):
        return "PASS"
    return "REVIEW: send to a person"


# ---------------------------------------------------------------- single image


def analyse(image: np.ndarray | None, box_conf: float, t_pass: float, t_fail: float):
    if image is None:
        return None, "Upload an image."
    result = MODEL.predict(image, conf=0.01, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    score = violation_score(boxes, classes, confs)

    drawn = image.copy()
    for (x1, y1, x2, y2), c, p in zip(boxes.astype(int), classes, confs):
        if p < box_conf:
            continue
        colour = COLOURS[CLASS_NAMES[c]]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(drawn, f"{CLASS_NAMES[c]} {p:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    shown = pd.Series(classes[confs >= box_conf]).map(dict(enumerate(CLASS_NAMES))).value_counts()
    summary = ", ".join(f"{n} {k}" for k, n in shown.items()) or "nothing"
    text = (
        f"### {decide(score, t_pass, t_fail)}\n"
        f"Violation score **{score:.3f}** (pass below {min(t_pass, t_fail):.2f}, fail at {t_fail:.2f} or above).\n\n"
        f"Boxes shown at confidence >= {box_conf:.2f}: {summary}."
    )
    return drawn, text


# --------------------------------------------------------------- operating point


def cost_curve(c: Costs, t_pass: float, t_fail: float) -> go.Figure:
    rows = sweep(SCORES, LABELS, c, GRID)
    x = [r["threshold"] for r in rows]
    y = [r["cost_per_1000"] for r in rows]
    best = min(rows, key=lambda r: r["cost_per_1000"])
    current = 1000 * cost_per_case(outcome_rates(SCORES, LABELS, t_pass, t_fail), c)
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name="Model alone (one threshold)")
    fig.add_vline(x=CHOSEN, line_dash="dash", line_color="firebrick", annotation_text=f"chosen t = {CHOSEN:.2f}")
    fig.add_scatter(x=[best["threshold"]], y=[best["cost_per_1000"]], mode="markers", marker_size=10, name=f"minimum at these costs (t = {best['threshold']:.2f})")
    fig.add_hline(y=current, line_dash="dot", line_color="seagreen", annotation_text=f"current band: EUR {current:,.0f}")
    fig.add_vrect(x0=min(t_pass, t_fail), x1=t_fail, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="review")
    fig.update_layout(
        xaxis_title="Violation-score threshold",
        yaxis_title="Expected cost per 1,000 entries (EUR)",
        yaxis_type="log",
        # Fixed range in log10 units: Plotly's autorange blows up on a log axis
        # once the review-band annotation is added.
        yaxis_range=[np.log10(max(min(y + [current]), 1) * 0.7), np.log10(max(y + [current]) * 1.5)],
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def operating_point(t_pass: float, t_fail: float, base_rate: float, fn: float, fp: float, entries: float):
    c = costs_from(base_rate, fn, fp)
    rates = outcome_rates(SCORES, LABELS, t_pass, t_fail)
    counts = confusion_counts(SCORES, LABELS, t_pass, t_fail)
    table = pd.DataFrame(counts).T.rename_axis("truth (test images)").reset_index()
    share = review_share(rates, c)
    ops = Operations(**{**RESULTS["operations"], "entries_per_day": int(entries)})
    staff = staffing(share, ops)
    cost = 1000 * cost_per_case(rates, c)
    missed = 1000 * c.base_rate * (rates["pos"]["pass"] + rates["pos"]["review"] * c.reviewer_miss)
    summary = (
        f"| | |\n|---|---|\n"
        f"| Expected cost per 1,000 entries | **EUR {cost:,.0f}** |\n"
        f"| Share sent to review | **{share:.1%}** |\n"
        f"| Violations caught (fail + review) | {rates['pos']['fail'] + rates['pos']['review']:.1%} |\n"
        f"| Compliant workers stopped | {rates['neg']['fail']:.1%} |\n"
        f"| Unprotected entries let through per 1,000 | {missed:.1f} |\n"
        f"| Reviews per day ({int(entries):,} entries) | {staff['reviews_per_day']:,.0f} |\n"
        f"| Reviewer hours per day | {staff['review_hours_per_day']:.1f} |\n"
        f"| Reviewers needed in the shift-start rush | {staff['reviewers_at_peak']:.1f} |\n"
    )
    return table, summary, cost_curve(c, t_pass, t_fail)


# ----------------------------------------------------------------- static tables


def per_class_table() -> pd.DataFrame:
    rows = []
    for model, m in RESULTS["detection"].items():
        for name in CLASS_NAMES:
            r = m["per_class"][name]
            rows.append({"model": model, "class": name, "test instances": r["instances"], "precision@0.25": round(r["precision"], 3), "recall@0.25": round(r["recall"], 3), "AP50": round(r["ap50"], 3), "AP50-95": round(r["ap50_95"], 3)})
        rows.append({"model": model, "class": "ALL (mean)", "test instances": None, "precision@0.25": None, "recall@0.25": None, "AP50": round(m["map50"], 3), "AP50-95": round(m["map50_95"], 3)})
    return pd.DataFrame(rows)


def policy_table() -> pd.DataFrame:
    df = pd.DataFrame(RESULTS["decision"]["policies"])
    df["review_share"] = (100 * df["review_share"]).round(1).astype(str) + "%"
    return df.round(1)


def composition_table() -> pd.DataFrame:
    rows = []
    for split, s in RESULTS["dataset"]["composition"].items():
        rows.append({"split": split, "images": s["images"], **s["instances"], "images with people": s["images_with_people"], "images with a helmetless person": s["images_with_violation"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- layout

costs = RESULTS["costs"]
with gr.Blocks(title="Site gate helmet check") as demo:
    gr.Markdown(
        "# Site gate helmet check\n"
        f"Finetuned `{WEIGHTS.name}` on Roboflow `{RESULTS['dataset']['workspace']}/{RESULTS['dataset']['project']}` "
        f"v{RESULTS['dataset']['version']}. Positive class: **a person entering without a helmet**. "
        "Every number below is recomputed from the test split, never retrained."
    )
    with gr.Row():
        t_pass = gr.Slider(0, 1, value=BAND["t_pass"], step=0.01, label="Pass below (lower edge of review band)")
        t_fail = gr.Slider(0, 1, value=BAND["t_fail"], step=0.01, label="Fail at or above (upper edge of review band)")

    with gr.Tab("Operating point"):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.30, value=costs["base_rate"], step=0.005, label="Base rate: share of entries without a helmet")
            fn = gr.Number(value=costs["fn"], label="Cost of a missed violation (EUR)")
            fp = gr.Number(value=costs["fp"], label="Cost of stopping a worker (EUR)")
            entries = gr.Number(value=RESULTS["operations"]["entries_per_day"], label="Gate entries per day")
        with gr.Row():
            with gr.Column():
                confusion = gr.Dataframe(label="Test-split outcomes at these thresholds")
                summary = gr.Markdown()
            plot = gr.Plot(label="Cost against threshold")
        inputs = [t_pass, t_fail, base_rate, fn, fp, entries]
        for control in inputs:
            control.change(operating_point, inputs, [confusion, summary, plot])
        demo.load(operating_point, inputs, [confusion, summary, plot])

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Gate image")
            image_out = gr.Image(type="numpy", label="Detections")
        box_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Show boxes at confidence >=")
        verdict = gr.Markdown()
        examples = sorted(str(p) for p in (ARTIFACTS / "examples").glob("*.jpg"))
        if examples:
            gr.Examples(examples, inputs=image_in)
        for control in (image_in, box_conf, t_pass, t_fail):
            control.change(analyse, [image_in, box_conf, t_pass, t_fail], [image_out, verdict])

    with gr.Tab("Per-class results"):
        gr.Markdown("Detection metrics on the published test split (211 images), for the finetuned model and both baselines.")
        gr.Dataframe(per_class_table())
        gr.Image(str(ARTIFACTS / "confusion_matrix.png"), label="Detection confusion matrix (rows predicted, columns true)")
        gr.Markdown("Dataset composition after removing train/valid images that duplicate a test image.")
        gr.Dataframe(composition_table())

    with gr.Tab("Policies compared"):
        gr.Markdown(f"At the assumed base rate of {costs['base_rate']:.0%}, missed violation EUR {costs['fn']:.0f}, stop EUR {costs['fp']:.0f}, review EUR {costs['review']:.2f}.")
        gr.Dataframe(policy_table())

demo.launch()
