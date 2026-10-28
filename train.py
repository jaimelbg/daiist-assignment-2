"""
Assignment 2: training pipeline for the site-gate helmet check.

    uv run python main.py train

Start to finish, no manual steps:

1. Download hx-hezqh / ppe-detection-yfmym / v1 from Roboflow Universe.
2. Remove near-duplicates of test images from train and valid. The published
   splits leak: 185 of the 211 test images are near-copies of a training image.
   The test split itself is left exactly as published.
3. Finetune yolo11n from the COCO checkpoint and save the best weights to
   weights/.
4. Evaluate the finetuned model, COCO yolo11n zero-shot and a predict-nothing
   baseline on the same test split: per-class P/R/AP and a confusion matrix.
5. Turn detections into a per-image violation score, sweep the threshold,
   cost it, pick the review band and work out staffing.
6. Write everything the dashboard needs to artifacts/.

Set EPOCHS to override the epoch count (EPOCHS=1 for a smoke test).
Set SKIP_TRAIN=1 to reuse the weights in weights/ and only redo steps 4-6, e.g.
after changing the costs in ppe_decision.py.
"""

import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from api_key import load_api_key
from ppe_decision import (
    CLASS_NAMES,
    HUMAN,
    Costs,
    Operations,
    best_band,
    confusion_counts,
    cost_per_case,
    is_violation,
    outcome_rates,
    review_share,
    staffing,
    sweep,
    violation_score,
)

ROOT = Path(__file__).resolve().parent
WORKSPACE, PROJECT, VERSION = "hx-hezqh", "ppe-detection-yfmym", 1
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
SPLIT_DIR = ROOT / "datasets" / "leak-free"
WEIGHTS = ROOT / "weights" / "ppe_yolo11n.pt"
ARTIFACTS = ROOT / "artifacts"

BASE_MODEL = "yolo11n.pt"
EPOCHS = int(os.environ.get("EPOCHS", "50"))
SKIP_TRAIN = os.environ.get("SKIP_TRAIN") == "1"
IMGSZ, BATCH, PATIENCE, SEED = 640, 8, 15, 0
DUPLICATE_BITS = 10  # of 256: below this two images count as the same photo
EVAL_CONF = 0.001  # keep low-confidence boxes so AP and the sweep see everything
REPORT_CONF = 0.25  # the confidence the per-class P/R and confusion matrix use
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
COCO_PERSON = 0


def device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- data


def download() -> Path:
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already at {DATA_DIR}")
        return DATA_DIR
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11", location=str(DATA_DIR))
    return DATA_DIR


def images(split: str) -> list[Path]:
    return sorted((DATA_DIR / split / "images").glob("*.jpg"))


def average_hash(path: Path) -> np.ndarray:
    """16x16 average hash: survives the resize/re-encode Roboflow's copies went through."""
    gray = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (16, 16), interpolation=cv2.INTER_AREA)
    return (gray > gray.mean()).flatten()


def leak_free_split() -> tuple[Path, dict]:
    """Drop every train/valid image that is a near-copy of a test image."""
    test_hashes = np.array([average_hash(p) for p in images("test")])
    kept, stats = {}, {}
    for split in ("train", "valid"):
        paths = images(split)
        keep = [p for p in paths if (test_hashes != average_hash(p)).sum(1).min() > DUPLICATE_BITS]
        kept[split] = keep
        stats[split] = {"published": len(paths), "kept": len(keep), "dropped": len(paths) - len(keep)}
        print(f"{split}: kept {len(keep)} of {len(paths)} images")
    stats["test"] = {"published": len(images("test")), "kept": len(images("test")), "dropped": 0}

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, keep in kept.items():
        (SPLIT_DIR / f"{split}.txt").write_text("\n".join(str(p) for p in keep) + "\n")
    yaml = SPLIT_DIR / "data.yaml"
    yaml.write_text(
        f"train: {SPLIT_DIR / 'train.txt'}\n"
        f"val: {SPLIT_DIR / 'valid.txt'}\n"
        f"test: {DATA_DIR / 'test' / 'images'}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    return yaml, stats


def ground_truth(image: Path) -> tuple[np.ndarray, np.ndarray]:
    """Labelled boxes in pixel xyxy, with their classes."""
    h, w = cv2.imread(str(image)).shape[:2]
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, int)
    data = np.array(rows, float)
    cx, cy, bw, bh = data[:, 1] * w, data[:, 2] * h, data[:, 3] * w, data[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return boxes, data[:, 0].astype(int)


def describe(paths_by_split: dict) -> dict:
    """Instances per class, images per class and box size relative to the image."""
    out = {}
    for split, paths in paths_by_split.items():
        counts = {n: 0 for n in CLASS_NAMES}
        image_counts = {n: 0 for n in CLASS_NAMES}
        areas = {n: [] for n in CLASS_NAMES}
        violations = with_people = 0
        for p in paths:
            boxes, classes = ground_truth(p)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (640 * 640)
            for c in set(classes.tolist()):
                image_counts[CLASS_NAMES[c]] += 1
            for c, a in zip(classes, area):
                counts[CLASS_NAMES[c]] += 1
                areas[CLASS_NAMES[c]].append(float(a))
            if (classes == HUMAN).any():
                with_people += 1
                violations += is_violation(boxes, classes)
        out[split] = {
            "images": len(paths),
            "images_with_people": with_people,
            "images_with_violation": violations,
            "instances": counts,
            "images_containing": image_counts,
            "median_box_area_pct": {n: round(100 * float(np.median(a)), 2) if a else None for n, a in areas.items()},
        }
    return out


# ------------------------------------------------------------------------ training


def train(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,
        device=device(),
        workers=2,
        project=str(ROOT / "runs"),
        name="ppe_yolo11n",
        exist_ok=True,
        plots=True,
    )
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    print(f"Saved {WEIGHTS}")
    return WEIGHTS


def trained_settings() -> dict:
    """What the weights in weights/ were actually trained with, read from the checkpoint."""
    args = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["train_args"]
    return {"base_model": args["model"], **{k: args[k] for k in ("epochs", "imgsz", "batch", "patience", "seed", "device")}}


# ---------------------------------------------------------------------- evaluation


def predict(model: YOLO | None, paths: list[Path], class_map: dict | None = None) -> list[dict]:
    """Boxes per test image. class_map translates a foreign model's classes to ours."""
    out = []
    for p in paths:
        if model is None:  # the trivial baseline: predicts nothing at all
            out.append({"boxes": np.zeros((0, 4)), "classes": np.zeros(0, int), "confs": np.zeros(0)})
            continue
        r = model.predict(str(p), conf=EVAL_CONF, imgsz=IMGSZ, device=device(), verbose=False, max_det=300)[0]
        boxes, classes, confs = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
        if class_map is not None:
            keep = np.isin(classes, list(class_map))
            boxes, confs = boxes[keep], confs[keep]
            classes = np.array([class_map[c] for c in classes[keep]], int)
        out.append({"boxes": boxes, "classes": classes, "confs": confs})
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def match(pred: dict, gt_boxes: np.ndarray, gt_classes: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """True-positive flags per prediction at each IoU threshold, greedy by confidence, same class."""
    tp = np.zeros((len(pred["confs"]), len(thresholds)), bool)
    overlaps = iou(pred["boxes"], gt_boxes)
    order = np.argsort(-pred["confs"])
    for k, t in enumerate(thresholds):
        taken = np.zeros(len(gt_boxes), bool)
        for i in order:
            candidates = (gt_classes == pred["classes"][i]) & ~taken & (overlaps[i] >= t)
            if candidates.any():
                j = np.argmax(np.where(candidates, overlaps[i], -1))
                taken[j] = True
                tp[i, k] = True
    return tp


def detection_metrics(preds: list[dict], gts: list[tuple]) -> dict:
    thresholds = np.linspace(0.5, 0.95, 10)
    tps, confs, classes, targets = [], [], [], []
    tp_at, fp_at, n_gt = np.zeros(len(CLASS_NAMES)), np.zeros(len(CLASS_NAMES)), np.zeros(len(CLASS_NAMES))
    for pred, (gt_boxes, gt_classes) in zip(preds, gts):
        tp = match(pred, gt_boxes, gt_classes, thresholds)
        tps.append(tp)
        confs.append(pred["confs"])
        classes.append(pred["classes"])
        targets.append(gt_classes)
        n_gt += np.bincount(gt_classes, minlength=len(CLASS_NAMES))
        kept = pred["confs"] >= REPORT_CONF
        tp_at += np.bincount(pred["classes"][kept][tp[kept, 0]], minlength=len(CLASS_NAMES))
        fp_at += np.bincount(pred["classes"][kept][~tp[kept, 0]], minlength=len(CLASS_NAMES))
    tp, conf, cls, target = map(np.concatenate, (tps, confs, classes, targets))

    ap = np.zeros((len(CLASS_NAMES), len(thresholds)))
    if len(conf):
        result = ap_per_class(tp, conf, cls, target, names=dict(enumerate(CLASS_NAMES)))
        for row, c in zip(result[5], result[6]):
            ap[int(c)] = row

    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        predicted = tp_at[c] + fp_at[c]
        per_class[name] = {
            "instances": int(n_gt[c]),
            "precision": float(tp_at[c] / predicted) if predicted else 0.0,
            "recall": float(tp_at[c] / n_gt[c]) if n_gt[c] else 0.0,
            "ap50": float(ap[c, 0]),
            "ap50_95": float(ap[c].mean()),
        }
    return {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean()), "per_class": per_class}


def confusion_matrix(preds: list[dict], gts: list[tuple]) -> np.ndarray:
    """Rows = predicted class (+ background), columns = true class (+ background), IoU 0.5."""
    n = len(CLASS_NAMES)
    matrix = np.zeros((n + 1, n + 1), int)
    for pred, (gt_boxes, gt_classes) in zip(preds, gts):
        keep = pred["confs"] >= REPORT_CONF
        boxes, classes, confs = pred["boxes"][keep], pred["classes"][keep], pred["confs"][keep]
        overlaps = iou(boxes, gt_boxes)
        matched_gt = np.zeros(len(gt_boxes), bool)
        for i in np.argsort(-confs):
            candidates = ~matched_gt & (overlaps[i] >= 0.5)
            if candidates.any():
                j = np.argmax(np.where(candidates, overlaps[i], -1))
                matched_gt[j] = True
                matrix[classes[i], gt_classes[j]] += 1
            else:
                matrix[classes[i], n] += 1  # false positive on background
        for j in np.where(~matched_gt)[0]:
            matrix[n, gt_classes[j]] += 1  # missed object
    return matrix


def plot_confusion(matrix: np.ndarray, path: Path) -> None:
    labels = CLASS_NAMES + ["background"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title(f"Finetuned yolo11n, test split, conf >= {REPORT_CONF}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------ decision


def decision_inputs(preds: list[dict], gts: list[tuple], paths: list[Path]) -> list[dict]:
    """One row per test image that contains a labelled person: the gate's unit of work."""
    rows = []
    for pred, (gt_boxes, gt_classes), path in zip(preds, gts, paths):
        if not (gt_classes == HUMAN).any():
            continue
        rows.append({
            "image": path.name,
            "violation": bool(is_violation(gt_boxes, gt_classes)),
            "score": violation_score(pred["boxes"], pred["classes"], pred["confs"]),
        })
    return rows


def policy_table(scores: np.ndarray, labels: np.ndarray, costs: Costs, ops: Operations, single_t: float, band: dict) -> list[dict]:
    """The five ways to run the gate, priced at the same base rate."""
    policies = [
        ("Do nothing (pass everyone)", 1.01, 1.01),
        ("Person checks everyone", 0.0, 1.01),
        ("Stop everyone", 0.0, 0.0),
        (f"Model alone (t = {single_t:.2f})", single_t, single_t),
        (f"Model + review ({band['t_pass']:.2f} to {band['t_fail']:.2f})", band["t_pass"], band["t_fail"]),
    ]
    table = []
    for name, t_pass, t_fail in policies:
        rates = outcome_rates(scores, labels, t_pass, t_fail)
        share = review_share(rates, costs)
        table.append({
            "policy": name,
            "cost_per_1000": 1000 * cost_per_case(rates, costs),
            "missed_violations_per_1000": 1000 * costs.base_rate * (rates["pos"]["pass"] + rates["pos"]["review"] * costs.reviewer_miss),
            "review_share": share,
            **{k: v for k, v in staffing(share, ops).items() if k in ("reviews_per_day", "reviewers_at_peak")},
        })
    return table


def sensitivity(scores: np.ndarray, labels: np.ndarray, base: Costs) -> dict:
    """How the chosen threshold moves when the two least certain assumptions move."""
    out = {"base_rate": [], "fn_cost": []}
    for rate in (0.01, 0.02, 0.05, 0.10, 0.18):
        c = Costs(**{**base.as_dict(), "base_rate": rate})
        rows = sweep(scores, labels, c, GRID)
        best = min(rows, key=lambda r: r["cost_per_1000"])
        band = best_band(scores, labels, c, GRID)
        out["base_rate"].append({"value": rate, "threshold": best["threshold"], "cost_per_1000": best["cost_per_1000"], **{k: band[k] for k in ("t_pass", "t_fail", "review_share")}})
    for fn in (25, 300, 1000, 2400, 10000):
        c = Costs(**{**base.as_dict(), "fn": fn})
        rows = sweep(scores, labels, c, GRID)
        best = min(rows, key=lambda r: r["cost_per_1000"])
        band = best_band(scores, labels, c, GRID)
        out["fn_cost"].append({"value": fn, "threshold": best["threshold"], "cost_per_1000": best["cost_per_1000"], **{k: band[k] for k in ("t_pass", "t_fail", "review_share")}})
    return out


def plot_sweep(rows: list[dict], chosen: float, band: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["threshold"] for r in rows], [r["cost_per_1000"] for r in rows], label="Model alone")
    best = min(rows, key=lambda r: r["cost_per_1000"])
    ax.axvline(chosen, color="C3", ls="--", label=f"Chosen t = {chosen:.2f} (EUR {best['cost_per_1000']:.0f})")
    ax.axhline(band["cost_per_1000"], color="C2", ls=":", label=f"Best review band (EUR {band['cost_per_1000']:.0f})")
    ax.set_xlabel("Fail threshold on violation score")
    ax.set_ylabel("Expected cost per 1,000 entries (EUR)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def main() -> None:
    download()
    data_yaml, split_stats = leak_free_split()
    composition = describe({
        "train (leak-free)": [Path(p) for p in (SPLIT_DIR / "train.txt").read_text().split("\n") if p],
        "valid (leak-free)": [Path(p) for p in (SPLIT_DIR / "valid.txt").read_text().split("\n") if p],
        "test": images("test"),
    })

    if not SKIP_TRAIN:
        train(data_yaml)

    test_paths = images("test")
    gts = [ground_truth(p) for p in test_paths]
    finetuned_preds = predict(YOLO(str(WEIGHTS)), test_paths)
    candidates = {
        "finetuned yolo11n": finetuned_preds,
        "COCO yolo11n zero-shot": predict(YOLO(BASE_MODEL), test_paths, class_map={COCO_PERSON: HUMAN}),
        "predict nothing": predict(None, test_paths),
    }
    detection = {name: detection_metrics(preds, gts) for name, preds in candidates.items()}
    for name, m in detection.items():
        print(f"{name:24s} mAP50 {m['map50']:.3f}  mAP50-95 {m['map50_95']:.3f}")

    ARTIFACTS.mkdir(exist_ok=True)
    matrix = confusion_matrix(finetuned_preds, gts)
    plot_confusion(matrix, ARTIFACTS / "confusion_matrix.png")

    costs, ops = Costs(), Operations()
    decisions = {name: decision_inputs(preds, gts, test_paths) for name, preds in candidates.items()}
    rows = decisions["finetuned yolo11n"]
    scores = np.array([r["score"] for r in rows])
    labels = np.array([r["violation"] for r in rows])

    cost_curve = sweep(scores, labels, costs, GRID)
    chosen = min(cost_curve, key=lambda r: r["cost_per_1000"])
    band = best_band(scores, labels, costs, GRID)
    band_rates = outcome_rates(scores, labels, band["t_pass"], band["t_fail"])
    plot_sweep(cost_curve, chosen["threshold"], band, ARTIFACTS / "cost_curve.png")

    baseline_decisions = {}
    for name, r in decisions.items():
        s, y = np.array([x["score"] for x in r]), np.array([x["violation"] for x in r])
        best = min(sweep(s, y, costs, GRID), key=lambda x: x["cost_per_1000"])
        baseline_decisions[name] = {"best_threshold": best["threshold"], "cost_per_1000": best["cost_per_1000"], "recall": best["recall"], "false_positive_rate": best["false_positive_rate"]}

    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "splits": split_stats, "composition": composition},
        "training": trained_settings(),
        "detection": detection,
        "confusion_matrix": {"labels": CLASS_NAMES + ["background"], "rows_predicted_cols_true": matrix.tolist(), "conf": REPORT_CONF},
        "costs": costs.as_dict(),
        "operations": ops.as_dict(),
        "decision": {
            "test_images_with_people": len(rows),
            "test_violations": int(labels.sum()),
            "single_threshold": chosen,
            "band": band,
            "band_confusion": confusion_counts(scores, labels, band["t_pass"], band["t_fail"]),
            "band_rates": band_rates,
            "staffing": staffing(band["review_share"], ops),
            "policies": policy_table(scores, labels, costs, ops, chosen["threshold"], band),
            "baselines": baseline_decisions,
            "sensitivity": sensitivity(scores, labels, costs),
        },
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    with (ARTIFACTS / "cost_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cost_curve[0]))
        writer.writeheader()
        writer.writerows(cost_curve)
    with (ARTIFACTS / "test_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "violation", "score"])
        writer.writeheader()
        writer.writerows(rows)

    # A few test images (CC BY 4.0) so the dashboard has something to try
    # without the gitignored dataset: the two clearest violations and two
    # clearest compliant entries, plus the most borderline case.
    examples = ARTIFACTS / "examples"
    shutil.rmtree(examples, ignore_errors=True)
    examples.mkdir()
    by_score = sorted(rows, key=lambda r: r["score"])
    violations = [r for r in by_score if r["violation"]][-2:]
    compliant = [r for r in by_score if not r["violation"]][:2]
    borderline = [min(rows, key=lambda r: abs(r["score"] - (band["t_pass"] + band["t_fail"]) / 2))]
    for r in violations + compliant + borderline:
        shutil.copy(DATA_DIR / "test" / "images" / r["image"], examples / r["image"])
    print(json.dumps(results["decision"]["policies"], indent=2))
    print(f"Chosen single threshold {chosen['threshold']:.2f}, band {band['t_pass']:.2f}-{band['t_fail']:.2f}")


if __name__ == "__main__":
    main()
