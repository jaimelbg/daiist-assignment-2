"""
Decision layer shared by train.py and app.py.

The detector finds boxes. The business decision is about people: is somebody
walking through the site gate without a helmet? This module turns boxes into a
per-image violation score, and turns scores into pass / review / fail and into
euros. Keeping it in one place means the dashboard can never cost things
differently from the pipeline that chose the thresholds.
"""

from dataclasses import asdict, dataclass

import numpy as np

CLASS_NAMES = ["boots", "gloves", "helmet", "human", "vest"]
HELMET, HUMAN = CLASS_NAMES.index("helmet"), CLASS_NAMES.index("human")

# A detected person only counts if the detector is reasonably sure it is a
# person. Held fixed so that the sweep is over one thing: helmet confidence.
HUMAN_CONF = 0.25

# Head region of a person box, as fractions of its height measured from the
# top edge. A helmet whose centre falls in here belongs to that person.
HEAD_TOP, HEAD_BOTTOM = -0.10, 0.35


@dataclass
class Costs:
    """Euro cost of each outcome for one gate entry. See REPORT.md for sources."""

    fn: float = 2400.0  # unprotected worker let on site: a LISOS 'falta grave' fine
    fp: float = 2.17  # compliant worker stopped at the gate: 10 min at EUR 13/h
    tp: float = 2.17  # non-compliant worker stopped: same stop, but warranted
    tn: float = 0.0
    review: float = 0.07  # 20 s of a safety coordinator at EUR 13/h
    reviewer_miss: float = 0.05  # reviewer passes a real violation
    reviewer_false_alarm: float = 0.02  # reviewer stops a compliant worker
    base_rate: float = 0.05  # share of gate entries with no helmet

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Operations:
    """Volume assumptions for the staffing calculation."""

    entries_per_day: int = 3000  # 10 sites x ~150 workers x 2 entries
    seconds_per_review: float = 20.0
    peak_share: float = 0.5  # half of all entries happen in the shift-start rush
    peak_minutes: float = 30.0
    reviewer_hours_per_day: float = 7.5

    def as_dict(self) -> dict:
        return asdict(self)


def _in_head(helmet_xyxy: np.ndarray, person_xyxy: np.ndarray) -> np.ndarray:
    """Which helmets have their centre inside this person's head region."""
    cx = (helmet_xyxy[:, 0] + helmet_xyxy[:, 2]) / 2
    cy = (helmet_xyxy[:, 1] + helmet_xyxy[:, 3]) / 2
    x1, y1, x2, y2 = person_xyxy
    h = y2 - y1
    return (cx >= x1) & (cx <= x2) & (cy >= y1 + HEAD_TOP * h) & (cy <= y1 + HEAD_BOTTOM * h)


def violation_score(boxes: np.ndarray, classes: np.ndarray, confs: np.ndarray) -> float:
    """Probability-like score that someone in the image has no helmet.

    For each detected person: 1 - (best helmet confidence in their head
    region). The image takes the worst person. If no person is detected at all
    there is nobody to clear, so the image scores 1.0 and cannot be passed
    automatically.
    """
    boxes, classes, confs = np.asarray(boxes, float), np.asarray(classes), np.asarray(confs, float)
    people = boxes[(classes == HUMAN) & (confs >= HUMAN_CONF)]
    if len(people) == 0:
        return 1.0
    helmet_boxes = boxes[classes == HELMET]
    helmet_confs = confs[classes == HELMET]
    worst = 0.0
    for person in people:
        if len(helmet_boxes):
            mask = _in_head(helmet_boxes, person)
            best = helmet_confs[mask].max() if mask.any() else 0.0
        else:
            best = 0.0
        worst = max(worst, 1.0 - float(best))
    return worst


def is_violation(boxes: np.ndarray, classes: np.ndarray) -> bool:
    """Ground truth: some labelled person has no labelled helmet on their head."""
    boxes, classes = np.asarray(boxes, float), np.asarray(classes)
    helmets = boxes[classes == HELMET]
    for person in boxes[classes == HUMAN]:
        if len(helmets) == 0 or not _in_head(helmets, person).any():
            return True
    return False


def outcome_rates(scores: np.ndarray, labels: np.ndarray, t_pass: float, t_fail: float) -> dict:
    """Share of positives and negatives that end in pass / review / fail.

    score < t_pass -> pass, score >= t_fail -> fail, anything between -> review.
    With t_pass == t_fail there is no review band: a single-threshold model.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    t_pass = min(t_pass, t_fail)
    out = {}
    for name, group in (("pos", scores[labels]), ("neg", scores[~labels])):
        n = max(len(group), 1)
        fail = (group >= t_fail).sum() / n
        passed = (group < t_pass).sum() / n
        out[name] = {"pass": passed, "review": 1.0 - fail - passed, "fail": fail}
    return out


def cost_per_case(rates: dict, c: Costs) -> float:
    """Expected euros per gate entry at the assumed base rate, not the test split's."""
    pos, neg = rates["pos"], rates["neg"]
    review_pos = c.review + c.reviewer_miss * c.fn + (1 - c.reviewer_miss) * c.tp
    review_neg = c.review + c.reviewer_false_alarm * c.fp + (1 - c.reviewer_false_alarm) * c.tn
    cost_pos = pos["pass"] * c.fn + pos["fail"] * c.tp + pos["review"] * review_pos
    cost_neg = neg["pass"] * c.tn + neg["fail"] * c.fp + neg["review"] * review_neg
    return c.base_rate * cost_pos + (1 - c.base_rate) * cost_neg


def review_share(rates: dict, c: Costs) -> float:
    return c.base_rate * rates["pos"]["review"] + (1 - c.base_rate) * rates["neg"]["review"]


def confusion_counts(scores: np.ndarray, labels: np.ndarray, t_pass: float, t_fail: float) -> dict:
    """Raw counts on the test images, rows = truth, columns = pass/review/fail."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    t_pass = min(t_pass, t_fail)
    table = {}
    for name, group in (("violation", scores[labels]), ("compliant", scores[~labels])):
        table[name] = {
            "pass": int((group < t_pass).sum()),
            "review": int(((group >= t_pass) & (group < t_fail)).sum()),
            "fail": int((group >= t_fail).sum()),
        }
    return table


def sweep(scores: np.ndarray, labels: np.ndarray, c: Costs, grid: np.ndarray) -> list[dict]:
    """Single-threshold sweep: fail if score >= t, pass otherwise."""
    rows = []
    for t in grid:
        r = outcome_rates(scores, labels, t, t)
        rows.append({
            "threshold": float(t),
            "recall": r["pos"]["fail"],
            "false_positive_rate": r["neg"]["fail"],
            "cost_per_1000": 1000 * cost_per_case(r, c),
        })
    return rows


def best_band(scores: np.ndarray, labels: np.ndarray, c: Costs, grid: np.ndarray) -> dict:
    """Cheapest (t_pass, t_fail) pair over the grid, review cost included."""
    best = None
    for i, t_pass in enumerate(grid):
        for t_fail in grid[i:]:
            r = outcome_rates(scores, labels, t_pass, t_fail)
            cost = cost_per_case(r, c)
            if best is None or cost < best["cost_per_1000"] / 1000 - 1e-12:
                best = {
                    "t_pass": float(t_pass),
                    "t_fail": float(t_fail),
                    "cost_per_1000": 1000 * cost,
                    "review_share": review_share(r, c),
                }
    return best


def staffing(review_fraction: float, ops: Operations) -> dict:
    """How many reviewers the review band needs, on average and in the rush."""
    per_day = review_fraction * ops.entries_per_day
    hours = per_day * ops.seconds_per_review / 3600
    peak_cases = per_day * ops.peak_share
    peak_people = peak_cases * ops.seconds_per_review / (ops.peak_minutes * 60)
    return {
        "reviews_per_day": per_day,
        "review_hours_per_day": hours,
        "reviewers_average": hours / ops.reviewer_hours_per_day,
        "reviewers_at_peak": peak_people,
    }
