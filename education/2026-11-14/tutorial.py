"""Deterministic score-semantics arithmetic; no data download or model fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


USER = [1.0, 2.0]
ITEMS = {"A": [0.4, 0.2], "B": [-0.1, 0.2], "C": [0.2, -0.2]}
SCALE, SHIFT = 10.0, 7.0


def sigmoid(value: float) -> float:
    """Stable logistic function. Bounded output alone does not show calibration."""
    if not math.isfinite(value):
        raise ValueError("a finite score is required")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def rank(scores: dict[str, float]) -> list[str]:
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("a nonempty collection of finite scores is required")
    return sorted(scores, key=lambda item: (-scores[item], item))


def calculate() -> dict:
    scores = {item: math.fsum(a * b for a, b in zip(USER, vector, strict=True))
              for item, vector in ITEMS.items()}
    transformed = {item: SCALE * value + SHIFT for item, value in scores.items()}
    shifted = {item: value + SHIFT for item, value in scores.items()}
    margin = scores["A"] - scores["B"]
    affine_margin = transformed["A"] - transformed["B"]
    return {
        "kind": "authored synthetic arithmetic; no fitted model or outcome labels",
        "inputs": {"user": USER, "items": ITEMS},
        "transform": {"scale": SCALE, "shift": SHIFT, "operation": "10*s+7"},
        "rows": [{"item": item, "item_vector": ITEMS[item], "score": value,
                  "transformed_score": transformed[item],
                  "single_item_sigmoid": sigmoid(value),
                  "transformed_single_item_sigmoid": sigmoid(transformed[item])}
                 for item, value in scores.items()],
        "rank_original": rank(scores), "rank_transformed": rank(transformed),
        "rank_sigmoid": rank({item: sigmoid(v) for item, v in scores.items()}),
        "pair_A_over_B": {
            "margin": margin, "sigmoid_margin": sigmoid(margin),
            "transformed_margin": affine_margin,
            "transformed_sigmoid_margin": sigmoid(affine_margin),
            "shift_only_margin": shifted["A"] - shifted["B"],
            "negative_log_sigmoid": -math.log(sigmoid(margin)),
            "transformed_negative_log_sigmoid": -math.log(sigmoid(affine_margin)),
            "interpretation": "BPR pairwise model illustration, not click probability"
        },
        "limitations": [
            "No observed event frequencies; calibration is unmeasured.",
            "Fixed user and candidate set; no inference about another model or user.",
            "Positive rescaling preserves this ordering, not the BPR training objective.",
            "Sigmoid saturation can merge distinct extreme floating-point scores."
        ],
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "dependencies": "standard library only", "seed": None,
                        "model_fits": 0},
        "sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ["tutorial.py", "PROTOCOL.md"]}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calculate()
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
