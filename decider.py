"""Turn an anomaly score (and the image) into an action for the arm.

    from decider import get_decider
    decider = get_decider("rule")              # or "vlm", "hybrid"
    action, confidence = decider.decide(image, score)
    decider.last_reason                        # human-readable why, for the log

Every Decider returns an action from settings.ACTIONS and a confidence in
[0.5, 1]. Swap implementations without touching sorter.py:

  RuleDecider    threshold on the score (default; microseconds)
  VLMDecider     a vision-language model on OpenVINO GenAI picks the action
                 from settings.ACTIONS given the image and the score
  HybridDecider  rule when the score is far from the threshold, VLM only for
                 the uncertain band -> VLM latency only on hard items

Confidence for the rule: 0.5 at the threshold, 0.75 at UNCERTAIN_MARGIN away,
1.0 at twice the margin. `is_uncertain(conf)` is True below 0.75; sorter.py
uses it to grab extra views before committing.
"""

import argparse
import re
from abc import ABC, abstractmethod

import numpy as np

import settings

UNCERTAIN_CONFIDENCE = 0.75


def is_uncertain(confidence: float) -> bool:
    return confidence < UNCERTAIN_CONFIDENCE


class Decider(ABC):
    last_reason: str = ""

    @abstractmethod
    def decide(self, image, score: float) -> tuple[str, float]:
        """Return (action, confidence). action is one of settings.ACTIONS."""


class RuleDecider(Decider):
    def __init__(self, threshold=None, margin=None):
        self.threshold = settings.THRESHOLD if threshold is None else threshold
        self.margin = settings.UNCERTAIN_MARGIN if margin is None else margin

    def decide(self, image, score):
        verdict = "defect" if score >= self.threshold else "good"
        distance = abs(score - self.threshold)
        confidence = 0.5 + 0.5 * min(1.0, distance / (2 * self.margin))
        self.last_reason = f"score {score:.3f} {'>=' if verdict == 'defect' else '<'} threshold {self.threshold}"
        return settings.ACTION_FOR[verdict], round(confidence, 3)


class VLMDecider(Decider):
    """Vision-language model via OpenVINO GenAI. NOT YET TESTED end to end.

    Setup (once, before the event if possible - models are 1-5 GB):
        pip install openvino-genai==<same minor as openvino>
        optimum-cli export openvino --model <hf-vlm-id> --weight-format int4 models/vlm
      or download a pre-converted OpenVINO IR from huggingface.co/OpenVINO.
    Run it on GPU so the NPU/CPU stay free for the anomaly model.
    """

    def __init__(self, model_dir="models/vlm", device="GPU", max_new_tokens=24, fallback=None):
        import openvino_genai  # lazy: only needed when this decider is used

        self.genai = openvino_genai
        props = {"CACHE_DIR": str(settings.CACHE_DIR / "vlm")} if device == "GPU" else {}
        self.pipe = openvino_genai.VLMPipeline(str(model_dir), device, **props)
        self.config = openvino_genai.GenerationConfig()
        self.config.max_new_tokens = max_new_tokens
        self.fallback = fallback or RuleDecider()

    def prompt(self, score):
        actions = ", ".join(settings.ACTIONS)
        return (
            "You are a quality inspector on a sorting line. An anomaly detector scored this item "
            f"{score:.2f} (0 = normal, 1 = defective, threshold {settings.THRESHOLD}). "
            f"Look at the item and choose exactly one action from: {actions}. "
            "Reply in the form: ACTION: <action>; CONFIDENCE: <0-1>; REASON: <few words>"
        )

    def decide(self, image, score):
        from openvino import Tensor
        from detector import load_image

        rgb = np.asarray(load_image(image), dtype=np.uint8)
        text = self.pipe.generate(self.prompt(score), images=[Tensor(rgb)], generation_config=self.config).texts[0]
        action, confidence = self.parse(text)
        if action is None:  # unparseable answer: never let the VLM stall the line
            action, confidence = self.fallback.decide(image, score)
            self.last_reason = f"VLM unparseable ({text!r}); {self.fallback.last_reason}"
        else:
            self.last_reason = f"VLM: {text.strip()}"
        return action, confidence

    @staticmethod
    def parse(text):
        lowered = text.lower()
        action = next((a for a in settings.ACTIONS if re.search(rf"\b{re.escape(a.lower())}\b", lowered)), None)
        m = re.search(r"confidence\W+([01](?:\.\d+)?)", lowered)
        confidence = float(m.group(1)) if m else 0.75
        return action, round(min(1.0, max(0.5, confidence)), 3)


class HybridDecider(Decider):
    """Rule for confident scores; consult the VLM only inside the uncertain band."""

    def __init__(self, vlm: Decider, rule: Decider | None = None):
        self.rule, self.vlm = rule or RuleDecider(), vlm

    def decide(self, image, score):
        action, confidence = self.rule.decide(image, score)
        if not is_uncertain(confidence):
            self.last_reason = self.rule.last_reason
            return action, confidence
        action, confidence = self.vlm.decide(image, score)
        self.last_reason = f"uncertain -> {self.vlm.last_reason}"
        return action, confidence


def get_decider(kind="rule", **kw) -> Decider:
    if kind == "rule":
        return RuleDecider(**kw)
    if kind == "vlm":
        return VLMDecider(**kw)
    if kind == "hybrid":
        return HybridDecider(VLMDecider(**kw))
    raise ValueError(f"unknown decider {kind!r}")


def main():
    ap = argparse.ArgumentParser(description="Show RuleDecider output for a sweep of scores.")
    ap.add_argument("scores", nargs="*", type=float, default=[0.0, 0.3, 0.42, 0.5, 0.55, 0.7, 1.0])
    args = ap.parse_args()
    d = RuleDecider()
    for s in args.scores:
        action, conf = d.decide(None, s)
        print(f"score={s:.2f} -> {action:7} confidence={conf:.3f} uncertain={is_uncertain(conf)}  ({d.last_reason})")
    assert VLMDecider.parse("ACTION: reject; CONFIDENCE: 0.9; REASON: crack") == ("reject", 0.9)
    assert VLMDecider.parse("I would box it")[0] == "box"
    print("parser self-test ok")


if __name__ == "__main__":
    main()
