"""What each tuning constant actually buys.

Every constant in this project was set by one measured failure and then left
alone. That makes each of them individually defensible and collectively
unknown: nobody can say which are sitting on a plateau, where a change costs
nothing, and which are on a cliff, where the next document that differs
slightly from the one they were tuned against falls off.

This answers that by moving one constant at a time and re-scoring the whole
corpus. A flat column is a constant nobody needs to be careful about. A column
that drops sharply on one side is where the fragility lives, and is what should
be written down next to the number.

Run: `python -m tests.corpus.sweep` (slow -- it masks the corpus once per
value), or `python -m tests.corpus.sweep above_gap` for one knob.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

from pii_mask_lito import detect, ocr

from .score import mean_precision, mean_recall, score_corpus


@contextmanager
def _patched(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def _adjacency(**overrides):
    """A detector factory with one Adjacency field overridden.

    Adjacency is a dataclass, so its defaults are captured in the generated
    __init__ at class-creation time -- patching the class attribute does not
    reach new instances. Building the object explicitly does.
    """

    def make():
        detector = detect.Detector()
        detector.spatial.adj = detect.Adjacency(**overrides)
        detector.structural.spatial = detector.spatial
        return detector

    return make


# Each entry is (label, values, runner). The runner takes one value and returns
# the scored corpus.
def _module_knob(module, name):
    def run(value):
        with _patched(module, name, value):
            return score_corpus()

    return run


def _adjacency_knob(field):
    def run(value):
        return score_corpus(make_detector=_adjacency(**{field: value}))

    return run


def _dpi_knob():
    def run(value):
        return score_corpus(dpi=value)

    return run


# Ranges deliberately bracket the operating point on BOTH sides, including
# values low enough to break the thing. A first pass swept only 0.015-0.08 for
# above_gap and reported a dead-flat line, which reads as "this constant does
# not matter" -- when in fact the cliff is at 0.01 and recall falls off it by
# 35 points. A sweep that never leaves the plateau measures nothing.
KNOBS = {
    "above_gap": ([0.0, 0.005, 0.01, 0.025, 0.08], _adjacency_knob("above_gap")),
    "left_gap": ([0.0, 0.02, 0.05, 0.18, 0.40], _adjacency_knob("left_gap")),
    "window": ([1, 2, 4, 10, 20], _adjacency_knob("window")),
    "fuzzy_ratio": ([0.60, 0.75, 0.86, 0.95, 1.0], _module_knob(detect, "_FUZZY_RATIO")),
    "min_propagate": ([2, 4, 8, 12, 99], _module_knob(detect, "_MIN_PROPAGATE")),
    "low_confidence": ([0, 40, 70, 95], _module_knob(ocr, "LOW_CONFIDENCE")),
    "dpi": ([100, 150, 200, 300, 400], _dpi_knob()),
}


def sweep(name: str) -> None:
    values, run = KNOBS[name]
    print(f"\n{name}")
    print(f"  {'value':>10}{'recall':>10}{'precision':>12}")
    for value in values:
        results = run(value)
        print(f"  {value:>10}{mean_recall(results):>10.4f}{mean_precision(results):>12.4f}")


def main() -> None:
    names = sys.argv[1:] or list(KNOBS)
    unknown = [n for n in names if n not in KNOBS]
    if unknown:
        raise SystemExit(f"unknown knob(s) {unknown}; choose from {list(KNOBS)}")
    for name in names:
        sweep(name)


if __name__ == "__main__":
    main()
