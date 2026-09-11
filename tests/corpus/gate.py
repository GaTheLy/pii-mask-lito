"""Fail the build when masking quality regresses.

The project rule is that a detection change must be justified by a corpus
score. That rule only holds if something checks it, and a number pasted into a
pull request is not something that checks it.

Thresholds are floors, not targets, and sit below the checked-in corpus's
current result to tolerate small OCR variation. Raise them when repeatable
measurements justify it.

Run: `python -m tests.corpus.gate`
"""

from __future__ import annotations

import sys

from .score import (mean_map_precision, mean_map_recall, mean_precision,
                     mean_recall, score_corpus)

# Two floors, answering two different questions.
#
# mAP is the object-detection score: is each identifier *detected* (a mask
# agrees with it at IoU >= 0.5) and is each mask a real detection. It is the
# stable headline, because it does not move when two OCR engines disagree on
# box shape by convention rather than correctness.
#
# area recall is the privacy floor: how much of the identifier's *ink* is
# covered. Detection at IoU 0.5 is not enough for this tool -- a value that is
# 50% covered is 50% printable -- so coverage is floored separately and much
# higher. The per-identifier version of it is the under_masked list.
MIN_MEAN_MAP_RECALL = 0.90
MIN_MEAN_MAP_PRECISION = 0.80
MIN_MEAN_AREA_RECALL = 0.88
# No single document may collapse while the mean stays up. The worst document
# in the corpus may not collapse while the mean stays high.
MIN_DOCUMENT_MAP_RECALL = 0.80
# An identifier printed partly in the clear is a leak; there is no floor low
# enough to be acceptable, so any under-masked value fails the gate outright.
MIN_COVERAGE = 0.90


def main() -> int:
    results = score_corpus()
    failures = []

    for row in results:
        if "error" in row:
            failures.append(f"{row['document']}: ERROR {row['error'][:80]}")
            continue
        if row["map_recall"] < MIN_DOCUMENT_MAP_RECALL:
            failures.append(
                f"{row['document']}: mAP recall {row['map_recall']:.3f} "
                f"< {MIN_DOCUMENT_MAP_RECALL} -- {', '.join(row['missed'][:3])}"
            )
        for value in row["under_masked"]:
            failures.append(
                f"{row['document']}: {value} is under {MIN_COVERAGE:.0%} covered"
            )

    map_r, map_p = mean_map_recall(results), mean_map_precision(results)
    area_r, area_p = mean_recall(results), mean_precision(results)
    if map_r < MIN_MEAN_MAP_RECALL:
        failures.append(f"mean mAP recall {map_r:.4f} < {MIN_MEAN_MAP_RECALL}")
    if map_p < MIN_MEAN_MAP_PRECISION:
        failures.append(f"mean mAP precision {map_p:.4f} < {MIN_MEAN_MAP_PRECISION}")
    if area_r < MIN_MEAN_AREA_RECALL:
        failures.append(f"mean area recall {area_r:.4f} < {MIN_MEAN_AREA_RECALL}")

    print(f"mAP   recall {map_r:.4f} precision {map_p:.4f}")
    print(f"area  recall {area_r:.4f} precision {area_p:.4f}")
    if not failures:
        print("quality gates passed")
        return 0
    print("\nQUALITY GATES FAILED", file=sys.stderr)
    for line in failures:
        print(f"  - {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
