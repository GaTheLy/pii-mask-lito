"""Character-area scoring against known ground truth.

Scored by *area covered*, not by span. Span-level scoring hides important
partial-coverage cases: a mask that covers four of a name's seven
letters scores as a hit, and a mask three times wider than the value it replaces
scores as a hit too. Area makes each visible as a number.

    recall     fraction of ground-truth identifier area that a mask covers
    precision  fraction of masked area that sits over a ground-truth identifier

Precision is the weaker of the two by design: a mask is drawn at the exact size
of an OCR token box, which is not the exact size of the printed glyphs, so
perfect precision is not achievable and not the goal. What matters is the
*curve* -- whether changing a constant moves either number, and by how much.

Run: `python -m tests.corpus.score`
"""

from __future__ import annotations

import json
from pathlib import Path

from pii_mask_lito.detect import Detector
from pii_mask_lito.pipeline import mask
from pii_mask_lito.registry import TagRegistry

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"

# Entities the ground truth records but which are provider-side or structural
# and deliberately never masked. Kept explicit so a change of policy shows up
# here as an edit rather than as a mysterious score movement.
IGNORED: set[str] = set()


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


# A ground-truth identifier counts as *detected* when a mask covers at least
# this share of its area, in the IoU sense. This is the mAP convention (0.5)
# and is deliberately a hit/miss rather than a continuous ratio: a mask that is
# the right shape but two lines tall still scores as a hit instead of half a
# hit. The continuous number -- how much of the identifier's ink is covered --
# is reported separately as recall, and is what the privacy gate watches.
_DETECT_IOU = 0.5


def score_document(src: Path, truth: list[list[dict]], make_detector=Detector, **kwargs) -> dict:
    """Mask one document and compare its mask boxes to the ground truth.

    `make_detector` is how sweep.py varies a constant that lives on the detector
    rather than in a module global.
    """
    dest = OUT / f"_scored_{src.name}"
    try:
        report = mask(str(src), str(dest), detector=make_detector(),
                      registry=TagRegistry(), verify=False, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"document": src.name, "error": str(exc)}

    masks: dict[int, list] = {}
    for finding in report.findings:
        if finding.bbox is not None:
            masks.setdefault(finding.page or 0, []).append(finding.bbox)

    covered = wanted = 0.0
    hit = drawn = 0.0
    missed = []
    under_masked = []
    detected_ids = 0
    total_ids = 0
    tp_masks = 0
    total_masks = 0
    for page, entries in enumerate(truth):
        boxes = masks.get(page, [])
        for entry in entries:
            if entry["entity"] in IGNORED:
                continue
            target = entry["bbox"]
            area = _area(target)
            wanted += area
            total_ids += 1
            # Union approximated by the sum of overlaps. Masks are drawn per
            # token and do not overlap each other, so the sum is the union;
            # clamped to the target in case OCR splits a word. A name masked
            # one box per word is the reason this is summed, not maxed: the
            # largest single overlap scores a three-word name at one third.
            got = min(sum(_overlap(target, b) for b in boxes), area)
            covered += got
            # Detection is mAP's question: is the identifier covered enough to
            # count as found, yes or no. 0.5 is the object-detection convention
            # and, unlike a single best-box IoU, it works for a name spread over
            # three word-masks, which no one box covers half of.
            if area and got / area >= _DETECT_IOU:
                detected_ids += 1
            else:
                missed.append(f"p{page + 1} {entry['entity']} {entry['value']}")
            # The privacy floor is tighter than the detection floor: a value
            # that is 60% covered still has 40% of itself printable. Non-textual
            # entities are exempt from *this* floor: a signature or a face is
            # not something a reader can extract a value from, so a sliver of
            # it left at 89% is not a leak, while a sliver of a name or an SSN
            # is. They are still held to the detection floor above.
            if area and got / area < 0.9 and entry["entity"] not in {"SIGNATURE", "FACE"}:
                under_masked.append(
                    f"p{page + 1} {entry['entity']} {entry['value']} "
                    f"({got / area:.0%})"
                )
        for box in boxes:
            box_area = _area(box)
            drawn += box_area
            total_masks += 1
            overlap = sum(_overlap(box, e["bbox"]) for e in entries
                          if e["entity"] not in IGNORED)
            hit += min(overlap, box_area)
            # Binary precision, mAP's question for the boxes we drew: is this
            # mask over a real identifier, or over paper? A mask over an SSN two
            # lines tall is a true positive, not half of one -- its shape is a
            # layout concern, reported separately by the continuous precision.
            if overlap > 0:
                tp_masks += 1

    return {
        "document": src.name,
        # continuous area: coverage (recall) and shape-sensitive precision
        "recall": round(covered / wanted, 4) if wanted else 1.0,
        "precision": round(hit / drawn, 4) if drawn else 1.0,
        # mAP-style hit/miss: the stable headline numbers
        "map_recall": round(detected_ids / total_ids, 4) if total_ids else 1.0,
        "map_precision": round(tp_masks / total_masks, 4) if total_masks else 1.0,
        "masks": len(report.findings),
        "missed": missed,
        "under_masked": under_masked,
    }


def score_corpus(**kwargs) -> list[dict]:
    """Score every document in the corpus. Shared with sweep.py."""
    if not (OUT / "index.json").exists():
        raise SystemExit("no corpus: run `python -m tests.corpus.build` first")
    results = []
    for name in json.loads((OUT / "index.json").read_text()):
        src = OUT / name
        truth = json.loads((OUT / f"{src.stem}.truth.json").read_text())
        results.append(score_document(src, truth, **kwargs))
    return results


def mean_recall(results: list[dict]) -> float:
    scored = [r for r in results if "recall" in r]
    return sum(r["recall"] for r in scored) / len(scored) if scored else 0.0


def mean_precision(results: list[dict]) -> float:
    scored = [r for r in results if "precision" in r]
    return sum(r["precision"] for r in scored) / len(scored) if scored else 0.0


def mean_map_recall(results: list[dict]) -> float:
    scored = [r for r in results if "map_recall" in r]
    return sum(r["map_recall"] for r in scored) / len(scored) if scored else 0.0


def mean_map_precision(results: list[dict]) -> float:
    scored = [r for r in results if "map_precision" in r]
    return sum(r["map_precision"] for r in scored) / len(scored) if scored else 0.0


def main() -> None:
    results = score_corpus()

    print(f"{'document':<26}{'mAP-R':>8}{'mAP-P':>8}{'area-R':>8}{'area-P':>8}  missed")
    for row in results:
        if "error" in row:
            print(f"{row['document']:<26}{'ERROR':>8}  {row['error'][:60]}")
            continue
        missed = ", ".join(row["missed"][:3]) or "-"
        print(f"{row['document']:<26}{row['map_recall']:>8.3f}{row['map_precision']:>8.3f}"
              f"{row['recall']:>8.3f}{row['precision']:>8.3f}  {missed}")

    scored = [r for r in results if "recall" in r]
    print(f"\nmAP    recall {mean_map_recall(results):.4f} "
          f"precision {mean_map_precision(results):.4f}")
    print(f"area   recall {mean_recall(results):.4f} "
          f"precision {mean_precision(results):.4f} over {len(scored)} documents")
    return results


if __name__ == "__main__":
    main()
