"""Benchmark text detection against Microsoft's MIT-licensed synthetic corpus.

The dataset is maintained separately at:
https://github.com/microsoft/presidio-research
The documented reproducible checkout is commit
f1deaaf3dfaf69f9a803d9ae72b185c752fa217d.

No dataset content is bundled here. Clone that repository separately, then run:

    python -m tests.benchmark.presidio_synth \
        /path/to/presidio-research/data/synth_dataset_v2.json --limit 200

The scorer maps the external label vocabulary onto this project's public entity
types and ignores only TITLE and NRP, which this project does not claim to mask.
It measures alphanumeric character coverage so harmless spaces and punctuation
inside a multi-token replacement do not count as leaked PII.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from pii_mask_lito.detect import Detector
from pii_mask_lito.model import TokenText

ENTITY_MAP = {
    "PERSON": "PERSON",
    "STREET_ADDRESS": "LOCATION",
    "GPE": "LOCATION",
    "ZIP_CODE": "LOCATION",
    "ORGANIZATION": "ORGANIZATION",
    "CREDIT_CARD": "CREDIT_CARD",
    "DATE_TIME": "DATE",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "AGE": "AGE",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "DOMAIN_NAME": "URL",
    "IBAN_CODE": "IBAN_CODE",
    "US_SSN": "US_SSN",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
}
IGNORED = {"TITLE", "NRP"}


def _positions(text: str, start: int, end: int) -> set[int]:
    """Alphanumeric character positions covered by a half-open span."""
    return {
        index
        for index in range(max(start, 0), min(end, len(text)))
        if text[index].isalnum()
    }


def score(rows: list[dict], show_values: bool = False,
          spacy_model: str = "en_core_web_lg", min_score: float = 0.4) -> dict:
    truth_chars: set[tuple[int, int]] = set()
    predicted_chars: set[tuple[int, int]] = set()
    truth_by_entity: Counter[str] = Counter()
    missed_by_entity: Counter[str] = Counter()
    missed_examples: dict[str, list[dict]] = {}

    shared = Detector(
        mask_organizations=True, spacy_model=spacy_model, min_score=min_score
    )
    analyzer = shared.analyzer
    for row_number, row in enumerate(rows):
        text = row["full_text"]
        detector = Detector(
            mask_organizations=True, spacy_model=spacy_model, min_score=min_score
        )
        detector._analyzer = analyzer
        predicted = detector.detect(TokenText.from_text(text))

        for span in predicted:
            predicted_chars.update(
                (row_number, position)
                for position in _positions(text, span.start, span.end)
            )

        for span in row.get("spans", []):
            entity = span.get("entity_type")
            if entity in IGNORED:
                continue
            value = span.get("entity_value", text[
                int(span["start_position"]):int(span["end_position"])
            ])
            if entity == "DATE_TIME" and re.fullmatch(r"\d{4}", value.strip()):
                # The public policy deliberately preserves bare years.
                continue
            mapped = ENTITY_MAP.get(entity)
            if mapped is None:
                raise ValueError(f"unmapped ground-truth entity: {entity}")
            positions = _positions(
                text,
                int(span["start_position"]),
                int(span["end_position"]),
            )
            truth_chars.update((row_number, position) for position in positions)
            truth_by_entity[mapped] += 1
            covered = sum(
                (row_number, position) in predicted_chars
                for position in positions
            )
            if positions and covered / len(positions) < 0.5:
                missed_by_entity[mapped] += 1
                examples = missed_examples.setdefault(mapped, [])
                if show_values and len(examples) < 3:
                    examples.append({
                        "value": value,
                        "text": text,
                    })

    overlap = truth_chars & predicted_chars
    return {
        "samples": len(rows),
        "truth_characters": len(truth_chars),
        "predicted_characters": len(predicted_chars),
        "recall": len(overlap) / len(truth_chars) if truth_chars else 1.0,
        "precision": len(overlap) / len(predicted_chars) if predicted_chars else 1.0,
        "truth_spans": sum(truth_by_entity.values()),
        "missed_spans": sum(missed_by_entity.values()),
        "truth_by_entity": dict(truth_by_entity.most_common()),
        "missed_by_entity": dict(missed_by_entity.most_common()),
        "missed_examples": missed_examples,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="presidio_synth")
    parser.add_argument("dataset", help="path to synth_dataset_v2.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spacy-model", default="en_core_web_lg")
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="print a few synthetic missed values and their source sentences",
    )
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    payload = dataset.read_bytes()
    rows = json.loads(payload)
    if args.limit and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    if not 0 <= args.min_score <= 1:
        parser.error("--min-score must be between 0 and 1")
    result = score(
        rows,
        show_values=args.show_values,
        spacy_model=args.spacy_model,
        min_score=args.min_score,
    )
    print(f"dataset sha256 {hashlib.sha256(payload).hexdigest()}")
    print(
        f"samples {result['samples']}  spans {result['truth_spans']}  "
        f"character recall {result['recall']:.4f}  "
        f"precision {result['precision']:.4f}"
    )
    print(
        f"spans below 50% coverage {result['missed_spans']}\n"
        f"truth by entity {result['truth_by_entity']}\n"
        f"missed by entity {result['missed_by_entity']}"
    )
    if args.show_values:
        for entity, examples in result["missed_examples"].items():
            for example in examples:
                print(f"{entity}: {example['value']!r} in {example['text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
