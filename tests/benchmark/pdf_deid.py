"""Benchmark against an external PDF de-identification dataset checkout.

The upstream project describes the corpus as synthetic:
https://github.com/JohnSnowLabs/pdf-deid-dataset

The checkout inspected during development did not include a license file. This
runner therefore consumes a separately obtained checkout but does not vendor,
redistribute, or make a licensing claim about it. Verify upstream provenance
and terms before use.

Why this matters as a second scoreboard: this project's own corpus is written
by the same hand as the detectors, so it can only ever confirm what its author
already thought of. This one was not, its layouts are different, and its dates
are day-first -- which is precisely the kind of assumption a home corpus cannot
falsify.

The ground truth is a list of PHI *strings* per file, with no coordinates. This
tool writes a rendered page plus a tag-only text layer, so pulling text out of
the output would report every value "gone" and prove nothing. The output is
therefore read back the way a recipient reads it: rendered and run through OCR.
A value counts as masked when OCR can no longer find it on the page.

    leak rate     PHI values still readable after masking / all PHI values
    over-masking  words readable before but not after, that are not PHI

Run: `python -m tests.benchmark.pdf_deid /path/to/pdf-deid-dataset --level Easy`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from pii_mask_lito import ocr, pdf
from pii_mask_lito.detect import Detector
from pii_mask_lito.pipeline import mask
from pii_mask_lito.registry import TagRegistry

LEVELS = {"Easy": "pdf_deid_gts_easy.json",
          "Medium": "pdf_deid_gts_medium.json",
          "Hard": "pdf_deid_gts_hard.json"}

# Tag text this tool writes onto the page. It is not a leak and not a word the
# document lost, so it is excluded from both numbers.
_TAG = re.compile(r"<[A-Z]+#\d+>")


def _norm(text: str) -> str:
    """Fold to the form OCR can actually be compared in.

    Case and punctuation both survive masking badly enough to produce false
    leaks: a phone number printed "(402) 738-5912" comes back from Tesseract as
    "402 738 5912" often enough that comparing raw strings measures OCR, not
    masking.
    """
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _readable_counts(text: str, values: list[str]) -> Counter[str]:
    """Count readable ground-truth occurrences without exceeding annotations.

    Ground truth repeats values that appear in multiple headers and footers.
    Counting every annotation as leaked when only one copy survives exaggerates
    failures; counting only unique values hides repeated failures. Bound the
    observed count by the number of annotated occurrences instead.
    """
    expected = Counter(_norm(value) for value in values if _norm(value))
    labels = {_norm(value): value for value in values if _norm(value)}
    normalized = _norm(text)

    def observed(value: str) -> int:
        if len(value) <= 3 and value.isalnum():
            # State codes and ages must not match inside ordinary words or
            # longer numbers (``OH`` in ``Alcohol``, or ``56`` in an SSN).
            raw = labels[value].strip()
            return len(re.findall(
                rf"(?<![A-Za-z0-9]){re.escape(raw)}(?![A-Za-z0-9])",
                text,
                re.I,
            ))
        return normalized.count(value)

    counts = Counter()
    for value, limit in expected.items():
        count = observed(value)
        if count:
            counts[value] = min(limit, count)
    return counts


def _page_text(path: str, engine) -> str:
    """Everything legible on the rendered page, as a recipient sees it."""
    chunks = []
    batch, numbers = [], []

    def read_batch() -> None:
        try:
            chunks.extend(
                token.text
                for page in ocr.read_pages(engine, batch, numbers)
                for token in page
            )
        finally:
            for image in batch:
                image.close()

    for page_number, image in enumerate(pdf.iter_page_images(path)):
        batch.append(image)
        numbers.append(page_number)
        if len(batch) == 4:
            read_batch()
            batch, numbers = [], []
    if batch:
        read_batch()
    return " ".join(chunks)


def score_file(src: Path, values: list[str], measure_engine,
               keep: Path | None = None, ocr_name: str = "auto",
               **policy) -> dict:
    with tempfile.TemporaryDirectory(prefix="pii-mask-benchmark-") as work:
        out = Path(work) / src.name
        try:
            # The selected engine must perform masking as well as measurement.
            # loop_ocr="same" keeps engine comparisons even-handed: each engine
            # runs its own recheck loop rather than silently substituting a
            # cheaper loop engine for only some candidates.
            mask(str(src), str(out), detector=Detector(**policy),
                 registry=TagRegistry(), verify=False, ocr_engine=ocr_name,
                 loop_ocr="same")
        except Exception as exc:  # noqa: BLE001
            return {"file": src.name, "error": str(exc)[:120]}

        before_raw = _page_text(str(src), measure_engine)
        before = _norm(before_raw)
        after_raw = _page_text(str(out), measure_engine)
        after = _norm(_TAG.sub(" ", after_raw))

        expected = Counter(_norm(value) for value in values if _norm(value))
        labels = {_norm(value): value for value in values if _norm(value)}
        before_counts = _readable_counts(before_raw, values)
        after_counts = _readable_counts(_TAG.sub(" ", after_raw), values)
        checked = sum(before_counts.values())
        unreadable = sum(expected.values()) - checked
        leaked_counts = Counter({
            value: min(count, after_counts[value])
            for value, count in before_counts.items()
            if after_counts[value]
        })
        leaked = sum(leaked_counts.values())
        # Over-masking: words the page lost that were never PHI.
        phi_chars = "".join(_norm(v) for v in values)
        source_words = re.findall(r"[A-Za-z]{4,}", before_raw)
        lost = [w for w in source_words
                if _norm(w) not in after and _norm(w) not in phi_chars]
        total_words = len(source_words) or 1

        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            out.replace(keep / src.name)
        return {
            "file": src.name,
            "phi": checked,
            "leaked": leaked,
            "recall": round(1 - leaked / checked, 4) if checked else 1.0,
            "unreadable_in_source": unreadable,
            "over_masked": round(len(lost) / total_words, 4),
            "examples": [labels[value] for value in leaked_counts][:6],
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pdf_deid")
    ap.add_argument("dataset", help="checkout of JohnSnowLabs/pdf-deid-dataset")
    ap.add_argument("--level", default="Easy", choices=sorted(LEVELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--file",
        action="append",
        default=[],
        help="score one exact PDF filename (repeatable; overrides --limit)",
    )
    ap.add_argument("--keep", default=None, help="directory to keep masked output in")
    ap.add_argument(
        "--show-values",
        action="store_true",
        help="print up to six synthetic values still readable in each output",
    )
    ap.add_argument("--ocr", default="auto", choices=["auto", "onnx", "doctr", "rapidocr", "paddle", "tesseract"],
                    help="which OCR engine does the MASKING")
    ap.add_argument("--measure-ocr", default="onnx", choices=["auto", "onnx", "doctr", "rapidocr", "paddle", "tesseract"],
                    help="which engine reads the result back (fixed across engines "
                         "so a weak masker cannot hide its own leaks)")
    # The dataset counts clinician identity, facility identity and every age as
    # PHI. Safe Harbor does not. Both numbers are worth reporting; neither is
    # the "real" one on its own.
    ap.add_argument("--profile", default="general",
                    choices=["general", "hipaa-safe-harbor"])
    ap.add_argument("--mask-providers", action=argparse.BooleanOptionalAction,
                    default=None)
    ap.add_argument(
        "--mask-organizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mask organizations counted by this dataset (default: enabled)",
    )
    ap.add_argument("--min-masked-age", type=int, default=None)
    a = ap.parse_args(argv)

    root = Path(a.dataset)
    truth = json.loads((root / "Mapping" / "all_phi" / LEVELS[a.level]).read_text())
    measure_engine = ocr.get(a.measure_ocr)
    level_dir = root / "PDF_Original" / a.level
    available_names = [name for name in sorted(truth) if (level_dir / name).exists()]
    if a.file:
        missing = [name for name in a.file if name not in available_names]
        if missing:
            ap.error(f"file not found in {a.level}: {', '.join(missing)}")
        names = list(dict.fromkeys(a.file))
    else:
        names = available_names[: a.limit or None]

    rows = []
    for index, name in enumerate(names, 1):
        src = level_dir / name
        row = score_file(src, truth[name], measure_engine,
                         keep=Path(a.keep) if a.keep else None,
                         ocr_name=a.ocr,
                         profile=a.profile,
                         mask_providers=a.mask_providers,
                         mask_organizations=a.mask_organizations,
                         min_masked_age=a.min_masked_age)
        rows.append(row)
        note = row.get("error") or (
            f"recall {row['recall']:.4f}  leaked {row['leaked']}/{row['phi']}"
            f"  over-mask {row['over_masked']:.3f}")
        print(f"[{index}/{len(names)}] {name:<42} {note}", flush=True)
        if a.show_values and row.get("examples"):
            print(f"       readable synthetic values: {row['examples']}", flush=True)

    scored = [r for r in rows if "recall" in r]
    if scored:
        recall = sum(r["recall"] for r in scored) / len(scored)
        leaked = sum(r["leaked"] for r in scored)
        phi = sum(r["phi"] for r in scored)
        over = sum(r["over_masked"] for r in scored) / len(scored)
        unread = sum(r["unreadable_in_source"] for r in scored)
        print(f"\n{a.level}: mean recall {recall:.4f}  "
              f"{leaked} leaked of {phi} readable PHI  "
              f"over-mask {over:.4f}  ({unread} PHI unreadable in source)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
