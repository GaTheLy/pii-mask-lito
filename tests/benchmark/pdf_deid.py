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


def _page_text(path: str, engine) -> str:
    """Everything legible on the rendered page, as a recipient sees it."""
    pages = ocr.read_pages(engine, pdf.page_images(path))
    return " ".join(token.text for page in pages for token in page)


def score_file(src: Path, values: list[str], measure_engine,
               keep: Path | None = None, ocr_name: str = "auto",
               **policy) -> dict:
    out = Path(tempfile.mkdtemp()) / src.name
    try:
        # The engine under test has to do the *masking*, not just the reading.
        # It did not, for several rounds of engine comparisons: --ocr only
        # reached the read-back below, so every engine scored the same masking
        # run and the numbers compared readers, not maskers.
        # loop_ocr="same" keeps the comparison even-handed: every engine runs
        # its own recheck loop. The default "auto" would downgrade paddle's loop
        # to tesseract but let onnx/tesseract loop on themselves, which scores
        # different pipelines rather than different engines.
        mask(str(src), str(out), detector=Detector(**policy),
             registry=TagRegistry(), verify=False, ocr_engine=ocr_name,
             loop_ocr="same")
    except Exception as exc:  # noqa: BLE001
        return {"file": src.name, "error": str(exc)[:120]}

    before_raw = _page_text(str(src), measure_engine)
    before = _norm(before_raw)
    after_raw = _page_text(str(out), measure_engine)
    after = _norm(_TAG.sub(" ", after_raw))

    leaked, missing_from_source = [], []
    for value in values:
        needle = _norm(value)
        if not needle:
            continue
        if needle not in before:
            # OCR could not read it in the ORIGINAL either. Counting it as
            # masked would flatter the tool; counting it as leaked would blame
            # the tool for Tesseract. It is reported separately.
            missing_from_source.append(value)
            continue
        if needle in after:
            leaked.append(value)

    checked = len(values) - len(missing_from_source)
    # Over-masking: words the page lost that were never PHI.
    phi_chars = "".join(_norm(v) for v in values)
    source_words = re.findall(r"[A-Za-z]{4,}", before_raw)
    lost = [w for w in source_words
            if _norm(w) not in after and _norm(w) not in phi_chars]
    total_words = len(source_words) or 1

    if keep:
        keep.mkdir(parents=True, exist_ok=True)
        Path(out).replace(keep / src.name)
    return {
        "file": src.name,
        "phi": checked,
        "leaked": len(leaked),
        "recall": round(1 - len(leaked) / checked, 4) if checked else 1.0,
        "unreadable_in_source": len(missing_from_source),
        "over_masked": round(len(lost) / total_words, 4),
        "examples": leaked[:6],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pdf_deid")
    ap.add_argument("dataset", help="checkout of JohnSnowLabs/pdf-deid-dataset")
    ap.add_argument("--level", default="Easy", choices=sorted(LEVELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", default=None, help="directory to keep masked output in")
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
    ap.add_argument("--mask-organizations", action=argparse.BooleanOptionalAction,
                    default=None)
    ap.add_argument("--min-masked-age", type=int, default=None)
    a = ap.parse_args(argv)

    root = Path(a.dataset)
    truth = json.loads((root / "Mapping" / "all_phi" / LEVELS[a.level]).read_text())
    measure_engine = ocr.get(a.measure_ocr)
    level_dir = root / "PDF_Original" / a.level
    names = [name for name in sorted(truth) if (level_dir / name).exists()]
    names = names[: a.limit or None]

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
