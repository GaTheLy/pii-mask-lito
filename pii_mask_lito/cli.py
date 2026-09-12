"""Command line entry point."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .detect import Detector, DetectorConfigurationError
from .model import TokenText, merge_spans
from .pipeline import MaskingError, mask
from .policy import PROFILES
from .registry import TagRegistry


def _probability(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1") from exc
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _safe_name(name: str, detector: Detector, registry: TagRegistry) -> str:
    """Mask known PII in an output filename.

    Document names often repeat a person's name or identifier. Propagation
    reuses values learned from document content so the destination filename
    does not disclose them.

    Underscores, hyphens and dots are separators in a filename, not part of the
    values, so they are spaced out before detection and restored afterwards.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    spaced = re.sub(r"[_\-.]+", " ", stem)
    tt = TokenText.from_text(spaced)
    spans = merge_spans(detector.detect(tt) + detector.propagate(tt))
    if not spans:
        return name
    parts, cursor = [], 0
    for span in spans:
        parts.append(spaced[cursor : span.start])
        parts.append(registry.tag(span.entity, span.text))
        cursor = span.end
    parts.append(spaced[cursor:])
    return re.sub(r"\s+", "_", "".join(parts).strip()) + suffix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii_mask_lito",
        description="Mask PII in documents while preserving format and layout.",
    )
    parser.add_argument("src", nargs="+", help="input file(s)")
    parser.add_argument("-o", "--out", required=True, help="output file or directory")
    parser.add_argument(
        "--profile",
        default="general",
        choices=sorted(PROFILES),
        help="masking policy profile (default: general)",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        default=None,
        metavar="ENTITY",
        help="override the selected profile with an explicit entity list",
    )
    parser.add_argument(
        "--ocr",
        default="auto",
        choices=["auto", "onnx", "doctr", "rapidocr", "paddle", "tesseract"],
        help="OCR engine for scanned PDFs and images. Default 'auto' selects "
             "the first installed engine from the documented preference order",
    )
    parser.add_argument(
        "--loop-ocr",
        default="auto",
        choices=["auto", "same", "onnx", "doctr", "rapidocr", "paddle", "tesseract"],
        help="engine for the recheck loop, which re-reads already-masked pages "
             "and accounts for most of the OCR a document costs. Default 'auto' "
             "drops to a fast engine when pages were read with a slow one; "
             "'same' keeps the primary throughout",
    )
    parser.add_argument("--dpi", type=int, default=300,
        help="raster DPI for OCR and PDF output (below 300 costs OCR accuracy)")
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_lg",
        help="installed spaCy model package or local model path used for NER "
             "(default: en_core_web_lg)",
    )
    parser.add_argument(
        "--min-score",
        type=_probability,
        default=0.4,
        help="minimum Presidio/NER confidence from 0 to 1 (default: 0.4; "
             "higher favors precision, lower favors recall)",
    )
    parser.add_argument("--report", help="write the masking report here (JSON)")
    parser.add_argument(
        "--report-values",
        action="store_true",
        help="include the original values in the report. WARNING: this makes the "
             "report itself a plaintext PII file; handle it like the source "
             "document, not like a log",
    )
    parser.add_argument(
        "--keep-filename",
        action="store_true",
        help="do not mask PII in the output filename; filenames can contain "
             "names or identifiers and are masked by default",
    )
    parser.add_argument(
        "--agents",
        nargs="?",
        const="gemma4:31b",
        metavar="MODEL",
        help="read the document with a VLM agent pipeline alongside the rules: "
             "classify, read fields and owners, adjudicate, then ground "
             "deterministically. MODEL picks the host as well as the model -- an "
             "Ollama tag stays local (gemma4:31b, qwen2.5vl:7b), while "
             "'gemini-3.8-flash', 'openrouter:MODEL', 'vllm:MODEL' or "
             "'openai-compatible:MODEL@https://host/v1' run it remotely. Hosted "
             "providers read the key from GEMINI_API_KEY / OPENAI_API_KEY / "
             "OPENROUTER_API_KEY and send page images off the machine; approve "
             "the provider and transfer before using sensitive documents.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="add an agent pass over the finished page looking for PII no "
             "detector proposed (requires --agents)",
    )
    parser.add_argument(
        "--ollama-host", default="http://localhost:11434",
        help="Ollama endpoint, for local --agents models"
    )
    parser.add_argument(
        "--long-tags",
        action="store_true",
        help="use full entity names (<MEDICAL_RECORD_NUMBER#0>) instead of short "
             "ones; tags may then not fit inside an exact-size mask",
    )
    parser.add_argument(
        "--mask-unread-ink",
        action="store_true",
        help="cover every region of ink that no OCR engine could read, instead "
             "of only flagging the page for review. This is the strict answer "
             "to handwriting and unreadable form fonts -- it masks what cannot "
             "be checked -- and it will also cover logos and stamps",
    )
    professional = parser.add_mutually_exclusive_group()
    professional.add_argument(
        "--mask-providers",
        dest="mask_providers",
        action="store_true",
        default=None,
        help="mask professional identities (already enabled by the general profile)",
    )
    professional.add_argument(
        "--preserve-professional-identities",
        dest="mask_providers",
        action="store_false",
        help="retain people explicitly identified as professionals",
    )
    parser.add_argument(
        "--mask-organizations",
        action="store_true",
        default=None,
        help="also mask organization names; disabled by default because an "
             "organization is not normally personal data",
    )
    parser.add_argument(
        "--min-masked-age",
        type=int,
        default=None,
        help="lowest age to mask (default comes from the selected profile)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the read-back verification pass (not recommended)",
    )
    return parser


def _unwritable(args, out: Path) -> list[str]:
    """Path problems worth refusing before an expensive run, as messages.

    Deliberately not exhaustive -- a disk can still fill up mid-write. It covers
    the mistakes an argument line actually makes: a destination directory that
    does not exist, and a report aimed inside one.
    """
    problems = []
    parent = out if out.is_dir() else out.parent
    if str(parent) and not parent.exists():
        problems.append("the --out parent directory does not exist")
    if args.report:
        report_dir = Path(args.report).parent
        if str(report_dir) and not report_dir.exists():
            if report_dir == out and not out.exists():
                problems.append(
                    "the --report path is inside a missing --out path. With one "
                    "input, --out is treated as a file, not a directory"
                )
            else:
                problems.append("the --report parent directory does not exist")
        elif report_dir.exists() and not report_dir.is_dir():
            problems.append("the --report parent path is not a directory")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if len(args.src) > 1 and not out.is_dir():
        print("error: --out must be a directory for multiple inputs", file=sys.stderr)
        return 2

    # Everything knowable about the paths is checked before any masking starts.
    # Masking a multi-page document can take minutes, and finding out afterwards
    # that the report has nowhere to go means paying for it twice -- and the
    # first run has already written its output by then, so the failure looks
    # like a crash on top of a success.
    for missing in _unwritable(args, out):
        print(f"error: {missing}", file=sys.stderr)
        return 2

    detector = Detector(entities=args.entities, spacy_model=args.spacy_model,
                        min_score=args.min_score,
                        mask_providers=args.mask_providers,
                        mask_organizations=args.mask_organizations,
                        min_masked_age=args.min_masked_age,
                        profile=args.profile)
    # One registry across every input, so the same person gets the same index in
    # a PDF and its companion spreadsheet.
    registry = TagRegistry(short=not args.long_tags)
    reviewer = None
    if args.agents:
        from . import agents

        reviewer = agents.build(
            detector, model_name=args.agents, host=args.ollama_host, audit=args.audit
        )

    reports, failed = [], 0
    rename = out.is_dir() and not args.keep_filename
    for position, src in enumerate(args.src):
        # When the name is ours to choose, write under a neutral one and rename
        # once the lexicon knows which person is sensitive. Going straight to the
        # masked name is not possible -- it is not known yet -- and going to the
        # source name first would put the PHI on disk, which is the leak being
        # closed. A name given explicitly with -o is the user's, and is kept.
        dest = out / (f"_masked_{position}{Path(src).suffix}" if rename
                      else Path(src).name) if out.is_dir() else out
        try:
            report = mask(
                src,
                str(dest),
                detector=detector,
                registry=registry,
                ocr_engine=args.ocr,
                dpi=args.dpi,
                verify=not args.no_verify,
                vlm=reviewer,
                loop_ocr=args.loop_ocr,
                mask_unread_ink=args.mask_unread_ink,
            )
        except (MaskingError, DetectorConfigurationError) as exc:
            print(f"FAIL input {position + 1}: {exc}", file=sys.stderr)
            failed += 1
            continue
        if rename:
            final = out / _safe_name(Path(src).name, detector, registry)
            dest.replace(final)
            report.output = dest = str(final)
        reports.append(report)
        status = "verified" if report.verified else "unverified"
        kind = f", {report.doc_type}" if report.doc_type else ""
        print(
            f"ok   input {position + 1}: output written "
            f"({len(report.findings)} masked, {status}{kind})"
        )
        if report.review:
            print(
                f"       review: {len(report.review)} item(s) require attention; "
                "details are omitted from ordinary logs"
            )
        for step in report.trace:
            detail = ", ".join(f"{k}={v}" for k, v in step.items() if k not in ("agent", "seconds"))
            print(f"       {step['agent']:<14} {step['seconds']:>6.1f}s  {detail}")

    if args.report and reports:
        import json

        Path(args.report).write_text(
            "[" + ",".join(r.to_json(values=args.report_values) for r in reports) + "]",
            encoding="utf-8",
        )
        if args.report_values:
            print(f"report: {args.report}  (CONTAINS PHI -- handle as such)")
        else:
            print(f"report: {args.report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
