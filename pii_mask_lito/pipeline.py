"""Dispatch, fail-closed verification, and privacy-safe masking reports.

The output is read back after rendering. A value which remains readable turns
the run into a failure instead of allowing a partially masked document to be
published. Reports and ordinary console messages omit source values and paths
by default because operational metadata is itself a common disclosure path.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import images, office, pdf, symbols, vision
from .detect import Detector
from .model import Span, TokenText, mask_rects, merge_spans, reading_order
from .registry import TagRegistry, normalize

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
OFFICE_SUFFIXES = {".txt": "txt", ".csv": "csv", ".docx": "docx", ".xlsx": "xlsx"}
_TAG = re.compile(r"<[A-Z_]+#\d+>")
# Above this size, keeping both rendered and masked 300-DPI pages in memory is
# needlessly expensive. The disk-backed path keeps behavior identical while
# bounding resident page buffers for long documents.
_SPOOL_PAGE_THRESHOLD = 8
_PAGE_BATCH = 4
_SAFE_FINDING_SOURCES = {
    "agent", "barcode", "beside-name", "date", "pattern", "presidio",
    "propagated", "recheck", "spatial", "structural", "vision",
}


def _public_entity(entity: str) -> str:
    return entity if re.fullmatch(r"[A-Z][A-Z0-9_]*", entity or "") else "CUSTOM"


def _public_source(source: str) -> str:
    return source if source in _SAFE_FINDING_SOURCES else "custom"


class MaskingError(RuntimeError):
    """Raised instead of writing output that still contains PHI."""


@dataclass
class Finding:
    entity: str
    value: str
    replacement: str
    score: float
    source: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class Report:
    source: str
    output: str
    engine: str
    findings: list[Finding] = field(default_factory=list)
    verified: bool = False
    # Gate 1: a value we claimed to mask is still readable. Rendering broke.
    leaked: list[str] = field(default_factory=list)
    # Gate 2: an identifier-shaped string is readable that no detector ever
    # proposed. Detection broke. Kept separate because the two failures need
    # different responses.
    leaked_patterns: list[str] = field(default_factory=list)
    # Neither pass nor fail: things a human should look at. Pass/fail is too
    # coarse for a document class where the honest answer is often "this page
    # contains ink we could not read".
    review: list[str] = field(default_factory=list)
    doc_type: str = ""
    trace: list[dict] = field(default_factory=list)

    def summary(self) -> list[dict]:
        """Value-free finding counts and aggregate normalized mask area."""
        counts: Counter[tuple[str, str]] = Counter()
        areas: Counter[tuple[str, str]] = Counter()
        for finding in self.findings:
            key = (_public_entity(finding.entity), _public_source(finding.source))
            counts[key] += 1
            if finding.bbox is not None:
                x0, y0, x1, y1 = finding.bbox
                areas[key] += max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        return [
            {
                "entity": entity,
                "source": source,
                "count": counts[(entity, source)],
                "normalized_box_area": round(areas[(entity, source)], 6),
            }
            for entity, source in sorted(counts)
        ]

    def to_json(self, indent: int = 2, values: bool = False) -> str:
        """Serialize the report, withholding sensitive content by default.

        Paths often contain names or record identifiers, while leak and review
        messages can contain verbatim source text. ``values=True`` is therefore
        an explicit opt-in for *all* sensitive report content, not only the
        ``Finding.value`` fields.
        """
        data = asdict(self)
        if not values:
            data["source"] = f"<input>{Path(self.source).suffix.lower()}"
            data["output"] = f"<output>{Path(self.output).suffix.lower()}"
            for finding in data["findings"]:
                finding["value"] = ""
                finding["entity"] = _public_entity(finding["entity"])
                finding["source"] = _public_source(finding["source"])
            data["leaked_count"] = len(data["leaked"])
            data["leaked_pattern_count"] = len(data["leaked_patterns"])
            data["review_count"] = len(data["review"])
            data["trace_count"] = len(data["trace"])
            data["leaked"] = []
            data["leaked_patterns"] = []
            data["review"] = []
            # Agent trace dictionaries are extension points and may contain
            # model-returned strings. Counts prove which stages ran without
            # treating arbitrary trace payloads as safe log metadata.
            data["trace"] = []
            data["doc_type"] = ""
        # `default` so a stray numpy scalar from a vision detector cannot kill
        # the audit trail of a run that already masked and verified correctly.
        # The coordinates are floats; anything else becomes its repr rather than
        # a TypeError raised after a minute of OCR.
        return json.dumps(data, indent=indent,
                          default=lambda o: float(o) if hasattr(o, "__float__") else repr(o))


def mask(
    src: str,
    dest: str,
    detector: Detector | None = None,
    registry: TagRegistry | None = None,
    ocr_engine: str = "auto",
    dpi: int = pdf.DEFAULT_DPI,
    verify: bool = True,
    vlm=None,
    loop_ocr: str = "auto",
    mask_unread_ink: bool = False,
) -> Report:
    """Mask one file, writing `dest` only if verification passes."""
    detector = detector or Detector()
    registry = registry or TagRegistry()
    suffix = Path(src).suffix.lower()
    report = Report(source=src, output=dest, engine=ocr_engine)
    if vlm is not None and hasattr(vlm, "begin_document"):
        vlm.begin_document()

    # Write to a temporary file first: a document that fails verification must
    # never appear at the destination path, even briefly.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    engine = None
    try:
        if suffix in PDF_SUFFIXES:
            engine = _ocr_or_none(ocr_engine, src, report)
            _mask_pdf(src, tmp.name, detector, registry, report, engine, dpi, vlm,
                      loop=_loop_engine(engine, ocr_engine, loop_ocr, report),
                      mask_unread_ink=mask_unread_ink)
        elif suffix in IMAGE_SUFFIXES:
            engine = _ocr(ocr_engine)
            _mask_image(src, tmp.name, detector, registry, report, engine, vlm,
                        mask_unread_ink=mask_unread_ink)
        elif suffix in OFFICE_SUFFIXES:
            kind = OFFICE_SUFFIXES[suffix]
            applied = getattr(office, f"mask_{kind}")(src, tmp.name, detector, registry)
            for span, tag in applied:
                report.findings.append(
                    Finding(span.entity, span.text, tag, span.score, span.source)
                )
        else:
            raise MaskingError(f"unsupported file type: {suffix or src!r}")

        if verify:
            report.leaked, report.leaked_patterns, flagged = _verify(tmp.name, report, engine)
            report.review += flagged
            # Gate 2 found identifier shapes nothing proposed. Before refusing,
            # give them back to the detector and rebuild once.
            #
            # `_recheck` reads the masked page images; verification reads the file
            # as written, after reportlab has re-encoded every page. OCR
            # does not agree across those two renderings, so a line reading
            # "Date: 09-08-2023" can exist only on the second -- no
            # detector ever saw the token, and no amount of recheck could.
            # Seeding the lexicon makes the finished document the last word.
            if report.leaked_patterns and suffix in PDF_SUFFIXES and not report.leaked:
                if vlm is not None and hasattr(vlm, "lock_values"):
                    vlm.lock_values(report.leaked_patterns)
                for value in report.leaked_patterns:
                    detector.lexicon.setdefault(normalize(value), _pattern_entity(value))
                report.findings.clear()
                if vlm is not None and hasattr(vlm, "begin_pass"):
                    vlm.begin_pass()
                _mask_pdf(src, tmp.name, detector, registry, report, engine, dpi, vlm,
                          loop=_loop_engine(engine, ocr_engine, loop_ocr, report),
                          mask_unread_ink=mask_unread_ink)
                report.leaked, report.leaked_patterns, flagged = _verify(
                    tmp.name, report, engine
                )
                report.review += flagged
            # Anything a detector argued away rather than masked. Suppression is
            # the only decision in this tool whose failure mode is a disclosure,
            # so it never happens silently -- a reviewer sees each one.
            report.review += detector.suppressed
            # The rebuild verifies twice, so the same flag arrives twice. Order
            # is kept: a review queue is read top to bottom.
            report.review = list(dict.fromkeys(report.review))
            report.verified = not (report.leaked or report.leaked_patterns)
            if report.leaked:
                raise MaskingError(
                    f"gate 1: {len(report.leaked)} detected value(s) still readable in "
                    "output; refusing to write the destination"
                )
            if report.leaked_patterns:
                raise MaskingError(
                    f"gate 2: {len(report.leaked_patterns)} identifier-shaped value(s) "
                    f"readable in output that no detector proposed; refusing to write "
                    "the destination"
                )
        if vlm is not None and getattr(vlm, "auditor", None) is not None:
            _audit_output(tmp.name, vlm, report)
        if vlm is not None and getattr(vlm, "trace", None):
            report.trace = vlm.trace.steps
            for step in report.trace:
                if step.get("doc_type"):
                    report.doc_type = step["doc_type"]
            failed_pages = [
                step for step in report.trace
                if step.get("agent") == "semantic_page" and step.get("failed")
            ]
            if failed_pages:
                report.review.append(
                    f"semantic assistance failed on {len(failed_pages)} page(s); "
                    "those pages used rules-only fallback"
                )
        os.replace(tmp.name, dest)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        if vlm is not None and hasattr(vlm, "end_document"):
            vlm.end_document()
    return report


def _ocr(name: str):
    from . import ocr as ocr_module

    return ocr_module.get(name)


def _ocr_or_none(name: str, src: str, report: Report):
    """OCR every page when an engine exists; require one only when it must.

    OCR runs on every page when an engine is available (see ``pdf.extract``),
    including small raster regions embedded in otherwise native PDFs.

    The engine itself is picked by what the pages actually need. A scan has no
    text layer, so the accurate engine is the standard -- whatever paddle fails
    to read becomes a token that never exists. A native PDF's text layer is
    authoritative and its boxes are exact; OCR there only supplements it and
    powers the read-back verification, so the fast engine runs instead. See
    ocr.cheap() for the geometry reason behind that decision.

    A missing engine is still not always fatal, though. A PDF whose every page
    has a solid text layer can be masked without one, and ocr.py promises that a
    user who only masks native PDFs installs neither engine. So a missing engine
    downgrades to a review flag unless some page genuinely cannot be read.
    """
    try:
        if name == "auto" and not pdf.requires_ocr(src):
            from . import ocr as ocr_module

            return ocr_module.cheap()
        return _ocr(name)
    except RuntimeError:
        if pdf.requires_ocr(src):
            raise
        report.review.append(
            "no OCR engine installed: pages read from the text layer only, and "
            "the output could not be verified through its pixels"
        )
        return None


def _loop_engine(engine, ocr_engine: str, loop_ocr: str, report: Report):
    """The engine that drives the recheck loop, which need not be the fast one.

    Iterative rechecks can dominate runtime on an expensive engine. A cheaper
    installed engine can search the already-learned lexicon there without
    changing the primary extraction engine, where OCR accuracy is decisive.
    """
    if engine is None:
        return None
    primary = ocr_engine if ocr_engine != "auto" else getattr(
        engine, "name", type(engine).__name__.lower()
    )
    primary = "paddle" if "paddle" in primary else ("tesseract" if "tess" in primary else primary)
    name = ocr_module_loop_name(primary, loop_ocr)
    if name == primary:
        return engine
    try:
        loop = _ocr(name)
    except Exception:  # noqa: BLE001
        return engine
    report.review.append(
        f"recheck ran on {name} while pages were read with {primary}: the loop "
        f"re-reads already-masked pages, and {primary} is too slow to run it "
        f"sixty times"
    )
    return loop


def ocr_module_loop_name(primary: str, requested: str) -> str:
    from . import ocr as ocr_module

    return ocr_module.loop_engine_name(primary, requested)


def _symbol_spans(tokens: list, first: int) -> list:
    """Spans for barcode payloads that are identifiers on their face.

    `first` is the index the symbol tokens start at. A payload that matches the
    lexicon or a pattern is caught by the ordinary detectors like any other
    token; this covers the rest, because a form barcode encoding a 6+
    digit run is an account number whether or not anything recognises the value.
    """
    return [
        Span(
            entity="ACCOUNT_NUMBER",
            start=0,
            end=0,
            score=0.9,
            text=token.text,
            tokens=[first + offset],
            source="barcode",
        )
        for offset, token in enumerate(tokens)
        if symbols.payload_is_identifier(token.text)
    ]


def _vision_boxes(image, tokens, entities, report, page: int,
                  mask_unread_ink: bool = False) -> list[tuple[str, tuple]]:
    """Faces and signature blocks on one page: PHI that is never text.

    Reported once per document when the dependency is missing, not once per
    page -- a review flag repeated ten times is noise, and noise is how a review
    queue stops being read.

    Unread ink is the third thing looked for and the only one that does not mask
    by default. A region of ink no engine could read might be a handwritten
    address or might be a logo, and this detector cannot tell -- so the honest
    output is a flag saying the page carries writing we could not check.
    `mask_unread_ink` is for callers who would rather lose the logo.
    """
    if not vision.available():
        if page == 0 and (entities & {"FACE", "SIGNATURE"}):
            report.review.append(
                "opencv not installed: faces and signature blocks were not "
                "looked for; visual identifiers require manual review"
            )
        return []
    found = []
    if "FACE" in entities:
        found += [("FACE", box) for box in vision.faces(image)]
    if "SIGNATURE" in entities:
        found += [("SIGNATURE", box) for box in vision.signatures(image, tokens)]
        unread = vision.unread_ink(image, tokens)
        if unread and mask_unread_ink:
            found += [("SIGNATURE", box) for box in unread]
        elif unread:
            # One flag per page, not one per region: the reviewer opens the page
            # either way, and a queue of forty lines is a queue nobody reads.
            report.review.append(
                f"page {page + 1} has {len(unread)} region(s) of ink no OCR "
                f"engine could read; they were not masked (--mask-unread-ink "
                f"covers them)"
            )
    return found


def _mask_pdf(src, dest, detector, registry, report, engine, dpi, vlm,
              recheck=True, loop=None, mask_unread_ink=False) -> None:
    if pdf.page_count(src) < _SPOOL_PAGE_THRESHOLD:
        rendered = pdf.page_images(src, dpi)
        return _mask_pdf_images(
            src, dest, detector, registry, report, engine, dpi, vlm,
            rendered, {}, recheck, loop, mask_unread_ink,
        )

    # Long documents are rendered once too, but their reusable page images are
    # kept as PNGs instead of simultaneous RGB buffers. This trades bounded,
    # local disk I/O for memory that no longer grows by roughly two full page
    # images per page.
    with tempfile.TemporaryDirectory(prefix="pii-mask-pages-") as work:
        rendered = pdf.spool_page_images(src, Path(work) / "source", dpi)
        masked_images = pdf.DiskImageStore(Path(work) / "masked")
        return _mask_pdf_images(
            src, dest, detector, registry, report, engine, dpi, vlm,
            rendered, masked_images, recheck, loop, mask_unread_ink,
        )


def _mask_pdf_images(src, dest, detector, registry, report, engine, dpi, vlm,
                     rendered, masked_images, recheck=True, loop=None,
                     mask_unread_ink=False) -> None:
    pages_tokens, _sizes, provenance = pdf.extract(
        src, ocr=engine, dpi=dpi, rendered=rendered, with_provenance=True
    )
    text_layers, page_boxes = {}, {}
    entities = set(detector.entities)

    # Barcode payloads join the token stream before detection, so the lexicon
    # learns them and they propagate to damaged printed copies of themselves.
    symbol_spans: dict[int, list] = {}
    if symbols.available():
        for index in range(len(rendered)):
            image = rendered[index]
            try:
                found = symbols.decode(image, page=index)
                if found:
                    symbol_spans[index] = _symbol_spans(found, len(pages_tokens[index]))
                    pages_tokens[index] = pages_tokens[index] + found
            finally:
                if isinstance(rendered, pdf.DiskImageStore):
                    image.close()
    else:
        report.review.append(
            "zxing-cpp not installed: barcodes and 2D symbols were not decoded, "
            "and a symbol encoding an identifier survives rasterization intact"
        )

    # Pass 1 detects contextual values and learns them. Pass 2 propagates those
    # values to occurrences that may appear without their original labels.
    tts = [TokenText(tokens) for tokens in pages_tokens]
    page_spans = []
    for index, tt in enumerate(tts):
        if vlm is None:
            spans = detector.detect(tt)
        else:
            image = rendered[index]
            try:
                spans = vlm.detect(image, tt)
            finally:
                if isinstance(rendered, pdf.DiskImageStore):
                    image.close()
        detector.learn(spans, tt)
        # A barcode encodes cleanly what OCR often reads badly, so its payload
        # goes into the lexicon whether or not a detector proposed it.
        detector.learn(symbol_spans.get(index, []), tt)
        page_spans.append(spans)

    for index, tt in enumerate(tts):
        propagated = detector.propagate(tt)
        if vlm is not None and hasattr(vlm, "filter_late_spans"):
            propagated = vlm.filter_late_spans(index, tt, propagated)
        spans = merge_spans(
            page_spans[index] + propagated + symbol_spans.get(index, [])
        )
        boxes, layer = [], []
        for span in spans:
            tag = registry.tag(span.entity, span.text)
            for page_no, bbox in mask_rects(tt, span):
                boxes.append((bbox, tag))
                layer.append((bbox, tag))
                report.findings.append(
                    Finding(span.entity, span.text, tag, span.score, span.source, page_no, bbox)
                )
        image = rendered[index]
        try:
            for entity, bbox in _vision_boxes(
                image, pages_tokens[index], entities, report, index,
                mask_unread_ink,
            ):
                # Indexed on position, not value: there is no value to key on, and
                # two photographs on one page are two different people.
                tag = registry.tag(entity, f"p{index}:{bbox[0]:.3f},{bbox[1]:.3f}")
                boxes.append((bbox, tag))
                layer.append((bbox, tag))
                report.findings.append(
                    Finding(entity, "", tag, 0.8, "vision", index, bbox)
                )
            masked_images[index] = images.mask(image, boxes) if boxes else image
        finally:
            if isinstance(rendered, pdf.DiskImageStore):
                image.close()
        text_layers[index] = layer
        page_boxes[index] = boxes

    # A gate-2 rebuild starts from the original render, so it must run the
    # convergence loop again or masks discovered during recheck would be lost.
    if engine is not None and recheck:
        active_pages = _active_recheck_pages(provenance, page_boxes)
        if active_pages:
            _recheck(loop or engine, detector, registry, report,
                     masked_images, text_layers, page_boxes,
                     originals=dict(enumerate(tts)), active_pages=active_pages,
                     semantic=vlm)

    pdf.write(src, dest, masked_images, text_layers, dpi)


def _active_recheck_pages(provenance, page_boxes) -> set[int]:
    """OCR-backed pages whose pixels actually changed during masking."""
    return {
        index for index, source in enumerate(provenance)
        if (source["full_ocr"] or source["ocr_supplements"])
        and bool(page_boxes.get(index))
    }


def _already_masked(tt, span, boxes, threshold: float = 0.5) -> bool:
    rects = tt.rects_for(span)
    if not rects:
        return False
    for _page, (x0, y0, x1, y1) in rects:
        covered = False
        for (bx0, by0, bx1, by1), _tag in boxes:
            inter = max(0.0, min(x1, bx1) - max(x0, bx0)) * max(0.0, min(y1, by1) - max(y0, by0))
            area = (x1 - x0) * (y1 - y0)
            if area > 0 and inter / area > threshold:
                covered = True
                break
        if not covered:
            return False
    return True


def _draw(tt, spans, index, registry, report, text_layers, page_boxes, source):
    """Paint spans onto a page and record them. Returns whether anything landed."""
    spans = [s for s in spans if not _already_masked(tt, s, page_boxes.get(index, []))]
    if not spans:
        return False
    boxes = page_boxes.setdefault(index, [])
    drawn = False
    for span in spans:
        tag = registry.tag(span.entity, span.text)
        for page_no, bbox in mask_rects(tt, span):
            boxes.append((bbox, tag))
            text_layers[index].append((bbox, tag))
            report.findings.append(
                Finding(span.entity, span.text, tag, span.score, source, page_no, bbox)
            )
            drawn = True
    return drawn


def _recheck(engine, detector, registry, report, masked_images, text_layers, page_boxes,
             originals=None, max_passes: int = 4, active_pages=None,
             semantic=None) -> None:
    """Re-OCR the masked pages and mask anything known that is still readable.

    OCR can change after masks alter the page image. Re-reading catches values
    that were missed or merged differently on the source rendering.

    Detection runs again too, not just propagation. OCR reads the same page
    differently once part of it is covered, and values it garbled the first time
    come back legible -- an SSN that was unreadable in the source pass, and so
    never entered the lexicon, is found on the second look.

    Runs to a fixpoint rather than a fixed count. Two passes was a guess dressed
    as a limit: what matters is whether the last pass still found something, and
    a page that was still gaining spans when the loop stopped has not converged.
    Hitting the cap is not a failure -- the document may well be fine -- but it
    is not a clean bill of health either, so the page is flagged for review.
    """
    from .model import assign_lines
    from .ocr import read_pages

    # Which pages are worth re-reading. Every masked page, to start with; after
    # that, only the ones the last pass actually drew on.
    #
    # A page that gained nothing last time reads the same this time, because
    # nothing about it changed -- the image is identical and the engine is
    # deterministic on identical input. Re-reading it is the only work in this
    # function guaranteed to find nothing.
    #
    # The thing that CAN change for a quiet page is the lexicon, grown by some
    # other page. That is covered without any OCR at all, by the propagation
    # pass over the original tokens below, which still runs for every page every
    # time -- and a page it draws on rejoins the re-read set. So the fixpoint is
    # unchanged; only the reads that could not have found anything are gone.
    active = set(masked_images) if active_pages is None else set(active_pages)

    def remask(index: int) -> None:
        image = masked_images[index]
        try:
            masked_images[index] = images.mask(image, page_boxes[index])
        finally:
            if isinstance(masked_images, pdf.DiskImageStore):
                image.close()

    for remaining in range(max_passes, 0, -1):
        changed = False
        pages = sorted(active)
        reads = {}
        for start in range(0, len(pages), _PAGE_BATCH):
            batch_pages = pages[start : start + _PAGE_BATCH]
            batch_images = [masked_images[index] for index in batch_pages]
            try:
                batch_reads = read_pages(engine, batch_images, batch_pages)
                reads.update(zip(batch_pages, batch_reads))
            finally:
                if isinstance(masked_images, pdf.DiskImageStore):
                    for image in batch_images:
                        image.close()
        for index in sorted(masked_images):
            tokens = reads.get(index)
            if tokens is None:
                # Converged on the image; the lexicon may still have news for it.
                propagated = detector.propagate(originals[index]) \
                    if originals and index in originals else []
                if (propagated and semantic is not None
                        and hasattr(semantic, "filter_late_spans")):
                    propagated = semantic.filter_late_spans(
                        index, originals[index], propagated
                    )
                drawn = bool(propagated and _draw(
                    originals[index], propagated, index, registry, report,
                    text_layers, page_boxes, "recheck"))
                if drawn:
                    remask(index)
                    active.add(index)
                    changed = True
                continue
            assign_lines(tokens)
            tokens = reading_order(tokens)
            tt = TokenText(tokens)
            found = detector.detect(tt)
            # Do not learn new structural names from a degraded, partly masked
            # rendering. Previously learned names remain covered by propagation.
            found = [s for s in found
                     if not (s.entity == "PERSON" and s.source == "structural")]
            if semantic is not None and hasattr(semantic, "filter_late_spans"):
                found = semantic.filter_late_spans(index, tt, found)
            # Learn only from recognizers with stable evidence on a degraded,
            # partly masked rendering.
            detector.learn([s for s in found if s.source in _TRUSTED_ON_RERENDER], tt)
            propagated = detector.propagate(tt)
            if semantic is not None and hasattr(semantic, "filter_late_spans"):
                propagated = semantic.filter_late_spans(index, tt, propagated)
            spans = merge_spans(found + propagated)
            drawn = _draw(tt, spans, index, registry, report,
                          text_layers, page_boxes, "recheck")

            # Also re-propagate over the page as it was *originally* read.
            #
            # Without this the loop's progress is tied to the image changing:
            # every value it can find must be one the re-OCR of a part-masked
            # render happens to surface. Draw fewer boxes and the image changes
            # less, OCR returns the same tokens, the loop calls it converged --
            # and values the lexicon has since learned are never applied to the
            # accurate original token stream at all. The original tokens have better
            # geometry anyway, so this is the cheaper half of the pass.
            if originals and index in originals:
                source = originals[index]
                propagated = detector.propagate(source)
                if semantic is not None and hasattr(semantic, "filter_late_spans"):
                    propagated = semantic.filter_late_spans(index, source, propagated)
                drawn |= _draw(source, propagated, index, registry, report,
                               text_layers, page_boxes, "recheck")
            if not drawn:
                active.discard(index)
                continue
            remask(index)
            active.add(index)
            changed = True
            if remaining == 1:
                report.review.append(
                    f"page {index + 1} was still gaining masks when the recheck "
                    f"limit of {max_passes} passes was reached; it has not converged"
                )
        if not changed:
            return


def _mask_image(src, dest, detector, registry, report, engine, vlm,
                mask_unread_ink: bool = False) -> None:
    """Mask a standalone image with the PDF page's pixel-level detectors."""
    image = images.load(src)
    tokens = engine.words(image)
    from .model import assign_lines

    assign_lines(tokens)
    tokens = reading_order(tokens)

    entities = set(detector.entities)
    # Barcode payloads join the token stream before detection, exactly as in
    # _mask_pdf, so a symbol encoding an identifier is masked and the value it
    # carries is learned and propagated to printed copies of itself.
    symbol_extra = []
    if symbols.available():
        found = symbols.decode(image, page=0)
        if found:
            symbol_extra = _symbol_spans(found, len(tokens))
            tokens = tokens + found
    else:
        report.review.append(
            "zxing-cpp not installed: barcodes and 2D symbols were not decoded, "
            "and a symbol encoding an identifier survives rasterization intact"
        )

    tt = TokenText(tokens)
    spans = vlm.detect(image, tt) if vlm is not None else detector.detect(tt)
    detector.learn(spans, tt)
    propagated = detector.propagate(tt)
    if vlm is not None and hasattr(vlm, "filter_late_spans"):
        propagated = vlm.filter_late_spans(0, tt, propagated)
    spans = merge_spans(spans + propagated + symbol_extra)
    boxes = []
    for span in spans:
        tag = registry.tag(span.entity, span.text)
        for page_no, bbox in mask_rects(tt, span):
            boxes.append((bbox, tag))
            report.findings.append(
                Finding(span.entity, span.text, tag, span.score, span.source, page_no, bbox)
            )
    for entity, bbox in _vision_boxes(image, tokens, entities, report, 0, mask_unread_ink):
        tag = registry.tag(entity, f"p0:{bbox[0]:.3f},{bbox[1]:.3f}")
        boxes.append((bbox, tag))
        report.findings.append(Finding(entity, "", tag, 0.8, "vision", 0, bbox))
    images.save(images.mask(image, boxes), dest, src)


# Gate 2's rules. Deliberately the high-precision shapes only: a hit here has to
# stand on its own, with no lexicon and no spatial context behind it.
_LONG_DIGITS = re.compile(r"(?<!\d)\d{9,}(?!\d)")
# Entity types detect.learn() deliberately keeps out of the lexicon, and which
# gate 1 therefore cannot ask a job-wide question about. Kept in step with it.
_NOT_PROPAGATED = {"DATE", "AGE", "LOCATION", "FACE", "SIGNATURE"}
# Detector sources whose hits are structural enough to trust on a re-rendered,
# part-masked page. See _recheck.
_TRUSTED_ON_RERENDER = {"pattern", "spatial", "date", "barcode"}


def _readback_data(path: str, engine,
                   symbol_leaks: list[str] | None = None) -> tuple[str, list[TokenText]]:
    """Everything legible in the written file.

    For a PDF this is the crux, and it is why an OCR engine is threaded all the
    way down here. `pdf.write` authors the output's text layer itself, and it
    writes only the replacement tags -- so reading that layer back and finding
    no PHI proves nothing whatsoever. It cannot contain any. Verification of a
    raster PDF has to go through the *pixels*, which means rendering the file we
    just wrote and OCRing it, exactly as a recipient would see it.

    The text layer is read too, cheaply, because it catches the separate class
    of bug where the tag stamped into the layer is not the tag drawn on the
    page.
    """
    suffix = Path(path).suffix.lower()
    if suffix in OFFICE_SUFFIXES:
        text = office.text_of(path)
        return text, [TokenText.from_text(text)]
    chunks = []
    pages: list[TokenText] = []
    if suffix in PDF_SUFFIXES:
        chunks.append(pdf.text_of(path))
    if engine is None:
        return "\n".join(chunks), pages
    from .model import assign_lines

    from .ocr import read_pages

    if suffix in PDF_SUFFIXES:
        def append_batch(page_batch, number_batch) -> None:
            try:
                if symbol_leaks is not None:
                    for rendered, page_number in zip(page_batch, number_batch):
                        symbol_leaks.extend(
                            _identifier_symbols(rendered, page_number)
                        )
                for tokens in read_pages(engine, page_batch, number_batch):
                    assign_lines(tokens)
                    page = TokenText(reading_order(tokens))
                    pages.append(page)
                    chunks.append(page.text)
            finally:
                for rendered in page_batch:
                    rendered.close()

        page_batch, number_batch = [], []
        for page_number, image in enumerate(pdf.iter_page_images(path)):
            page_batch.append(image)
            number_batch.append(page_number)
            if len(page_batch) < _PAGE_BATCH:
                continue
            append_batch(page_batch, number_batch)
            page_batch, number_batch = [], []
        if page_batch:
            append_batch(page_batch, number_batch)
    else:
        image = images.load(path)
        try:
            tokens = read_pages(engine, [image])[0]
            assign_lines(tokens)
            page = TokenText(reading_order(tokens))
            pages.append(page)
            chunks.append(page.text)
        finally:
            image.close()
    return "\n".join(chunks), pages


def _readback(path: str, engine) -> str:
    """Compatibility wrapper returning only read-back text."""
    return _readback_data(path, engine)[0]


def _verify(path: str, report: Report, engine=None) -> tuple[list[str], list[str], list[str]]:
    """Read the written file back and check it twice.

    Gate 1 looks for any value we claimed to mask. It is the stronger check but
    it can only ever fail on the detectors' own homework -- it says nothing
    about what they never proposed.

    Gate 2 is independent: the high-precision patterns run again over the
    re-read output, with no reference to the lexicon or to what was found on the
    way in. A hit means detection broke, not rendering, and the two need
    different responses, so they are returned and reported separately.

    Returns (gate 1 leaks, gate 2 leaks, review flags).
    """
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES and engine is None:
        return [], [], ["image output not verified: no OCR engine"]
    combined_symbols = (
        [] if suffix in PDF_SUFFIXES and engine is not None and symbols.available()
        else None
    )
    try:
        raw, pattern_pages = _readback_data(path, engine, combined_symbols)
    except Exception:
        # An unreadable output cannot be shown to be clean, so treat it as dirty.
        return ["<output unreadable>"], [], []
    # Strip our own replacement tags before searching. Otherwise a masked column
    # header "Email" is "found" inside the <EMAIL_ADDRESS#0> that replaced it,
    # and a correctly masked file is rejected.
    stripped = _TAG.sub(" ", raw)
    haystack = normalize(stripped)
    leaked, flagged_names = [], []
    for finding in report.findings:
        # Gate 1 asks "was this value masked *everywhere*", which is only a fair
        # question about values propagation actually carries job-wide. Dates,
        # ages and place fragments are deliberately kept out of the lexicon --
        # they are too generic to match bare -- so demanding that such a value
        # appear nowhere else would contradict the propagation policy.
        if finding.entity in _NOT_PROPAGATED:
            continue
        needle = normalize(finding.value)
        # Word-boundary match: a short value must not hit inside a longer word.
        if len(needle) < 4 or not re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
            continue
        if finding.entity == "PERSON" and " " not in needle:
            # One-word names collide with ordinary vocabulary, so surface them
            # for review instead of turning an ambiguous OCR match into a hard
            # failure.
            # The guarantee is not weakened by much: identifiers still fail
            # hard, gate 2 re-derives the high-stakes shapes independently, and
            # job-wide name coverage is propagation's job -- which now runs to a
            # fixpoint and flags the pages where it did not converge.
            flagged_names.append(f"name still readable in output: {finding.replacement}")
            continue
        leaked.append(finding.value)
    patterns, flagged = [], []
    pattern_inputs = pattern_pages or [TokenText.from_text(stripped)]
    for page in pattern_inputs:
        page_patterns, page_flags = _verify_patterns(page)
        patterns.extend(page_patterns)
        flagged.extend(page_flags)
    patterns += (
        combined_symbols if combined_symbols is not None else _verify_symbols(path)
    )
    return sorted(set(leaked)), sorted(set(patterns)), flagged + sorted(set(flagged_names))


def _verify_symbols(path: str) -> list[str]:
    """Gate 3: does anything on the finished page still decode?

    Cheap and absolute. A symbol is either readable by a scanner or it is not,
    and there is no ambiguity to weigh, so unlike the digit-run rule this one
    fails the run rather than flagging it.
    """
    if Path(path).suffix.lower() not in PDF_SUFFIXES or not symbols.available():
        return []
    leaked = []
    for index, image in enumerate(pdf.iter_page_images(path)):
        try:
            leaked.extend(_identifier_symbols(image, index))
        finally:
            image.close()
    return leaked


def _identifier_symbols(image, page: int) -> list[str]:
    """Value-free verification failures for identifier-bearing symbols."""
    return [
        f"decodable symbol on page {page + 1}"
        for token in symbols.decode(image, page=page)
        if symbols.payload_is_identifier(token.text)
    ]


def _audit_output(path: str, semantic, report: Report) -> None:
    """Run an optional semantic review over finished pixels, as review flags."""
    import time

    suffix = Path(path).suffix.lower()
    if suffix in PDF_SUFFIXES:
        rendered = enumerate(pdf.iter_page_images(path))
    elif suffix in IMAGE_SUFFIXES:
        rendered = enumerate([images.load(path)])
    else:
        return
    for page, image in rendered:
        started = time.time()
        try:
            remaining = semantic.auditor.run(image)
        except Exception as exc:  # noqa: BLE001 - audit is advisory
            semantic.trace.add(
                "auditor", time.time() - started,
                {"page": page + 1, "failed": type(exc).__name__},
            )
            report.review.append(
                f"agent audit page {page + 1} failed; manual review required"
            )
            continue
        finally:
            image.close()
        semantic.trace.add(
            "auditor", time.time() - started,
            {"page": page + 1, "remaining": len(remaining)},
        )
        report.review.extend(
            f"agent audit page {page + 1}: possible remaining identifier: {value}"
            for value in remaining
        )


def _table_noise(span) -> bool:
    """Return whether a verification hit is common table or OCR noise.

    Detection may conservatively over-mask, while verification failures block
    output. The verification filter therefore rejects two unambiguous artifacts:

      "150.00 4904"  an amount and short code, which satisfies the SSN shape
                     exactly. Real SSNs use one separator throughout; a match
                     mixing a decimal point with a space is a table, not a
                     number.
      "92@01"        an OCR artefact satisfying the email shape. Every real
                     address has a letter in it.
      "ING@GBAX"     OCR debris off a masked page. The email pattern is
                     deliberately tolerant because OCR can destroy dots and
                     domains, so a dot cannot be required. A conservative
                     minimum length filters short debris.
    """
    if span.entity in ("US_SSN", "PHONE_NUMBER"):
        return "." in span.text and " " in span.text
    if span.entity == "EMAIL_ADDRESS":
        return not any(c.isalpha() for c in span.text) or len(span.text) < 12
    return False


def _verify_patterns(text: str | TokenText) -> tuple[list[str], list[str]]:
    """Gate 2: high-precision identifier shapes in the finished output.

    This deliberately derives identifier patterns from the finished document,
    independently of the values proposed during detection.

    What fails and what merely flags is the whole design of this function. It
    fails only on shapes that cannot be anything else: an SSN, a phone number,
    an email address, or a date written with separators. Everything ambiguous is
    a review flag instead:

      - Bare 9+ digit runs. Organization and payee identifiers can trip this
        legitimately, and read-back text is flat, so the spatial role context
        used during detection is unavailable here.
      - Separator-less dates (MMDDYY, MMDDYYYY). A six-digit account fragment
        satisfies MMDDYY often enough that failing on it would reject ordinary
        documents containing numeric account fragments.

    A gate that fails on everything gets switched off, and a gate that is
    switched off catches nothing.
    """
    from .detect import DateDetector, PatternDetector

    tt = text if isinstance(text, TokenText) else TokenText.from_text(text)
    hits = PatternDetector().detect(tt)
    leaked = [s.text for s in hits if not _table_noise(s)]
    # Suppressed, not discarded. Something that looked like an identifier and
    # was argued away should still be visible to whoever reviews the run.
    noise = [f"{s.entity} shape suppressed as table noise: {s.text}"
             for s in hits if _table_noise(s)]
    dates = DateDetector().detect(tt)
    leaked += [s.text for s in dates if any(c in s.text for c in "/-")]

    flagged = [f"unmasked date-shaped run: {s.text}" for s in dates
               if not any(c in s.text for c in "/-")]
    flagged += [
        f"unmasked {len(m.group(0))}-digit run in output"
        for m in _LONG_DIGITS.finditer(tt.text)
    ]
    return sorted(set(leaked)), sorted(set(flagged + noise))


def _pattern_entity(value: str) -> str:
    """Entity type of a Gate 2 value, using the same recognizers as the gate."""
    from .detect import DateDetector, PatternDetector

    tt = TokenText.from_text(value)
    entities = {span.entity for span in PatternDetector().detect(tt)}
    for entity in ("US_SSN", "EMAIL_ADDRESS", "PHONE_NUMBER"):
        if entity in entities:
            return entity
    if DateDetector().detect(tt):
        return "DATE"
    return "ACCOUNT_NUMBER"
