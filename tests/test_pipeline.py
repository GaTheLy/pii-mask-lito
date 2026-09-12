"""Runnable checks: `python -m tests.test_pipeline` or `pytest tests/`.

Each check targets a safety or geometry invariant using invented data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pii_mask_lito.detect import DateDetector, Detector
from pii_mask_lito.model import (Span, Token, TokenText, assign_lines,
                                 merge_spans, reading_order)
from pii_mask_lito.pipeline import Finding, Report, _already_masked, _verify
from pii_mask_lito.registry import TagRegistry, normalize


def test_offset_token_roundtrip():
    tokens = [
        Token("Customer", 0, (0.1, 0.1, 0.2, 0.12), 0),
        Token("CALDER,", 0, (0.21, 0.1, 0.3, 0.12), 0),
        Token("MIRA", 0, (0.31, 0.1, 0.4, 0.12), 0),
    ]
    tt = TokenText(tokens)
    assert tt.text == "Customer CALDER, MIRA"

    start = tt.text.index("CALDER")
    end = tt.text.index("MIRA") + len("MIRA")
    assert tt.tokens_for(start, end) == [1, 2]

    span = Span("PERSON", start, end, 0.9, tt.text[start:end], [1, 2])
    rects = tt.rects_for(span)
    # One line crossed -> one rectangle, spanning both words and nothing else.
    assert len(rects) == 1
    page, (x0, y0, x1, y1) = rects[0]
    assert page == 0
    assert (round(x0, 2), round(x1, 2)) == (0.21, 0.4)
    assert x0 > tokens[0].bbox[2], "must not cover the field label"


def test_multiline_entity_yields_one_rect_per_line():
    tokens = [
        Token("MIRA", 0, (0.8, 0.10, 0.95, 0.12), 0),
        Token("CALDER", 0, (0.05, 0.14, 0.20, 0.16), 1),
    ]
    tt = TokenText(tokens)
    span = Span("PERSON", 0, len(tt.text), 0.9, tt.text, [0, 1])
    rects = tt.rects_for(span)
    # Two boxes, not one tall box swallowing the right margin and the gap.
    assert len(rects) == 2
    assert all(r[1][3] - r[1][1] < 0.05 for r in rects), "boxes must stay line-height"


def test_assign_lines_groups_by_baseline():
    tokens = [
        Token("CUSTOMER", 0, (0.10, 0.100, 0.20, 0.120)),
        Token("RECORD", 0, (0.21, 0.101, 0.31, 0.121)),
        Token("SUMMARY", 0, (0.10, 0.200, 0.22, 0.220)),
    ]
    assign_lines(tokens)
    assert tokens[0].line == tokens[1].line
    assert tokens[2].line != tokens[0].line


def test_reading_order_is_geometric_not_extraction_order():
    """Separate PDF objects may be extracted column-by-column."""
    tokens = [
        Token("LEFT-TOP", 0, (0.05, 0.10, 0.20, 0.12)),
        Token("LEFT-BOTTOM", 0, (0.05, 0.20, 0.22, 0.22)),
        Token("RIGHT-TOP", 0, (0.70, 0.10, 0.90, 0.12)),
        Token("RIGHT-BOTTOM", 0, (0.70, 0.20, 0.93, 0.22)),
    ]
    assign_lines(tokens)
    ordered = reading_order(tokens)
    assert [token.text for token in ordered] == [
        "LEFT-TOP", "RIGHT-TOP", "LEFT-BOTTOM", "RIGHT-BOTTOM"
    ]


def test_already_masked_requires_every_line_to_be_covered():
    tokens = [
        Token("SYNTHETIC", 0, (0.10, 0.10, 0.30, 0.12), 0),
        Token("IDENTITY", 0, (0.10, 0.20, 0.28, 0.22), 1),
    ]
    tt = TokenText(tokens)
    span = Span("PERSON", 0, len(tt.text), 0.9, tt.text, [0, 1])
    first_only = [((0.10, 0.10, 0.30, 0.12), "<NAME#0>")]
    both = first_only + [((0.10, 0.20, 0.28, 0.22), "<NAME#1>")]
    assert not _already_masked(tt, span, first_only)
    assert _already_masked(tt, span, both)


def test_explicit_empty_detector_configuration_stays_empty():
    assert Detector(entities=[]).entities == []
    assert Detector(allowlist=[]).allowlist == set()
    assert Detector(entities=[], mask_providers=True).entities == []


def test_invalid_calendar_date_is_not_detected():
    assert DateDetector().detect(TokenText.from_text("31/02/2024")) == []


def test_registry_is_stable_across_pages():
    reg = TagRegistry()
    # Same surname, three surface forms, three different pages.
    assert reg.tag("PERSON", "CALDER,") == "<NAME#0>"
    assert reg.tag("PERSON", "Calder") == "<NAME#0>"
    assert reg.tag("PERSON", " calder ") == "<NAME#0>"
    # A different value gets the next index in the same type.
    assert reg.tag("PERSON", "MIRA") == "<NAME#1>"
    # Counters are per entity type, so MRNs start again at 0. The displayed
    # name is the short one by default, so the tag fits an exact-size mask.
    assert reg.tag("GENERIC_ID", "CUS-4827-9136") == "<ID#0>"
    assert len(reg) == 3


def test_normalize_preserves_internal_structure():
    # Edge punctuation goes; the internals of an email must survive.
    assert normalize("(MIRA.CALDER@EXAMPLE.NET)") == "mira.calder@example.net"
    assert normalize("CALDER,  MIRA") == "calder mira"


def test_merge_spans_collapses_overlaps():
    spans = [
        Span("US_SSN", 10, 21, 0.85, "000-00-0000"),
        Span("PHONE_NUMBER", 10, 21, 0.40, "000-00-0000"),
        Span("EMAIL_ADDRESS", 30, 50, 0.99, "a@b.com"),
    ]
    kept = merge_spans(spans)
    assert len(kept) == 2
    assert kept[0].entity == "US_SSN", "higher score wins an exact overlap"
    assert kept[1].entity == "EMAIL_ADDRESS"


def test_merge_spans_prefers_longer_match():
    spans = [
        Span("PERSON", 0, 4, 0.85, "MIRA"),
        Span("PERSON", 0, 11, 0.60, "MIRA CALDER"),
    ]
    kept = merge_spans(spans)
    # Longer wins even at lower score: masking too much is safe, too little is a leak.
    assert len(kept) == 1 and kept[0].end == 11


def test_verifier_catches_a_leak(tmp="/tmp/piimask_verify_probe.txt"):
    """The safety net itself. If this breaks, every other guarantee is a claim."""
    Path(tmp).write_text("Customer <PERSON#0> SSN 000-00-0000 remains readable\n")
    report = Report(source="x", output=tmp, engine="none")
    report.findings.append(Finding("US_SSN", "000-00-0000", "<US_SSN#0>", 0.9, "presidio"))
    leaked, _patterns, _flagged = _verify(tmp, report)
    assert leaked == ["000-00-0000"]


def test_verifier_ignores_its_own_tags(tmp="/tmp/piimask_verify_probe2.txt"):
    """Regression: masking a column header "Email" was then "found" inside the

    <EMAIL_ADDRESS#0> that replaced it, so a correctly masked file was rejected.
    """
    Path(tmp).write_text("Customer,ID,<EMAIL_ADDRESS#0>,Balance\n")
    report = Report(source="x", output=tmp, engine="none")
    report.findings.append(Finding("PERSON", "Email", "<EMAIL_ADDRESS#0>", 0.6, "presidio"))
    assert _verify(tmp, report) == ([], [], [])


def test_gate2_fails_on_what_no_detector_proposed(tmp="/tmp/piimask_gate2.txt"):
    """C4. Gate 1 can only ever check the detectors' own homework.

    An empty findings list means gate 1 has nothing to look for and passes
    trivially -- which is exactly the state a detection miss produces. Gate 2
    has to fail here with no lexicon and nothing found on the way in.
    """
    Path(tmp).write_text("Customer <NAME#0> SSN 000-00-0000 and 10/03/2026\n")
    report = Report(source="x", output=tmp, engine="none")
    leaked, patterns, _flagged = _verify(tmp, report)
    assert leaked == []
    assert "000-00-0000" in patterns
    assert "10/03/2026" in patterns


def test_gate2_flags_rather_than_fails_ambiguous_shapes(tmp="/tmp/piimask_gate2b.txt"):
    """A long digit run can be an organizational ID rather than personal PII.

    Failing on every unlabelled run would reject ordinary business documents,
    and a gate that fails on everything gets switched off.
    """
    Path(tmp).write_text("Organization ID 000123456 payee 000123456789\n")
    report = Report(source="x", output=tmp, engine="none")
    leaked, patterns, flagged = _verify(tmp, report)
    assert leaked == [] and patterns == []
    assert len(flagged) == 2


def test_report_withholds_values_by_default():
    """C3. The report is written next to the masked file and gets emailed."""
    report = Report(source="x", output="y", engine="none")
    report.findings.append(Finding("US_SSN", "000-00-0000", "<SSN#0>", 0.9, "pattern"))
    assert "000-00-0000" not in report.to_json()
    assert "<SSN#0>" in report.to_json()
    assert "000-00-0000" in report.to_json(values=True)


def test_gate2_ignores_table_artefacts():
    """Amounts and short OCR debris must not be treated as leaked PII."""
    from pii_mask_lito.pipeline import _verify_patterns

    leaked, _flagged = _verify_patterns("Charge 150.00 4904 ref 92@01\n")
    assert leaked == []
    # A real SSN on the same page still fails, one separator throughout.
    leaked, _flagged = _verify_patterns("Charge 150.00 4904 SSN 000-00-0000\n")
    assert leaked == ["000-00-0000"]


def test_barcode_payload_is_masked_and_stops_decoding():
    """C1. Rasterization preserves a symbol perfectly, and masking covered only

    the digits printed beside it. A verified, "clean" document could be decoded
    with a phone camera. Skipped when zxing-cpp is absent -- the pipeline
    records that case as a review flag rather than pretending it looked.
    """
    from pii_mask_lito import symbols

    if not symbols.available():
        return
    import numpy as np
    import zxingcpp
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    from pii_mask_lito import pdf as pdf_module
    from pii_mask_lito.pipeline import mask as mask_file

    src, dest = "/tmp/piimask_bc.pdf", "/tmp/piimask_bc_masked.pdf"
    barcode = zxingcpp.create_barcode("4827913650", zxingcpp.BarcodeFormat.Code128)
    image = Image.fromarray(
        np.array(zxingcpp.write_barcode_to_image(barcode, scale=4))
    ).convert("RGB").resize((600, 140), Image.Resampling.NEAREST)
    page = canvas.Canvas(src, pagesize=(612, 792))
    page.setFont("Helvetica", 11)
    page.drawString(60, 700, "Account #4827913650")
    page.drawImage(ImageReader(image), 60, 600, width=300, height=70)
    page.showPage()
    page.save()

    assert [r.text for r in zxingcpp.read_barcodes(pdf_module.page_images(src)[0])]
    report = mask_file(src, dest)
    assert any(f.source == "barcode" for f in report.findings)
    assert not [
        r.text
        for image in pdf_module.page_images(dest)
        for r in zxingcpp.read_barcodes(image)
    ], "output still decodes"


def test_signature_block_is_ink_without_words():
    """C5. Tesseract cannot read a signature, so it is invisible to detection,

    to propagation and to both verification gates. Geometry finds it instead:
    ink where OCR found no words, next to a label that says signature.
    """
    from PIL import Image, ImageDraw

    from pii_mask_lito import vision

    if not vision.available():
        return
    image = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(image)
    # A scribble to the right of the label, and a genuinely blank line below it.
    for offset in range(0, 200, 3):
        draw.line([(300 + offset, 60 + (offset % 30)), (310 + offset, 80)], fill="black", width=3)
    label = Token("Signature:", 0, (0.25, 0.13, 0.35, 0.17), 0)
    blank = Token("Witness:", 0, (0.25, 0.60, 0.35, 0.64), 1)
    found = vision.signatures(image, [label, blank])
    assert len(found) == 1, found
    assert found[0][1] < 0.5, "the signature block, not the empty witness line"


def test_typed_electronic_signature_text_is_not_a_signature_region():
    from PIL import Image, ImageDraw

    from pii_mask_lito import vision

    if not vision.available():
        return
    image = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(image)
    for offset in range(0, 180, 3):
        draw.line([(300 + offset, 70 + (offset % 30)), (310 + offset, 90)],
                  fill="black", width=3)
    tokens = [
        Token("Electronically", 0, (0.10, 0.20, 0.23, 0.25), 0),
        Token("signed", 0, (0.24, 0.20, 0.30, 0.25), 0),
    ]
    assert vision.signatures(image, tokens) == []


def test_yunet_weights_ship_and_haar_still_covers_for_them():
    """Safe Harbor #17 has no room for "we could not check".

    Both halves matter. The weights must actually be in the package -- a
    detector that silently fails to load reports every page face-free -- and the
    Haar fallback must still run without them, for a checkout or a trimmed wheel
    that has no package data.
    """
    from pathlib import Path

    from PIL import Image

    from pii_mask_lito import vision

    if not vision.available():
        return
    import cv2

    weights = Path(vision._weights())
    assert weights.exists(), f"YuNet weights missing from the package: {weights}"
    assert (weights.parent / "face_detection_yunet.LICENSE").exists(), \
        "vendored weights ship with their licence or they do not ship"
    # Loads, rather than merely existing on disk.
    assert cv2.FaceDetectorYN.create(str(weights), "", (64, 64), vision._FACE_SCORE)

    blank = Image.new("RGB", (400, 400), "white")
    assert vision.faces(blank) == [], "blank paper is not a face"
    original = vision._YUNET
    try:
        vision._YUNET = "absent.onnx"
        assert vision.faces(blank) == [], "the fallback must run, not raise"
    finally:
        vision._YUNET = original


def test_unread_ink_finds_handwriting_and_ignores_a_ruled_table():
    """The same "ink and no words" test as signatures(), off the label.

    The whole difficulty is false positives: a structured form is full of
    rules, borders and gridlines, and a detector that fires on those flags
    every page and gets ignored. The discriminator is spread in both axes -- a
    printed rule is one pixel row and every column, handwriting is neither.
    """
    from PIL import Image, ImageDraw

    from pii_mask_lito import vision

    if not vision.available():
        return
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    # A ruled table across the whole page: this must produce nothing.
    for y in range(100, 1500, 60):
        draw.line([(60, y), (1140, y)], fill="black", width=3)
    for x in range(60, 1141, 120):
        draw.line([(x, 100), (x, 1440)], fill="black", width=3)
    assert vision.unread_ink(image, []) == [], "ruled table is not handwriting"

    # A signature scrawled across a few cells of it.
    for offset in range(0, 260, 4):
        draw.line([(200 + offset, 300 + (offset % 70)), (215 + offset, 360)],
                  fill="black", width=5)
    found = vision.unread_ink(image, [])
    assert len(found) == 1, found
    x0, y0, x1, y1 = found[0]
    assert 0.1 < x0 < 0.25 and x1 > 0.35, found
    assert 0.15 < y0 < 0.25 and y1 < 0.35, found

    # Ink that OCR *did* read is not unread ink, whatever it looks like.
    covering = Token("scrawl", 0, (0.0, 0.0, 1.0, 1.0), 0)
    assert vision.unread_ink(image, [covering]) == []


def test_ocr_debris_over_a_scrawl_does_not_explain_its_ink():
    """A signature is ink OCR could not read, and OCR does not answer

    unreadable ink with silence -- it answers with garbage. Artificial
    low-confidence fragments exercise that failure mode: while a token was
    merely *present* or its box merely *covered* the strokes, debris could
    switch signature detection off. Confidence separates a word from a guess.
    """
    from PIL import Image, ImageDraw

    from pii_mask_lito import vision

    if not vision.available():
        return
    image = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(image)
    for offset in range(0, 260, 4):
        draw.line([(300 + offset, 90 + (offset % 60)), (315 + offset, 150)],
                  fill="black", width=5)
    label = Token("Signature:", 0, (0.10, 0.28, 0.22, 0.36), 0)
    # Debris blanketing the strokes, exactly as Tesseract reported it.
    debris = [
        Token("ALLL", 0, (0.33, 0.28, 0.47, 0.52), 0, confidence=0.52),
        Token("LIA", 0, (0.47, 0.28, 0.62, 0.52), 0, confidence=0.53),
    ]
    assert vision.signatures(image, [label, *debris]), \
        "a low-confidence guess must not count as having read the ink"

    # A real word there means it is a filled-in text field, not a signature.
    read = Token("CALDER", 0, (0.33, 0.28, 0.62, 0.52), 0, confidence=0.96)
    assert not vision.signatures(image, [label, read])


def test_bad_report_path_fails_before_any_work():
    """Masking a document can take a minute of OCR. A knowable path mistake

    from the argument line must not be discovered after paying for it -- the
    output has already been written by then, so it surfaces as a traceback
    stacked on top of a success.
    """
    import shutil

    from pii_mask_lito.cli import main as cli_main

    out = Path("/tmp/piimask_cli_probe")
    shutil.rmtree(out, ignore_errors=True)
    out.unlink(missing_ok=True)
    # --out does not exist, so with one input it is the output *file*, and the
    # report is aimed inside it.
    code = cli_main(["samples/note.txt", "-o", str(out), "--report", f"{out}/r.json"])
    assert code == 2
    assert not out.exists(), "refused runs must not leave output behind"

    out.mkdir(parents=True)
    assert cli_main(["samples/note.txt", "-o", str(out), "--report", f"{out}/r.json"]) == 0
    assert (out / "r.json").exists()
    shutil.rmtree(out, ignore_errors=True)


def test_cli_confidence_score_is_bounded():
    from pii_mask_lito.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["in.txt", "-o", "out.txt", "--min-score", "0.6"])
    assert args.min_score == 0.6
    with pytest.raises(SystemExit):
        parser.parse_args(["in.txt", "-o", "out.txt", "--min-score", "1.1"])


def test_cli_reports_missing_spacy_model_without_a_traceback(tmp_path, capsys):
    from pii_mask_lito.cli import main as cli_main

    out = tmp_path / "masked.txt"
    code = cli_main([
        "samples/note.txt", "-o", str(out),
        "--spacy-model", "pii_mask_missing_test_model",
    ])
    captured = capsys.readouterr()
    assert code == 1 and not out.exists()
    assert "FAIL input 1: spaCy model" in captured.err


def test_report_is_safe_by_default():
    """Paths, source values, and free-form review text may all contain PII."""
    import json

    from pii_mask_lito.pipeline import Finding, Report

    report = Report(
        source="/records/Jordan_Example_123456789.pdf",
        output="/exports/Jordan_Example_123456789_masked.pdf",
        engine="test",
        findings=[Finding("PERSON", "Jordan Example", "<NAME#0>", 0.9, "test")],
        leaked=["Jordan Example"],
        leaked_patterns=["123456789"],
        review=["possible address: 100 Example Avenue"],
    )

    safe = report.to_json()
    for sensitive in ("Jordan", "123456789", "100 Example Avenue", "/records", "/exports"):
        assert sensitive not in safe
    data = json.loads(safe)
    assert data["source"] == "<input>.pdf"
    assert data["output"] == "<output>.pdf"
    assert data["leaked_count"] == 1
    assert data["leaked_pattern_count"] == 1
    assert data["review_count"] == 1
    assert data["trace_count"] == 0

    unsafe = report.to_json(values=True)
    assert "Jordan Example" in unsafe
    assert "/records/Jordan_Example_123456789.pdf" in unsafe


def test_report_trace_is_withheld_by_default():
    report = Report(source="input.pdf", output="output.pdf", engine="test")
    report.doc_type = "Record for Sensitive Person"
    report.trace = [{"agent": "reader", "value": "Sensitive Person", "fields": 1}]
    safe = report.to_json()
    assert "Sensitive Person" not in safe
    assert '"trace": []' in safe
    assert '"trace_count": 1' in safe


def test_a_small_image_panel_is_still_read():
    """Every raster panel is eligible for OCR, regardless of page coverage."""
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    from pii_mask_lito import ocr
    from pii_mask_lito.detect import Detector
    from pii_mask_lito.pipeline import mask as mask_file

    if not ocr.available():
        return
    from pii_mask_lito.images import _font

    # Sized so the panel's pixels land roughly 1:1 at the default 300 DPI and
    # its text remains small but readable.
    panel = Image.new("RGB", (700, 130), "white")
    ImageDraw.Draw(panel).text((12, 30), "SSN 000-00-0000", fill="black", font=_font(64))
    src, dest = "/tmp/piimask_panel.pdf", "/tmp/piimask_panel_masked.pdf"
    page = canvas.Canvas(src, pagesize=(612, 792))
    page.setFont("Helvetica", 10)
    for row in range(60):  # text-rich, so the panel is a small fraction of it
        page.drawString(50, 740 - 11 * row, "Statement of account activity for the period. " * 2)
    # 168 x 31pt is under 1% of the page.
    page.drawImage(ImageReader(panel), 60, 60, width=168, height=31)
    page.showPage()
    page.save()

    report = mask_file(src, dest, detector=Detector(), verify=False)
    assert any(f.value == "000-00-0000" for f in report.findings), "panel never read"


def test_corpus_scores_above_its_floor():
    """P1. The end-to-end number the tuning constants are answerable to.

    Skipped until `python -m tests.corpus.build` has been run, so the normal
    suite stays fast and needs no generated fixtures.
    """
    from tests.corpus.score import OUT, mean_map_recall, score_corpus

    if not (OUT / "index.json").exists():
        return
    results = score_corpus()
    assert mean_map_recall(results) >= 0.90, results


def test_long_pdf_uses_disk_backed_page_storage(tmp_path, monkeypatch):
    """Long inputs must not retain every source and masked RGB page in RAM."""
    from reportlab.pdfgen import canvas

    from pii_mask_lito import pdf as pdf_module
    from pii_mask_lito.pipeline import _SPOOL_PAGE_THRESHOLD, mask as mask_file

    src, dest = tmp_path / "long.pdf", tmp_path / "masked.pdf"
    document = canvas.Canvas(str(src), pagesize=(180, 240))
    for page in range(_SPOOL_PAGE_THRESHOLD):
        document.drawString(12, 210, f"Synthetic long document page {page + 1}")
        document.showPage()
    document.save()

    called = []
    original = pdf_module.spool_page_images

    def observed(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pdf_module, "spool_page_images", observed)
    mask_file(str(src), str(dest), detector=Detector(entities=[]), verify=False)
    assert called and pdf_module.page_count(str(dest)) == _SPOOL_PAGE_THRESHOLD


def test_read_pages_keeps_the_real_page_number_on_a_subset():
    """The recheck loop reads a subset of pages; their numbers must survive.

    Page number is what the report anchors a finding to. Numbering a subset by
    its position in the sublist reports a hit on page 7 as a hit on page 0, and
    a reviewer chasing it looks at the wrong page. Checked in both modes,
    because only one of them goes through the thread pool.
    """
    from pii_mask_lito.ocr import read_pages

    class Engine:
        parallel_safe = True

        def words(self, image, page=0):
            return [Token(text=image, page=page)]

    class Serial(Engine):
        parallel_safe = False

    for engine in (Engine(), Serial()):
        got = read_pages(engine, ["a", "b", "c"], [2, 5, 7])
        assert [t[0].page for t in got] == [2, 5, 7], engine
        assert [t[0].text for t in got] == ["a", "b", "c"], "order must follow input"

    # Defaulting to 0..n-1 is right only when the images are the whole document.
    assert [t[0].page for t in read_pages(Engine(), ["a", "b"])] == [0, 1]

    try:
        read_pages(Engine(), ["a", "b"], [1])
        raise AssertionError("mismatched lengths should not be silently zipped")
    except ValueError:
        pass


def main():
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for check in checks:
        check()
        print(f"  ok  {check.__name__}")
    print(f"\n{len(checks)} checks passed")


if __name__ == "__main__":
    main()
