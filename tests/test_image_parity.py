"""Standalone images use the same text, symbol, and vision paths as PDFs."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pii_mask_lito import ocr, symbols, vision
from pii_mask_lito.detect import Detector
from pii_mask_lito.pipeline import mask
from pii_mask_lito.registry import TagRegistry

ACCOUNT = "4827913650"


def _page(tmp_path, *, barcode=False, signature=False):
    """A letter-shaped white page with printed text, optionally more."""
    from pii_mask_lito.images import _font

    image = Image.new("RGB", (1275, 1650), "white")
    draw = ImageDraw.Draw(image)
    body = _font(44)
    draw.text((60, 60), "CUSTOMER NAME: CALDER, MIRA", fill="black", font=body)
    draw.text((60, 125), "CUSTOMER ID: CUS-4827-9136", fill="black", font=body)
    if barcode:
        import zxingcpp

        try:
            symbol = zxingcpp.write_barcode_to_image(
                zxingcpp.create_barcode("1" + ACCOUNT, zxingcpp.BarcodeFormat.Code128)
            )
        except AttributeError:  # 2.x renamed the writer; both are in the wild
            symbol = zxingcpp.write_barcode(zxingcpp.BarcodeFormat.Code128, "1" + ACCOUNT)
        bar = Image.fromarray(np.array(symbol)).convert("RGB").resize((400, 120))
        image.paste(bar, (60, 200))
    if signature:
        draw.text((60, 800), "Signature:", fill="black")
        # Strokes, not a font: the point of the detector is that nothing reads it.
        points = [(230 + i * 6, 812 + int(28 * np.sin(i / 3.0))) for i in range(60)]
        draw.line(points, fill="black", width=4)
    path = tmp_path / "page.png"
    image.save(path)
    return path


def _mask(path, tmp_path, **kwargs):
    if not ocr.available():
        pytest.skip("an OCR extra is required for standalone image masking")
    out = tmp_path / "masked.png"
    return mask(str(path), str(out), detector=Detector(), registry=TagRegistry(),
                verify=False, **kwargs), out


@pytest.mark.skipif(not symbols.available(), reason="zxing-cpp not installed")
def test_barcode_in_a_png_is_decoded_and_masked(tmp_path):
    """The payload survives rasterization; a text-only path never sees it."""
    report, out = _mask(_page(tmp_path, barcode=True), tmp_path)
    assert any(ACCOUNT in (f.value or "") for f in report.findings), (
        "barcode payload was not detected on an image input"
    )
    # And the printed symbol must no longer decode out of the written file.
    assert not [d for d in symbols.decode(Image.open(out), page=0)
                if ACCOUNT in d.text]


@pytest.mark.skipif(not vision.available(), reason="opencv not installed")
def test_signature_in_a_png_is_masked(tmp_path):
    report, _ = _mask(_page(tmp_path, signature=True), tmp_path)
    assert any(f.entity == "SIGNATURE" for f in report.findings), (
        "signature ink on an image input was neither masked nor flagged"
    )


def test_image_input_still_masks_text(tmp_path):
    """The parity work must not cost the detection that already worked."""
    report, _ = _mask(_page(tmp_path), tmp_path)
    assert {f.entity for f in report.findings} & {"PERSON", "GENERIC_ID"}


def test_missing_symbol_reader_is_reported_not_silent(tmp_path, monkeypatch):
    """A capability that is absent must reach the report, not vanish."""
    monkeypatch.setattr(symbols, "available", lambda: False)
    report, _ = _mask(_page(tmp_path), tmp_path)
    assert any("zxing" in line for line in report.review)
