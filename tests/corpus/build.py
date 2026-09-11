"""Build a license-clean synthetic PDF corpus with exact ground truth.

The layouts are original fixtures for this project. They represent customer
support, employee administration, and retail ordering so one industry cannot
silently determine the detector's behavior.

Run: ``python -m tests.corpus.build``
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PAGE = (612.0, 792.0)
HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"

_ASCENT = 0.718
_DESCENT = 0.207
_DESCENDERS = set("gjpqy")


@dataclass(frozen=True)
class Field:
    label: str
    value: str
    entity: str | None = None


@dataclass(frozen=True)
class Fixture:
    stem: str
    title: str
    summary: str
    fields: tuple[Field, ...]
    scan: tuple[int, int, float]
    signature: bool = False


FIXTURES = (
    Fixture(
        stem="support_case",
        title="Customer support case",
        summary="Delivery window change requested through the customer portal.",
        fields=(
            Field("Customer name", "MIRA CALDER", "PERSON"),
            Field("Customer ID", "CUS-4827-9136", "GENERIC_ID"),
            Field("Case ID", "CASE-2048-7713", "GENERIC_ID"),
            Field("Email", "mira.calder@example.net", "EMAIL_ADDRESS"),
            Field("Phone", "(202) 555-0147", "PHONE_NUMBER"),
            Field("Opened", "2026-09-11", "DATE"),
            Field("Mailing address", "1847 WILLOWGLASS LANE", "LOCATION"),
            Field("City and postal code", "PORT HAVEN, OR 97035", "LOCATION"),
            Field("IP address", "192.0.2.44", "IP_ADDRESS"),
        ),
        scan=(300, 84, 0.0),
    ),
    Fixture(
        stem="employee_record",
        title="Employee equipment handover",
        summary="Equipment is prepared for courier delivery before the start date.",
        fields=(
            Field("Employee name", "DORIAN VALE", "PERSON"),
            Field("Employee ID", "EMP-7305-A19", "GENERIC_ID"),
            Field("Work email", "dorian.vale@example.net", "EMAIL_ADDRESS"),
            Field("Mobile phone", "+1 202-555-0183", "PHONE_NUMBER"),
            Field("Start date", "2026-10-12", "DATE"),
            Field("Home address", "62 JUNIPER ARCADE", "LOCATION"),
            Field("City and postal code", "NORTH EMBER, WA 98118", "LOCATION"),
            Field("Document ID", "DOC-6198-TEAL", "GENERIC_ID"),
        ),
        scan=(220, 68, 0.35),
        signature=True,
    ),
    Fixture(
        stem="order_receipt",
        title="Online order receipt",
        summary="One desk lamp is ready to ship from the regional warehouse.",
        fields=(
            Field("Customer", "NIA QUILL", "PERSON"),
            Field("Order ID", "ORDER-8451-2206", "GENERIC_ID"),
            Field("Account number", "ACCT-9082-4417", "ACCOUNT_NUMBER"),
            Field("Email", "nia.quill@example.net", "EMAIL_ADDRESS"),
            Field("Telephone", "+1 202-555-0166", "PHONE_NUMBER"),
            Field("Order date", "2026-08-29", "DATE"),
            Field("Ship to", "409 ORCHARD LANTERN ROAD", "LOCATION"),
            Field("City and postal code", "SILVER BAY, CA 94107", "LOCATION"),
            Field("Reference ID", "REF-5509-ZINC", "GENERIC_ID"),
        ),
        scan=(170, 52, -0.6),
    ),
)


def _truth(page: canvas.Canvas, x: float, baseline: float, text: str,
           size: float, entity: str) -> dict:
    width = page.stringWidth(text, "Helvetica", size)
    top = baseline + _ASCENT * size
    bottom = baseline - (_DESCENT * size if set(text) & _DESCENDERS else 0.0)
    return {
        "entity": entity,
        "value": text,
        "bbox": [
            x / PAGE[0],
            1 - top / PAGE[1],
            (x + width) / PAGE[0],
            1 - bottom / PAGE[1],
        ],
    }


def _draw_header(page: canvas.Canvas, fixture: Fixture) -> None:
    page.setFillColor(HexColor("#1F4E78"))
    page.rect(0, 720, PAGE[0], 72, stroke=0, fill=1)
    page.setFillColor(HexColor("#FFFFFF"))
    page.setFont("Helvetica-Bold", 20)
    page.drawString(48, 752, fixture.title)
    page.setFont("Helvetica", 9)
    page.drawString(48, 735, "Synthetic detector fixture")
    page.setFillColor(HexColor("#111827"))
    page.setFont("Helvetica", 10)
    page.drawString(48, 686, fixture.summary)


def _draw_fields(page: canvas.Canvas, fields: tuple[Field, ...]) -> list[dict]:
    truth: list[dict] = []
    left = 48.0
    column_width = 258.0
    start_y = 632.0
    row_height = 76.0
    for index, field in enumerate(fields):
        column = index % 2
        row = index // 2
        x = left + column * column_width
        y = start_y - row * row_height
        page.setFillColor(HexColor("#F3F6F8"))
        page.roundRect(x, y - 42, 226, 54, 5, stroke=0, fill=1)
        page.setFillColor(HexColor("#4B5563"))
        page.setFont("Helvetica-Bold", 7.5)
        page.drawString(x + 11, y - 3, field.label.upper())
        page.setFillColor(HexColor("#111827"))
        page.setFont("Helvetica", 10)
        baseline = y - 25
        page.drawString(x + 11, baseline, field.value)
        if field.entity:
            truth.append(_truth(page, x + 11, baseline, field.value, 10, field.entity))
    return truth


def _draw_activity(page: canvas.Canvas, y: float) -> None:
    page.setFont("Helvetica-Bold", 9)
    page.setFillColor(HexColor("#1F4E78"))
    page.drawString(48, y, "ACTIVITY")
    page.setFillColor(HexColor("#111827"))
    page.setFont("Helvetica", 8.5)
    rows = (
        ("SKU-LAMP-14", "Desk lamp", "1", "$64.00"),
        ("SHIP-GROUND", "Ground delivery", "1", "$8.00"),
        ("TAX-LOCAL", "Sales tax", "", "$5.76"),
    )
    page.drawString(48, y - 20, "CODE")
    page.drawString(174, y - 20, "DESCRIPTION")
    page.drawString(382, y - 20, "QTY")
    page.drawString(458, y - 20, "AMOUNT")
    for index, row in enumerate(rows):
        baseline = y - 40 - index * 19
        for x, value in zip((48, 174, 382, 458), row):
            page.drawString(x, baseline, value)


def _draw_signature(page: canvas.Canvas, y: float) -> dict:
    page.setFont("Helvetica", 9)
    page.drawString(48, y, "Recipient signature")
    left, right = 174.0, 360.0
    bottom, top = y - 13.0, y + 10.0
    page.setLineWidth(1.2)
    points = ((left, y - 3), (205, y + 8), (226, y - 8), (251, y + 5),
              (279, y - 5), (314, y + 7), (right, y - 2))
    for first, second in zip(points, points[1:]):
        page.line(first[0], first[1], second[0], second[1])
    page.setLineWidth(1)
    return {
        "entity": "SIGNATURE",
        "value": "",
        "bbox": [left / PAGE[0], 1 - top / PAGE[1], right / PAGE[0], 1 - bottom / PAGE[1]],
    }


def build_document(path: Path, fixture: Fixture) -> list[list[dict]]:
    page = canvas.Canvas(str(path), pagesize=PAGE, pageCompression=1)
    _draw_header(page, fixture)
    truth = _draw_fields(page, fixture.fields)
    _draw_activity(page, 236)
    if fixture.signature:
        truth.append(_draw_signature(page, 132))
    page.setFont("Helvetica", 7.5)
    page.setFillColor(HexColor("#6B7280"))
    page.drawString(48, 38, "All names and identifiers on this page are fictional.")
    page.showPage()
    page.save()
    return [truth]


def _rotate_truth(truth: list[list[dict]], degrees: float) -> list[list[dict]]:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotated: list[list[dict]] = []
    for page_entries in truth:
        entries = []
        for original in page_entries:
            x0, y0, x1, y1 = original["bbox"]
            xs, ys = [], []
            for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
                dx, dy = x - 0.5, y - 0.5
                xs.append(0.5 + cosine * dx + sine * dy)
                ys.append(0.5 - sine * dx + cosine * dy)
            entry = dict(original)
            entry["bbox"] = [
                max(0.0, min(xs)),
                max(0.0, min(ys)),
                min(1.0, max(xs)),
                min(1.0, max(ys)),
            ]
            entries.append(entry)
        rotated.append(entries)
    return rotated


def degrade(src: Path, dest: Path, dpi: int, quality: int, skew: float) -> None:
    from PIL import Image

    from pii_mask_lito import pdf

    output = canvas.Canvas(str(dest), pagesize=PAGE, pageCompression=1)
    for image in pdf.page_images(str(src), dpi):
        if skew:
            image = image.rotate(skew, expand=False, fillcolor="white")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        output.drawImage(ImageReader(Image.open(buffer)), 0, 0, width=PAGE[0], height=PAGE[1])
        output.showPage()
    output.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for fixture in FIXTURES:
        clean = OUT / f"{fixture.stem}.pdf"
        truth = build_document(clean, fixture)
        clean_truth = OUT / f"{fixture.stem}.truth.json"
        clean_truth.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
        index.append(clean.name)

        scan = OUT / f"{fixture.stem}_scan.pdf"
        degrade(clean, scan, *fixture.scan)
        scan_truth = _rotate_truth(truth, fixture.scan[2]) if fixture.scan[2] else truth
        (OUT / f"{fixture.stem}_scan.truth.json").write_text(
            json.dumps(scan_truth, indent=2) + "\n", encoding="utf-8"
        )
        index.append(scan.name)

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(index)} documents to {OUT}")


if __name__ == "__main__":
    main()
