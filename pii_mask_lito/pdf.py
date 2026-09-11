"""PDF in, PDF out. Extraction adapts per page; rendering never does.

Extraction: a page with a usable text layer yields exact word boxes from
`get_charbox()`; a page without one is rendered and handed to OCR. Mixed pages
can contain both native text and embedded raster regions, so the choice is made
per page and per image region.

Rendering: every page is rasterized, masked as pixels, and reassembled with an
invisible text layer carrying the *masked* text. Uniform on purpose. It means a
PHI string cannot survive in a text layer, annotation, form field, OCG layer or
embedded original, because none of those survive at all.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from pathlib import Path

import pypdfium2 as pdfium
from pypdfium2 import raw as pdfium_c
from PIL import Image
from reportlab import rl_config
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .model import Token, assign_lines, reading_order

# 300 DPI is a conservative default for compact text in scanned documents.
DEFAULT_DPI = 300
# Below this many characters a "text layer" is usually just a stray watermark
# or a scanner's failed attempt, and OCR gives better coverage.
MIN_CHARS_FOR_TEXT_LAYER = 20
# Keep OCR batches small enough that a long 300-DPI document does not turn into
# one full-document allocation. Four letter pages are roughly 100 MiB as RGB.
OCR_BATCH_IMAGES = 4


class DiskImageStore(MutableMapping[int, Image.Image]):
    """A dict-like image collection whose pixel buffers live on disk.

    Values are loaded as independent RGB images on access and saved as
    low-compression PNGs on assignment. The store deliberately implements the
    ordinary mapping interface used by the masking and PDF-writing code, so the
    long-document path changes storage rather than detection behavior.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._keys: set[int] = set()

    def _path(self, key: int) -> Path:
        return self.root / f"page-{key:06d}.png"

    def __getitem__(self, key: int) -> Image.Image:
        if key not in self._keys:
            raise KeyError(key)
        with Image.open(self._path(key)) as source:
            image = source.convert("RGB")
            image.load()
        return image

    def __setitem__(self, key: int, image: Image.Image) -> None:
        path = self._path(key)
        pending = path.with_suffix(".pending.png")
        try:
            image.save(pending, "PNG", compress_level=1)
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
        self._keys.add(key)

    def __delitem__(self, key: int) -> None:
        if key not in self._keys:
            raise KeyError(key)
        self._path(key).unlink(missing_ok=True)
        self._keys.remove(key)

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._keys))

    def __len__(self) -> int:
        return len(self._keys)


def _words_from_text_layer(page, page_index: int) -> list[Token]:
    """Group characters into whitespace-delimited words with union boxes."""
    width, height = page.get_size()
    textpage = page.get_textpage()
    try:
        n = textpage.count_chars()
        if n < MIN_CHARS_FOR_TEXT_LAYER:
            return []
        tokens: list[Token] = []
        buf: list[str] = []
        box: list[float] | None = None

        def flush():
            nonlocal buf, box
            if buf and box:
                # PDF space is y-up from bottom-left; normalize to y-down.
                tokens.append(
                    Token(
                        text="".join(buf),
                        page=page_index,
                        bbox=(
                            box[0] / width,
                            1 - box[3] / height,
                            box[2] / width,
                            1 - box[1] / height,
                        ),
                    )
                )
            buf, box = [], None

        for i in range(n):
            ch = textpage.get_text_range(i, 1)
            if not ch or ch.isspace():
                flush()
                continue
            x0, y0, x1, y1 = textpage.get_charbox(i)
            buf.append(ch)
            box = (
                [x0, y0, x1, y1]
                if box is None
                else [min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)]
            )
        flush()
        return tokens
    finally:
        textpage.close()


def _image_regions(page) -> list[tuple[float, float, float, float]]:
    """Normalized y-down rectangles of raster images painted on ``page``."""
    width, height = page.get_size()
    regions = []
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        x0, y0, x1, y1 = obj.get_bounds()
        left, right = sorted((x0 / width, x1 / width))
        top, bottom = sorted((1 - y1 / height, 1 - y0 / height))
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(1.0, right), min(1.0, bottom)
        # Degenerate decorative objects cannot contain readable content.
        if right - left > 1e-4 and bottom - top > 1e-4:
            regions.append((left, top, right, bottom))
    return _merge_regions(regions)


def _merge_regions(regions: list[tuple[float, float, float, float]]):
    """Merge overlapping image objects so the same pixels are OCR'd once."""
    merged: list[tuple[float, float, float, float]] = []
    for region in sorted(regions):
        rx0, ry0, rx1, ry1 = region
        changed = True
        while changed:
            changed = False
            kept = []
            for x0, y0, x1, y1 in merged:
                overlaps = rx0 < x1 and x0 < rx1 and ry0 < y1 and y0 < ry1
                if overlaps:
                    rx0, ry0 = min(rx0, x0), min(ry0, y0)
                    rx1, ry1 = max(rx1, x1), max(ry1, y1)
                    changed = True
                else:
                    kept.append((x0, y0, x1, y1))
            merged = kept
        merged.append((rx0, ry0, rx1, ry1))
    return merged


def _crop(image: Image.Image, region: tuple[float, float, float, float]):
    """Return a crop and its actual pixel-aligned normalized page bounds."""
    import math

    x0, y0, x1, y1 = region
    width, height = image.size
    pixels = (
        max(0, math.floor(x0 * width)),
        max(0, math.floor(y0 * height)),
        min(width, math.ceil(x1 * width)),
        min(height, math.ceil(y1 * height)),
    )
    actual = (
        pixels[0] / width,
        pixels[1] / height,
        pixels[2] / width,
        pixels[3] / height,
    )
    return image.crop(pixels), actual


def _map_crop_tokens(tokens: list[Token], page: int,
                     region: tuple[float, float, float, float]) -> list[Token]:
    """Map crop-relative OCR boxes back into normalized page coordinates."""
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    for token in tokens:
        token.page = page
        if token.bbox is not None:
            bx0, by0, bx1, by1 = token.bbox
            token.bbox = (
                x0 + bx0 * width,
                y0 + by0 * height,
                x0 + bx1 * width,
                y0 + by1 * height,
            )
    return tokens


def render_page(page, dpi: int = DEFAULT_DPI) -> Image.Image:
    return page.render(scale=dpi / 72).to_pil().convert("RGB")


def page_count(src: str) -> int:
    pdf = pdfium.PdfDocument(src)
    try:
        return len(pdf)
    finally:
        pdf.close()


def iter_page_images(src: str, dpi: int = DEFAULT_DPI) -> Iterator[Image.Image]:
    """Render pages one at a time, keeping at most the yielded page resident."""
    pdf = pdfium.PdfDocument(src)
    try:
        for index in range(len(pdf)):
            yield render_page(pdf[index], dpi)
    finally:
        pdf.close()


def spool_page_images(src: str, root: str | Path,
                      dpi: int = DEFAULT_DPI) -> DiskImageStore:
    """Render a reusable document into a disk-backed image mapping."""
    store = DiskImageStore(root)
    for index, image in enumerate(iter_page_images(src, dpi)):
        try:
            store[index] = image
        finally:
            image.close()
    return store


def requires_ocr(path: str) -> bool:
    """Does any page have no usable text layer, making OCR mandatory?

    Note what this does *not* decide: whether to OCR. `extract` OCRs every page
    it can, unconditionally. This answers the narrower question of whether the
    document can be masked at all with no OCR engine installed, which is the one
    case where a missing engine has to be fatal rather than a review flag.

    Image coverage is deliberately not a deciding factor. A small embedded ID
    panel on an otherwise text-rich page can still contain sensitive content,
    so every raster region is eligible for OCR.
    """
    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(len(pdf)):
            textpage = pdf[i].get_textpage()
            try:
                if textpage.count_chars() < MIN_CHARS_FOR_TEXT_LAYER:
                    return True
            finally:
                textpage.close()
        return False
    finally:
        pdf.close()


def _overlaps(a, b, threshold: float = 0.3) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return smaller > 0 and inter / smaller > threshold


def extract(path: str, ocr=None, dpi: int = DEFAULT_DPI, rendered=None,
            with_provenance: bool = False):
    """Per-page tokens, from the text layer and OCR together where needed.

    Returns (tokens_per_page, page_sizes_in_points).

    `rendered` lets the caller supply page images it already has. The masking
    pipeline renders every page anyway -- to draw on -- and rendering them a
    second time here was pure duplicated work, plus a second full set of 300 DPI
    images resident at once, which can consume gigabytes on a long document.
    """
    pdf = pdfium.PdfDocument(path)
    try:
        sizes, layers, regions = [], [], []
        for i in range(len(pdf)):
            page = pdf[i]
            sizes.append(page.get_size())
            layers.append(_words_from_text_layer(page, i))
            regions.append(_image_regions(page))
        if ocr is None:
            for tokens in layers:
                assign_lines(tokens)
            ordered = [reading_order(tokens) for tokens in layers]
            if with_provenance:
                return ordered, sizes, [
                    {"full_ocr": False, "image_regions": 0, "ocr_supplements": 0}
                    for _ in ordered
                ]
            return ordered, sizes

        from .ocr import read_pages

        images = rendered if rendered is not None else [
            render_page(pdf[i], dpi) for i in range(len(pdf))
        ]
        # A page without a usable text layer is OCR'd in full. On a native PDF,
        # only embedded raster regions need OCR: re-reading thousands of exact
        # text-layer glyphs is both expensive and a source of competing, less
        # accurate boxes. Small image panels are still read; they are cropped,
        # never ignored by an area threshold.
        jobs: list[tuple[int, tuple[float, float, float, float], Image.Image]] = []
        found_per_page: list[list[Token]] = [[] for _ in layers]
        provenance = []

        def flush_jobs() -> None:
            nonlocal jobs
            if not jobs:
                return
            current, jobs = jobs, []
            try:
                reads = read_pages(ocr, [job[2] for job in current])
                for (page_index, region, _crop_image), found in zip(current, reads):
                    found_per_page[page_index].extend(
                        _map_crop_tokens(found, page_index, region)
                    )
            finally:
                for _page_index, _region, crop_image in current:
                    crop_image.close()

        for page_index, (tokens, page_regions) in enumerate(zip(layers, regions)):
            image = images[page_index]
            try:
                full_ocr = not tokens
                selected = [(0.0, 0.0, 1.0, 1.0)] if full_ocr else page_regions
                provenance.append({
                    "full_ocr": full_ocr,
                    "image_regions": len(selected),
                    "ocr_supplements": 0,
                })
                for region in selected:
                    cropped, actual_region = _crop(image, region)
                    jobs.append((page_index, actual_region, cropped))
                    if len(jobs) >= OCR_BATCH_IMAGES:
                        flush_jobs()
            finally:
                # Disk-backed sequences return a fresh image on every access.
                # Closing here releases it immediately; callers supplying an
                # ordinary list retain ownership of their reusable images.
                if isinstance(images, DiskImageStore):
                    image.close()
        flush_jobs()

        pages = []
        for page_index, (tokens, found) in enumerate(zip(layers, found_per_page)):
            # Rendering paints the text layer too, so OCR re-reads whatever the
            # text layer already gave us. Keep the text-layer token -- its box
            # is exact -- and take OCR only for the rest.
            boxes = [t.bbox for t in tokens if t.bbox]
            supplements = [
                t for t in found
                if t.bbox and not any(_overlaps(t.bbox, b) for b in boxes)
            ]
            # Multiple image objects may render the same visual content. Keep
            # one OCR box for each location after native-text de-duplication.
            accepted = []
            for token in supplements:
                if not any(
                    token.bbox and other.bbox and _overlaps(token.bbox, other.bbox)
                    and token.text.casefold() == other.text.casefold()
                    for other in accepted
                ):
                    accepted.append(token)
            provenance[page_index]["ocr_supplements"] = len(accepted)
            tokens = tokens + accepted
            assign_lines(tokens)
            pages.append(reading_order(tokens))
        return (pages, sizes, provenance) if with_provenance else (pages, sizes)
    finally:
        pdf.close()


def write(
    src: str,
    dest: str,
    masked_images: MutableMapping[int, Image.Image],
    text_layers: dict[int, list[tuple[tuple[float, float, float, float], str]]],
    dpi: int = DEFAULT_DPI,
) -> None:
    """Reassemble a PDF from masked page images plus invisible masked text.

    `text_layers` maps page index to (normalized bbox, text) pairs. It is drawn
    in render mode 3 -- invisible -- so the output stays searchable and
    copy/pasteable while containing only already-masked text.
    """
    # Binary image streams are smaller and substantially quicker to encode than
    # ReportLab's optional ASCII85 wrapper; PDF readers support them directly.
    rl_config.useA85 = False
    pdf = pdfium.PdfDocument(src)
    try:
        out = canvas.Canvas(dest)
        for i in range(len(pdf)):
            page = pdf[i]
            w_pt, h_pt = page.get_size()
            stored = i in masked_images
            image = masked_images[i] if stored else render_page(page, dpi)
            try:
                out.setPageSize((w_pt, h_pt))
                out.drawImage(ImageReader(image), 0, 0, width=w_pt, height=h_pt)

                entries = text_layers.get(i) or []
                if entries:
                    text = out.beginText()
                    text.setTextRenderMode(3)
                    for (x0, y0, x1, y1), value in entries:
                        size = max((y1 - y0) * h_pt, 1)
                        text.setFont("Helvetica", size)
                        # Back to PDF's y-up origin, sitting on the baseline.
                        text.setTextOrigin(x0 * w_pt, (1 - y1) * h_pt)
                        text.textOut(value)
                    out.drawText(text)
                out.showPage()
            finally:
                if not stored or isinstance(masked_images, DiskImageStore):
                    image.close()
        out.save()
    finally:
        pdf.close()


def page_images(src: str, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    return list(iter_page_images(src, dpi))


def text_of(src: str) -> str:
    """Flat text of a PDF, for the read-back verification pass."""
    pdf = pdfium.PdfDocument(src)
    try:
        chunks = []
        for i in range(len(pdf)):
            tp = pdf[i].get_textpage()
            try:
                chunks.append(tp.get_text_range())
            finally:
                tp.close()
        return "\n".join(chunks)
    finally:
        pdf.close()
