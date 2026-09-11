"""Draw replacement masks and compact tags onto image pixels."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# The mask is drawn at the exact size of the content it covers -- a 3x3 value
# gets a 3x3 box -- so surrounding layout is never displaced. OCR may report a
# box tighter than the visible glyphs; callers can add proportional padding when
# needed to cover antialiasing and descenders.
PADDING = 0.0

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(size: int):
    size = max(int(size), 1)
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(draw: ImageDraw.ImageDraw, text: str, width: float, height: float):
    """Largest font fitting inside the exact box, on both axes.

    Legibility comes from short tag names rather than widening the mask. See
    ``policy.SHORT_NAMES`` for the compact display vocabulary.
    """
    lo, hi, best = 3, max(int(height * 1.6), 4), None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _font(mid)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        if x1 - x0 <= width and y1 - y0 <= height:
            best, lo = font, mid + 1
        else:
            hi = mid - 1
    return best or _font(3)


def denormalize(bbox, size: tuple[int, int]) -> tuple[float, float, float, float]:
    w, h = size
    return (bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h)


def mask(
    image: Image.Image,
    boxes: list[tuple[tuple[float, float, float, float], str]],
    fill: str = "white",
    outline: str = "black",
    text_fill: str = "black",
    padding: float = PADDING,
) -> Image.Image:
    """Paint a box the exact size of each value and stamp its tag inside.

    `boxes` are (page-normalized bbox, replacement tag) pairs.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for bbox, tag in boxes:
        x0, y0, x1, y1 = denormalize(bbox, out.size)
        pad = padding * (y1 - y0)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        # The box is the size of what it covers. Nothing is widened to fit a
        # tag, and neighbouring boxes are not merged, so surrounding layout is
        # never displaced or obscured.
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline)
        if not tag:
            continue
        font = _fit(draw, tag, x1 - x0, y1 - y0)
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), tag, font=font)
        draw.text(
            (x0 + ((x1 - x0) - (tx1 - tx0)) / 2 - tx0,
             y0 + ((y1 - y0) - (ty1 - ty0)) / 2 - ty0),
            tag,
            font=font,
            fill=text_fill,
        )
    return out


def load(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def save(image: Image.Image, path: str, source: str) -> None:
    """Write back in the source's own format, per the same-format requirement."""
    fmt = (Image.open(source).format or "PNG").upper()
    if fmt in {"JPEG", "JPG"}:
        image.save(path, "JPEG", quality=95, subsampling=0)
    else:
        image.save(path, fmt)
