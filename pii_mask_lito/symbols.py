"""Decode barcodes and 2D symbols into the ordinary token stream.

Machine-readable symbols can carry identifiers that are absent from text
extraction even when a human-readable caption is masked. Emitting each decoded
payload as a token lets the normal detection, propagation, tagging, and
verification paths handle it consistently.

The fix is deliberately not a new detector. A decoded symbol is emitted as an
ordinary `Token` carrying its payload and its own bounding box, which drops it
into the same token stream as everything else -- so detection, the lexicon,
propagation and the tag registry all pick it up with no further work.

Decoded payloads also provide a clean source for values that OCR may read
imperfectly elsewhere in the same job.

zxing-cpp is Apache-2.0. pyzbar was rejected: it wraps zbar, which is LGPL-2.1,
and this project's dependency policy is Apache/BSD/MIT only -- the same rule
that ruled out PyMuPDF.
"""

from __future__ import annotations

import re

from PIL import Image

from .model import Token

# A payload with a run this long is an identifier, not a routing code. Symbols
# matching it are treated as identifiers whether or not a text detector also
# recognizes the value.
IDENTIFIER_RUN = re.compile(r"\d{6,}")


def available() -> bool:
    try:
        import zxingcpp  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def decode(image: Image.Image, page: int = 0) -> list[Token]:
    """Every decodable symbol on the page, as page-normalized tokens.

    Returns an empty list rather than raising when zxing-cpp is absent; the
    caller records that as a review flag, since "we could not look" and "we
    looked and found nothing" are different claims.
    """
    try:
        import zxingcpp
    except Exception:  # noqa: BLE001
        return []

    width, height = image.size
    tokens = []
    for result in zxingcpp.read_barcodes(image):
        text = (result.text or "").strip()
        if not text:
            continue
        position = result.position
        corners = (
            position.top_left,
            position.top_right,
            position.bottom_left,
            position.bottom_right,
        )
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        tokens.append(
            Token(
                text=text,
                page=page,
                bbox=(
                    max(min(xs) / width, 0.0),
                    max(min(ys) / height, 0.0),
                    min(max(xs) / width, 1.0),
                    min(max(ys) / height, 1.0),
                ),
            )
        )
    return tokens


def payload_is_identifier(text: str) -> bool:
    return bool(IDENTIFIER_RUN.search(text))
