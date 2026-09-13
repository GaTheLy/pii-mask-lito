"""Detect visual content that text extraction cannot represent.

Faces, handwritten signatures, and other unread ink have no dependable text
span for an ordinary detector to match. This module reports page-normalized
regions for those pixels. Unknown ink is a review signal by default because its
meaning is ambiguous; callers may choose to cover it when risk tolerance favors
recall over preserving decorative content.

opencv-python is Apache-2.0.
"""

from __future__ import annotations

import re

from PIL import Image

# YuNet, with the Haar cascade kept as a fallback. YuNet is a small CNN: far
# fewer false positives than Haar, and it handles profiles and angled faces.
# The small weights are vendored rather than fetched at runtime because a
# masking tool that silently degrades when it cannot reach the network is a
# masking tool that silently under-masks.
#
# face_detection_yunet_2023mar.onnx, OpenCV Zoo. Licence checked before
# vendoring and it is MIT (Copyright (c) 2020 Shiqi Yu) -- not Apache-2.0 as the
# rest of opencv_zoo's tooling is -- which is inside this project's
# Apache/BSD/MIT dependency policy. The text is in pii_mask_lito/data/ beside it.
#
# Haar stays reachable for an install where the weights are missing: a
# source checkout without package data, or a trimmed wheel. Its failure mode is
# the safe one -- it over-detects, and an over-detection costs one white box.
_YUNET = "face_detection_yunet_2023mar.onnx"
_CASCADE = "haarcascade_frontalface_default.xml"
# YuNet reports a confidence per detection. 0.7 is OpenCV's own default and is
# kept at the model's documented default. A very low threshold turns ordinary
# page graphics into face detections; unread-ink review remains an independent
# signal for images the face model cannot classify.
_FACE_SCORE = 0.7
# Below this fraction of a page's width a "face" is noise in a logo or a form
# rule, not a photograph.
_MIN_FACE = 0.02
_YUNET_CACHE: tuple[str, object] | None = None
_HAAR_CACHE: tuple[str, object] | None = None

# Ink coverage above which a region with no readable words in it is treated as
# handwriting rather than as blank paper. Printed text sits far higher than
# this, but printed text is excluded by the no-tokens test before we get here,
# so the only thing this has to separate is ink from an empty ruled line.
_INK_FRACTION = 0.015
_INK_LEVEL = 200  # 8-bit grey below this counts as ink
# Confidence at which a token counts as having *read* the ink under it. Well
# above ocr.LOW_CONFIDENCE (0.40) on purpose: that floor is a "keep it, it might
# be a word" threshold and errs towards keeping debris, while this is a "trust
# it, that ink is accounted for" threshold and must err the other way.
_READ_CONFIDENCE = 0.6


def _looks_read(text: str) -> bool:
    """Could this token be a genuine reading, or debris over ink nobody read?

    Confidence alone is not enough: an OCR engine can confidently turn a
    handwritten stroke into random alphanumeric text. The additional shape
    check accepts words and digit-heavy identifiers while treating mixed,
    letter-heavy fragments as unexplained ink.
    """
    stripped = re.sub(r"\W", "", text)
    if len(stripped) < 2:
        return False
    alpha = sum(c.isalpha() for c in stripped)
    digits = sum(c.isdigit() for c in stripped)
    if alpha == 0 or digits == 0:
        # All one kind: a word or a number.
        return True
    # A letter/digit mix is a genuine reading only when digits clearly
    # outnumber letters, as many structured identifiers do. Letter-heavy mixed
    # fragments are treated as unexplained ink even when OCR confidence is high.
    return digits > alpha

SIGNATURE_LABELS = ("signature", "signed", "sign here", "authorized by")
_SIGNATURE_LABEL = re.compile(r"\b(?:signature|signed|sign\s+here|authorized\s+by)\b")
_ELECTRONIC_SIGNATURE = {"electronically", "digitally"}


def _is_signature_label(index: int, tokens: list) -> bool:
    """Whether a token labels nearby handwritten ink, not typed e-sign text."""
    token = tokens[index]
    folded = token.text.casefold()
    if not _SIGNATURE_LABEL.search(folded):
        return False
    if _ELECTRONIC_SIGNATURE & set(re.findall(r"[a-z]+", folded)):
        return False
    # OCR normally splits a line into words, so "electronically signed" may be
    # two tokens. A typed e-sign statement does not imply adjacent handwriting.
    return not any(
        other.page == token.page
        and other.line == token.line
        and other.text.casefold().strip(" .,:;") in _ELECTRONIC_SIGNATURE
        for other in tokens
    )


def available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _weights() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent / "data" / _YUNET)


def _yunet_detector(cv2, size: tuple[int, int]):
    """Reuse YuNet model state while updating its page-specific input size."""
    global _YUNET_CACHE

    weights = _weights()
    if _YUNET_CACHE is not None and _YUNET_CACHE[0] == weights:
        detector = _YUNET_CACHE[1]
        try:
            detector.setInputSize(size)
            return detector
        except Exception:  # noqa: BLE001 - recreate below
            _YUNET_CACHE = None
    detector = cv2.FaceDetectorYN.create(weights, "", size, _FACE_SCORE)
    _YUNET_CACHE = (weights, detector)
    return detector


def _haar_detector(cv2):
    """Load OpenCV's fallback cascade once per process."""
    global _HAAR_CACHE

    path = cv2.data.haarcascades + _CASCADE
    if _HAAR_CACHE is None or _HAAR_CACHE[0] != path:
        _HAAR_CACHE = (path, cv2.CascadeClassifier(path))
    return _HAAR_CACHE[1]


def faces(image: Image.Image) -> list[tuple[float, float, float, float]]:
    """Page-normalized boxes around every detected face.

    YuNet when its weights are present, Haar when they are not. Both paths are
    live so an installation missing optional package data still checks faces.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return []

    width, height = image.size
    array = np.array(image.convert("RGB"))
    try:
        detector = _yunet_detector(cv2, (width, height))
        # YuNet takes BGR, like the rest of OpenCV.
        _retval, found = detector.detect(array[:, :, ::-1])
        boxes = [] if found is None else [row[:4] for row in found]
        # No size floor here, and the absence is load-bearing. _MIN_FACE is
        # Haar's noise filter, expressed as a minimum square because Haar
        # returns no score and geometry is the only handle it offers. YuNet
        # returns a confidence and usually boxes the face more tightly. Applying
        # Haar's geometry-only minimum to scored YuNet results can discard valid
        # small faces, so only the fallback uses the size floor.
    except Exception:  # noqa: BLE001
        # Missing weights, or an OpenCV too old for FaceDetectorYN.
        cascade = _haar_detector(cv2)
        if cascade.empty():
            return []
        minimum = int(_MIN_FACE * width)
        grey = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        boxes = cascade.detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(minimum, minimum)
        )
    # float(), not the numpy scalars OpenCV hands back. YuNet returns float32
    # and Haar returns int32; either one reaches Finding.bbox and then the
    # masking report, where json.dumps refuses it and prevents the audit trail
    # from being written.
    return [
        (float(x) / width, float(y) / height,
         float(x + w) / width, float(y + h) / height)
        for x, y, w, h in boxes
    ]


def _ink_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> float:
    """Fraction of a page-normalized region that is dark enough to be ink."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return 0.0

    width, height = image.size
    x0, y0, x1, y1 = (
        max(int(box[0] * width), 0),
        max(int(box[1] * height), 0),
        min(int(box[2] * width), width),
        min(int(box[3] * height), height),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = np.asarray(image.convert("L").crop((x0, y0, x1, y1)))
    return float((crop < _INK_LEVEL).mean())


def _unread_ink_fraction(image: Image.Image, tokens: list, box) -> float:
    """Ink in a region that no token accounts for, as a fraction of the region.

    OCR often returns debris rather than nothing for unreadable handwriting.
    Token presence alone therefore cannot explain the pixels. Only sufficiently
    confident, word-like token boxes are subtracted; uncertain fragments remain
    available to the signature and unknown-ink detectors.
    """
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return 0.0

    width, height = image.size
    x0, y0, x1, y1 = (
        max(int(box[0] * width), 0),
        max(int(box[1] * height), 0),
        min(int(box[2] * width), width),
        min(int(box[3] * height), height),
    )
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ink = np.asarray(image.convert("L").crop((x0, y0, x1, y1))) < _INK_LEVEL
    for token in tokens:
        if (token.bbox is None or token.confidence < _READ_CONFIDENCE
                or not _looks_read(token.text)):
            # The shape check is specific to label-anchored signature detection.
            # It is not applied to the broader unread_ink() path, which already
            # favors review over classifying ambiguous content.
            continue
        ink[
            max(int(token.bbox[1] * height) - y0, 0) : max(int(token.bbox[3] * height) - y0, 0),
            max(int(token.bbox[0] * width) - x0, 0) : max(int(token.bbox[2] * width) - x0, 0),
        ] = False
    return float(ink.mean())


def signatures(image: Image.Image, tokens: list) -> list[tuple[float, float, float, float]]:
    """Regions beside a signature label that hold ink but no readable words.

    The two conditions together are the whole idea. Ink alone is just printed
    text; no-words alone is just blank paper. Ink *and* no words, next to
    something that says "signature", is a signature.

    "No words" is measured as ink the tokens do not account for rather than as
    the absence of tokens -- see _unread_ink_fraction for why the absence test
    could be defeated by a single hallucinated fragment.
    """
    boxes = []
    for index, token in enumerate(tokens):
        if token.bbox is None:
            continue
        if not _is_signature_label(index, tokens):
            continue
        x0, y0, x1, y1 = token.bbox
        line = max(y1 - y0, 1e-6)
        # The area a signature can occupy: to the right of the label and a few
        # lines below it, which covers "Signature: ____" and a name written
        # above a printed rule alike.
        region = (x0, y0, min(x0 + 0.35, 1.0), min(y1 + 3 * line, 1.0))
        # Every token, the label included: the label is a readable word and its
        # own ink sits inside the region it anchors.
        page_tokens = [t for t in tokens if t.page == token.page]
        if _unread_ink_fraction(image, page_tokens, region) >= _INK_FRACTION:
            boxes.append(region)
    return boxes


# How far to the left of a region its label may sit, as a fraction of the page.
# A signature rule is printed hard against its label; a value in the next column
# over is further away than this.
_LABEL_GAP = 0.06
# A label's box often runs a hair into the region beside it once the region has
# been tightened to the ink, so "to the left of" has to tolerate slight overlap.
_LABEL_OVERLAP = 0.02


def signature_ink(image: Image.Image, tokens: list) -> list[tuple[float, float, float, float]]:
    """Unread ink beside a label the OCR engine itself could not read.

    A degraded scan may make a signature caption unreadable. A nearby uncertain
    token supplies positive evidence of a caption without requiring its exact
    text. Requiring that anchor avoids classifying isolated logos, stamps, or
    garbled headers as signatures merely because no readable word precedes them.

    Regions come from `unread_ink`, so they are already tightened to the ink and
    already filtered for writing-like proportions. This only decides which of
    them have an unreadable caption next to them.
    """
    out = []
    for region in unread_ink(image, tokens):
        x0, y0, x1, y1 = region
        height = y1 - y0
        for index, token in enumerate(tokens):
            if token.bbox is None:
                continue
            tx0, ty0, tx1, ty1 = token.bbox
            line = min(ty1 - ty0, height)
            if line <= 0 or min(ty1, y1) - max(ty0, y0) < 0.5 * line:
                continue  # not on the same line as the ink
            if not -_LABEL_OVERLAP <= x0 - tx1 <= _LABEL_GAP:
                continue  # not immediately to its left
            if _is_signature_label(index, tokens):
                break  # readable caption: signatures() already owns this region
            if (token.confidence < _READ_CONFIDENCE
                    or not _looks_read(token.text)):
                # A garbled caption. This is the one positive signal the
                # function exists for.
                out.append(region)
                break
    return out


# The page is divided into cells and each is judged on its own. Cells rather
# than pixels because the question is about *regions* -- resolution-independent,
# so a 150 DPI scan and a 600 DPI one are cut the same way and no constant here
# has to know the DPI. 24 across is roughly a third of an inch on US Letter,
# which is a little under one line of text.
_GRID_COLS = 24
# Cells a region must occupy before it is reported. A signature runs several
# cells wide; a single inky cell is a bullet, a stamp edge or a logo corner.
_MIN_REGION_CELLS = 4
# Ink coverage inside a cell. Lower than _INK_FRACTION because a cell is small
# enough that a few handwritten strokes cross only part of it.
_CELL_INK = 0.01
# Second guard, after the rules are gone: ink must be spread across a good
# share of both the rows and the columns of its cell. A dashed rule and a row of
# dot leaders survive line removal -- each dash is too short to open away -- and
# both put ink in one band of rows and nowhere else. Handwriting is genuinely
# two-dimensional and passes easily.
_MIN_SPREAD = 0.25
# Writing usually runs along a line while repeated page furniture often runs
# down a margin. Requiring a region to be wider than tall suppresses narrow
# stacks of icons and checkboxes.
#
# The cost is honest and worth stating: a handwritten block several lines deep
# is taller than one line of writing, and a narrow one is rejected here. That
# case keeps the review flag -- it is only masking that this gates -- so the
# page is still reported as carrying ink nobody could read.
_MIN_ASPECT = 1.5
# A run of ink this long, perfectly straight, is printed rule and not writing.
# Expressed as a fraction of the page width so it holds at any DPI: a twelfth of
# US Letter is about 0.7in, and no handwritten stroke is 0.7in of contiguous ink
# on a single pixel row.
_RULE_LENGTH = 1 / 12
# Padding added around the tight ink bounds of an unread-ink region, as a
# fraction of the page. See the comment where it is used in unread_ink().
_INK_PAD = 0.002


def _without_rules(mask, cv2, np):
    """The ink, minus every long straight line in it.

    Two-dimensional spread rejects a line but not a *crossing*: where a table's
    horizontal and vertical rules meet, the cell has ink in every row (from the
    vertical) and every column (from the horizontal), so an empty table cell can
    otherwise resemble handwriting.

    Morphological opening with a long thin kernel keeps only ink that runs
    straight for `_RULE_LENGTH` of the page, which is what a rule does and what
    a pen does not. Removing that first leaves the writing behind, and the
    spread test then only has to separate writing from dashes.
    """
    length = max(int(mask.shape[1] * _RULE_LENGTH), 3)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    return mask & ~(horizontal | vertical)


def unread_ink(image: Image.Image, tokens: list) -> list[tuple[float, float, float, float]]:
    """Regions holding ink that no OCR engine turned into a word.

    This is `signatures()` without the label: the same "ink and no words" test,
    applied to the whole page. It answers the question the rest of the tool
    cannot -- what is on this page that we were unable to read at all -- and
    every caller has to decide separately what to do about the answer, because
    the honest description of a hit is "unknown ink", not "personal data".
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return []

    width, height = image.size
    if width < _GRID_COLS or height < _GRID_COLS:
        return []
    ink = _without_rules(
        (np.asarray(image.convert("L")) < _INK_LEVEL).astype(np.uint8), cv2, np
    ).astype(bool)
    # Erase everything OCR read. Printed text is ink too, and the whole premise
    # is that what is left over is what nothing could read. "Read" means the
    # same here as in _unread_ink_fraction -- read *and stood behind* -- because
    # a low-confidence guess sprayed over a scrawl is the ink this exists to
    # find, not an explanation of it.
    for token in tokens:
        if token.bbox is None or token.confidence < _READ_CONFIDENCE:
            continue
        x0, y0, x1, y1 = token.bbox
        ink[
            max(int(y0 * height), 0) : max(int(y1 * height), 0),
            max(int(x0 * width), 0) : max(int(x1 * width), 0),
        ] = False

    cell_w = width / _GRID_COLS
    rows = max(int(round(height / cell_w)), 1)
    cell_h = height / rows
    grid = np.zeros((rows, _GRID_COLS), np.uint8)
    for r in range(rows):
        for c in range(_GRID_COLS):
            cell = ink[
                int(r * cell_h) : int((r + 1) * cell_h),
                int(c * cell_w) : int((c + 1) * cell_w),
            ]
            if cell.size == 0 or cell.mean() < _CELL_INK:
                continue
            # Spread in both axes: see _MIN_SPREAD.
            if (cell.any(axis=1).mean() >= _MIN_SPREAD
                    and cell.any(axis=0).mean() >= _MIN_SPREAD):
                grid[r, c] = 1

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(grid, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:  # row 0 is the background
        if area < _MIN_REGION_CELLS:
            continue
        # The grid answers *where* to look; it is the wrong shape to mask. A
        # cell is a third of an inch, so a one-line signature lands in a band of
        # cells taller than the ink itself. Tightening to ink bounds preserves
        # neighboring page content without changing which strokes are covered.
        px0, py0 = int(x * cell_w), int(y * cell_h)
        px1, py1 = min(int((x + w) * cell_w), width), min(int((y + h) * cell_h), height)
        patch = ink[py0:py1, px0:px1]
        rows = np.flatnonzero(patch.any(axis=1))
        cols = np.flatnonzero(patch.any(axis=0))
        if rows.size and cols.size:
            py0, py1 = py0 + int(rows[0]), py0 + int(rows[-1]) + 1
            px0, px1 = px0 + int(cols[0]), px0 + int(cols[-1]) + 1
        # Antialiased stroke edges sit just above _INK_LEVEL and are not in
        # `ink`, so the tight bounds stop one shade short of the visible mark.
        # A pad of a fifth of a percent of the page is under a point at any DPI:
        # invisible against a mask, and the difference between covering a
        # descender and clipping it.
        if (px1 - px0) < _MIN_ASPECT * (py1 - py0):
            continue
        pad_x, pad_y = max(int(_INK_PAD * width), 1), max(int(_INK_PAD * height), 1)
        # float() for the same reason as in faces(): connectedComponentsWithStats
        # returns numpy ints, and a numpy scalar in a bbox kills the report.
        boxes.append((
            float(max(px0 - pad_x, 0) / width),
            float(max(py0 - pad_y, 0) / height),
            float(min(px1 + pad_x, width) / width),
            float(min(py1 + pad_y, height) / height),
        ))
    return boxes
