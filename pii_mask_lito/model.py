"""Format-agnostic intermediate representation.

Every input format (PDF text layer, OCR words, docx runs, spreadsheet cells) is
normalized to a list of `Token`. Detection and tagging are written once against
that list rather than once per format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Token:
    """One positioned piece of text.

    bbox is page-normalized (0..1, origin top-left) so coordinates survive
    rasterization at any DPI. It is None for flat formats (txt/csv/docx/xlsx)
    where there is nothing to draw a box on.

    confidence is how sure the reader was, 0..1, and defaults to certain --
    a text layer, a docx run and a decoded barcode all are. Only OCR sets it
    lower, and only vision.py reads it: detection deliberately does not, because
    a low-confidence token is still ink that might be a name and dropping it
    guarantees it can never be masked (see ocr.LOW_CONFIDENCE). What it is for
    is the opposite question -- deciding whether ink has been *read*, where a
    garbage word confidently reported over a scrawl is the failure mode.
    """

    text: str
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    line: int = 0
    confidence: float = 1.0


@dataclass
class Span:
    """A detected PII/PHI value, located in both text and page space."""

    entity: str
    start: int
    end: int
    score: float
    text: str
    tokens: list[int] = field(default_factory=list)
    source: str = "presidio"

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


class TokenText:
    """Joins tokens into searchable text, keeping a char-offset -> token map.

    This is the hinge of the whole pipeline: detectors work in character
    offsets, renderers work in rectangles, and this is what converts between
    them.
    """

    def __init__(self, tokens: list[Token], sep: str = " "):
        self.tokens = tokens
        self.sep = sep
        self.offsets: list[tuple[int, int]] = []
        pos = 0
        for tok in tokens:
            self.offsets.append((pos, pos + len(tok.text)))
            pos += len(tok.text) + len(sep)
        self.text = sep.join(t.text for t in tokens)

    @classmethod
    def from_text(cls, text: str) -> "TokenText":
        """Tokenize a plain string, keeping offsets into the original.

        Used by the flat formats. Offsets index the real string rather than a
        rejoined one, so replacement preserves the source's exact whitespace.
        """
        obj = cls.__new__(cls)
        obj.sep = " "
        obj.text = text
        obj.tokens = []
        obj.offsets = []
        for match in re.finditer(r"\S+", text):
            # Line numbers matter even without geometry: they stop a pattern
            # from matching across a newline in a plain-text note.
            line = text.count("\n", 0, match.start())
            obj.tokens.append(Token(text=match.group(), line=line))
            obj.offsets.append((match.start(), match.end()))
        return obj

    def tokens_for(self, start: int, end: int) -> list[int]:
        """Token indices overlapping the character range [start, end)."""
        return [i for i, (s, e) in enumerate(self.offsets) if s < end and e > start]

    def rects_for(self, span: Span) -> list[tuple[int, tuple[float, float, float, float]]]:
        """(page, bbox) rectangles covering a span, one per line it crosses.

        Grouping by line matters: a name wrapping across two lines must produce
        two tight boxes, not one tall box that also swallows whatever sits
        between them.
        """
        by_line: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
        for i in span.tokens:
            tok = self.tokens[i]
            if tok.bbox is None:
                continue
            by_line.setdefault((tok.page, tok.line), []).append(tok.bbox)
        rects = []
        for (page, _line), boxes in by_line.items():
            rects.append(
                (
                    page,
                    (
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes),
                    ),
                )
            )
        return rects


# Money is sacred, and so are the codes that sit beside it in a charge table.
# This was previously a rail on the agent path only, which made the guarantee
# depend on which detector happened to win merge_spans -- a multi-token address
# span running along a charge row could still swallow the amount printed in it.
# Enforced at render time instead, it holds for every detector unconditionally.
# Only shapes carrying a decimal point or a thousands separator. Bare 4- and
# 5-digit runs are indistinguishable from a house number, so treating them as
# money would quietly punch a hole in masked street addresses. Short codes are
# already structurally safe --
# _CANDIDATE will not accept one without a label, and DateDetector rejects them
# on an impossible day -- so this only has to cover money itself.
_MONEY = re.compile(r"^\$?\d{1,3}(?:,\d{3})+(?:\.\d{2})?$|^\$?\d+\.\d{2}$")


def protect_money(tt: "TokenText", span: Span) -> Span:
    """Drop money tokens from a span before it is drawn.

    Returns the span unchanged when it holds one token or nothing to protect:
    a value that is *itself* money was never going to be detected as PHI, and
    this only trims multi-token spans that grew across a charge row.
    """
    if len(span.tokens) <= 1:
        return span
    kept = [i for i in span.tokens if not _MONEY.match(tt.tokens[i].text.strip())]
    if len(kept) == len(span.tokens):
        return span
    return Span(span.entity, span.start, span.end, span.score, span.text, kept, span.source)


def mask_rects(
    tt: "TokenText",
    span: Span,
    max_width: float | None = None,
    padding: float = 0.12,
):
    """The rectangles it is safe to paint for this span.

    Money protection applies uniformly.  Width limiting is available to a
    caller with a document-specific policy, but the general path never drops a
    valid wide field solely because of its page-relative size.
    """
    rects = []
    for page, (x0, y0, x1, y1) in tt.rects_for(protect_money(tt, span)):
        height = max(y1 - y0, 0.0)
        margin = height * padding
        # OCR occasionally returns a syntactically valid date while clipping
        # its final glyph (for example, ``2026-09-1`` for ``2026-09-11``).  In
        # that case the detector quite correctly masks what it could read, but
        # the last printed digit can sit outside the reported word box.  Native
        # PDF boxes use the exact text-layer geometry and must stay tight; only
        # an OCR-derived date gets a conservative, one-glyph right overhang.
        # Token confidence defaults to exactly 1.0 and only OCR readers replace
        # it, so this distinction does not require format-specific knowledge in
        # the detector.
        page_tokens = [
            tt.tokens[index]
            for index in span.tokens
            if tt.tokens[index].page == page and tt.tokens[index].bbox is not None
        ]
        right_margin = margin
        if span.source == "date" and any(token.confidence < 1.0 for token in page_tokens):
            right_margin = max(right_margin, height * 1.5)
        rects.append(
            (page, (
                max(0.0, x0 - margin),
                max(0.0, y0 - margin),
                min(1.0, x1 + right_margin),
                min(1.0, y1 + margin),
            ))
        )
    if max_width is None:
        return rects
    return [(page, box) for page, box in rects if box[2] - box[0] <= max_width]


def assign_lines(tokens: list[Token], tolerance: float = 0.5) -> None:
    """Fill in `line` for engines that only return per-word boxes.

    Two tokens share a line when their vertical centres are closer together
    than `tolerance` of the shorter token's height. Mutates in place.
    """
    lines: dict[int, list[tuple[float, float, int]]] = {}
    # PDF text extraction order is not necessarily visual order. In a header
    # with a logo at the left and metadata at the right it commonly walks one
    # object completely before the other. Assigning line ids in that order can
    # make geometrically adjacent words look several lines apart and, later,
    # create a character span across unrelated intervening tokens. Process the
    # boxes top-to-bottom and left-to-right so line ids describe the page.
    positioned = [tok for tok in tokens if tok.bbox is not None]
    positioned.sort(
        key=lambda tok: (
            tok.page,
            (tok.bbox[1] + tok.bbox[3]) / 2,
            tok.bbox[0],
        )
    )
    for tok in positioned:
        if tok.bbox is None:
            continue
        centre = (tok.bbox[1] + tok.bbox[3]) / 2
        height = tok.bbox[3] - tok.bbox[1]
        page_lines = lines.setdefault(tok.page, [])
        for idx, (line_centre, line_height, line_id) in enumerate(page_lines):
            if abs(centre - line_centre) <= tolerance * min(height, line_height):
                tok.line = line_id
                # Widen the running average so drifting baselines stay together.
                page_lines[idx] = (
                    (line_centre + centre) / 2,
                    max(line_height, height),
                    line_id,
                )
                break
        else:
            tok.line = len(page_lines)
            page_lines.append((centre, height, tok.line))


def reading_order(tokens: list[Token]) -> list[Token]:
    """Return positioned tokens in stable geometric reading order.

    ``assign_lines`` must run first. Tokens without geometry retain their input
    order after positioned tokens; PDF/OCR callers normally have none, while
    flat-format callers should keep using :meth:`TokenText.from_text`.
    """
    indexed = list(enumerate(tokens))
    return [
        tok
        for _index, tok in sorted(
            indexed,
            key=lambda pair: (
                pair[1].page,
                pair[1].line if pair[1].bbox is not None else 10**9,
                pair[1].bbox[0] if pair[1].bbox is not None else pair[0],
                pair[0],
            ),
        )
    ]


def merge_spans(spans: list[Span]) -> list[Span]:
    """Resolve overlapping detections, keeping the longest then highest-scoring.

    Two recognizers firing on the same text is normal (a SSN pattern and a
    generic number pattern, say). Rendering both would double-stamp the same
    rectangle, so overlaps collapse to a single winner.
    """
    ordered = sorted(spans, key=lambda s: (-(s.end - s.start), -s.score, s.start))
    kept: list[Span] = []
    for span in ordered:
        if not any(span.overlaps(k) for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)
