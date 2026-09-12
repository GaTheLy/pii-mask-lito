"""Document-neutral PII detection over text and page geometry.

Presidio supplies general named-entity and validated-pattern recognition.
Additional deterministic recognizers cover high-confidence patterns, while a
spatial detector relates identifier-shaped values to labels printed beside or
above them.  That geometry-aware context works for forms, tables, scans, and
ordinary prose without relying on extractor-specific text order.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from datetime import date

from .model import Span, Token, TokenText, merge_spans
from .policy import DEFAULT_ALLOWLIST, profile as policy_profile
from .registry import normalize


class DetectorConfigurationError(RuntimeError):
    """Raised for an unusable detector configuration before analysis starts."""

# --------------------------------------------------------------------------
# Label -> entity. Matched against the text spatially adjacent to a candidate.
# --------------------------------------------------------------------------
LABELS: dict[str, tuple[str, ...]] = {
    "GENERIC_ID": (
        "unique id", "identifier", "employee id", "customer id", "client id",
        "user id", "record id", "case id", "reference id", "document id",
        "order id", "transaction id", "ticket id", "asset id",
        "hospital id", "doctor id", "form", "form id", "form number",
    ),
    "MEDICAL_RECORD_NUMBER": (
        "mrn",
        "med rec",
        "medical record",
        "medical rec",
        "med record",
    ),
    "ACCOUNT_NUMBER": (
        "account",
        "acct",
        "patient account",
        "patient control",
        "pat cntl",
    ),
    "HEALTH_PLAN_ID": (
        "health plan id",
        "member id",
        "insured's unique id",
        "insureds unique id",
        "subscriber id",
        "subscriber number",
        "policy id",
        "policy number",
        "group id",
        "group number",
        "patient id",
        "certificate",
    ),
    "CLAIM_NUMBER": ("claim number", "claim no", "claim #", "claim id"),
    "CREDIT_CARD": ("credit card", "card number", "card no", "card", "cc"),
    "LOCATION": (
        "address", "home address", "mailing address", "street address",
        "location", "postal code", "zip code", "united states",
    ),
}

PERSON_LABELS = (
    "name", "full name", "customer", "customer name", "employee name", "contact name",
    "recipient name", "applicant name", "account holder", "patient name",
)

# The key PROVIDER_LABELS competes under in _nearest_label. Not an entity: a win
# here means suppress, not mask. Deliberately not a string that could ever be a
# real entity name, so a mix-up is a KeyError rather than a silent mask.
_PROVIDER = object()

# Optional structural organization detection complements general NER when OCR
# preserves capitalization but not sentence context.
_ORG_WORD = r"[A-Z][A-Za-z&'\-]+"
_FACILITY = re.compile(
    rf"\b(?:{_ORG_WORD}(?:\s+{_ORG_WORD}){{1,3}}\s+"
    r"(?i:HOSPITAL|CLINIC|INSTITUTE|LABORATORY|LABS|PHARMACY|PHYSICIANS"
    r"|CENTER|CENTRE|HEALTHCARE|MEDICAL|HEALTH)"
    rf"|{_ORG_WORD}(?:\s+{_ORG_WORD}){{0,2}}\s+"
    r"(?i:ASSOCIATES|TECHNOLOGIES|TECHNOLOGY|LINES|CORPORATION|CORP|ORCHESTRA))"
    r"(?:(?:\s+|,\s*)(?i:INSTITUTE|CENTER|CENTRE|GROUP|SYSTEM|INC|LLC|LLP|CORP|PC))?\b"
)


# Age labels are word-bounded so substrings such as "page" and "coverage" do
# not claim nearby numbers. The profile supplies the minimum age.
AGE_LABEL = re.compile(
    r"\b(?:age|aged|dob|d\.o\.b|yrs|years?\s+old|y/o|birth|turn|turned|turning"
    r"|(?:he|she|they)\s+(?:is|was))\b"
)
_MIN_MASKED_AGE = 90
_MAX_PLAUSIBLE_AGE = 130
_AGE_VALUE = re.compile(r"^(\d{1,3})(?:[-\s]?(?:years?|yrs?)(?:[-\s]?old)?)?$", re.I)

# Candidate identifiers may contain common system separators but must contain
# a digit and be vouched for by a nearby label.
_CANDIDATE = re.compile(
    r"^(?=.{3,64}$)(?=.*\d)[A-Z0-9][A-Z0-9._:/\-]*$", re.I
)
_HAS_DECIMAL = re.compile(r"\d\.\d")
_TOKEN_EDGE = " \t.,;:!?()[]{}#<>|'\"/\\\u201c\u201d\u2018\u2019\u00ab\u00bb*~^\u00b0_-"
# Common OCR artifacts inside otherwise identifier-shaped tokens.
_OCR_NOISE = re.compile(r"[\\|/_^~`\u00b7\u2022]")

# Date shapes. Safe Harbor identifier #3 covers dates more precise than a year.
_DATE_PATTERNS = [
    re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"),
    re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"),
]
# OCR can split the four-digit year at a visual gap while preserving every
# glyph, for example ``17/08/1 974``. The two year pieces must total exactly
# four digits and still form a valid calendar date.
_SPLIT_YEAR_DATE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{1,3})\s+(\d{1,3})")
_FULL_DATE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")
_DATE_LABEL_REMNANT = re.compile(r"(?:d?ate|dated|dob|born|on|recorded)\W*$", re.I)
_COMPACT_DATE = re.compile(r"\b(\d{2})(\d{2})(\d{4})\b")
_SHORT_DATE = re.compile(r"\b(\d{2})(\d{2})(\d{2})\b")
_RELATIVE_WEEKDAY = re.compile(
    r"(?i:\b(?:meet|meeting|appointment|scheduled|on)[ \t]+)"
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.I,
)


def _is_date(month: int, day: int, year: int) -> bool:
    """Is this a calendar date, under either field order?

    Both month-first and day-first numeric dates are accepted. Masking only
    needs to establish that a string is a valid calendar date, not infer its
    locale-specific interpretation.
    """
    if not 1900 <= year <= 2100:
        return False
    for candidate_month, candidate_day in ((month, day), (day, month)):
        try:
            date(year, candidate_month, candidate_day)
            return True
        except ValueError:
            continue
    return False


@dataclass
class Adjacency:
    """How far a label may sit from its value, in page-normalized units."""

    left_gap: float = 0.18  # same line, label to the left
    above_gap: float = 0.025  # label in the box directly above, not three rows up
    line_tol: float = 0.6  # fraction of token height counted as "same line"
    column_tol: float = 0.10  # horizontal overlap needed for "above"
    window: int = 10  # nearest N tokens considered; keeps distant labels out
    linear_window: int = 4  # preceding words used when there is no geometry


class SpatialContextDetector:
    """Finds labelled identifiers by page geometry rather than text order."""

    def __init__(self, adjacency: Adjacency | None = None,
                 provider_names: set[str] | None = None,
                 mask_providers: bool = True,
                 mask_organizations: bool = False,
                 min_masked_age: int = 0):
        self.adj = adjacency or Adjacency()
        self.mask_providers = mask_providers
        self.mask_organizations = mask_organizations
        self.min_masked_age = min_masked_age
        # Values seen under a provider label, shared with the Detector. Written
        # here and by StructuralDetector; suppression and learn() read it.
        self.provider_names = provider_names if provider_names is not None else set()

    def context_for(self, tokens: list[Token], i: int) -> str:
        """Text of the tokens spatially adjacent to token i (left or above).

        Two properties matter and are easy to get wrong. The window is limited
        to the nearest few tokens, so a label three rows away cannot claim a
        value. And the result is emitted in *reading order*, not distance order
        -- sorting by proximity turns "MED REC #" into "# REC MED" and no label
        phrase ever matches again.
        """
        target = tokens[i]
        if target.bbox is None:
            # Flat formats (txt, docx, cells) have no geometry, so fall back to
            # linear proximity using the preceding few words.
            return " ".join(
                t.text for t in tokens[max(0, i - self.adj.linear_window) : i]
            ).casefold()
        tx0, ty0, tx1, ty1 = target.bbox
        height = max(ty1 - ty0, 1e-6)
        near = []
        for j, other in enumerate(tokens):
            if j == i or other.bbox is None or other.page != target.page:
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            same_line = abs((oy0 + oy1) / 2 - (ty0 + ty1) / 2) <= self.adj.line_tol * height
            if same_line and 0 <= tx0 - ox1 <= self.adj.left_gap:
                near.append((tx0 - ox1, other.line, ox0, other.text))
                continue
            overlaps_column = ox0 < tx1 + self.adj.column_tol and ox1 > tx0 - self.adj.column_tol
            if overlaps_column and 0 <= ty0 - oy1 <= self.adj.above_gap:
                near.append((ty0 - oy1, other.line, ox0, other.text))
        near.sort()
        window = near[: self.adj.window]
        # Re-sort the survivors into reading order. Line index, not raw y:
        # digits and letters on one baseline report slightly different box
        # tops, and sorting on those turns "MED REC #" into "REC # 56 MED".
        return " ".join(
            text for _, _, _, text in sorted(window, key=lambda n: (n[1], n[2]))
        ).casefold()

    def context_distances(self, tokens: list[Token], i: int) -> tuple[str, list[float]]:
        """`context_for`, plus how far away each character of it was found.

        Distances make the nearest matching label win when dense forms place
        several candidate labels in the same context window.

        Returned as (text, distance-per-character) so a phrase match can be
        scored by the closest token it actually covers.
        """
        target = tokens[i]
        if target.bbox is None:
            # No geometry: rank by how recently the word was read instead, so
            # the immediately preceding word still outranks one four back.
            parts, dists = [], []
            window = tokens[max(0, i - self.adj.linear_window) : i]
            for offset, other in enumerate(window):
                text = other.text.casefold()
                if parts:
                    parts.append(" ")
                    dists.append(float(len(window) - offset))
                parts.append(text)
                dists.extend([float(len(window) - offset)] * len(text))
            return "".join(parts), dists

        tx0, ty0, tx1, ty1 = target.bbox
        height = max(ty1 - ty0, 1e-6)
        near = []
        for j, other in enumerate(tokens):
            if j == i or other.bbox is None or other.page != target.page:
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            same_line = abs((oy0 + oy1) / 2 - (ty0 + ty1) / 2) <= self.adj.line_tol * height
            if same_line and 0 <= tx0 - ox1 <= self.adj.left_gap:
                # Scored as a fraction of the gap this direction is allowed, not
                # in raw page units. A label to the left may sit 0.18 away and a
                # label above only 0.025, so the two are seven times apart in
                # what "adjacent" means -- and compared raw, a heading on the
                # line above always beats the label printed beside the value.
                # Otherwise a heading above can outrank the label beside a value.
                near.append(((tx0 - ox1) / self.adj.left_gap, other.line, ox0, other.text))
                continue
            overlaps_column = ox0 < tx1 + self.adj.column_tol and ox1 > tx0 - self.adj.column_tol
            if overlaps_column and 0 <= ty0 - oy1 <= self.adj.above_gap:
                near.append(((ty0 - oy1) / self.adj.above_gap, other.line, ox0, other.text))
        near.sort()
        window = near[: self.adj.window]
        parts, dists = [], []
        for distance, _line, _x, text in sorted(window, key=lambda n: (n[1], n[2])):
            folded = text.casefold()
            if parts:
                parts.append(" ")
                dists.append(distance)
            parts.append(folded)
            dists.extend([distance] * len(folded))
        return "".join(parts), dists


    def detect(self, tt: TokenText, hint: str = "") -> list[Span]:
        spans = []
        for i, tok in enumerate(tt.tokens):
            # Strip wrapping punctuation before testing candidate shape.
            stripped = tok.text.strip(_TOKEN_EDGE)
            offset = tok.text.find(stripped)
            # Shape is tested on the de-noised form; the span still covers the
            # token as printed, so the mask lands on the real ink.
            clean = _OCR_NOISE.sub("", stripped)

            # Age is tested first because it is the one candidate shape smaller
            # than _CANDIDATE permits: "94" is two characters.
            age_match = _AGE_VALUE.match(stripped)
            if age_match:
                age = int(age_match.group(1))
                if self.min_masked_age <= age <= _MAX_PLAUSIBLE_AGE:
                    context = f"{hint} {self.context_for(tt.tokens, i)}".strip().casefold()
                    compound = "year" in stripped.casefold() or "yr" in stripped.casefold()
                    following = " ".join(
                        token.text
                        for token in tt.tokens[i + 1:i + 4]
                        if (token.page, token.line) == (tok.page, tok.line)
                    ).casefold()
                    if compound or AGE_LABEL.search(context) or AGE_LABEL.search(following):
                        start = tt.offsets[i][0] + offset
                        spans.append(
                            Span(
                                entity="AGE",
                                start=start,
                                end=start + len(stripped),
                                score=0.8,
                                text=stripped,
                                tokens=[i],
                                source="spatial",
                            )
                        )
                        continue

            if not _CANDIDATE.match(clean) or _HAS_DECIMAL.search(clean):
                continue
            context, dists = self.context_distances(tt.tokens, i)
            if hint:
                # A CSV or spreadsheet column header *is* the label, so it sits
                # at distance zero -- there is no geometry for it to lose to.
                folded = hint.casefold()
                context = f"{folded} {context}"
                dists = [0.0] * (len(folded) + 1) + dists
            if not context.strip():
                continue

            # Nearest label wins, positive or negative. A professional-side
            # label is not a page-wide veto, and a distant identifier label is
            # not evidence for the value in another field. Weighting the whole
            # context window equally creates both under- and over-masking.
            groups = dict(LABELS)
            if not self.mask_providers:
                groups[_PROVIDER] = PROVIDER_LABELS
            match = _nearest_label(context, dists, groups)
            if match is None:
                continue
            entity, _distance = match
            if entity is _PROVIDER:
                # Remember the explicit role classification so recheck passes
                # apply the same policy even if OCR loses the label.
                self.provider_names.add(normalize(clean))
                continue
            start = tt.offsets[i][0] + offset
            spans.append(
                Span(
                    entity=entity,
                    start=start,
                    end=start + len(stripped),
                    score=0.8,
                    text=stripped,
                    tokens=[i],
                    source="spatial",
                )
            )
        return spans


def _nearest_label(text: str, dists: list[float], groups: dict) -> tuple[str, float] | None:
    """Which label group sits closest to the value, and how close.

    Nearest wins, rather than first-found or negative-beats-positive. That is
    what lets a label beside a value outrank a different field several rows or
    columns away. It also makes negative policy context local rather than a
    global veto.
    """
    best: tuple[str, float] | None = None
    for name, labels in groups.items():
        for label in labels:
            pattern = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)")
            for found in pattern.finditer(text):
                at = found.start()
                span = dists[at : at + len(label)]
                if span:
                    distance = min(span)
                    if best is None or distance < best[1]:
                        best = (name, distance)
    return best

# Structural patterns complement prose-oriented NER on tables and forms. The
# surname-first pattern includes a single-letter middle initial.
_NAME_LAST_FIRST = re.compile(
    r"\b([A-Z][A-Za-z'\-]{1,20}),\s+([A-Z][A-Za-z'\-]{1,20})"
    r"(?:\s+([A-Z][A-Za-z'\-]{0,20})\.?)?\b"
)
_NAME_FIRST_LAST = re.compile(
    r"[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3}$"
)
_STREET = re.compile(
    r"\b\d{1,6}\s+(?:[A-Za-z0-9'.\-]+\s+){0,4}"
    r"(?:ST|STREET|DR|DRIVE|RD|ROAD|LN|LANE|CT|COURT|AVE|AVENUE|BLVD|BOULEVARD"
    r"|WAY|PL|PLACE|TER|TERRACE|TRL|PKWY|PARKWAY|CIR|CIRCLE|HWY|HIGHWAY"
    r"|RIDGE|GLEN|RUN|ARCADE|PLAZA|SQUARE|QUAY|WALK|ROW)\b\.?",
    re.I,
)
_STATE = (
    r"A[LKZR]|C[AOT]|D[EC]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]"
    r"|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AT]|W[AIVY]"
)
_CITY_STATE_ZIP = re.compile(
    r"\b([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3}),?\s+"
    rf"({_STATE})\s+\d{{5}}(?:-\d{{4}})?\b"
)
_CITY_STATE = re.compile(
    r"\b([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3}),\s*"
    rf"({_STATE})\b"
)
_LABELLED_LOCATION = re.compile(
    r"(?i:\b(?:location|city|state|country)\b)\s*:\s+"
    r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,3}?)"
    r"(?=\s+(?i:(?:billing(?:\s+no)?|mrn|procedure\s+date|received\s+date"
    r"|report\s+date|date|dob|age|ssn|id|phone|email|name|provider|doctor))\s*:|$)"
)
_LABELLED_NAME = re.compile(
    r"(?i:\b(?:name|full\s+name|customer\s+name|employee\s+name|patient\s+name)\b)"
    r"\s*:\s*([^\W\d_][\w'’\-]*(?:\s+[^\W\d_][\w'’\-]*){0,3})$"
)
_LABELLED_ADDRESS = re.compile(
    r"(?i:\b(?:address|home\s+address|mailing\s+address|street\s+address)\b)"
    r"\s*:\s*(\S(?:.*\S)?)$"
)
_PROSE_ADDRESS = re.compile(
    r"(?i:\b(?:address|home\s+address|mailing\s+address|street\s+address)\s+is)"
    r"\s+(\S(?:.*\S)?)$"
)
_UNIT_LINE = re.compile(r"(?i:\b(?:apt|apartment|suite|unit)\.?\s+[A-Za-z0-9-]+\b)")
_INTERNATIONAL_STREET = re.compile(
    r"\b[A-Z][\w'’.-]*(?:\s+[A-Za-z][\w'’.-]*){0,3}\s+"
    r"(?i:STREET|ST|ROAD|RD|AVENUE|AVE|LANE|LN|DRIVE|DR|BOULEVARD|BLVD|WAY"
    r"|COURT|CT|PLACE|PL|TERRACE|TRAIL|PARKWAY|CIRCLE|HIGHWAY|TEE|RUE|VIA"
    r"|CALLE|STRASSE|STRAßE|UL)\.?\s+\d{1,6}\b"
)
_POSTAL_LINE = re.compile(
    r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,2}\s+\d{5}(?:-\d{4})?$"
)
_CONTEXT_PERSON = (
    re.compile(
        r"(?i:\b(?:gender[ \t]+of|called|named|name[ \t]+is|directed[ \t]+by"
        r"|written[ \t]+by|created[ \t]+by|signed[ \t]+by))[ \t]*:?[ \t]+"
        r"([^\W\d_][\w'’\-]{1,30}(?:[ \t]+[A-ZÀ-ÖØ-Þ][\w'’\-]{1,30}){0,3})"
    ),
    re.compile(
        r"(?i:\b(?:user|customer|client|applicant))[ \t]*:[ \t]*"
        r"([^\W\d_][\w'’\-]*(?:[ \t]+[A-ZÀ-ÖØ-Þ][\w'’\-]*){0,3})"
    ),
    re.compile(
        r"\b([^\W\d_][\w'’\-]{1,30})['’]s[ \t]+"
        r"(?i:daughter|son|wife|husband|mother|father|sister|brother|partner)\b"
    ),
    re.compile(r"(?m)^([^\W\d_][\w'’\-]{1,30}):(?=\s|[\"'])"),
)
_CONTEXT_ORGANIZATION = (
    re.compile(
        r"(?i:\b(?:work(?:ed)?[ \t]+for|employed[ \t]+by|company|organization))"
        r"[ \t]+"
        r"([A-Z][A-Z0-9&.\-]{1,20})\b"
    ),
    re.compile(r"\b([A-Z][\w&'’\-]+)[ \t]+is[ \t]+a[ \t]+501\(c\)3\b"),
)
_CONTEXT_LOCATION = (
    re.compile(
        r"(?i:\b(?:home[ \t]+city|located[ \t]+in|based[ \t]+in|lives?[ \t]+in"
        r"|moved[ \t]+to|born[ \t]+in))[ \t]+"
        r"([^\W\d_][\w'’\-]*(?:[ \t]+[A-ZÀ-ÖØ-ÞΑ-Ω][\w'’\-]*){0,4})"
        r"(?=[ \t]*[:,.;]|$)"
    ),
    re.compile(
        r"(?i:\b(?:year|years)[ \t]+in)[ \t]+"
        r"([^\W\d_][\w'’\-]*(?:[ \t]+[A-ZÀ-ÖØ-ÞΑ-Ω][\w'’\-]*){0,3})"
        r"(?=[ \t]*[,.;]|$)"
    ),
)


def _flat_address_blocks(tt: TokenText) -> list[Span]:
    """Multiline flat-text address blocks with unit and postal evidence."""
    if any(token.bbox is not None for token in tt.tokens) or "\n" not in tt.text:
        return []
    lines = list(re.finditer(r"[^\n]+", tt.text))
    spans = []
    for start_index, first in enumerate(lines):
        prose = re.search(
            r"(?i:\baddress(?:\s+of\s+.{1,80}?)?\s+is\s+)",
            first.group(0),
        )
        first_text = re.sub(r"^\s*[^\w]+\s*", "", first.group(0))
        if prose is None and not re.match(r"\d{1,6}\b", first_text):
            continue
        for end_index in range(start_index + 1, min(start_index + 7, len(lines))):
            end = lines[end_index]
            block = tt.text[first.start():end.end()]
            end_text = re.sub(r"^\s*[^\w]+\s*", "", end.group(0)).strip()
            if (not _UNIT_LINE.search(block)
                    or not re.search(r"\b\d{5}(?:-\d{4})?$", end_text)):
                continue
            start = first.start() + (prose.end() if prose is not None else 0)
            stop = end.end()
            spans.append(Span(
                "LOCATION", start, stop, 0.85, tt.text[start:stop],
                tt.tokens_for(start, stop), "structural",
            ))
            break
    return spans


# A city/state pair can resemble a surname-first name.
_STATE_ONLY = re.compile(rf"^({_STATE})$")

# Common professional-role labels used only when a profile explicitly retains
# professional identities.
PROVIDER_LABELS = (
    "attending", "operating", "referring", "rendering", "other", "npi",
    "physician", "provider", "surgeon", "added by", "counselor", "qual",
    "ordering", "admitting", "signature", "signed by", "taxonomy",
    "taxpayer", "payee", "clearing house", "clearinghouse", "facility",
)

# Entity types eligible for the explicit professional-identity exception.
_PROVIDER_SUPPRESSIBLE = {
    "PERSON", "PHONE_NUMBER", "US_BANK_NUMBER", "US_ITIN", "US_PASSPORT",
    "MEDICAL_LICENSE", "US_DRIVER_LICENSE", "IBAN_CODE",
}

# Tokens of a span whose surroundings are checked for a label. Bounded because
# context_for scans the page, and a name is a few tokens at most.
_LABEL_PROBE = 4


def _near_label(spatial: "SpatialContextDetector", tt: TokenText, tokens: list[int],
                labels) -> bool:
    """Return whether any token in a span is adjacent to a listed label."""
    for index in tokens[:_LABEL_PROBE]:
        context = spatial.context_for(tt.tokens, index)
        if any(label in context for label in labels):
            return True
    return False

# The mirror image: labels that prove a name *is* the patient's. Used to settle
# the conflict when a value carries a provider label on one page and a patient
# label on another -- see _suppressed() . Without this the provider test is a
# veto, and a wrong veto is an unmasked patient name.
PATIENT_LABELS = (
    "patient name", "pat name", "patient's name", "guarantor",
    "insured's name", "insureds name", "subscriber name", "bill to",
    "responsible party",
)

# The same idea for addresses. Kept separate from PATIENT_LABELS rather than
# appended to it: that tuple also decides which *names* outrank the provider
# veto, and "mailing address" has nothing to say about a name.
PATIENT_ADDRESS_LABELS = PATIENT_LABELS + (
    "patient address", "patient addr", "pat addr", "patient's address",
    "home address", "mailing address", "street address", "resides",
)
ADDRESS_BLOCK_LABELS = PATIENT_ADDRESS_LABELS + ("address", "location", "contact")

# Organisations. Split out of FACILITY_LABELS because the two halves cannot be
# used for the same job: the street furniture below is printed *inside* every
# address, so a rule that consults it about a LOCATION suppresses every address
# on the page including the patient's. Only the org nouns carry information
# about whose address it is.
ORG_LABELS = (
    "medical center", "hospital", "clinic", "insurance", "health plan",
    "healthcare", "health system", "laboratory", "pharmacy", "payer",
)

# Organization and address terms that disambiguate surname collisions when the
# professional-identity exception is active.
FACILITY_LABELS = ORG_LABELS + (
    "street", "avenue", "ave", "blvd", "road", "drive", "suite",
)

class StructuralDetector:
    """Shape-and-position detection for names and addresses."""

    def __init__(self, spatial: "SpatialContextDetector", provider_names: set[str] | None = None):
        self.spatial = spatial
        # Names proven to sit under a provider label, shared with the Detector.
        # Written here, read there. See _is_provider.
        self.provider_names = provider_names if provider_names is not None else set()

    def detect(self, tt: TokenText) -> list[Span]:
        """Match within each line separately.

        Page text is a single stream; matching within visual line segments
        prevents structural regexes from crossing rows or columns.
        """
        spans = _flat_address_blocks(tt)
        for pattern in _CONTEXT_PERSON:
            for match in pattern.finditer(tt.text):
                start, end = match.span(1)
                spans.append(Span(
                    "PERSON", start, end, 0.8, match.group(1),
                    tt.tokens_for(start, end), "structural",
                ))
        if self.spatial.mask_organizations:
            for pattern in _CONTEXT_ORGANIZATION:
                for match in pattern.finditer(tt.text):
                    start, end = match.span(1)
                    spans.append(Span(
                        "ORGANIZATION", start, end, 0.8, match.group(1),
                        tt.tokens_for(start, end), "structural",
                    ))
        for pattern in _CONTEXT_LOCATION:
            for match in pattern.finditer(tt.text):
                start, end = match.span(1)
                spans.append(Span(
                    "LOCATION", start, end, 0.8, match.group(1),
                    tt.tokens_for(start, end), "structural",
                ))
        for text, offsets, indices in _lines(tt):
            # A comma inside a digit-led address is not surname-first syntax:
            # "731 Lantern Quays, North Ember" is one address line.
            address_line = bool(re.match(r"^\s*\d{1,6}\b", text))
            organization_line = bool(re.search(
                r"\b(?:INC|LLC|LLP|CORP|ASSOCIATES)\b", text, re.I
            ))
            if (address_line and indices
                    and _near_label(self.spatial, tt, indices[:1], ADDRESS_BLOCK_LABELS)):
                # Address blocks often use generated or regional street suffixes
                # that no finite suffix list can cover. A digit-led value directly
                # under an explicit label is stronger evidence than vocabulary.
                # Stop before an inline phone field so unrelated contact data keeps
                # its own entity type and tag.
                covered = []
                for index in indices:
                    word = normalize(tt.tokens[index].text).rstrip(":")
                    if word in {"tel", "telephone", "phone", "ph", "fax"}:
                        break
                    covered.append(index)
                if covered:
                    start = tt.offsets[covered[0]][0]
                    end = tt.offsets[covered[-1]][1]
                    spans.append(Span(
                        "LOCATION", start, end, 0.8,
                        " ".join(tt.tokens[index].text for index in covered),
                        covered, "structural",
                    ))
            for m in _NAME_LAST_FIRST.finditer(text):
                if address_line or organization_line:
                    continue
                # A city with a damaged ZIP, not a person. See _STATE_ONLY.
                if _STATE_ONLY.match(m.group(2)):
                    continue
                span = _localize(tt, m, offsets, indices, "PERSON", 0.75)
                if self._is_provider(tt, span.tokens):
                    # Remember the exception per word so later OCR-only passes
                    # apply the same explicit policy consistently.
                    self.provider_names.update(
                        normalize(w) for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", span.text)
                    )
                    continue
                spans += _per_token(tt, span)
            labelled = _NAME_FIRST_LAST.fullmatch(text)
            if labelled and _near_label(self.spatial, tt, indices, PERSON_LABELS):
                span = _localize(tt, labelled, offsets, indices, "PERSON", 0.8)
                if not _generic_name(span):
                    spans += _per_token(tt, span)
            if self.spatial.mask_organizations:
                for m in _FACILITY.finditer(text):
                    spans.append(
                        _localize(tt, m, offsets, indices, "ORGANIZATION", 0.7)
                    )
            for m in _LABELLED_NAME.finditer(text):
                spans.append(
                    _localize(tt, m, offsets, indices, "PERSON", 0.8, group=1)
                )
            for m in _LABELLED_ADDRESS.finditer(text):
                spans.append(
                    _localize(tt, m, offsets, indices, "LOCATION", 0.8, group=1)
                )
            for m in _PROSE_ADDRESS.finditer(text):
                spans.append(
                    _localize(tt, m, offsets, indices, "LOCATION", 0.8, group=1)
                )
            for m in _LABELLED_LOCATION.finditer(text):
                spans.append(
                    _localize(tt, m, offsets, indices, "LOCATION", 0.8, group=1)
                )
            for pattern in (
                _STREET, _CITY_STATE_ZIP, _CITY_STATE, _UNIT_LINE,
                _INTERNATIONAL_STREET, _POSTAL_LINE,
            ):
                for m in pattern.finditer(text):
                    spans.append(_localize(tt, m, offsets, indices, "LOCATION", 0.7))
        return spans

    def _is_provider(self, tt: TokenText, tokens: list[int]) -> bool:
        if self.spatial.mask_providers:
            # The caller has asked for a policy under which a clinician's name
            # is PHI too, so there is nothing here to suppress.
            return False
        return _near_label(self.spatial, tt, tokens, PROVIDER_LABELS)

    @staticmethod
    def _span(tt: TokenText, m: re.Match, entity: str, score: float) -> Span:
        return Span(
            entity=entity,
            start=m.start(),
            end=m.end(),
            score=score,
            text=m.group(0),
            tokens=tt.tokens_for(m.start(), m.end()),
            source="structural",
        )


# --------------------------------------------------------------------------
# High-confidence patterns are detected locally in addition to Presidio.
# --------------------------------------------------------------------------
_SSN = re.compile(r"\b\d{3}[-\s.]\d{2}[-\s.]\d{4}\b")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[-\s.])?\(?\d{3}\)?[-\s.]\d{3}[-\s.]\d{4}"
    r"(?:\s*(?:x|ext\.?)\s*\d{1,6})?\b",
    re.I,
)
_INTERNATIONAL_PHONE = re.compile(
    r"(?<![\w/])(?:\+\d{1,7}|\(\d{1,4}\)|\d{2,4})"
    r"(?:[ .-]\d{2,6}){1,4}(?![\w/])"
)
_E164_PHONE = re.compile(r"(?<!\w)\+\d{7,15}\b")
_TRAILING_LABEL_PHONE = re.compile(
    r"\b\d{7,15}(?=[ \t]*-[ \t]*(?:fax|mobile|phone|office)\b)",
    re.I,
)


# Toll-free US numbers usually identify organizations rather than people. This
# conservative exception is only applied by the local pattern recognizer;
# callers can include them through other recognizers or explicit fields.
_TOLL_FREE = re.compile(r"^\(?(?:800|833|844|855|866|877|888)\)?[-\s.]")


def toll_free(text: str) -> bool:
    return bool(_TOLL_FREE.match(text.strip()))


def money_shaped(text: str) -> bool:
    """Is this SSN/phone match really a charge row?

    An amount followed by a short numeric code can satisfy the SSN pattern. The
    discriminator is that a real SSN or phone number uses one separator
    throughout: a match mixing a decimal point with a space, or carrying a
    thousands comma, is tabular numeric data.

    Shared with the verification gate, which needs the identical rule -- it runs
    the same recognizers over the finished page and would otherwise refuse to
    write documents containing ordinary financial tables.
    """
    return ("." in text and " " in text) or "," in text
# Anything with an @ between two runs of word characters is treated as an
# address, even when OCR has destroyed the domain.
_EMAIL = re.compile(r"\b[\w.+-]{2,}@[\w.-]{2,}\b")
# Requiring letters in the local part limits OCR punctuation false positives.
_EMAIL_LOCAL_LETTERS = 2


def email_shaped(text: str) -> bool:
    local = text.split("@", 1)[0]
    return sum(c.isalpha() for c in local) >= _EMAIL_LOCAL_LETTERS


def _geometrically_contiguous(tt: TokenText, indices: list[int]) -> bool:
    """Whether matched tokens occupy one visual segment rather than columns."""
    if len(indices) < 2:
        return True
    for left_index, right_index in zip(indices, indices[1:]):
        left, right = tt.tokens[left_index], tt.tokens[right_index]
        if (left.page, left.line) != (right.page, right.line):
            return False
        if left.bbox is None or right.bbox is None:
            continue
        height = max(
            left.bbox[3] - left.bbox[1],
            right.bbox[3] - right.bbox[1],
            1e-6,
        )
        if right.bbox[0] - left.bbox[2] > max(0.03, 3 * height):
            return False
    return True


class PatternDetector:
    """Deterministic recognizers for the entities that must never be missed."""

    PATTERNS = (
        ("US_SSN", _SSN, 0.9),
        ("PHONE_NUMBER", _PHONE, 0.85),
        ("PHONE_NUMBER", _INTERNATIONAL_PHONE, 0.82),
        ("PHONE_NUMBER", _E164_PHONE, 0.9),
        ("PHONE_NUMBER", _TRAILING_LABEL_PHONE, 0.85),
        ("EMAIL_ADDRESS", _EMAIL, 0.9),
    )

    def detect(self, tt: TokenText) -> list[Span]:
        spans = []
        for entity, pattern, score in self.PATTERNS:
            for m in pattern.finditer(tt.text):
                if entity in ("US_SSN", "PHONE_NUMBER") and money_shaped(m.group(0)):
                    continue
                digits = sum(character.isdigit() for character in m.group(0))
                if entity == "PHONE_NUMBER" and not 7 <= digits <= 15:
                    continue
                indices = tt.tokens_for(m.start(), m.end())
                if entity == "PHONE_NUMBER" and not _geometrically_contiguous(tt, indices):
                    continue
                if entity == "PHONE_NUMBER" and toll_free(m.group(0)):
                    continue
                if entity == "EMAIL_ADDRESS" and not email_shaped(m.group(0)):
                    continue
                spans.append(
                    Span(
                        entity=entity,
                        start=m.start(),
                        end=m.end(),
                        score=score,
                        text=m.group(0),
                        tokens=indices,
                        source="pattern",
                    )
                )
        return spans


class DateDetector:
    """Masks dates precise to a day; leaves bare years and money alone.

    Bare numbers are retained so amounts and quantities are not mistaken for
    identifiers. Requiring a *valid date shape* masks 05/17/2023 while leaving
    3,058.85 and a short numeric code alone.
    """

    def detect(self, tt: TokenText) -> list[Span]:
        spans = []
        for pattern in _DATE_PATTERNS:
            for m in pattern.finditer(tt.text):
                a, b, c = (int(g) for g in m.groups())
                ok = _is_date(a, b, c) if c > 31 else _is_date(b, c, a)
                if ok:
                    spans.append(self._span(tt, m))
        for m in _SPLIT_YEAR_DATE.finditer(tt.text):
            year_text = m.group(3) + m.group(4)
            if len(year_text) != 4:
                continue
            covered = tt.tokens_for(m.start(), m.end())
            if len({(tt.tokens[index].page, tt.tokens[index].line) for index in covered}) != 1:
                continue
            first, second, year = int(m.group(1)), int(m.group(2)), int(year_text)
            if _is_date(first, second, year):
                spans.append(self._span(tt, m))
        for m in _COMPACT_DATE.finditer(tt.text):
            month, day, year = (int(g) for g in m.groups())
            if _is_date(month, day, year):
                spans.append(self._span(tt, m))
        for m in _SHORT_DATE.finditer(tt.text):
            month, day, yy = (int(g) for g in m.groups())
            if _is_date(month, day, 2000 + yy):
                spans.append(self._span(tt, m))
        for m in _RELATIVE_WEEKDAY.finditer(tt.text):
            spans.append(self._span(tt, m, group=1))
        for span in self._fragmented(tt):
            if not any(span.overlaps(existing) for existing in spans):
                spans.append(span)
        return spans

    @staticmethod
    def _fragmented(tt: TokenText) -> list[Span]:
        """Reassemble a date split across up to three adjacent OCR tokens."""
        spans = []
        for start, first in enumerate(tt.tokens):
            tail = re.search(r"(\d[\d/-]*)$", first.text)
            if tail is None:
                continue
            prefix = first.text[:tail.start()]
            if prefix and not _DATE_LABEL_REMNANT.search(prefix):
                continue
            candidate = tail.group(1)
            covered = [start]
            previous = first
            for index in range(start + 1, min(start + 3, len(tt.tokens))):
                token = tt.tokens[index]
                if (token.page, token.line) != (first.page, first.line):
                    break
                if previous.bbox is not None and token.bbox is not None:
                    height = max(
                        previous.bbox[3] - previous.bbox[1],
                        token.bbox[3] - token.bbox[1],
                        1e-6,
                    )
                    if token.bbox[0] - previous.bbox[2] > 1.5 * height:
                        break
                if not re.fullmatch(r"[\d/-]+", token.text):
                    break
                candidate += token.text
                covered.append(index)
                previous = token
                match = _FULL_DATE.fullmatch(candidate)
                if match is None:
                    continue
                first_part, second_part, year = (int(group) for group in match.groups())
                if not _is_date(first_part, second_part, year):
                    continue
                start_offset = tt.offsets[start][0] + tail.start()
                spans.append(Span(
                    "DATE", start_offset, tt.offsets[index][1], 0.85,
                    candidate, covered.copy(), "date",
                ))
                break
        return spans

    @staticmethod
    def _span(tt: TokenText, m: re.Match, group: int = 0) -> Span:
        start, end = m.span(group)
        return Span(
            entity="DATE",
            start=start,
            end=end,
            score=0.85,
            text=m.group(group),
            tokens=tt.tokens_for(start, end),
            source="date",
        )


class Detector:
    """Runs every detector over one page and reconciles the results."""

    def learn(self, spans: list[Span], tt: TokenText | None = None) -> None:
        learn(self, spans, tt)

    def propagate(self, tt: TokenText) -> list[Span]:
        return merge_spans(propagate(self, tt))

    def __init__(
        self,
        entities: list[str] | None = None,
        spacy_model: str = "en_core_web_lg",
        allowlist: list[str] | None = None,
        min_score: float = 0.4,
        mask_providers: bool | None = None,
        min_masked_age: int | None = None,
        profile: str = "general",
        mask_organizations: bool | None = None,
    ):
        # Profiles choose policy without changing detector capability.  The
        # domain-neutral default masks professional people and all ages; the
        # healthcare-specific profile can retain the narrower Safe Harbor age
        # threshold.  Explicit arguments always win over profile defaults.
        selected_policy = policy_profile(profile)
        self.entities = list(selected_policy.entities) if entities is None else list(entities)
        mask_professional_people = (
            selected_policy.mask_professional_people
            if mask_providers is None
            else mask_providers
        )
        if mask_organizations is None:
            mask_organizations = selected_policy.mask_organizations
        if entities is None and mask_organizations and "ORGANIZATION" not in self.entities:
            self.entities.append("ORGANIZATION")
        self.min_score = min_score
        self.mask_providers = mask_professional_people
        self.mask_organizations = mask_organizations
        min_masked_age = (
            selected_policy.min_masked_age if min_masked_age is None else min_masked_age
        )
        selected_allowlist = DEFAULT_ALLOWLIST if allowlist is None else allowlist
        self.allowlist = {normalize(a) for a in selected_allowlist}
        # Values seen under a provider label anywhere in the job -- names from
        # the structural detector, identifiers from the spatial one. Both write
        # it; suppression and learn() read it.
        self.provider_names: set[str] = set()
        self.spatial = SpatialContextDetector(provider_names=self.provider_names,
                                              mask_providers=mask_professional_people,
                                              mask_organizations=mask_organizations,
                                              min_masked_age=min_masked_age)
        self.structural = StructuralDetector(self.spatial, self.provider_names)
        self.dates = DateDetector()
        self.patterns = PatternDetector()
        # Values proven to be PHI somewhere in this job. See learn()/propagate().
        self.lexicon: dict[str, str] = {}
        # Lexicon keys folded through the OCR confusion classes, for numeric
        # identifiers. See _fold() and C2 in the audit.
        self.numeric_index: dict[str, str] = {}
        # Names proven under a patient label somewhere. These outrank the
        # provider/facility suppression at propagation time. See _suppressed().
        self.patient_names: set[str] = set()
        # Addresses proven to be a patient's. The lexicon-wins guard for
        # _payer_address, and the exact mirror of patient_names.
        self.patient_locations: set[str] = set()
        # Doubts worth a human's attention: things masked that may not be PHI,
        # and things a rule argued away. Under-masking is a disclosure, so a
        # judgement call in that direction is never made silently.
        self.suppressed: list[str] = []
        self._analyzer = None
        self._spacy_model = spacy_model

    @property
    def analyzer(self):
        # Built lazily: loading a spaCy model costs seconds, and the flat-text
        # formats can be masked without ever touching it.
        if self._analyzer is None:
            import importlib.util
            from pathlib import Path

            model_path = Path(self._spacy_model)
            try:
                installed = importlib.util.find_spec(self._spacy_model) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                installed = False
            if not model_path.exists() and not installed:
                raise DetectorConfigurationError(
                    f"spaCy model {self._spacy_model!r} is not installed; install it "
                    f"with `python -m spacy download {self._spacy_model}` or pass "
                    "--spacy-model /path/to/a/compatible-model"
                )
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": self._spacy_model}],
                }
            )
            self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        return self._analyzer

    def detect(self, tt: TokenText, hint: str = "") -> list[Span]:
        if not tt.text.strip():
            return []
        spans: list[Span] = []

        presidio_wanted = [e for e in self.entities if e in _PRESIDIO_SUPPORTED]
        if presidio_wanted:
            for r in self.analyzer.analyze(
                text=tt.text, entities=presidio_wanted, language="en"
            ):
                if r.score < self.min_score:
                    continue
                value = tt.text[r.start : r.end]
                if not self._plausible(r.entity_type, value, tt, r.start, r.end):
                    continue
                candidate = Span(
                    entity=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                    text=value,
                    tokens=tt.tokens_for(r.start, r.end),
                    source="presidio",
                )
                # NER can join a name in one form column to an identifier in
                # the next because its input is flat text. Names are indexed
                # per word elsewhere too; split them here and discard only the
                # numeric piece so one broad NER span cannot create a page-wide
                # mask or teach an identifier as a person's name.
                if r.entity_type == "PERSON":
                    spans.extend(
                        piece for piece in _per_token(tt, candidate)
                        if not any(character.isdigit() for character in piece.text)
                    )
                else:
                    spans.append(candidate)

        spans += [s for s in self.patterns.detect(tt) if s.entity in self.entities]
        spans += [s for s in self.structural.detect(tt) if s.entity in self.entities]
        if any(e in self.entities for e in (*LABELS, "AGE")):
            spans += [s for s in self.spatial.detect(tt, hint) if s.entity in self.entities]
        if "DATE" in self.entities:
            spans += self.dates.detect(tt)

        allowed = _allowlisted_ranges(tt, self.allowlist)
        spans = [
            s for s in spans
            if not _inside(s, allowed)
            and normalize(s.text) not in self.allowlist
            and not _generic_name(s)
            and not self._known_provider(s)
        ]
        return merge_spans(spans)

    def _known_provider(self, span: Span) -> bool:
        """Was this name already settled as a clinician's, on a better read?

        The lexicon still wins: a value proven under a patient label somewhere
        is never suppressed, because a wrong veto is an unmasked patient name.
        """
        if self.mask_providers or not self.provider_names:
            return False
        if span.entity != "PERSON":
            # Identifiers are settled whole: an NPI is the provider's or it is
            # not, and there are no parts to weigh.
            key = normalize(_OCR_NOISE.sub("", span.text))
            return key in self.provider_names and key not in self.patient_names
        words = {normalize(w) for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", span.text)}
        if words & self.patient_names:
            return False
        return bool(words) and words <= self.provider_names

    def _plausible(self, entity: str, value: str, tt: TokenText, start: int, end: int) -> bool:
        """Reject Presidio matches that structured tables produce by accident.

        An amount beside a short code can satisfy the US phone pattern. Some
        professional identifiers can as well. Both are filtered on shape and,
        when the selected profile requests it, on local role context.
        """
        if entity == "PHONE_NUMBER" and money_shaped(value):
            return False
        if (entity == "URL" and not re.match(r"(?i)(?:https?://|www\.)", value)
                and re.search(r"\.[A-Z]", value)):
            # OCR frequently glues a sentence boundary into ``word.Next``.
            # A capitalized apparent TLD without a URL marker is prose, not a
            # bare domain.
            return False
        if not self.mask_providers and entity in _PROVIDER_SUPPRESSIBLE:
            tokens = tt.tokens_for(start, end)
            if _near_label(self.spatial, tt, tokens, PROVIDER_LABELS):
                return False
        return True


# Presidio provides both NER-backed recognizers and validated identifier
# recognizers.  Deterministic local recognizers still run as a second source;
# reconciliation removes overlaps while retaining the strongest evidence.
_PRESIDIO_SUPPORTED = {
    "PERSON",
    "LOCATION",
    "PHONE_NUMBER",
    # Email is handled by PatternDetector above. Presidio's email validator
    # asks tldextract to refresh the public-suffix list on first use, which is
    # an unnecessary outbound network attempt during local document masking.
    "US_SSN",
    "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE",
    "URL",
    "IP_ADDRESS",
    "US_PASSPORT",
    "US_ITIN",
    "US_BANK_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "CRYPTO",
}


def _per_token(tt: TokenText, span: Span) -> list[Span]:
    """Split a name into independently indexed word spans.

    Per-word identity lets abbreviated and fully written occurrences share
    stable tags across pages.
    """
    if len(span.tokens) <= 1:
        return [span]
    out = []
    for i in span.tokens:
        start, end = tt.offsets[i]
        text = tt.text[start:end].strip(" ,;")
        # Keep initials; discard punctuation-only tokenizer fragments.
        if not any(character.isalnum() for character in text):
            continue
        out.append(
            Span(
                entity=span.entity,
                start=start,
                end=start + len(text),
                score=span.score,
                text=text,
                tokens=[i],
                source=span.source,
            )
        )
    return out or [span]


def _lines(tt: TokenText):
    """Yield visually contiguous line segments in reading order.

    PDF extractors commonly assign one line number to multiple columns. A
    large horizontal gap therefore ends a segment so recognizers cannot join
    unrelated cells.
    """
    grouped: dict[tuple[int, int], list[int]] = {}
    for i, tok in enumerate(tt.tokens):
        grouped.setdefault((tok.page, tok.line), []).append(i)
    for key in sorted(grouped):
        indices = sorted(
            grouped[key],
            key=lambda i: tt.tokens[i].bbox[0] if tt.tokens[i].bbox else i,
        )
        segments: list[list[int]] = [[]]
        for index in indices:
            if segments[-1]:
                previous = tt.tokens[segments[-1][-1]]
                current = tt.tokens[index]
                if previous.bbox is not None and current.bbox is not None:
                    height = max(
                        previous.bbox[3] - previous.bbox[1],
                        current.bbox[3] - current.bbox[1],
                        1e-6,
                    )
                    if current.bbox[0] - previous.bbox[2] > max(0.03, 3 * height):
                        segments.append([])
            segments[-1].append(index)
        for segment in segments:
            yield _line_text(tt, segment)


def _line_text(tt: TokenText, indices: list[int]):
    """Build searchable text and offsets for a sequence of global tokens."""
    parts, offsets, pos = [], [], 0
    for i in indices:
        text = tt.tokens[i].text
        parts.append(text)
        offsets.append((pos, pos + len(text)))
        pos += len(text) + 1
    return " ".join(parts), offsets, indices


def _localize(tt: TokenText, m, offsets, indices, entity: str, score: float,
              group: int = 0) -> Span:
    """Convert a per-line match back to page-level offsets and token indices."""
    local_start, local_end = m.span(group)
    covered = [
        indices[k]
        for k, (start, end) in enumerate(offsets)
        if start < local_end and end > local_start
    ]
    if not covered:
        return Span(entity, 0, 0, score, m.group(group), [], "structural")
    start = min(tt.offsets[i][0] for i in covered)
    end = max(tt.offsets[i][1] for i in covered)
    text = " ".join(tt.tokens[i].text for i in covered)
    return Span(entity, start, end, score, text, covered, "structural")


def _allowlisted_ranges(tt: TokenText, allowlist: set[str]) -> list[tuple[int, int]]:
    """Character ranges for normalized multi-token allowlist phrases."""
    phrases = [p.split() for p in allowlist if p]
    if not phrases:
        return []
    words = [normalize(t.text) for t in tt.tokens]
    ranges = []
    for phrase in phrases:
        n = len(phrase)
        for i in range(len(words) - n + 1):
            if words[i : i + n] == phrase:
                ranges.append((tt.offsets[i][0], tt.offsets[i + n - 1][1]))
    return ranges


def _inside(span: Span, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= span.start and span.end <= end for start, end in ranges)


def _generic_name(span: Span) -> bool:
    """Return whether a PERSON/ORGANIZATION span is document vocabulary."""
    if span.entity not in {"PERSON", "ORGANIZATION"}:
        return False
    words = normalize(span.text).split()
    return bool(words) and all(
        w in UI_NOISE or w in _TOO_COMMON or _common_compound(w)
        for w in words
    )


# --------------------------------------------------------------------------
# Cross-page propagation: a contextual value detected once remains sensitive
# when it appears elsewhere without its original label.
# --------------------------------------------------------------------------

# Words too generic to propagate on. A surname that is also a common noun would
# otherwise mask unrelated body text wherever it appeared.
_TOO_COMMON = {
    "account", "address", "amount", "balance", "code", "date", "description",
    "document", "email", "name", "number", "page", "phone", "record", "status",
    "total", "type", "value", "work", "home", "other", "none", "general",
    "po", "box", "apt", "suite", "unit", "floor", "mail",
    "llc", "inc", "corp", "company", "ltd", "group", "id",
    "customer", "client", "employee", "recipient", "applicant", "holder",
    "case", "support", "patient", "doctor", "encounter", "participant",
    "summary", "information", "notes", "service", "confirmation", "contact",
    "location", "electronically", "signed", "dob", "unique", "subjective",
    "observations", "vitamin", "level", "levels", "coordinator", "for",
    "healthplans", "xray", "diagnostic", "form", "report",
}

UI_NOISE = {"select", "filter", "export", "update", "none", "total", "balance"}
_MIN_PROPAGATE = 2


def _common_compound(word: str) -> bool:
    """Whether an OCR-concatenated token consists only of document terms."""
    if len(word) < 8:
        return False
    reachable = {0}
    vocabulary = UI_NOISE | _TOO_COMMON
    for start in range(len(word)):
        if start not in reachable:
            continue
        reachable.update(
            start + len(part)
            for part in vocabulary
            if word.startswith(part, start)
        )
    return len(word) in reachable

# OCR confusion folding is restricted to long alphanumeric identifiers.
_CONFUSION = str.maketrans("OoIiLlSsBbZzGg", "00111155882266")
_FOLDABLE = re.compile(r"^[0-9OoIiLlSsBbZzGg]+$")
# Below this length the folded form collides with many ordinary four- and
# five-digit codes. Above it, plus the identical-length test at lookup, the
# remaining collision surface is much smaller.
_MIN_FOLD = 6


def _fold(key: str) -> str | None:
    """Fold a damaged numeric identifier onto its confusion-class form.

    Returns None for anything that is not a foldable numeric identifier. The
    requirement that the key already contain a digit is what stops an ordinary
    word from folding into a number -- "globes" would otherwise become a
    six-digit lookup key.
    """
    if len(key) < _MIN_FOLD or not _FOLDABLE.match(key):
        return None
    if not any(c.isdigit() for c in key):
        return None
    folded = key.translate(_CONFUSION)
    return folded if folded.isdigit() else None


def _has_label(detector: "Detector", tt: TokenText, i: int, labels) -> bool:
    """Does the text spatially adjacent to token i contain one of `labels`?"""
    context = detector.spatial.context_for(tt.tokens, i)
    return any(label in context for label in labels)


def _suppressed(detector: "Detector", tt: TokenText, i: int, key: str) -> bool:
    """Apply an explicitly selected professional-identity exception."""
    if detector.mask_providers:
        return False
    if key in detector.patient_names:
        return False
    if key in detector.provider_names:
        return True
    return _has_label(detector, tt, i, PROVIDER_LABELS + FACILITY_LABELS)


def learn(detector: "Detector", spans: list[Span], tt: TokenText | None = None) -> None:
    """Remember values proven to be PHI, for propagation across the job."""
    for span in spans:
        if span.entity in {"DATE", "AGE", "LOCATION"}:
            # Dates, ages and place fragments are far too generic to match bare.
            continue
        if span.entity == "PERSON":
            # Names are indexed and propagated per word so shortened forms can
            # retain stable identity tags.
            parts = re.findall(r"[A-Za-z][A-Za-z'\-]+", span.text)
            patient = tt is not None and span.tokens and _has_label(
                detector, tt, span.tokens[0], PATIENT_LABELS
            )
        else:
            parts = [span.text]
            patient = False
        for part in parts:
            key = normalize(part)
            if len(key) < _MIN_PROPAGATE or key in _TOO_COMMON:
                continue
            # A value settled as provider-side on a clean read never enters the
            # lexicon, however a degraded re-read later relabels it.
            if key in detector.provider_names and key not in detector.patient_names:
                continue
            detector.lexicon.setdefault(key, span.entity)
            if patient:
                detector.patient_names.add(key)
            folded = _fold(key)
            if folded:
                detector.numeric_index.setdefault(folded, span.entity)


# A conservative fuzzy match can recover a known name after minor OCR damage.
_FUZZY_RATIO = 0.86
_FUZZY_MIN_LEN = 5


def _letters(text: str) -> str:
    """Reduce to letters so punctuation artifacts do not block matching."""
    return "".join(c for c in text if c.isalpha())


def _fuzzy_entity(detector: "Detector", key: str) -> str | None:
    reduced = _letters(key)
    if len(reduced) < _FUZZY_MIN_LEN:
        return None
    best, score = None, 0.0
    for known, entity in detector.lexicon.items():
        if entity != "PERSON" or not known.isalpha():
            continue
        if abs(len(known) - len(reduced)) > 2:
            continue
        ratio = difflib.SequenceMatcher(None, reduced, known).ratio()
        if ratio > score:
            best, score = entity, ratio
    return best if score >= _FUZZY_RATIO else None


def propagate(detector: "Detector", tt: TokenText) -> list[Span]:
    """Mask every occurrence of an already-proven value on this page."""
    if not detector.lexicon:
        return []
    allowed = _allowlisted_ranges(tt, detector.allowlist)
    spans = []
    for i, token in enumerate(tt.tokens):
        stripped = token.text.strip(_TOKEN_EDGE)
        key = normalize(stripped)
        entity = detector.lexicon.get(key)
        if entity is None and key not in _TOO_COMMON:
            # Exact miss. Try the OCR confusion classes before the letter-level
            # fuzzy match: a damaged MRN is numeric, and _fuzzy_entity only ever
            # considers PERSON entries.
            # Folding preserves length, so a hit here is already a same-length
            # match -- no separate length test is needed.
            folded = _fold(key)
            if folded is not None:
                entity = detector.numeric_index.get(folded)
            if entity is None:
                entity = _fuzzy_entity(detector, key)
        if not entity or entity not in detector.entities:
            continue
        if entity == "PERSON":
            if key in UI_NOISE or _suppressed(detector, tt, i, key):
                continue
        start = tt.offsets[i][0] + token.text.find(stripped)
        if any(s <= start and start + len(stripped) <= e for s, e in allowed):
            continue
        spans.append(
            Span(
                entity=entity,
                start=start,
                end=start + len(stripped),
                score=0.75,
                text=stripped,
                tokens=[i],
                source="propagated",
            )
        )
    return spans
