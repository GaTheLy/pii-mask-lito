"""Independent tests for domain-neutral text and geometry detection."""

from types import SimpleNamespace

from pii_mask_lito.detect import (
    DateDetector,
    Detector,
    PatternDetector,
    SpatialContextDetector,
    StructuralDetector,
    _fold,
)
from pii_mask_lito.model import Span, Token, TokenText, mask_rects


def test_spatial_label_above_identifier():
    tokens = [
        Token("Customer", 0, (0.10, 0.10, 0.18, 0.12), 0),
        Token("ID", 0, (0.19, 0.10, 0.22, 0.12), 0),
        Token("CUS-4827-9136", 0, (0.10, 0.13, 0.25, 0.15), 1),
    ]
    found = {(span.entity, span.text) for span in SpatialContextDetector().detect(TokenText(tokens))}
    assert ("GENERIC_ID", "CUS-4827-9136") in found


def test_flat_text_uses_preceding_label():
    tt = TokenText.from_text("Reference ID REF-5509-ZINC")
    found = {(span.entity, span.text) for span in SpatialContextDetector().detect(tt)}
    assert ("GENERIC_ID", "REF-5509-ZINC") in found


def test_unlabelled_identifier_shape_is_not_enough():
    assert SpatialContextDetector().detect(TokenText.from_text("Quantity 48279136")) == []


def test_nearest_label_wins_in_dense_form():
    tokens = [
        Token("Account", 0, (0.02, 0.10, 0.10, 0.12), 0),
        Token("number", 0, (0.11, 0.10, 0.18, 0.12), 0),
        Token("Customer", 0, (0.25, 0.10, 0.34, 0.12), 0),
        Token("ID", 0, (0.35, 0.10, 0.38, 0.12), 0),
        Token("CUS-8172-A4", 0, (0.39, 0.10, 0.50, 0.12), 0),
    ]
    found = SpatialContextDetector().detect(TokenText(tokens))
    assert [(span.entity, span.text) for span in found] == [("GENERIC_ID", "CUS-8172-A4")]


def test_age_profile_thresholds():
    tt = TokenText.from_text("Age 46")
    assert [span.text for span in SpatialContextDetector(min_masked_age=0).detect(tt)] == ["46"]
    assert SpatialContextDetector(min_masked_age=90).detect(tt) == []


def test_compound_age_is_detected_without_label():
    tt = TokenText.from_text("The account holder is 46-year-old")
    assert any(span.entity == "AGE" for span in SpatialContextDetector().detect(tt))


def test_calendar_validation_accepts_both_orders():
    tt = TokenText.from_text("Dates 12/31/2026 31/12/2026 02/30/2026")
    values = {span.text for span in DateDetector().detect(tt)}
    assert {"12/31/2026", "31/12/2026"} <= values
    assert "02/30/2026" not in values


def test_deterministic_contact_patterns():
    tt = TokenText.from_text(
        "Reach mira.calder@example.net at 202-555-0147; tax ID 000-00-0000"
    )
    found = {(span.entity, span.text) for span in PatternDetector().detect(tt)}
    assert ("EMAIL_ADDRESS", "mira.calder@example.net") in found
    assert ("PHONE_NUMBER", "202-555-0147") in found
    assert ("US_SSN", "000-00-0000") in found


def test_money_is_not_a_contact_pattern():
    assert PatternDetector().detect(TokenText.from_text("Total 2,080.10 0730")) == []


def test_surname_first_structural_name_keeps_initial():
    spans = StructuralDetector(SpatialContextDetector()).detect(
        TokenText.from_text("CALDER, MIRA Q.")
    )
    assert [span.text for span in spans if span.entity == "PERSON"] == ["CALDER", "MIRA", "Q."]


def test_structural_match_does_not_cross_columns():
    tokens = [
        Token("CALDER,", 0, (0.05, 0.10, 0.12, 0.12), 0),
        Token("MIRA", 0, (0.13, 0.10, 0.18, 0.12), 0),
        Token("Report", 0, (0.62, 0.10, 0.68, 0.12), 0),
        Token("Status", 0, (0.69, 0.10, 0.75, 0.12), 0),
    ]
    names = [
        span.text
        for span in StructuralDetector(SpatialContextDetector()).detect(TokenText(tokens))
        if span.entity == "PERSON"
    ]
    assert names == ["CALDER", "MIRA"]


def test_general_and_hipaa_profiles_are_distinct():
    general = Detector()
    hipaa = Detector(profile="hipaa-safe-harbor")
    assert general.mask_providers is True and general.spatial.min_masked_age == 0
    assert hipaa.mask_providers is False and hipaa.spatial.min_masked_age == 90
    assert "GENERIC_ID" in general.entities and "GENERIC_ID" not in hipaa.entities


def test_explicit_empty_configuration_stays_empty():
    detector = Detector(entities=[], allowlist=[])
    assert detector.entities == [] and detector.allowlist == set()


def test_explicit_allowlist_handles_multiword_value():
    detector = Detector(entities=["PERSON"], allowlist=["Northwind Group"])
    tt = TokenText.from_text("NORTHWIND, GROUP")
    assert detector.detect(tt) == []


def test_cross_page_propagation_masks_known_values():
    detector = Detector(entities=["PERSON", "GENERIC_ID"])
    source = TokenText.from_text("CALDER, MIRA Customer ID CUS-4827-9136")
    spans = detector.detect(source)
    detector.learn(spans, source)
    repeated = TokenText.from_text("Calder CUS-4827-9136")
    found = {(span.entity, span.text) for span in detector.propagate(repeated)}
    assert ("PERSON", "Calder") in found
    assert ("GENERIC_ID", "CUS-4827-9136") in found


def test_numeric_ocr_confusion_folding_is_bounded():
    assert _fold("73186o104") == "731860104"
    assert _fold("total") is None


def test_wide_identifier_is_not_silently_dropped():
    tt = TokenText([Token("Mira", 0, (0.05, 0.10, 0.80, 0.12), 0)])
    span = Span("PERSON", 0, 4, 0.8, "Mira", [0], "presidio")
    assert mask_rects(tt, span)


def test_mask_width_limit_is_available_only_when_explicit():
    tt = TokenText([Token("Mira", 0, (0.05, 0.10, 0.80, 0.12), 0)])
    span = Span("PERSON", 0, 4, 0.8, "Mira", [0], "presidio")
    assert mask_rects(tt, span, max_width=0.3) == []


def test_ocr_date_box_has_trailing_glyph_allowance():
    token = Token(
        "2026-09-1", 0, (0.50, 0.40, 0.59, 0.41), 0, confidence=0.999
    )
    tt = TokenText([token])
    span = Span("DATE", 0, len(token.text), 0.9, token.text, [0], "date")
    _page, box = mask_rects(tt, span)[0]
    assert round(box[2], 6) == 0.605


def test_native_date_box_keeps_standard_padding():
    token = Token("2026-09-11", 0, (0.50, 0.40, 0.60, 0.41), 0)
    tt = TokenText([token])
    span = Span("DATE", 0, len(token.text), 0.9, token.text, [0], "date")
    _page, box = mask_rects(tt, span)[0]
    assert round(box[2], 6) == 0.6012


def test_person_ner_cannot_absorb_an_identifier_from_the_next_column():
    tokens = [
        Token("MIRA", 0, (0.10, 0.20, 0.16, 0.22), 0),
        Token("CALDER", 0, (0.17, 0.20, 0.26, 0.22), 0),
        Token("CUS-4827-9136", 0, (0.60, 0.20, 0.75, 0.22), 0),
    ]
    tt = TokenText(tokens)

    class Analyzer:
        def analyze(self, **_kwargs):
            return [SimpleNamespace(
                entity_type="PERSON", start=0, end=len(tt.text), score=0.91
            )]

    detector = Detector(entities=["PERSON"])
    detector._analyzer = Analyzer()
    found = detector.detect(tt)
    assert [span.text for span in found] == ["MIRA", "CALDER"]
    assert all(not any(character.isdigit() for character in span.text) for span in found)
