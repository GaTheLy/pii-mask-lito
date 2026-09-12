"""Independent tests for domain-neutral text and geometry detection."""

from types import SimpleNamespace

from pii_mask_lito.detect import (
    DateDetector,
    Detector,
    PatternDetector,
    SpatialContextDetector,
    StructuralDetector,
    _fold,
    _generic_name,
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


def test_spatial_location_context_masks_short_house_and_postal_numbers():
    tokens = [
        Token("Location:", 0, (0.10, 0.10, 0.20, 0.12), 0),
        Token("731", 0, (0.10, 0.13, 0.14, 0.15), 1),
        Token("United", 0, (0.30, 0.13, 0.37, 0.15), 1),
        Token("States", 0, (0.38, 0.13, 0.44, 0.15), 1),
        Token("55962", 0, (0.45, 0.13, 0.51, 0.15), 1),
    ]
    found = {(span.entity, span.text) for span in SpatialContextDetector().detect(TokenText(tokens))}
    assert ("LOCATION", "731") in found
    assert ("LOCATION", "55962") in found


def test_form_number_is_a_contextual_identifier():
    tt = TokenText.from_text("Diagnostic Form FORM-731")
    found = {(span.entity, span.text) for span in SpatialContextDetector().detect(tt)}
    assert ("GENERIC_ID", "FORM-731") in found


def test_surname_first_pattern_does_not_parse_a_digit_led_address():
    tt = TokenText.from_text("731 Lantern Quays, North Ember")
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert not [span for span in spans if span.entity == "PERSON"]


def test_structural_address_block_uses_label_and_stops_before_phone():
    tokens = [
        Token("Contact", 0, (0.10, 0.10, 0.18, 0.12), 0),
        Token("5891", 0, (0.10, 0.13, 0.15, 0.15), 1),
        Token("Kenneth", 0, (0.16, 0.13, 0.24, 0.15), 1),
        Token("Ports,", 0, (0.25, 0.13, 0.31, 0.15), 1),
        Token("97375", 0, (0.32, 0.13, 0.38, 0.15), 1),
        Token("Tel:", 0, (0.39, 0.13, 0.43, 0.15), 1),
        Token("(253)", 0, (0.44, 0.13, 0.49, 0.15), 1),
    ]
    spans = StructuralDetector(SpatialContextDetector()).detect(TokenText(tokens))
    locations = [span for span in spans if span.entity == "LOCATION"]
    assert [span.text for span in locations] == ["5891 Kenneth Ports, 97375"]
    assert locations[0].tokens == [1, 2, 3, 4]


def test_facility_pattern_does_not_match_health_prefix_inside_a_word():
    tt = TokenText.from_text("Coordinator For Healthworks")
    spans = StructuralDetector(
        SpatialContextDetector(mask_organizations=True)
    ).detect(tt)
    assert not [span for span in spans if span.entity == "ORGANIZATION"]


def test_organization_suffix_is_not_parsed_as_surname_first():
    tt = TokenText.from_text("APOLLO PATHOLOGY ASSOCIATES, INC")
    spans = StructuralDetector(
        SpatialContextDetector(mask_organizations=True)
    ).detect(tt)
    assert not [span for span in spans if span.entity == "PERSON"]
    assert [span.entity for span in spans] == ["ORGANIZATION"]


def test_document_phrases_are_not_short_organization_names():
    tt = TokenText.from_text("Her medical\nCurrent health\nContact clinic\nPast Hospital")
    spans = StructuralDetector(
        SpatialContextDetector(mask_organizations=True)
    ).detect(tt)
    assert not [span for span in spans if span.entity == "ORGANIZATION"]


def test_city_state_without_zip_is_still_a_location():
    tt = TokenText.from_text("TOLEDO, OH (419) 555-8923")
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert [span.text for span in spans if span.entity == "LOCATION"] == ["TOLEDO, OH"]


def test_labelled_location_stops_before_the_next_dense_field():
    tt = TokenText.from_text("LOCATION: North Ember BILLING NO: 9509645")
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert [span.text for span in spans if span.entity == "LOCATION"] == ["North Ember"]

    heading = TokenText.from_text("CITY AND POSTAL CODE")
    assert not [
        span for span in StructuralDetector(SpatialContextDetector()).detect(heading)
        if span.entity == "LOCATION"
    ]


def test_concatenated_document_heading_is_not_a_person():
    span = Span("PERSON", 0, 20, 0.8, "DIAGNOSTICFORMREPORT", [0])
    assert _generic_name(span)


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
    tt = TokenText.from_text("Age 46; the applicant is aged 56")
    assert [span.text for span in SpatialContextDetector(min_masked_age=0).detect(tt)] == [
        "46", "56"
    ]
    assert SpatialContextDetector(min_masked_age=90).detect(tt) == []


def test_age_can_be_explained_by_following_words():
    tt = TokenText.from_text(
        "The applicant is 36 year old and just turned 54; when he was 78"
    )
    assert [(span.entity, span.text) for span in SpatialContextDetector().detect(tt)] == [
        ("AGE", "36"), ("AGE", "54"), ("AGE", "78")
    ]


def test_compound_age_is_detected_without_label():
    tt = TokenText.from_text("The account holder is 46-year-old")
    assert any(span.entity == "AGE" for span in SpatialContextDetector().detect(tt))


def test_calendar_validation_accepts_both_orders():
    tt = TokenText.from_text("Dates 12/31/2026 31/12/2026 02/30/2026")
    values = {span.text for span in DateDetector().detect(tt)}
    assert {"12/31/2026", "31/12/2026"} <= values
    assert "02/30/2026" not in values


def test_weekday_is_a_date_only_with_scheduling_context():
    found = DateDetector().detect(TokenText.from_text("We'll meet Tuesday at noon"))
    assert [(span.entity, span.text) for span in found] == [("DATE", "Tuesday")]
    assert DateDetector().detect(TokenText.from_text("Tuesday report totals")) == []


def test_ocr_split_year_is_reassembled_with_both_token_boxes():
    tt = TokenText([
        Token("19/04/1", 0, (0.10, 0.20, 0.16, 0.22), 0, confidence=0.99),
        Token("976", 0, (0.17, 0.20, 0.20, 0.22), 0, confidence=0.99),
    ])
    spans = DateDetector().detect(tt)
    assert [(span.text, span.tokens) for span in spans] == [("19/04/1 976", [0, 1])]

    tt.tokens[1].line = 1
    assert DateDetector().detect(tt) == []


def test_ocr_date_fragments_can_recover_a_digit_glued_to_its_label():
    tt = TokenText([
        Token("ate:0", 0, (0.10, 0.20, 0.15, 0.22), 0, confidence=0.99),
        Token("9/12", 0, (0.155, 0.20, 0.19, 0.22), 0, confidence=0.99),
        Token("/2026", 0, (0.195, 0.20, 0.24, 0.22), 0, confidence=0.99),
    ])
    spans = DateDetector().detect(tt)
    assert [(span.text, span.tokens) for span in spans] == [
        ("09/12/2026", [0, 1, 2])
    ]


def test_deterministic_contact_patterns():
    tt = TokenText.from_text(
        "Reach mira.calder@example.net at 202-555-0147; tax ID 000-00-0000"
    )
    found = {(span.entity, span.text) for span in PatternDetector().detect(tt)}
    assert ("EMAIL_ADDRESS", "mira.calder@example.net") in found
    assert ("PHONE_NUMBER", "202-555-0147") in found
    assert ("US_SSN", "000-00-0000") in found


def test_international_grouped_phone_numbers_are_supported():
    tt = TokenText.from_text(
        "Call (62) 000-147, 62 000 0147, +620000000147, +620000 014 700, "
        "+1-202-555-0199x731, or 2025550199-Fax"
    )
    found = {(span.entity, span.text) for span in PatternDetector().detect(tt)}
    assert ("PHONE_NUMBER", "(62) 000-147") in found
    assert ("PHONE_NUMBER", "62 000 0147") in found
    assert ("PHONE_NUMBER", "+620000000147") in found
    assert ("PHONE_NUMBER", "+620000 014 700") in found
    assert ("PHONE_NUMBER", "+1-202-555-0199x731") in found
    assert ("PHONE_NUMBER", "2025550199") in found


def test_phone_pattern_does_not_cross_distant_form_columns():
    tokens = [
        Token("+1", 0, (0.10, 0.20, 0.12, 0.22), 0),
        Token("202-555-0166", 0, (0.13, 0.20, 0.25, 0.22), 0),
        Token("2026", 0, (0.60, 0.20, 0.65, 0.22), 0),
    ]
    found = [
        span for span in PatternDetector().detect(TokenText(tokens))
        if span.entity == "PHONE_NUMBER"
    ]
    assert [(span.text, span.tokens) for span in found] == [
        ("+1 202-555-0166", [0, 1])
    ]


def test_labelled_unvalidated_card_number_is_contextual():
    tt = TokenText.from_text("card number 123456789012345? cc 503802053770")
    found = {(span.entity, span.text) for span in SpatialContextDetector().detect(tt)}
    assert ("CREDIT_CARD", "123456789012345") in found
    assert ("CREDIT_CARD", "503802053770") in found


def test_explicit_name_and_address_fields_do_not_require_ner():
    tt = TokenText.from_text(
        "Name: Mira Calder\nAddress: the corner of ul. Lanternowa 127 and Harbor Bypass"
    )
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    found = {(span.entity, span.text) for span in spans}
    assert ("PERSON", "Mira Calder") in found
    assert (
        "LOCATION", "the corner of ul. Lanternowa 127 and Harbor Bypass"
    ) in found

    moved = TokenText.from_text(
        "Please update my new address is the corner of 159 Eleftheriou Venizelou str"
    )
    found = StructuralDetector(SpatialContextDetector()).detect(moved)
    assert [span.text for span in found if span.entity == "LOCATION"] == [
        "the corner of 159 Eleftheriou Venizelou str"
    ]


def test_apartment_unit_line_is_location_content():
    tt = TokenText.from_text("Apt. 931")
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert [(span.entity, span.text) for span in spans] == [("LOCATION", "Apt. 931")]


def test_international_street_and_postal_lines_are_structural_locations():
    tt = TokenText.from_text("Lantern tee 87\nEstoria 62031")
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert {(span.entity, span.text) for span in spans} == {
        ("LOCATION", "Lantern tee 87"),
        ("LOCATION", "Estoria 62031"),
    }


def test_multiline_address_block_uses_unit_and_postal_evidence():
    text = "6614 Lantern tee 87\nApt. 931\nSälumere\nEstoria 62031"
    spans = StructuralDetector(SpatialContextDetector()).detect(TokenText.from_text(text))
    blocks = [span for span in spans if span.entity == "LOCATION" and "Sälumere" in span.text]
    assert len(blocks) == 1 and blocks[0].text == text

    decorated = "??? 688 rue du centre 320\n??? apt. 169\n??? marke\n??? belgium 97466"
    spans = StructuralDetector(SpatialContextDetector()).detect(TokenText.from_text(decorated))
    assert any(span.entity == "LOCATION" and "marke" in span.text for span in spans)

    prose = (
        "The address of Northwind is 6750 Glasskatu 25 Apt. 864\n"
        "Harboros\n\nEstoria 64677"
    )
    spans = StructuralDetector(SpatialContextDetector()).detect(TokenText.from_text(prose))
    assert any(
        span.entity == "LOCATION" and span.text.startswith("6750 Glasskatu")
        and span.text.endswith("Estoria 64677")
        for span in spans
    )


def test_high_signal_person_context_supports_lowercase_and_unicode_names():
    tt = TokenText.from_text(
        "the gender of mirabel is unknown\n"
        "cassia: I'm dorian's daughter\n"
        "Héldor: What a wife.\n"
        "He was called Árven Þórsen\n"
        "My name is Mireya\n"
        "Directed By: Dorian Vale"
    )
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert {span.text for span in spans if span.entity == "PERSON"} == {
        "mirabel", "cassia", "dorian", "Héldor", "Árven Þórsen",
        "Mireya", "Dorian Vale",
    }


def test_high_signal_organization_context_masks_acronym_and_nonprofit_name():
    tt = TokenText.from_text("I worked for ZETA. Glasswing is a 501(c)3")
    spans = StructuralDetector(
        SpatialContextDetector(mask_organizations=True)
    ).detect(tt)
    assert {span.text for span in spans if span.entity == "ORGANIZATION"} == {
        "ZETA", "Glasswing"
    }

    suffixes = TokenText.from_text("Northwind Technologies\nHarbor Van Lines")
    spans = StructuralDetector(
        SpatialContextDetector(mask_organizations=True)
    ).detect(suffixes)
    assert {span.text for span in spans if span.entity == "ORGANIZATION"} == {
        "Northwind Technologies", "Harbor Van Lines"
    }

    suffixes = TokenText.from_text("Lantern Corporation\nSilverglass Orchestra")
    spans = StructuralDetector(
        SpatialContextDetector(mask_organizations=True)
    ).detect(suffixes)
    assert {span.text for span in spans if span.entity == "ORGANIZATION"} == {
        "Lantern Corporation", "Silverglass Orchestra"
    }


def test_explicit_location_phrases_support_unicode_place_names():
    tt = TokenText.from_text(
        "Our home city ΝΕΑ ΑΚΤΗ: is nearby.\n"
        "Celebrating its tenth year in Žaromira, we returned."
    )
    spans = StructuralDetector(SpatialContextDetector()).detect(tt)
    assert {span.text for span in spans if span.entity == "LOCATION"} == {
        "ΝΕΑ ΑΚΤΗ", "Žaromira"
    }


def test_sentence_boundary_glued_by_ocr_is_not_a_bare_url():
    detector = Detector(entities=[])
    tt = TokenText.from_text("regimen.Patient")
    assert not detector._plausible("URL", "regimen.Pa", tt, 0, 10)
    assert detector._plausible("URL", "example.co", tt, 0, 10)


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


def test_rechecked_ocr_date_keeps_trailing_glyph_allowance():
    token = Token(
        "17/08/1974", 0, (0.10, 0.20, 0.20, 0.21), 0, confidence=0.95
    )
    tt = TokenText([token])
    span = Span("DATE", 0, len(token.text), 0.9, token.text, [0], "recheck")
    _page, box = mask_rects(tt, span)[0]
    assert round(box[2], 6) == 0.215


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
