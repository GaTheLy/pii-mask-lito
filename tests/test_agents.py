"""Tests for the deterministic boundaries around optional model assistance."""

import pytest

from pii_mask_lito.agents import (
    AgenticDetector,
    Field,
    Ollama,
    OpenAICompatible,
    _coerce_fields,
    _first_json,
    _safe_doc_type,
    canonical_type,
    resolve_value,
    transport,
)
from pii_mask_lito.model import Span, Token, TokenText


def test_canonical_type_maps_cross_domain_labels():
    assert canonical_type("employee id") == "GENERIC_ID"
    assert canonical_type("customer name") == "PERSON"
    assert canonical_type("postal code") == "LOCATION"
    assert canonical_type("unknown category") is None


def test_first_json_ignores_model_wrapping():
    assert _first_json('Result follows: {"doc_type": "invoice"}\nDone.') == {
        "doc_type": "invoice"
    }


def test_document_type_is_reduced_to_a_safe_log_category():
    assert _safe_doc_type("Invoice for Mira Calder") == "invoice"
    assert _safe_doc_type("Mira Calder personal dossier") == "unknown"


def test_malformed_field_output_is_ignored():
    reply = {
        "fields": [
            "Email",
            {"label": "Customer ID", "value": "CUS-4827-9136", "type": "identifier"},
            {"value": "missing label"},
        ]
    }
    assert _coerce_fields(reply) == [
        Field(label="Customer ID", value="CUS-4827-9136", entity="GENERIC_ID")
    ]


def test_semantic_field_action_and_reason_are_coerced():
    fields = _coerce_fields({"fields": [{
        "label": "Category",
        "value": "Order",
        "type": "other",
        "action": "KEEP",
        "reason": "ordinary heading",
    }]})
    assert fields == [
        Field(label="Category", value="Order", decision="keep",
              reason="ordinary heading")
    ]


def test_value_resolution_uses_text_not_model_coordinates():
    tt = TokenText.from_text("Customer ID: CUS-4827-9136")
    indices = resolve_value(tt, "Customer ID CUS-4827-9136", "GENERIC_ID")
    assert [tt.tokens[index].text for index in indices] == ["CUS-4827-9136"]


def test_value_resolution_tolerates_punctuation():
    tt = TokenText.from_text("Email (mira.calder@example.net)")
    indices = resolve_value(tt, "mira.calder@example.net", "EMAIL_ADDRESS")
    assert indices


def test_strict_union_reconciliation_is_additive():
    rule = Span("PERSON", 0, 4, 0.8, "Mira", [0], "presidio")
    added = Span("GENERIC_ID", 5, 18, 0.7, "CUS-4827-9136", [1], "agent")
    detector = AgenticDetector.__new__(AgenticDetector)
    detector.mode = "strict-union"
    merged = detector._reconcile([rule], [added], {0})
    assert {(span.entity, span.text) for span in merged} == {
        ("PERSON", "Mira"),
        ("GENERIC_ID", "CUS-4827-9136"),
    }


def test_hybrid_vetoes_soft_candidate_but_not_validated_identifier():
    soft = Span("PERSON", 0, 8, 0.7, "Category", [0], "presidio")
    hard = Span("EMAIL_ADDRESS", 9, 31, 0.9, "mira.calder@example.net", [1], "pattern")
    detector = AgenticDetector.__new__(AgenticDetector)
    detector.mode = "hybrid"
    merged = detector._reconcile([soft, hard], [], set(), {0, 1})
    assert [(span.entity, span.text) for span in merged] == [
        ("EMAIL_ADDRESS", "mira.calder@example.net")
    ]


def test_hybrid_keep_region_blocks_late_soft_spans_only():
    tt = TokenText([
        Token("Category", 0, (0.10, 0.20, 0.20, 0.23), 0),
    ])
    soft = Span("PERSON", 0, 8, 0.7, "Category", [0], "propagated")
    hard = Span("EMAIL_ADDRESS", 0, 8, 0.9, "Category", [0], "pattern")
    detector = AgenticDetector.__new__(AgenticDetector)
    detector.mode = "hybrid"
    detector.keep_regions = {0: [(0.10, 0.20, 0.20, 0.23)]}
    assert detector.filter_late_spans(0, tt, [soft, hard]) == [hard]
    assert detector.filter_late_spans(1, tt, [soft]) == [soft]


class _FakeRules:
    mask_providers = True
    mask_organizations = False

    class spatial:
        min_masked_age = 0

    def detect(self, _tt):
        return [Span("PERSON", 0, 8, 0.7, "Category", [0], "presidio")]


class _FakeSemanticModel:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def ask(self, _prompt, _image=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return {
            "doc_type": "order form",
            "candidate_decisions": [
                {"candidate_id": "0", "action": "keep", "reason": "heading"}
            ],
            "fields": [{
                "label": "Reference ID",
                "value": "REF-5509-ZINC",
                "owner": "customer",
                "type": "identifier",
                "action": "mask",
                "reason": "customer identifier",
            }],
        }


def test_hybrid_uses_one_call_to_keep_soft_candidate_and_add_field():
    model = _FakeSemanticModel()
    detector = AgenticDetector(
        _FakeRules(), model=model, verbose=False, mode="hybrid"
    )
    found = detector.detect(object(), TokenText.from_text("Category REF-5509-ZINC"))
    assert model.calls == 1
    assert [(span.entity, span.text) for span in found] == [
        ("GENERIC_ID", "REF-5509-ZINC")
    ]
    assert detector.trace.steps[0]["agent"] == "semantic_page"
    assert detector.trace.steps[0]["doc_type"] == "order form"


def test_rules_only_never_calls_model_and_model_failure_keeps_rules():
    text = TokenText.from_text("Category")
    rules_only_model = _FakeSemanticModel(fail=True)
    rules_only = AgenticDetector(
        _FakeRules(), model=rules_only_model, verbose=False, mode="rules-only"
    )
    assert rules_only.detect(object(), text)[0].text == "Category"
    assert rules_only_model.calls == 0

    failing_model = _FakeSemanticModel(fail=True)
    hybrid = AgenticDetector(
        _FakeRules(), model=failing_model, verbose=False, mode="hybrid"
    )
    assert hybrid.detect(object(), text)[0].text == "Category"
    assert failing_model.calls == 1


def test_begin_document_clears_page_local_state():
    detector = AgenticDetector(
        _FakeRules(), model=_FakeSemanticModel(), verbose=False, mode="hybrid"
    )
    detector.page = 4
    detector.keep_regions = {0: [(0.1, 0.1, 0.2, 0.2)]}
    detector.trace.add("semantic_page", 0.1, {"fields": 1})

    detector.begin_document()

    assert detector.page == 0
    assert detector.keep_regions == {}
    assert detector.trace.steps == []


def test_unknown_agent_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown masking mode"):
        AgenticDetector(_FakeRules(), model=_FakeSemanticModel(), mode="unknown")


def test_grounding_never_masks_money():
    tt = TokenText.from_text("Amount $72.00")
    detector = AgenticDetector.__new__(AgenticDetector)
    spans, kept = detector._ground(
        tt,
        [Field(label="Amount", value="$72.00", entity="GENERIC_ID", decision="mask")],
    )
    assert spans == [] and kept == set()


def test_unknown_model_tag_defaults_to_local_transport():
    assert isinstance(transport("small-vision:latest"), Ollama)


def test_explicit_compatible_transport_requires_base_url():
    try:
        transport("openai-compatible:model")
    except ValueError as exc:
        assert "base URL" in str(exc)
    else:
        raise AssertionError("missing base URL should fail")


def test_explicit_compatible_transport_uses_supplied_endpoint():
    model = transport("openai-compatible:model@https://models.example.net/v1")
    assert isinstance(model, OpenAICompatible)
