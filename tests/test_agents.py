"""Tests for the deterministic boundaries around optional model assistance."""

from pii_mask_lito.agents import (
    AgenticDetector,
    Field,
    Ollama,
    OpenAICompatible,
    _coerce_fields,
    _first_json,
    canonical_type,
    resolve_value,
    transport,
)
from pii_mask_lito.model import Span, TokenText


def test_canonical_type_maps_cross_domain_labels():
    assert canonical_type("employee id") == "GENERIC_ID"
    assert canonical_type("customer name") == "PERSON"
    assert canonical_type("postal code") == "LOCATION"
    assert canonical_type("unknown category") is None


def test_first_json_ignores_model_wrapping():
    assert _first_json('Result follows: {"doc_type": "invoice"}\nDone.') == {
        "doc_type": "invoice"
    }


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


def test_value_resolution_uses_text_not_model_coordinates():
    tt = TokenText.from_text("Customer ID: CUS-4827-9136")
    indices = resolve_value(tt, "Customer ID CUS-4827-9136", "GENERIC_ID")
    assert [tt.tokens[index].text for index in indices] == ["CUS-4827-9136"]


def test_value_resolution_tolerates_punctuation():
    tt = TokenText.from_text("Email (mira.calder@example.net)")
    indices = resolve_value(tt, "mira.calder@example.net", "EMAIL_ADDRESS")
    assert indices


def test_agent_reconciliation_is_additive():
    rule = Span("PERSON", 0, 4, 0.8, "Mira", [0], "presidio")
    added = Span("GENERIC_ID", 5, 18, 0.7, "CUS-4827-9136", [1], "agent")
    detector = AgenticDetector.__new__(AgenticDetector)
    merged = detector._reconcile([rule], [added], {0})
    assert {(span.entity, span.text) for span in merged} == {
        ("PERSON", "Mira"),
        ("GENERIC_ID", "CUS-4827-9136"),
    }


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
