"""Structural names must include every name token, including initials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pii_mask_lito.detect import SpatialContextDetector, StructuralDetector
from pii_mask_lito.model import Token, TokenText


def _name_line(*words):
    """One printed line of name tokens, laid out left to right."""
    tokens = []
    x = 0.10
    for word in words:
        tokens.append(Token(word, 0, (x, 0.20, x + 0.02 * len(word), 0.212), 0))
        x += 0.02 * len(word) + 0.008
    return TokenText(tokens)


def _masked(*words):
    spans = StructuralDetector(SpatialContextDetector()).detect(_name_line(*words))
    return {s.text for s in spans if s.entity == "PERSON"}


def test_middle_initial_is_masked():
    assert "Q" in _masked("CALDER,", "MIRA", "Q")


def test_middle_initial_with_a_period_is_masked():
    assert _masked("VALE,", "DORIAN", "P.") & {"P", "P."}


def test_full_middle_name_still_masked():
    """The fix must not cost the case that already worked."""
    assert "MARIS" in _masked("QUILL,", "NIA", "MARIS")


def test_surname_and_forename_still_separate_tags():
    """Per-word spans are what give a person stable indices across pages."""
    assert {"CALDER", "MIRA"} <= _masked("CALDER,", "MIRA", "Q")


def test_punctuation_alone_is_not_a_name_token():
    """What the length guard was actually for: the tokenizer's stray commas."""
    assert not (_masked("QUILL,", "NIA", ",") & {",", ""})
