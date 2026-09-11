"""Vision helpers distinguish meaningful OCR text from stroke-like debris."""

from pii_mask_lito.vision import _looks_read


def test_words_and_identifiers_count_as_read():
    for text in ("Signature:", "Customer", "Email", "CUS48279136", "A14", "record id"):
        assert _looks_read(text), text


def test_letter_digit_soup_is_debris():
    for text in ("15FW5FH", "5FH5FH", "X7Q9W8P0"):
        assert not _looks_read(text), text


def test_short_or_blank_is_not_a_reading():
    for text in ("", "L", "7", "#"):
        assert not _looks_read(text), text
