from tests.benchmark.pdf_deid import _norm, _readable_counts


def test_readable_counts_tracks_occurrences_without_exceeding_ground_truth():
    values = ["Northwind Harbor Institute", "Northwind Harbor Institute", "24/05/1977"]

    counts = _readable_counts(
        "Northwind Harbor Institute appears once; date 24-05-1977.",
        values,
    )

    assert counts[_norm("Northwind Harbor Institute")] == 1
    assert counts[_norm("24/05/1977")] == 1


def test_readable_counts_caps_duplicate_ocr_reads_at_annotations():
    value = "Northwind Harbor Institute"
    counts = _readable_counts(f"{value} {value} {value}", [value, value])
    assert counts[_norm(value)] == 2


def test_short_values_require_word_boundaries():
    counts = _readable_counts("Alcohol OH 156", ["OH", "56"])
    assert counts[_norm("OH")] == 1
    assert counts[_norm("56")] == 0
