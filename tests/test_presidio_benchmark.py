from tests.benchmark.presidio_synth import _positions


def test_benchmark_positions_count_sensitive_glyphs_not_separators():
    text = "Mira Calder"
    assert _positions(text, 0, len(text)) == set(range(4)) | set(range(5, 11))


def test_benchmark_positions_are_clamped_to_the_source_text():
    assert _positions("ID-42", -10, 100) == {0, 1, 3, 4}
