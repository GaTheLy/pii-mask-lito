from pathlib import Path

from tools.privacy_gate import _artifact_issue, _scan_text


def test_privacy_gate_accepts_reserved_synthetic_identifiers():
    text = "Contact a@example.test from the generated fixture."
    assert _scan_text(Path("samples/fixture.txt"), text) == []


def test_privacy_gate_reports_categories_without_echoing_values():
    text = "\n".join([
        "owner=" + "person@" + "corp.local",
        "path=" + "/" + "Users/fictional/private.pdf",
        "source=" + "sample" + "-22.pdf",
        "key=" + "AKIA" + "A" * 16,
    ])
    issues = _scan_text(Path("probe.txt"), text)
    categories = [category for _line, category in issues]
    assert categories == [
        "non-reserved email address",
        "absolute user-home path",
        "private sample marker",
        "credential/private-key pattern",
    ]
    encoded = repr(issues)
    assert "fictional" not in encoded and "corp.local" not in encoded


def test_privacy_gate_rejects_output_and_session_filenames():
    assert _artifact_issue(Path("masked-output.pdf")) == "masked output artifact"
    assert _artifact_issue(Path("job-report.json")) == "masking report artifact"
    assert _artifact_issue(Path("trace.log")) == "operational log/session artifact"
    assert _artifact_issue(Path("state.ses")) == "operational log/session artifact"
    assert _artifact_issue(Path("tests/fixture.json")) is None
