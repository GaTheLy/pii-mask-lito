"""Fail CI when tracked release content resembles private operational data.

The gate reports only file, line, and category. It deliberately never echoes a
matched value: a secret scanner must not turn CI logs into a second disclosure.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {
    "", ".cfg", ".csv", ".dockerignore", ".gitignore", ".html", ".ini",
    ".json", ".md", ".py", ".rst", ".toml", ".txt", ".xml", ".yaml",
    ".yml",
}
OUTPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".xlsx", ".csv"}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)
HOME_PATHS = (
    re.compile("/" + "Users/"),
    re.compile("/" + "home/"),
    re.compile(r"[A-Z]:\\" + r"Users\\", re.I),
)
PRIVATE_MARKERS = (re.compile("sample" + r"[-_ ]?22", re.I),)
SECRET_PATTERNS = (
    re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)\b"
        r"\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{12,}['\"]"
    ),
)
RESERVED_EMAIL_DOMAINS = {
    "example.com", "example.net", "example.org", "example.test",
    "users.noreply.github.com",
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / raw.decode() for raw in result.stdout.split(b"\0") if raw]


def _artifact_issue(path: Path) -> str | None:
    name = path.name.casefold()
    if path.suffix.casefold() in {".log", ".ses"}:
        return "operational log/session artifact"
    if path.suffix.casefold() == ".json" and "report" in name:
        return "masking report artifact"
    if path.suffix.casefold() in OUTPUT_SUFFIXES and (
        name.startswith(("masked-", "masked_", "masked."))
        or "-masked." in name
    ):
        return "masked output artifact"
    return None


def _scan_text(path: Path, text: str) -> list[tuple[int, str]]:
    issues: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in HOME_PATHS):
            issues.append((line_number, "absolute user-home path"))
        if any(pattern.search(line) for pattern in PRIVATE_MARKERS):
            issues.append((line_number, "private sample marker"))
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            issues.append((line_number, "credential/private-key pattern"))
        for match in EMAIL.finditer(line):
            if match.group(1).casefold() not in RESERVED_EMAIL_DOMAINS:
                issues.append((line_number, "non-reserved email address"))
    return issues


def scan(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT) if path.is_absolute() else path
        artifact = _artifact_issue(relative)
        if artifact:
            problems.append(f"{relative}: {artifact}")
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        # Third-party notices can legitimately carry public copyright contact
        # details; they are provenance, not project operational data.
        if path.name == "LICENSE" or path.name.endswith(".LICENSE"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line, category in _scan_text(relative, text):
            problems.append(f"{relative}:{line}: {category}")
    return problems


def main() -> int:
    problems = scan(_tracked_files())
    if not problems:
        print("privacy gate passed")
        return 0
    print("privacy gate failed; matched values are intentionally omitted")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
