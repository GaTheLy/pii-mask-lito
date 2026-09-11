"""Stable indexed replacement tags.

Each normalized value receives one index that is reused throughout a job. This
preserves document-level relationships without retaining the original value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EDGE_PUNCT = " \t\r\n.,;:#()[]{}/\\'\"<>|*"
# Separator punctuation is dropped even mid-string so superficial punctuation
# differences share an index. ``@``, ``.``, ``-``, ``_``, and ``/`` are kept
# because they can carry meaning inside emails, identifiers, and dates.
_SEPARATORS = re.compile(r"[,;|]+")


def normalize(value: str) -> str:
    """Fold a value to its registry key.

    Casefold, drop separator punctuation, collapse whitespace, strip edge
    punctuation so case, spacing, and separator variants resolve to one index,
    while structural punctuation inside an email or identifier is retained.
    """
    return " ".join(_SEPARATORS.sub(" ", value).split()).strip(_EDGE_PUNCT).casefold()


@dataclass(frozen=True)
class Entry:
    entity: str
    value: str
    index: int
    tag: str


class TagRegistry:
    """Assigns `<TYPE#N>` tags, N counted per entity type, stable within a job.

    One registry spans every page and every file in a run, so the same person
    appearing in a PDF and its companion spreadsheet gets the same index.
    """

    def __init__(self, template: str = "<{entity}#{index}>", short: bool = True):
        self.template = template
        # Short names keep the tag inside a box the exact size of the original.
        self.short = short
        self._index: dict[tuple[str, str], int] = {}
        self._next: dict[str, int] = {}
        self._first_seen: dict[tuple[str, str], str] = {}

    def _display(self, entity: str) -> str:
        from .policy import SHORT_NAMES

        return SHORT_NAMES.get(entity, entity) if self.short else entity

    def tag(self, entity: str, value: str) -> str:
        key = (entity, normalize(value))
        if key not in self._index:
            self._index[key] = self._next.get(entity, 0)
            self._next[entity] = self._index[key] + 1
            self._first_seen[key] = value
        return self.template.format(entity=self._display(entity), index=self._index[key])

    def entries(self) -> list[Entry]:
        """Every value seen, for the masking report."""
        return [
            Entry(
                entity=entity,
                value=self._first_seen[(entity, norm)],
                index=idx,
                tag=self.template.format(entity=self._display(entity), index=idx),
            )
            for (entity, norm), idx in sorted(self._index.items(), key=lambda kv: (kv[0][0], kv[1]))
        ]

    def values(self) -> list[str]:
        """Original values, for the read-back verification pass."""
        return list(self._first_seen.values())

    def __len__(self) -> int:
        return len(self._index)
