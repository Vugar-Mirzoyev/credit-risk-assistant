"""The policy corpus: eCFR XML turned into sections that can be cited.

The regulations arrive as XML with real structure, and that structure is the
most valuable thing about them. Each ``DIV8[TYPE=SECTION]`` carries its own
number, so a passage does not have to be located by guesswork: it knows it came
from § 1002.9, and an answer built on it can say so.

That matters more here than it would for most corpora. An assistant that
answers a compliance question without a citation is worse than one that
declines, because the reader has no way to check it and every incentive to
believe it. Every unit this module produces carries the part, the section
number and the heading it came from, and the retrieval evaluation scores
whether the right *section* was found, not whether some plausible text was.

Text extraction is deliberately conservative. Regulatory XML is full of
non-breaking spaces, section signs and soft hyphens that survive naive
extraction and then poison both keyword matching and embedding: ``§ 1002.9``
with a non-breaking space does not match ``§ 1002.9`` typed by a person. Those
are normalised once, here, so that every downstream component sees the same
text.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

#: eCFR wraps each numbered section in one of these.
SECTION_TAG = "DIV8"
APPENDIX_TAG = "DIV9"

#: Characters that regulatory XML uses freely and that break naive matching.
_WHITESPACE = dict.fromkeys(map(ord, "     "), " ")
_SOFT_HYPHEN = dict.fromkeys(map(ord, "­​"), None)


@dataclass(frozen=True)
class Section:
    """One numbered section of a regulation, with the text of its paragraphs."""

    document: str
    part: str
    number: str
    heading: str
    paragraphs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def citation(self) -> str:
        """What an answer would put in brackets: `12 CFR 1002.9`."""
        return f"12 CFR {self.number}" if self.number else f"12 CFR {self.part}"

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Corpus:
    """Every section of every regulation, in reading order."""

    sections: tuple[Section, ...]
    issue_date: str = ""

    def __len__(self) -> int:
        return len(self.sections)

    @property
    def words(self) -> int:
        return sum(s.words for s in self.sections)

    def by_citation(self, citation: str) -> Section | None:
        return next((s for s in self.sections if s.citation == citation), None)

    def summary(self) -> str:
        parts: dict[str, list[Section]] = {}
        for section in self.sections:
            parts.setdefault(section.part, []).append(section)
        lines = [
            f"issue date              {self.issue_date or 'unknown'}",
            f"sections                {len(self.sections)}",
            f"words                   {self.words:,}",
        ]
        for part, group in sorted(parts.items()):
            words = sum(s.words for s in group)
            lines.append(
                f"  part {part:<17}{len(group):>4} sections, {words:>9,} words"
            )
        return "\n".join(lines)


def normalise(text: str) -> str:
    """One spelling of every character, so matching is not a lottery."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_WHITESPACE).translate(_SOFT_HYPHEN)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _element_text(element: ET.Element) -> str:
    return normalise(" ".join(t for t in element.itertext() if t))


def _sections_of(root: ET.Element, document: str, part: str) -> list[Section]:
    sections: list[Section] = []
    for node in root.iter():
        if node.tag not in (SECTION_TAG, APPENDIX_TAG):
            continue
        head = node.find("HEAD")
        heading = _element_text(head) if head is not None else ""
        paragraphs = [
            _element_text(p)
            for p in node.iter()
            if p.tag in ("P", "FP", "FP-DASH")
        ]
        paragraphs = [p for p in paragraphs if p]
        if not paragraphs:
            continue
        sections.append(
            Section(
                document=document,
                part=part,
                number=(node.get("N") or "").strip(),
                heading=heading,
                paragraphs=tuple(paragraphs),
            )
        )
    return sections


def load_document(path: Path | str) -> list[Section]:
    """Parse one eCFR part into its sections."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No policy document at {path}. Run scripts/fetch_policies.py to "
            "download the corpus."
        )
    root = ET.fromstring(path.read_bytes())
    part_node = next(
        (e for e in root.iter() if e.get("TYPE") == "PART" and e.get("N")), None
    )
    part = (part_node.get("N") if part_node is not None else path.stem).strip()
    return _sections_of(root, document=path.name, part=part)


def load_corpus(directory: Path | str, *, issue_date: str = "") -> Corpus:
    """Every part in the directory, in filename order."""
    directory = Path(directory)
    files = sorted(directory.glob("*.xml"))
    if not files:
        raise FileNotFoundError(
            f"No policy documents in {directory}. Run "
            "scripts/fetch_policies.py first; the corpus is the Layer 2 "
            "equivalent of the raw extract and nothing works without it."
        )
    sections: list[Section] = []
    for path in files:
        sections.extend(load_document(path))
    return Corpus(sections=tuple(sections), issue_date=issue_date)
