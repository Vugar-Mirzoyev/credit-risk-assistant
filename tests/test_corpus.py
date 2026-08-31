"""Tests for the policy corpus.

Two things are worth protecting. A section that loses its number cannot be
cited, and an answer without a citation is worse than no answer in a
compliance setting. And regulatory XML is full of characters that look like
ordinary spaces and are not: if those survive, a search for "§ 1002.9" fails
against text that appears to contain exactly that.
"""

from __future__ import annotations

import pytest

from src.rag.corpus import Section, load_corpus, load_document, normalise

XML = """<?xml version="1.0" encoding="UTF-8"?>
<ECFR>
  <DIV5 TYPE="PART" N="1002">
    <HEAD>PART 1002—EQUAL CREDIT OPPORTUNITY ACT</HEAD>
    <DIV8 TYPE="SECTION" N="1002.2">
      <HEAD>§ 1002.2 Definitions.</HEAD>
      <P>(p) Empirically derived and other credit­scoring systems.</P>
      <P>A credit scoring system evaluates an applicant mechanically.</P>
    </DIV8>
    <DIV8 TYPE="SECTION" N="1002.9">
      <HEAD>§ 1002.9 Notifications.</HEAD>
      <P>(a) Notification of action taken within 30 days.</P>
    </DIV8>
    <DIV8 TYPE="SECTION" N="1002.99">
      <HEAD>§ 1002.99 Reserved.</HEAD>
    </DIV8>
    <DIV9 TYPE="APPENDIX" N="Appendix B">
      <HEAD>Appendix B to Part 1002</HEAD>
      <P>Model form C-1.</P>
    </DIV9>
  </DIV5>
</ECFR>
"""


def _write(tmp_path, body=XML, name="part1002.xml"):
    path = tmp_path / name
    path.write_bytes(body.encode("utf-8"))
    return path


def test_sections_keep_the_number_that_makes_them_citable():
    section = Section(
        document="x.xml", part="1002", number="1002.9", heading="Notifications"
    )

    assert section.citation == "12 CFR 1002.9"


def test_every_section_is_found_with_its_heading(tmp_path):
    sections = load_document(_write(tmp_path))

    numbers = [s.number for s in sections]
    assert "1002.2" in numbers
    assert "1002.9" in numbers
    assert all(s.part == "1002" for s in sections)
    notifications = next(s for s in sections if s.number == "1002.9")
    assert "Notifications" in notifications.heading


def test_appendices_are_part_of_the_corpus(tmp_path):
    # Model adverse action forms live in Appendix B. Dropping appendices would
    # lose exactly the passages a notice question needs.
    sections = load_document(_write(tmp_path))

    assert any("Appendix B" in s.heading for s in sections)


def test_a_section_with_no_text_is_skipped(tmp_path):
    sections = load_document(_write(tmp_path))

    assert "1002.99" not in [s.number for s in sections]


def test_paragraphs_survive_as_separate_units(tmp_path):
    sections = load_document(_write(tmp_path))
    definitions = next(s for s in sections if s.number == "1002.2")

    assert len(definitions.paragraphs) == 2
    assert definitions.text.count("\n\n") == 1
    assert definitions.words > 5


def test_characters_that_look_like_spaces_are_normalised():
    # A non-breaking space between the section sign and the number makes a
    # search for the obvious string fail against text that appears to contain
    # it. Soft hyphens do the same inside a word.
    assert normalise("§ 1002.9") == "§ 1002.9"
    assert normalise("credit­scoring") == "creditscoring"
    assert normalise("a​b") == "ab"
    assert normalise("  spaced   out  ") == "spaced out"


def test_the_section_sign_itself_is_kept():
    assert "§" in normalise("§ 1002.9 Notifications.")


def test_normalisation_reaches_the_loaded_text(tmp_path):
    sections = load_document(_write(tmp_path))
    joined = "\n".join(s.heading + s.text for s in sections)

    for bad in (" ", "­", "​"):
        assert bad not in joined


def test_a_corpus_can_be_looked_up_by_citation(tmp_path):
    corpus = load_corpus(_write(tmp_path).parent)

    found = corpus.by_citation("12 CFR 1002.9")
    assert found is not None
    assert "Notification of action taken" in found.text
    assert corpus.by_citation("12 CFR 9999.9") is None


def test_corpus_totals_add_up(tmp_path):
    corpus = load_corpus(_write(tmp_path).parent)

    assert len(corpus) == len(corpus.sections)
    assert corpus.words == sum(s.words for s in corpus.sections)
    assert "sections" in corpus.summary()


SUPPLEMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ECFR>
  <DIV5 TYPE="PART" N="1002">
    <HEAD>PART 1002</HEAD>
    <DIV9 TYPE="APPENDIX" N="Supplement I to Part 1002">
      <HEAD>Supplement I to Part 1002—Official Interpretations</HEAD>
      <HD1>Introduction</HD1>
      <P>1. Official status of the commentary.</P>
      <HD2>Section 1002.2—Definitions</HD2>
      <P>2(c) Adverse action.</P>
      <HD3>2(p) Empirically Derived and Other Credit Scoring Systems</HD3>
      <P>1. Purpose of definition. The definition sets the criteria.</P>
      <P>2. Periodic revalidation. The regulation does not specify how often.</P>
      <HD3>Paragraph 4(b)</HD3>
      <P>1. Discouraging applications is prohibited.</P>
    </DIV9>
  </DIV5>
</ECFR>
"""


def test_official_interpretations_are_split_on_their_own_headings(tmp_path):
    # Left whole, Supplement I to Regulation B is one 49,000-word unit and the
    # interpretation of the credit-scoring definition cannot be addressed at
    # all. The headings inside it are already the keys a lawyer would cite.
    sections = load_document(_write(tmp_path, SUPPLEMENT_XML, "supp.xml"))
    numbers = [s.number for s in sections]

    assert "1002 Supp. I 2(p)" in numbers
    assert "1002 Supp. I 4(b)" in numbers

    scoring = next(s for s in sections if s.number == "1002 Supp. I 2(p)")
    assert "Periodic revalidation" in scoring.text
    assert "Discouraging applications" not in scoring.text
    assert scoring.citation == "12 CFR 1002 Supp. I 2(p)"


def test_a_split_supplement_keeps_the_section_it_sits_under(tmp_path):
    sections = load_document(_write(tmp_path, SUPPLEMENT_XML, "supp.xml"))
    scoring = next(s for s in sections if s.number == "1002 Supp. I 2(p)")

    assert "Section 1002.2" in scoring.heading
    assert "Empirically Derived" in scoring.heading


def test_splitting_a_supplement_loses_no_text(tmp_path):
    # The check that matters: the split reorganises, it must not drop or
    # duplicate a single paragraph of regulation.
    sections = load_document(_write(tmp_path, SUPPLEMENT_XML, "supp.xml"))
    joined = " ".join(s.text for s in sections)

    for sentence in (
        "Official status of the commentary",
        "Adverse action",
        "Purpose of definition",
        "Periodic revalidation",
        "Discouraging applications",
    ):
        assert joined.count(sentence) == 1


def test_an_appendix_without_inner_headings_stays_whole(tmp_path):
    sections = load_document(_write(tmp_path))
    appendix = [s for s in sections if "Appendix B" in s.heading]

    assert len(appendix) == 1


def test_a_missing_document_points_at_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_policies.py"):
        load_document(tmp_path / "absent.xml")


def test_an_empty_corpus_directory_points_at_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_policies.py"):
        load_corpus(tmp_path)


def test_the_committed_corpus_loads_and_carries_what_layer_one_needs():
    # The real files, no network. This is the property D6 buys: a reviewer can
    # clone the repository and reproduce the retrieval numbers immediately.
    corpus = load_corpus("data/policies")

    assert len(corpus) > 100
    assert corpus.words > 500_000
    assert {"1002", "1022", "1026"} <= {s.part for s in corpus.sections}

    # The three units the whole project points at.
    notifications = corpus.by_citation("12 CFR 1002.9")
    definitions = corpus.by_citation("12 CFR 1002.2")
    commentary = corpus.by_citation("12 CFR 1002 Supp. I 2(p)")

    assert notifications is not None and "Notifications" in notifications.heading
    assert definitions is not None
    assert "empirically derived" in definitions.text.lower()
    # The official interpretation of the provision that defines what Layer 1
    # built. Before the supplement was split this was buried in a single
    # 49,000-word unit and could not be retrieved on its own.
    assert commentary is not None
    assert commentary.words < 2000
    assert "revalidation" in commentary.text.lower()
