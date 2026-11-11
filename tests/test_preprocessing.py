import json
from pathlib import Path

import pytest

from preprocessing import (
    _feats_to_str,
    load_preprocessed_corpus,
    preprocess,
    preprocess_corpus,
)


# ============================================================================
# Helpers
# ============================================================================


CONLLU_TEXT = """\
# sent_id = 1
# text = Katten sover.
1\tKatten\tkatt\tNOUN\t_\tNumber=Sing|Definite=Def\t0\troot\t_\t_
2\tsover\tsove\tVERB\t_\tVerbForm=Fin|Tense=Pres\t1\tconj\t_\t_
3\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_

# sent_id = 2
# text = Katten og hunden.
1-2\tKatten\t_\t_\t_\t_\t_\t_\t_\t_
1\tKatten\tkatt\tNOUN\t_\tNumber=Sing|Definite=Def\t0\troot\t_\t_
2\tog\tog\tCCONJ\t_\t_\t0\tcc\t_\t_
3\thunden\thund\tNOUN\t_\tNumber=Sing|Definite=Def\t0\tconj\t_\t_

# sent_id = 3
# text = skrevet skrevet
1\tskrevet\tskrive\tVERB\t_\tVerbForm=Part\t0\troot\t_\t_
2\tskrevet\tskrive\tVERB\t_\tVerbForm=Inf\t1\tconj\t_\t_
"""

EMPTY_CONLLU_TEXT = ""

RULES_CSV_HEADER = (
    "rule_id,reform_id,reform_year,lemma,old_form,new_form,language"
)

# Rules referencing "katten" and "skrevet" (both present in
# CONLLU_TEXT), so tests can check that other forms ("hunden",
# "sover", ".", "og") are excluded when restrict_to_rules is True.
RULES_CSV_ROWS = (
    "r1,reform1,1900,katt,katten,katta,nb",
    "r2,reform1,1900,skrive,skrevet,skriven,nn",
)


def write_conllu(path: Path, name: str, text: str = CONLLU_TEXT) -> Path:
    """Write a small CONLLU fixture file.

    Args:
        path: Directory to write the file into.
        name: Filename (without directory) for the fixture.
        text: CONLLU file contents. Defaults to CONLLU_TEXT.

    Returns:
        Path to the written file.
    """

    file_path = path / name
    file_path.write_text(text, encoding="utf-8")

    return file_path


def write_rules_csv(tmp_path: Path) -> Path:
    """Write a minimal rules CSV fixture (independent of data/rules.csv).

    Args:
        tmp_path: Directory to write the fixture into.

    Returns:
        Path to the written rules.csv file.
    """

    file_path = tmp_path / "rules.csv"
    lines = [RULES_CSV_HEADER, *RULES_CSV_ROWS]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return file_path


# ============================================================================
# _feats_to_str
# ============================================================================


def test_feats_to_str_sorts_and_joins():

    assert _feats_to_str(
        {"Number": "Sing", "Definite": "Def"}
    ) == "Definite=Def|Number=Sing"


def test_feats_to_str_empty_or_none():

    assert _feats_to_str(None) == ""
    assert _feats_to_str({}) == ""


# ============================================================================
# preprocess
# ============================================================================


def test_preprocess_counts_and_total_words(tmp_path):

    file_path = write_conllu(
        tmp_path, "vestby_avis_1980_19801127_vestby_avis.conllu"
    )

    counts, total_words = preprocess(file_path)

    noun_feats = "Definite=Def|Number=Sing"

    # 2x ("NOUN", "katten", noun_feats), 1x each for the rest; the
    # multiword token "1-2" must be skipped. "skrevet" appears twice
    # with the same surface form but different FEATS (participle vs
    # infinitive), and must be counted separately.
    assert counts[("NOUN", "katten", noun_feats)] == 2
    assert counts[("VERB", "sover", "Tense=Pres|VerbForm=Fin")] == 1
    assert counts[("PUNCT", ".", "")] == 1
    assert counts[("CCONJ", "og", "")] == 1
    assert counts[("NOUN", "hunden", noun_feats)] == 1
    assert counts[("VERB", "skrevet", "VerbForm=Part")] == 1
    assert counts[("VERB", "skrevet", "VerbForm=Inf")] == 1

    assert total_words == 8


def test_preprocess_with_allowed_forms_restricts_counts_only(tmp_path):

    file_path = write_conllu(
        tmp_path, "vestby_avis_1980_19801127_vestby_avis.conllu"
    )

    allowed_forms = {"katten", "skrevet"}

    counts, total_words = preprocess(file_path, allowed_forms)

    noun_feats = "Definite=Def|Number=Sing"

    # Only forms in allowed_forms are kept in counts, but
    # total_words still reflects every token in the file.
    assert counts == {
        ("NOUN", "katten", noun_feats): 2,
        ("VERB", "skrevet", "VerbForm=Part"): 1,
        ("VERB", "skrevet", "VerbForm=Inf"): 1,
    }
    assert total_words == 8


# ============================================================================
# preprocess_corpus
# ============================================================================


def test_preprocess_corpus_json_rules_only_defaults(tmp_path):

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "19801127_vestby_avis.conllu")

    rules_path = write_rules_csv(tmp_path)

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        rules_path=rules_path,
        n_jobs=1,
    )

    assert output_path.is_file()

    with output_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 1

    record = records[0]
    noun_feats = "Definite=Def|Number=Sing"

    # Only "katten"/"skrevet" (referenced by the fixture rules) are
    # kept; "hunden", "sover", ".", "og" are excluded. total_words
    # still reflects every token in the issue.
    assert record["counts"] == {
        "NOUN": {"katten": {noun_feats: 2}},
        "VERB": {
            "skrevet": {"VerbForm=Part": 1, "VerbForm=Inf": 1},
        },
    }
    assert record["total_words"] == 8
    assert record["journal"] == "vestby_avis"
    assert record["year"] == 1980
    assert record["date"] == "19801127"


def test_preprocess_corpus_json_full_dictionary(tmp_path):

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "19801127_vestby_avis.conllu")

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    assert len(loaded) == 1

    issue = loaded[0]
    noun_feats = "Definite=Def|Number=Sing"

    assert issue["counts"] == {
        "NOUN": {
            "katten": {noun_feats: 2},
            "hunden": {noun_feats: 1},
        },
        "VERB": {
            "sover": {"Tense=Pres|VerbForm=Fin": 1},
            "skrevet": {"VerbForm=Part": 1, "VerbForm=Inf": 1},
        },
        "PUNCT": {".": {"": 1}},
        "CCONJ": {"og": {"": 1}},
    }
    assert issue["total_words"] == 8


def test_preprocess_corpus_json_empty_file_empty_counts(tmp_path):

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(
        corpus_dir,
        "19801128_vestby_avis.conllu",
        text=EMPTY_CONLLU_TEXT,
    )

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    assert len(loaded) == 1
    assert loaded[0]["counts"] == {}
    assert loaded[0]["total_words"] == 0


# ============================================================================
# load_preprocessed_corpus: up to two issues per publication date
# ============================================================================
#
# Filtering happens in load_preprocessed_corpus (not
# preprocess_corpus), so it applies uniformly to caches written
# before this filter existed too, with no need to re-run
# preprocess_corpus over the raw CONLLU corpus.


def test_preprocess_corpus_same_date_keeps_up_to_two_records(tmp_path):

    corpus_dir = tmp_path / "corpus"

    dir_a = corpus_dir / "vestby_avis" / "1980"
    dir_b = corpus_dir / "another_avis" / "1980"
    dir_c = corpus_dir / "third_avis" / "1980"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    dir_c.mkdir(parents=True)

    write_conllu(dir_a, "19801127_vestby_avis.conllu")
    write_conllu(dir_b, "19801127_another_avis.conllu")
    write_conllu(dir_c, "19801127_third_avis.conllu")

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=corpus_dir,
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    # Scan order via Path.rglob is filesystem-dependent, so only the
    # count is asserted, not which two journals survive.
    assert len(loaded) == 2
    assert {r["date"] for r in loaded} == {"19801127"}


def test_preprocess_corpus_undated_file_dropped(tmp_path):

    corpus_dir = tmp_path / "corpus" / "someJournal" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "undated_someJournal.conllu")

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    assert len(loaded) == 0
    assert all(r["journal"] != "someJournal" for r in loaded)


def test_preprocess_corpus_different_dates_keeps_both(tmp_path):

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "19801127_vestby_avis.conllu")
    write_conllu(corpus_dir, "19801128_vestby_avis.conllu")

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    assert len(loaded) == 2
    assert {r["date"] for r in loaded} == {"19801127", "19801128"}


def test_load_preprocessed_corpus_filters_preexisting_cache(tmp_path):

    # A cache written directly (as if by an older preprocess_corpus
    # that had no per-date cap) still gets filtered on load, with no
    # need to re-run preprocessing over the raw corpus.
    output_path = tmp_path / "output" / "counts.json"
    output_path.parent.mkdir(parents=True)

    records = [
        {
            "journal": "vestby_avis",
            "year": 1980,
            "date": "19801127",
            "file": "a.conllu",
            "counts": {},
            "total_words": 1,
        },
        {
            "journal": "another_avis",
            "year": 1980,
            "date": "19801127",
            "file": "b.conllu",
            "counts": {},
            "total_words": 2,
        },
        {
            "journal": "third_avis",
            "year": 1980,
            "date": "19801127",
            "file": "e.conllu",
            "counts": {},
            "total_words": 5,
        },
        {
            "journal": "vestby_avis",
            "year": 1980,
            "date": None,
            "file": "c.conllu",
            "counts": {},
            "total_words": 3,
        },
        {
            "journal": "vestby_avis",
            "year": 1980,
            "date": "19801128",
            "file": "d.conllu",
            "counts": {},
            "total_words": 4,
        },
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f)

    loaded = load_preprocessed_corpus(output_path)

    # First two records for 19801127 are kept ("e.conllu" is the
    # third and is dropped); the None-date record is always dropped.
    assert [r["file"] for r in loaded] == ["a.conllu", "b.conllu", "d.conllu"]


def test_load_preprocessed_corpus_custom_max_per_date(tmp_path):

    output_path = tmp_path / "output" / "counts.json"
    output_path.parent.mkdir(parents=True)

    records = [
        {
            "journal": journal,
            "year": 1980,
            "date": "19801127",
            "file": f"{journal}.conllu",
            "counts": {},
            "total_words": 0,
        }
        for journal in ("a", "b", "c")
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f)

    assert len(
        load_preprocessed_corpus(output_path, max_issues_per_date=1)
    ) == 1
    assert len(
        load_preprocessed_corpus(output_path, max_issues_per_date=3)
    ) == 3


# ============================================================================
# load_preprocessed_corpus
# ============================================================================


def test_load_preprocessed_corpus_round_trip(tmp_path):

    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "19801127_vestby_avis.conllu")

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    assert len(loaded) == 1

    issue = loaded[0]

    expected_file = str(corpus_dir / "19801127_vestby_avis.conllu")

    noun_feats = "Definite=Def|Number=Sing"

    expected_counts = {
        "NOUN": {
            "katten": {noun_feats: 2},
            "hunden": {noun_feats: 1},
        },
        "VERB": {
            "sover": {"Tense=Pres|VerbForm=Fin": 1},
            "skrevet": {"VerbForm=Part": 1, "VerbForm=Inf": 1},
        },
        "PUNCT": {".": {"": 1}},
        "CCONJ": {"og": {"": 1}},
    }

    assert issue == {
        "journal": "vestby_avis",
        "year": 1980,
        "date": "19801127",
        "file": expected_file,
        "counts": expected_counts,
        "total_words": 8,
    }

    # Guard against non-int leakage (e.g. from json.load).
    assert isinstance(issue["year"], int)
    assert isinstance(issue["total_words"], int)
    for word_to_feats in issue["counts"].values():
        for feats_to_count in word_to_feats.values():
            for count in feats_to_count.values():
                assert isinstance(count, int)


def test_load_preprocessed_corpus_null_date_round_trip(tmp_path):

    # Full-corpus layout (.../<journal>/<year>/<filename>.conllu)
    # with a filename that does not start with an 8-digit date, so
    # core.metadata.extract_metadata returns date=None. Such records
    # are dropped by _select_up_to_n_issues_per_date on load.
    corpus_dir = tmp_path / "corpus" / "vestby_avis" / "1980"
    corpus_dir.mkdir(parents=True)

    write_conllu(corpus_dir, "undated_vestby_avis.conllu")

    output_path = tmp_path / "output" / "counts.json"

    preprocess_corpus(
        corpus_path=tmp_path / "corpus",
        output_path=output_path,
        restrict_to_rules=False,
        n_jobs=1,
    )

    loaded = load_preprocessed_corpus(output_path)

    assert len(loaded) == 0


def test_load_preprocessed_corpus_missing_file(tmp_path):

    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):

        load_preprocessed_corpus(missing_path)


def test_load_preprocessed_corpus_rejects_legacy_shape(tmp_path):

    output_path = tmp_path / "output" / "counts.json"
    output_path.parent.mkdir(parents=True)

    legacy_records = [
        {
            "journal": "vestby_avis",
            "year": 1980,
            "date": "19801127",
            "file": "19801127_vestby_avis.conllu",
            "counts": {"VERB": {"skulde": 2}},
            "total_words": 3,
        }
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(legacy_records, f)

    with pytest.raises(ValueError) as excinfo:

        load_preprocessed_corpus(output_path)

    message = str(excinfo.value)
    assert str(output_path) in message
    assert "re-run" in message or "rerun" in message
