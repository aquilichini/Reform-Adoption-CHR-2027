from pathlib import Path

import config
from core.metadata import extract_metadata
from orthography_change.analyzer import OrthographyAnalyzer
from orthography_change.models.rule import OrthographyRule
from preprocessing import preprocess


# ============================================================================
# Helpers
# ============================================================================


CONLLU_TEXT = """\
# sent_id = 1
# text = Skulde hun skulle.
1\tSkulde\tskulle\tVERB\t_\t_\t0\troot\t_\t_
2\thun\thun\tPRON\t_\t_\t1\tnsubj\t_\t_
3\tSkulle\tskulle\tVERB\t_\t_\t1\tconj\t_\t_
4\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_

# sent_id = 2
# text = Skulde hus.
1\tSkulde\tskulle\tVERB\t_\t_\t0\troot\t_\t_
2\thus\thus\tNOUN\t_\t_\t1\tobj\t_\t_

# sent_id = 3
# text = skrevet skrevet
1\tskrevet\tskrive\tVERB\t_\tVerbForm=Part\t0\troot\t_\t_
2\tskrevet\tskrive\tVERB\t_\tVerbForm=Inf\t1\tconj\t_\t_
"""


def write_conllu(path: Path, name: str) -> Path:
    """Write a small CONLLU fixture file.

    Args:
        path: Directory to write the file into.
        name: Filename (without directory) for the fixture.

    Returns:
        Path to the written file.
    """

    file_path = path / name
    file_path.write_text(CONLLU_TEXT, encoding="utf-8")

    return file_path


def make_indexes() -> tuple[
    dict[str, OrthographyRule],
    dict[str, OrthographyRule],
]:
    """Build a small old/new rule index for testing.

    Includes a plain rule ("skulde" -> "skulle", no morphology
    filter) and a morphology-filtered rule ("skrevet" -> "skrive",
    feats_filter="VerbForm=Part") that must only count the
    participle reading of "skrevet", not the infinitive reading
    (same surface form, different FEATS).

    Returns:
        Tuple of (old_index, new_index) dicts keyed by spelling,
            each mapping to the corresponding OrthographyRule.
    """

    rule = OrthographyRule(
        rule_id="R1",
        reform_id="X1907",
        reform_year=1907,
        lemma="skulle",
        old_form="skulde",
        new_form="skulle",
        language="nb",
    )

    participle_rule = OrthographyRule(
        rule_id="R2",
        reform_id="nn_1901",
        reform_year=1901,
        lemma="skrive",
        old_form="skrevet",
        new_form="skrive",
        language="nn",
        feats_filter="VerbForm=Part",
    )

    old_index = {"skulde": rule, "skrevet": participle_rule}
    new_index = {"skulle": rule, "skrive": participle_rule}

    return old_index, new_index


# ============================================================================
# analyze
# ============================================================================


def test_analyze_matches_old_and_new_forms(tmp_path):

    file_path = write_conllu(
        tmp_path, "vestby_avis_1980_19801127_vestby_avis.conllu"
    )

    old_index, new_index = make_indexes()

    analyzer = OrthographyAnalyzer(old_index, new_index)

    result = analyzer.analyze(file_path)

    # "skulde" appears twice, "skulle" once; "hun", "hus", "."
    # do not match any rule. "skrevet" appears twice with the same
    # surface form but different FEATS (participle vs infinitive);
    # only the participle reading satisfies feats_filter and counts
    # toward the rule.
    assert result["counts"] == {
        "skulde": 2, "skulle": 1, "skrevet": 1,
    }

    assert result["journal"] == "vestby_avis"
    assert result["year"] == 1980
    assert result["file"] == file_path


# ============================================================================
# analyze_from_cache
# ============================================================================


def test_analyze_from_cache_matches_analyze(tmp_path):

    file_path = write_conllu(
        tmp_path, "vestby_avis_1980_19801127_vestby_avis.conllu"
    )

    old_index, new_index = make_indexes()

    analyzer = OrthographyAnalyzer(old_index, new_index)

    direct_result = analyzer.analyze(file_path)

    counts, total_words = preprocess(file_path)
    metadata = extract_metadata(file_path)

    nested_counts: dict[str, dict[str, dict[str, int]]] = {}
    for (pos, word, feats), count in counts.items():
        nested_counts.setdefault(pos, {}).setdefault(
            word, {}
        )[feats] = count

    record = {
        **metadata,
        "file": str(file_path),
        "counts": nested_counts,
        "total_words": total_words,
    }

    cached_result = analyzer.analyze_from_cache(record)

    assert cached_result["counts"] == direct_result["counts"]
    assert cached_result["journal"] == direct_result["journal"]
    assert cached_result["year"] == direct_result["year"]
    assert cached_result["date"] == direct_result["date"]
    assert cached_result["file"] == direct_result["file"]


def test_analyze_from_cache_applies_feats_filter():

    # Built directly, independent of analyze()/preprocess(), to
    # pin down the feats_filter behaviour on its own: the participle
    # reading of "skrevet" (VerbForm=Part) must count toward the
    # rule, while the infinitive reading (same surface form, VerbForm
    # =Inf) must not.
    old_index, new_index = make_indexes()

    analyzer = OrthographyAnalyzer(old_index, new_index)

    record = {
        "journal": "vestby_avis",
        "year": 1980,
        "date": "19801127",
        "file": "/corpus/vestby_avis/1980/19801127_vestby_avis.conllu",
        "counts": {
            "VERB": {
                "skrevet": {
                    "VerbForm=Part": 3,
                    "VerbForm=Inf": 5,
                },
            },
        },
        "total_words": 8,
    }

    cached_result = analyzer.analyze_from_cache(record)

    assert cached_result["counts"] == {"skrevet": 3}


# ============================================================================
# config smoke check
# ============================================================================


def test_config_corpus_dir_uses_full_corpus():

    assert config.CORPUS_DIR == config.FULL_CORPUS
