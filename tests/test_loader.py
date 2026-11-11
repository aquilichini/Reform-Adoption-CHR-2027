from pathlib import Path

import pandas as pd
import pytest

from orthography_change.rules.loader import (
    get_rule_by_new_form,
    get_rule_by_old_form,
    get_rules_for_lemma,
    load_rules,
    load_rules_by_lemma,
    load_rules_by_new_form,
    load_rules_by_old_form,
    validate_rules_dataframe,
)


# ============================================================================
# Helpers
# ============================================================================


def create_rules_csv(path: Path) -> Path:
    """
    Create a temporary rules CSV used for testing.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Path
    """

    dataframe = pd.DataFrame(
        {
            "rule_id": [
                "DK1948_0001",
                "DK1948_0002",
            ],
            "reform_id": [
                "DK1948",
                "DK1948",
            ],
            "reform_year": [
                1948,
                1948,
            ],
            "lemma": [
                "å",
                "skulle",
            ],
            "old_form": [
                "aa",
                "skulde",
            ],
            "new_form": [
                "å",
                "skulle",
            ],
            "language": [
                "nb",
                "nb",
            ],
            "comment": [
                "",
                "",
            ],
        }
    )

    csv_file = path / "rules.csv"

    dataframe.to_csv(
        csv_file,
        index=False,
    )

    return csv_file


def create_rules_csv_with_extra_rows(
    path: Path,
    extra_rows: list[dict],
) -> Path:
    """
    Create a temporary rules CSV with extra rows appended.

    Parameters
    ----------
    path : Path
    extra_rows : list[dict]
        Extra rows to append. Each dict must provide values for
        rule_id, reform_id, reform_year, lemma, old_form, new_form.

    Returns
    -------
    Path
    """

    dataframe = pd.DataFrame(
        {
            "rule_id": [
                "DK1948_0001",
                "DK1948_0002",
            ],
            "reform_id": [
                "DK1948",
                "DK1948",
            ],
            "reform_year": [
                1948,
                1948,
            ],
            "lemma": [
                "å",
                "skulle",
            ],
            "old_form": [
                "aa",
                "skulde",
            ],
            "new_form": [
                "å",
                "skulle",
            ],
            "language": [
                "nb",
                "nb",
            ],
            "comment": [
                "",
                "",
            ],
        }
    )

    extra_dataframe = pd.DataFrame(extra_rows)

    dataframe = pd.concat(
        [dataframe, extra_dataframe],
        ignore_index=True,
    )

    csv_file = path / "rules.csv"

    dataframe.to_csv(
        csv_file,
        index=False,
    )

    return csv_file


# ============================================================================
# Validation
# ============================================================================


def test_validate_dataframe_success():

    dataframe = pd.DataFrame(
        columns=[
            "rule_id",
            "reform_id",
            "reform_year",
            "lemma",
            "old_form",
            "new_form",
            "language",
        ]
    )

    validate_rules_dataframe(dataframe)


def test_validate_dataframe_missing_column():

    dataframe = pd.DataFrame(
        columns=[
            "rule_id",
            "lemma",
            "old_form",
        ]
    )

    with pytest.raises(ValueError):

        validate_rules_dataframe(dataframe)


# ============================================================================
# Loading
# ============================================================================


def test_load_rules(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    rules = load_rules(csv_file)

    assert len(rules) == 2

    assert rules[0].old_form == "aa"

    assert rules[1].new_form == "skulle"


def test_load_rules_language(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    rules = load_rules(csv_file)

    assert rules[0].language == "nb"

    assert rules[1].language == "nb"


def test_load_rules_pos_and_feats_filter_absent_columns(tmp_path):

    # pos_filter/feats_filter columns entirely absent from the CSV
    # must yield None for every rule (backward compat).
    csv_file = create_rules_csv(tmp_path)

    rules = load_rules(csv_file)

    for rule in rules:
        assert rule.pos_filter is None
        assert rule.feats_filter is None


def test_load_rules_pos_and_feats_filter_blank_cells(tmp_path):

    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "DK1959_0001",
                "reform_id": "DK1959",
                "reform_year": 1959,
                "lemma": "å",
                "old_form": "aa",
                "new_form": "å",
                "language": "nb",
                "comment": "",
                "pos_filter": "",
                "feats_filter": "",
            },
        ],
    )

    rules = load_rules(csv_file)

    for rule in rules:
        assert rule.pos_filter is None
        assert rule.feats_filter is None


def test_load_rules_pos_and_feats_filter_set(tmp_path):

    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "NN1901_0001",
                "reform_id": "nn_1901",
                "reform_year": 1901,
                "lemma": "skrive",
                "old_form": "skrevet",
                "new_form": "skrive",
                "language": "nn",
                "comment": "",
                "pos_filter": "VERB",
                "feats_filter": "VerbForm=Part",
            },
        ],
    )

    rules = load_rules(csv_file)

    filtered_rule = rules[-1]

    assert filtered_rule.old_form == "skrevet"
    assert filtered_rule.pos_filter == "VERB"
    assert filtered_rule.feats_filter == "VerbForm=Part"


# ============================================================================
# Index by old form
# ============================================================================


def test_load_rules_by_old_form(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_old_form(csv_file)

    assert "aa" in index

    assert "skulde" in index

    assert index["aa"].new_form == "å"


def test_load_rules_by_old_form_matching_duplicate(tmp_path):

    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "DK1959_0001",
                "reform_id": "DK1959",
                "reform_year": 1959,
                "lemma": "å",
                "old_form": "aa",
                "new_form": "å",
                "language": "nb",
                "comment": "",
            },
        ],
    )

    index = load_rules_by_old_form(csv_file)

    assert index["aa"].rule_id == "DK1948_0001"


def test_load_rules_by_old_form_and_new_form_normalize_keys(tmp_path):

    # old_form/new_form with mixed casing and surrounding whitespace
    # must still be found by their lowercased/stripped forms, while
    # the returned rule keeps the original CSV text untouched.
    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "NN1901_0001",
                "reform_id": "nn_1901",
                "reform_year": 1901,
                "lemma": "Visorna",
                "old_form": " Visorna ",
                "new_form": " Visorne ",
                "language": "nn",
                "comment": "",
            },
        ],
    )

    old_index = load_rules_by_old_form(csv_file)
    new_index = load_rules_by_new_form(csv_file)

    assert "visorna" in old_index
    assert "visorne" in new_index

    old_rule = old_index["visorna"]
    new_rule = new_index["visorne"]

    assert old_rule.old_form == " Visorna "
    assert new_rule.new_form == " Visorne "


def test_load_rules_by_old_form_conflicting_duplicate(tmp_path):

    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "DK1959_0001",
                "reform_id": "DK1959",
                "reform_year": 1959,
                "lemma": "å",
                "old_form": "aa",
                "new_form": "aa2",
                "language": "nb",
                "comment": "",
            },
        ],
    )

    with pytest.raises(ValueError):

        load_rules_by_old_form(csv_file)


# ============================================================================
# Index by new form
# ============================================================================


def test_load_rules_by_new_form(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_new_form(csv_file)

    assert "å" in index

    assert "skulle" in index

    assert index["å"].old_form == "aa"


def test_load_rules_by_new_form_matching_duplicate(tmp_path):

    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "DK1959_0001",
                "reform_id": "DK1959",
                "reform_year": 1959,
                "lemma": "å",
                "old_form": "aa",
                "new_form": "å",
                "language": "nb",
                "comment": "",
            },
        ],
    )

    index = load_rules_by_new_form(csv_file)

    assert index["å"].rule_id == "DK1948_0001"


def test_load_rules_by_new_form_many_to_one_no_conflict(tmp_path):

    # Several distinct old_form spellings legitimately normalize to
    # the same new_form within one reform (e.g. inflected variants
    # of a lemma). This must not raise, unlike the old_form index.

    csv_file = create_rules_csv_with_extra_rows(
        tmp_path,
        extra_rows=[
            {
                "rule_id": "DK1959_0001",
                "reform_id": "DK1959",
                "reform_year": 1959,
                "lemma": "aa2",
                "old_form": "aa2",
                "new_form": "å",
                "language": "nb",
                "comment": "",
            },
        ],
    )

    index = load_rules_by_new_form(csv_file)

    assert index["å"].rule_id == "DK1948_0001"

    assert index["å"].old_form == "aa"


# ============================================================================
# Index by lemma
# ============================================================================


def test_load_rules_by_lemma(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_lemma(csv_file)

    assert "å" in index

    assert "skulle" in index

    assert len(index["å"]) == 1

    assert len(index["skulle"]) == 1


# ============================================================================
# Convenience functions
# ============================================================================


def test_get_rule_by_old_form(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_old_form(csv_file)

    rule = get_rule_by_old_form("aa", index)

    assert rule is not None

    assert rule.new_form == "å"


def test_get_rule_by_new_form(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_new_form(csv_file)

    rule = get_rule_by_new_form("å", index)

    assert rule is not None

    assert rule.old_form == "aa"


def test_get_rules_for_lemma(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_lemma(csv_file)

    rules = get_rules_for_lemma("å", index)

    assert len(rules) == 1

    assert rules[0].old_form == "aa"


def test_get_rules_for_unknown_lemma(tmp_path):

    csv_file = create_rules_csv(tmp_path)

    index = load_rules_by_lemma(csv_file)

    rules = get_rules_for_lemma("unknown", index)

    assert rules == []
