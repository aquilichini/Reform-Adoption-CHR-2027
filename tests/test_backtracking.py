import pytest

from orthography_change.backtracking import (
    backtracking_pairs,
    find_backtracking_groups,
    group_rules_by_form_pair,
)
from orthography_change.models.rule import OrthographyRule


def _rule(
    rule_id, reform_id, reform_year, old_form, new_form, language="nb",
):
    return OrthographyRule(
        rule_id=rule_id,
        reform_id=reform_id,
        reform_year=reform_year,
        lemma=new_form,
        old_form=old_form,
        new_form=new_form,
        language=language,
    )


# ============================================================================
# find_backtracking_groups
# ============================================================================


def test_two_event_reversal():
    rules = [
        _rule("r1", "nb_1959", 1959, "mel", "mjøl"),
        _rule("r2", "nb_1981", 1981, "mjøl", "mel"),
    ]
    groups = find_backtracking_groups(rules)

    assert len(groups) == 1
    group = groups[0]
    assert group.old_form == "mel"
    assert group.new_form == "mjøl"
    assert [e.direction for e in group.events] == ["up", "down"]


def test_same_direction_reattestation_is_not_a_group():
    rules = [
        _rule("r1", "nb_1948", 1948, "aa", "å"),
        _rule("r2", "nb_1959", 1959, "aa", "å"),
    ]
    groups = find_backtracking_groups(rules)

    assert groups == []


def test_isolated_single_rule_is_not_a_group():
    rules = [_rule("r1", "nb_1959", 1959, "sen", "sein")]
    groups = find_backtracking_groups(rules)

    assert groups == []


def test_three_event_alternating_chain():
    rules = [
        _rule("r1", "nb_1959", 1959, "a", "b"),
        _rule("r2", "nb_1970", 1970, "b", "a"),
        _rule("r3", "nb_1981", 1981, "a", "b"),
    ]
    groups = find_backtracking_groups(rules)

    assert len(groups) == 1
    group = groups[0]
    assert group.old_form == "a"
    assert group.new_form == "b"
    assert [e.direction for e in group.events] == [
        "up", "down", "up",
    ]


def test_group_language_is_populated():
    rules = [
        _rule("r1", "nn_1959", 1959, "mel", "mjøl", language="nn"),
        _rule("r2", "nn_1981", 1981, "mjøl", "mel", language="nn"),
    ]
    groups = find_backtracking_groups(rules)

    assert len(groups) == 1
    assert groups[0].language == "nn"


def test_conflicting_language_raises():
    rules = [
        _rule("r1", "nb_1959", 1959, "mel", "mjøl", language="nb"),
        _rule("r2", "nn_1981", 1981, "mjøl", "mel", language="nn"),
    ]

    with pytest.raises(ValueError):
        find_backtracking_groups(rules)


def test_two_independent_reversal_pairs_stay_distinct():
    rules = [
        _rule("r1", "nb_1959", 1959, "mel", "mjøl"),
        _rule("r2", "nb_1981", 1981, "mjøl", "mel"),
        _rule("r3", "nb_1959", 1959, "sen", "sein"),
        _rule("r4", "nb_1981", 1981, "sein", "sen"),
    ]
    groups = find_backtracking_groups(rules)

    assert len(groups) == 2
    labels = {g.label for g in groups}
    assert labels == {"mel _--_ mjøl", "sen _--_ sein"}


# ============================================================================
# group_rules_by_form_pair
# ============================================================================


def test_group_rules_by_form_pair():
    rules = [
        _rule("r1", "nb_1959", 1959, "mel", "mjøl"),
        _rule("r2", "nb_1981", 1981, "mjøl", "mel"),
    ]
    groups = group_rules_by_form_pair(rules)

    assert list(groups.keys()) == [frozenset({"mel", "mjøl"})]
    assert len(groups[frozenset({"mel", "mjøl"})]) == 2


# ============================================================================
# backtracking_pairs
# ============================================================================


def test_backtracking_pairs():
    rules = [
        _rule("r1", "nb_1959", 1959, "mel", "mjøl"),
        _rule("r2", "nb_1981", 1981, "mjøl", "mel"),
        _rule("r3", "nb_1959", 1959, "sen", "sein"),
        _rule("r4", "nb_1981", 1981, "sein", "sen"),
    ]
    groups = find_backtracking_groups(rules)
    pairs = backtracking_pairs(groups)

    assert pairs == {
        frozenset({"mel", "mjøl"}),
        frozenset({"sen", "sein"}),
    }
