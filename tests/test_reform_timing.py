import numpy as np
import pandas as pd

from orthography_change.adoption import AdoptionCalculator
from orthography_change.models.rule import OrthographyRule
from orthography_change.reform_timing import (
    bootstrap_crossing_times,
    compute_reform_boxplot_data,
    group_rules_by_reform,
)


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
# group_rules_by_reform
# ============================================================================


def test_group_rules_by_reform_groups_and_orders():
    rules = [
        _rule("r1", "nb_1907", 1907, "a", "b"),
        _rule("r2", "nn_1917", 1917, "c", "d"),
        _rule("r3", "nb_1907", 1907, "e", "f"),
    ]
    groups = group_rules_by_reform(rules)

    assert list(groups.keys()) == ["nb_1907", "nn_1917"]
    assert [r.rule_id for r in groups["nb_1907"]] == ["r1", "r3"]
    assert [r.rule_id for r in groups["nn_1917"]] == ["r2"]


def test_group_rules_by_reform_dedups_identical_pairs():
    rules = [
        _rule("r1", "nb_1907", 1907, "a", "b"),
        _rule("r2", "nb_1907", 1907, "a", "b"),
        _rule("r3", "nb_1907", 1907, "c", "d"),
    ]
    groups = group_rules_by_reform(rules)

    assert [r.rule_id for r in groups["nb_1907"]] == ["r1", "r3"]


# ============================================================================
# bootstrap_crossing_times
# ============================================================================


def test_bootstrap_crossing_times_within_period_range():
    curve = pd.DataFrame(
        {
            "period": [1900.0, 1901.0, 1902.0, 1903.0],
            "old_count": [10, 5, 1, 0],
            "new_count": [0, 5, 9, 10],
            "total": [10, 10, 10, 10],
        }
    )

    rng = np.random.default_rng(0)
    crossings = bootstrap_crossing_times(
        curve=curve, threshold=0.8, calculator=AdoptionCalculator(),
        n_bootstrap=50, use_ewma=False, rng=rng,
    )

    assert len(crossings) > 0
    assert all(1900.0 <= c <= 1903.0 for c in crossings)


# ============================================================================
# compute_reform_boxplot_data
# ============================================================================


def test_compute_reform_boxplot_data_structure_and_zero_total():
    rules = [
        _rule("r1", "nb_1907", 1907, "skulde", "skulle"),
        _rule("r2", "nb_1907", 1907, "vilde", "ville"),
        _rule("r3", "nn_1917", 1917, "aa", "å"),
    ]

    df = pd.DataFrame(
        {
            "period": [1907.0, 1908.0, 1909.0, 1910.0],
            "skulde": [10, 5, 1, 0],
            "skulle": [0, 5, 9, 10],
            # "vilde"/"ville" never occur -> zero-total rule.
            "aa": [10, 5, 1, 0],
            "å": [0, 5, 9, 10],
        }
    )

    rng = np.random.default_rng(0)
    result = compute_reform_boxplot_data(
        df=df, rules=rules, threshold=0.8,
        calculator=AdoptionCalculator(), n_bootstrap=30,
        use_ewma=False, rng=rng,
    )

    assert set(result.keys()) == {"nb_1907", "nn_1917"}
    assert result["nb_1907"]["reform_year"] == 1907
    assert result["nn_1917"]["reform_year"] == 1917

    # skulde/skulle has occurrences and should cross 0.8 somewhere,
    # contributing samples; vilde/ville has zero total and
    # contributes none, so nb_1907's samples come only from r1.
    assert len(result["nb_1907"]["samples"]) > 0
    assert len(result["nn_1917"]["samples"]) > 0


def test_compute_reform_boxplot_data_language_dfs():
    # Two rules from different reform_ids/languages, each with
    # occurrences only in its own language's df: language_dfs must
    # route each reform group to its own df instead of a shared one.
    rules = [
        _rule("r1", "nb_1907", 1907, "skulde", "skulle", "nb"),
        _rule("r2", "nn_1917", 1917, "aa", "å", "nn"),
    ]

    nb_df = pd.DataFrame(
        {
            "period": [1907.0, 1908.0, 1909.0, 1910.0],
            "skulde": [10, 5, 1, 0],
            "skulle": [0, 5, 9, 10],
        }
    )
    nn_df = pd.DataFrame(
        {
            "period": [1917.0, 1918.0, 1919.0, 1920.0],
            "aa": [10, 5, 1, 0],
            "å": [0, 5, 9, 10],
        }
    )

    rng = np.random.default_rng(0)
    result = compute_reform_boxplot_data(
        rules=rules, threshold=0.8, calculator=AdoptionCalculator(),
        n_bootstrap=30, use_ewma=False, rng=rng,
        language_dfs={"nb": nb_df, "nn": nn_df},
    )

    assert set(result.keys()) == {"nb_1907", "nn_1917"}
    assert len(result["nb_1907"]["samples"]) > 0
    assert len(result["nn_1917"]["samples"]) > 0
