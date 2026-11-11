import numpy as np
import pandas as pd
from orthography_change.adoption import AdoptionCalculator, DEFAULT_EWMA_SPAN
from orthography_change.models.rule import OrthographyRule


def group_rules_by_reform(
    rules: list[OrthographyRule],
) -> dict[str, list[OrthographyRule]]:
    """Group rules by reform_id, deduplicating repeated form pairs.

    Args:
        rules: Rules to group.

    Returns:
        dict[str, list[OrthographyRule]]: Rules keyed by reform_id,
            in order of first appearance, each list deduplicated on
            (old_form, new_form), keeping the first occurrence.
    """

    groups: dict[str, list[OrthographyRule]] = {}
    seen_pairs: dict[str, set[tuple[str, str]]] = {}

    for rule in rules:
        pair = (rule.old_form, rule.new_form)
        pairs = seen_pairs.setdefault(rule.reform_id, set())
        if pair in pairs:
            continue
        pairs.add(pair)
        groups.setdefault(rule.reform_id, []).append(rule)

    return groups


def bootstrap_crossing_times(
    curve: pd.DataFrame,
    threshold: float,
    calculator: AdoptionCalculator,
    n_bootstrap: int,
    use_ewma: bool,
    rng: np.random.Generator,
    ewma_span: float = DEFAULT_EWMA_SPAN,
) -> list[float]:
    """Bootstrap-resample one rule's threshold-crossing period.

    Args:
        curve: One rule's raw curve, i.e. compute_curve()'s output, with "period", "old_count", "new_count", "total" columns (all raw regardless of use_ewma).
        threshold: Adoption-rate threshold in (0, 1).
        calculator: AdoptionCalculator used to rebuild a curve from each resampled count array.
        n_bootstrap: Number of bootstrap replicates to draw.
        use_ewma: Whether replicate curves are smoothed before locating the crossing.
        ewma_span: ewma window size
        rng: Random generator used for the binomial resampling.

    Returns:
        list[float]: Crossing periods found across replicates.
    """

    period = curve["period"].to_numpy()
    total = curve["total"].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        p_hat = np.where(total > 0, curve["new_count"] / total, 0.0)
    
    new_b = rng.binomial(
        total.astype(int), p_hat, size=(n_bootstrap, total.size)
    )
    old_b = total - new_b
    total_b = old_b + new_b

    if use_ewma:
        new_b = calculator._ewma_accumulate(new_b, ewma_span)
        old_b = calculator._ewma_accumulate(old_b, ewma_span)
        total_b = old_b + new_b

    with np.errstate(divide="ignore", invalid="ignore"):
        adoption = np.where(total_b > 0, new_b / total_b, np.nan)

    crossed = adoption >= threshold
    has_crossing = crossed.any(axis=1)
    first_crossing_idx = crossed.argmax(axis=1)[has_crossing]

    return period[first_crossing_idx].tolist()


def compute_reform_boxplot_data(
    rules: list[OrthographyRule],
    threshold: float,
    calculator: AdoptionCalculator,
    n_bootstrap: int,
    use_ewma: bool,
    rng: np.random.Generator,
    df: pd.DataFrame | None = None,
    language_dfs: dict[str, pd.DataFrame] | None = None,
    ewma_span: float = DEFAULT_EWMA_SPAN,
) -> dict[str, dict]:
    """Compute time-to-adoption samples per reform, for one boxplot.

    Exactly one of `df`/`language_dfs` should be given. With `df`,
    every rule is evaluated against that one shared curve source
    (the pooled/combined-corpus mode). With `language_dfs`, each
    reform group's df is instead resolved via
    `language_dfs[reform_rules[0].language]` -- safe because
    reform_id values are language-homogeneous (see
    backtracking.BacktrackingGroup), so one reform's rules always
    share one language.

    Args:
        rules: Rules to include, grouped internally by reform_id via group_rules_by_reform.
        threshold: Adoption-rate threshold in (0, 1).
        calculator: AdoptionCalculator used to compute curves.
        n_bootstrap: Number of bootstrap replicates drawn per rule.
        use_ewma: Whether curves are smoothed before locating each crossing.
        rng: Random generator used for bootstrap resampling.
        df: Output of Statistics.aggregate_by_period(), with a "period" column (float) and one column per orthographic form (occurrence counts). Used for every rule when given. Defaults to None.
        language_dfs: {language: df} mapping, one aggregated DataFrame per OrthographyRule.language value. Used instead of `df` when given. Defaults to None. 
        ewma_span: ewma window size

    Returns:
        dict[str, dict]: {reform_id: {"reform_year": int, "samples": list[float]}} for every reform_id present in `rules`. Each sample is a crossing period minus that reform's reform_year; samples may be an empty list.
    """

    groups = group_rules_by_reform(rules)
    result: dict[str, dict] = {}

    for reform_id, reform_rules in groups.items():
        reform_year = reform_rules[0].reform_year
        samples: list[float] = []
        reform_df = (
            language_dfs[reform_rules[0].language]
            if language_dfs is not None else df
        )

        for rule in reform_rules:
            curve = calculator.compute_curve(
                df=reform_df, old_form=rule.old_form,
                new_form=rule.new_form,
                use_ewma=use_ewma, ewma_span=ewma_span,
            )
            if curve["total"].sum() == 0:
                continue
            crossings = bootstrap_crossing_times(
                curve=curve, threshold=threshold, calculator=calculator,
                n_bootstrap=n_bootstrap, use_ewma=use_ewma,
                ewma_span=ewma_span, rng=rng,
            )
            samples.extend(
                crossing - reform_year for crossing in crossings
            )

        result[reform_id] = {
            "reform_year": reform_year,
            "samples": samples,
        }

    return result
