from collections import Counter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from config import (
    ADOPTION_THRESHOLDS,
    AGGREGATION_MONTHS,
    APPROX_LINE_WINDOW,
    BOOTSTRAP_SEED,
    CLASSIFIED_CACHE,
    CORPUS_DIR,
    EWMA_SPAN,
    LANGUAGE_LABEL_GROUPS,
    NORMATIVE_EFFECT_GROUPS,
    N_BOOTSTRAP,
    PLOT_N_JOBS,
    PREPROCESSED_CACHE_JSON,
    RULES_FILE,
    RULE_LANGUAGE_TO_GROUP,
    SHOW_APPROX_LINE,
    USE_EWMA,
)
from orthography_change.analyzer import OrthographyAnalyzer
from orthography_change.backtracking import (
    BacktrackingGroup,
    backtracking_pairs,
    find_backtracking_groups,
)
from orthography_change.models.rule import OrthographyRule
from orthography_change.statistics import (
    GRANULARITY_LABELS,
    Statistics,
)
from orthography_change.adoption import AdoptionCalculator
from orthography_change.plotter import Plotter
from orthography_change.reform_timing import compute_reform_boxplot_data
from orthography_change.rules.loader import (
    load_rules,
    load_rules_by_old_form,
    load_rules_by_new_form,
)
from preprocessing import preprocess_corpus, load_preprocessed_corpus
from language_classification import (
    attach_labels,
    classify_corpus,
    load_classified_corpus,
)

_worker_plotters: dict[str, Plotter] = {}


def _get_worker_plotter(output_directory: str) -> Plotter:
    """Return this worker process's cached Plotter for a directory.

    Args:
        output_directory: Output directory path; also the cache key.

    Returns:
        Plotter: Cached instance, constructing one on first use.
    """

    return _worker_plotters.setdefault(
        output_directory, Plotter(output_directory=output_directory),
    )


def _plot_backtracking_task(
    df, old_form, new_form, label, events, output_directory,
    x_label, use_ewma, ewma_span, group_name,
    show_approx_line, approx_window,
):
    """Compute and plot one backtracking curve in a worker process.

    Args:
        df: Aggregated period-count DataFrame, as returned by
            Statistics.aggregate_by_period.
        old_form: Anchor spelling A.
        new_form: Anchor spelling B.
        label: Plot title/filename stem, e.g. "mel ↔ mjøl".
        events: Ordered (reform_year, direction) pairs.
        output_directory: Directory the PNG is written to.
        x_label: x-axis label for the aggregation granularity.
        use_ewma: Whether to compute the curve from EWMA-smoothed
            counts.
        ewma_span: EWMA window span, in periods.
        group_name: Language group name, used only to format the
            "Skipping ..." message.
        show_approx_line: Whether to overlay a red moving-average
            approximation line.
        approx_window: Rolling-average window, in periods, for the
            approximation line.

    Returns:
        tuple[pandas.DataFrame | None, str | None]: (curve, None) if
            the curve has occurrences and was plotted, otherwise
            (None, message) with the "no occurrences" skip message.
    """

    calculator = AdoptionCalculator()
    curve = calculator.compute_curve(
        df=df, old_form=old_form, new_form=new_form,
        use_ewma=use_ewma, ewma_span=ewma_span,
    )
    if curve["adoption"].notna().sum() == 0:
        return None, f"[{group_name}] Skipping {label} (no occurrences)"
    plotter = _get_worker_plotter(output_directory)
    plotter.plot_backtracking_curve(
        curve=curve, label=label, events=events, x_label=x_label,
        show_approx_line=show_approx_line, approx_window=approx_window,
    )
    return curve, None


def _plot_adoption_rule_task(
    df, old_form, new_form, rule_label, reform_year, output_directory,
    x_label, use_ewma, ewma_span, group_name,
    show_approx_line, approx_window,
):
    """Compute and plot one adoption curve in a worker process.

    Args:
        df: Aggregated period-count DataFrame, as returned by
            Statistics.aggregate_by_period.
        old_form: Historical spelling.
        new_form: Modern spelling.
        rule_label: Rule label, used in the title and "Skipping ..."
            message.
        reform_year: Official reform year, for the vertical marker.
        output_directory: Directory the PNG is written to.
        x_label: x-axis label for the aggregation granularity.
        use_ewma: Whether to compute the curve from EWMA-smoothed
            counts.
        ewma_span: EWMA window span, in periods.
        group_name: Language group name, used only to format the
            "Skipping ..." message.
        show_approx_line: Whether to overlay a red moving-average
            approximation line.
        approx_window: Rolling-average window, in periods, for the
            approximation line.

    Returns:
        tuple[pandas.DataFrame | None, str | None]: (curve, None) if
            the curve has occurrences and was plotted, otherwise
            (None, message) with the "no occurrences" skip message.
    """

    calculator = AdoptionCalculator()
    curve = calculator.compute_curve(
        df=df, old_form=old_form, new_form=new_form,
        use_ewma=use_ewma, ewma_span=ewma_span,
    )
    if curve["adoption"].notna().sum() == 0:
        return None, (
            f"[{group_name}] Skipping {rule_label} (no occurrences)"
        )
    plotter = _get_worker_plotter(output_directory)
    plotter.plot_adoption_curve(
        curve=curve, rule_name=rule_label, reform_year=reform_year,
        x_label=x_label, ewma=use_ewma,
        show_approx_line=show_approx_line, approx_window=approx_window,
    )
    return curve, None


def _deduplicated_plot_rules(
    rules: list[OrthographyRule],
    bt_pairs: set[frozenset[str]],
    group_name: str,
) -> list[OrthographyRule]:
    """Filter rules down to one plot per (old_form, new_form) pair.

    Skips rules already covered by a combined backtracking plot
    (pair in bt_pairs) and rules re-attesting an earlier reform's
    exact (old_form, new_form) pair, printing a "[group_name]
    Skipping ..." message for each.

    Args:
        rules: Rules to filter.
        bt_pairs: {old_form, new_form} sets already covered by a
            backtracking plot (see backtracking.backtracking_pairs).
        group_name: Message-prefix label, e.g. "bokmaal" or
            "same_language".

    Returns:
        list[OrthographyRule]: One rule per distinct
            (old_form, new_form) pair, in order of first appearance.
    """

    seen_pairs = set()
    plot_rules = []

    for rule in rules:
        pair = (rule.old_form, rule.new_form)
        if frozenset(pair) in bt_pairs:
            print(f"[{group_name}] Skipping {rule.label} "
                  "(part of a combined backtracking plot)")
            continue
        if pair in seen_pairs:
            print(f"[{group_name}] Skipping {rule.label} "
                  "(duplicate of an earlier reform)")
            continue
        seen_pairs.add(pair)
        plot_rules.append(rule)

    return plot_rules


def run_group(
    group_name: str,
    labels: set[str],
    results: list[dict],
    x_label: str,
    rng: np.random.Generator,
) -> None:
    """Aggregate, compute, and plot all curves for one language group.

    Args:
        group_name: Output subdirectory name under "plots/", e.g.
            "bokmaal".
        labels: language_classification label values whose issues
            are included in this group (see
            config.LANGUAGE_LABEL_GROUPS).
        results: Per-issue analysis dicts (each with "year", "date",
            "counts", "label"), unfiltered — covers all labels.
        x_label: Axis label for the aggregation granularity.
        rng: Random generator used for the time-to-adoption
            boxplots' bootstrap resampling.
    """

    group_results = [r for r in results if r["label"] in labels]

    print(f"[{group_name}] {len(group_results)} issue(s) in this "
          "group")

    if len(group_results) == 0:
        print(f"[{group_name}] No issues in this group; skipping "
              "all curves. Check the label distribution above and "
              "attach_labels' warning (if any) to see why.")
        return

    df = statistics.aggregate_by_period(
        group_results, months=AGGREGATION_MONTHS,
    )

    output_directory = f"plots/{group_name}"
    plotter = Plotter(output_directory=output_directory)

    print('> Plotting boxplot')
    for threshold in ADOPTION_THRESHOLDS:
        boxplot_data = compute_reform_boxplot_data(
            df=df, rules=rules, threshold=threshold,
            calculator=calculator, n_bootstrap=N_BOOTSTRAP,
            use_ewma=USE_EWMA, ewma_span=EWMA_SPAN, rng=rng,
        )
        plotter.plot_adoption_time_boxplot(boxplot_data, threshold)

    print('> Plotting boxplot by normative effect')
    for effect_key, effect_values in NORMATIVE_EFFECT_GROUPS.items():
        effect_rules = [
            r for r in rules
            if effect_by_rule_id.get(r.rule_id) in effect_values
        ]
        effect_plotter = Plotter(
            output_directory=(
                f"plots/{group_name}/normative_boxplots/{effect_key}"
            ),
        )
        for threshold in ADOPTION_THRESHOLDS:
            boxplot_data = compute_reform_boxplot_data(
                df=df, rules=effect_rules, threshold=threshold,
                calculator=calculator, n_bootstrap=N_BOOTSTRAP,
                use_ewma=USE_EWMA, ewma_span=EWMA_SPAN, rng=rng,
            )
            if not any(v["samples"] for v in boxplot_data.values()):
                print(f"[{group_name}] Skipping {effect_key} "
                      f"boxplot for threshold {threshold:.0%} "
                      "(no occurrences)")
                continue
            effect_plotter.plot_adoption_time_boxplot(
                boxplot_data, threshold,
            )

    curves = []

    print('> Plotting backtracking curves')
    backtracking_results = Parallel(
        n_jobs=PLOT_N_JOBS, backend="loky",
    )(
        delayed(_plot_backtracking_task)(
            df=df, old_form=group.old_form, new_form=group.new_form,
            label=group.label,
            events=[(e.reform_year, e.direction) for e in group.events],
            output_directory=output_directory, x_label=x_label,
            use_ewma=USE_EWMA, ewma_span=EWMA_SPAN,
            group_name=group_name,
            show_approx_line=SHOW_APPROX_LINE,
            approx_window=APPROX_LINE_WINDOW,
        )
        for group in backtracking_groups
    )

    for curve, message in backtracking_results:
        if message is not None:
            print(message)
            continue
        curves.append(curve)

    print('> Plotting adoption curves')
    plot_rules = _deduplicated_plot_rules(rules, bt_pairs, group_name)

    adoption_results = Parallel(
        n_jobs=PLOT_N_JOBS, backend="loky",
    )(
        delayed(_plot_adoption_rule_task)(
            df=df, old_form=rule.old_form, new_form=rule.new_form,
            rule_label=rule.label, reform_year=rule.reform_year,
            output_directory=output_directory, x_label=x_label,
            use_ewma=USE_EWMA, ewma_span=EWMA_SPAN,
            group_name=group_name,
            show_approx_line=SHOW_APPROX_LINE,
            approx_window=APPROX_LINE_WINDOW,
        )
        for rule in plot_rules
    )

    for curve, message in adoption_results:
        if message is not None:
            print(message)
            continue
        curves.append(curve)

    if not curves:
        print(f"[{group_name}] Skipping meta curve (no curves)")
        return

    print('> Plotting meta-curve')
    meta_curve = calculator.compute_meta_curve(
        curves, use_ewma=USE_EWMA, ewma_span=EWMA_SPAN,
    )
    plotter.plot_meta_curve(
        meta_curve, x_label=x_label, ewma=USE_EWMA,
        show_approx_line=SHOW_APPROX_LINE,
        approx_window=APPROX_LINE_WINDOW,
    )


def run_same_language(
    rules: list[OrthographyRule],
    backtracking_groups: list[BacktrackingGroup],
    plot_rules: list[OrthographyRule],
    language_dfs: dict[str, pd.DataFrame],
    x_label: str,
    rng: np.random.Generator,
) -> None:
    """Compute and plot every reform's own-language adoption curves.

    Args:
        rules: Every loaded rule (unfiltered).
        backtracking_groups: Backtracking groups from
            find_backtracking_groups(rules).
        plot_rules: Deduplicated rules to plot individually, e.g.
            from _deduplicated_plot_rules(rules, bt_pairs,
            "same_language").
        language_dfs: {language: df} mapping, one aggregated
            DataFrame (Statistics.aggregate_by_period output) per
            OrthographyRule.language value.
        x_label: Axis label for the aggregation granularity.
        rng: Random generator used for the time-to-adoption
            boxplots' bootstrap resampling.
    """

    group_name = "same_language"
    output_directory = "plots/same_language"
    plotter = Plotter(output_directory=output_directory)

    print('> Plotting boxplot')
    for threshold in ADOPTION_THRESHOLDS:
        boxplot_data = compute_reform_boxplot_data(
            rules=rules, threshold=threshold, calculator=calculator,
            n_bootstrap=N_BOOTSTRAP, use_ewma=USE_EWMA,
            ewma_span=EWMA_SPAN, rng=rng, language_dfs=language_dfs,
        )
        plotter.plot_adoption_time_boxplot(boxplot_data, threshold)

    print('> Plotting boxplot by normative effect')
    for effect_key, effect_values in NORMATIVE_EFFECT_GROUPS.items():
        effect_rules = [
            r for r in rules
            if effect_by_rule_id.get(r.rule_id) in effect_values
        ]
        effect_plotter = Plotter(
            output_directory=(
                f"{output_directory}/normative_boxplots/{effect_key}"
            ),
        )
        for threshold in ADOPTION_THRESHOLDS:
            boxplot_data = compute_reform_boxplot_data(
                rules=effect_rules, threshold=threshold,
                calculator=calculator, n_bootstrap=N_BOOTSTRAP,
                use_ewma=USE_EWMA, ewma_span=EWMA_SPAN, rng=rng,
                language_dfs=language_dfs,
            )
            if not any(v["samples"] for v in boxplot_data.values()):
                print(f"[{group_name}] Skipping {effect_key} "
                      f"boxplot for threshold {threshold:.0%} "
                      "(no occurrences)")
                continue
            effect_plotter.plot_adoption_time_boxplot(
                boxplot_data, threshold,
            )

    curves = []

    print('> Plotting backtracking curves')
    backtracking_results = Parallel(
        n_jobs=PLOT_N_JOBS, backend="loky",
    )(
        delayed(_plot_backtracking_task)(
            df=language_dfs[group.language], old_form=group.old_form,
            new_form=group.new_form, label=group.label,
            events=[(e.reform_year, e.direction) for e in group.events],
            output_directory=output_directory, x_label=x_label,
            use_ewma=USE_EWMA, ewma_span=EWMA_SPAN,
            group_name=group_name,
            show_approx_line=SHOW_APPROX_LINE,
            approx_window=APPROX_LINE_WINDOW,
        )
        for group in backtracking_groups
    )

    for curve, message in backtracking_results:
        if message is not None:
            print(message)
            continue
        curves.append(curve)

    print('> Plotting adoption curves')
    adoption_results = Parallel(
        n_jobs=PLOT_N_JOBS, backend="loky",
    )(
        delayed(_plot_adoption_rule_task)(
            df=language_dfs[rule.language], old_form=rule.old_form,
            new_form=rule.new_form, rule_label=rule.label,
            reform_year=rule.reform_year,
            output_directory=output_directory, x_label=x_label,
            use_ewma=USE_EWMA, ewma_span=EWMA_SPAN,
            group_name=group_name,
            show_approx_line=SHOW_APPROX_LINE,
            approx_window=APPROX_LINE_WINDOW,
        )
        for rule in plot_rules
    )

    for curve, message in adoption_results:
        if message is not None:
            print(message)
            continue
        curves.append(curve)

    if not curves:
        print(f"[{group_name}] Skipping meta curve (no curves)")
        return

    print('> Plotting meta-curve')
    meta_curve = calculator.compute_meta_curve(
        curves, use_ewma=USE_EWMA, ewma_span=EWMA_SPAN,
    )
    plotter.plot_meta_curve(
        meta_curve, x_label=x_label, ewma=USE_EWMA,
        show_approx_line=SHOW_APPROX_LINE,
        approx_window=APPROX_LINE_WINDOW,
    )


if __name__ == "__main__":
    # ======================================================
    # Load rules
    # ======================================================

    print('Loading rules')
    rules = load_rules(RULES_FILE)
    old_index = load_rules_by_old_form(RULES_FILE)
    new_index = load_rules_by_new_form(RULES_FILE)

    rules_df = pd.read_csv(RULES_FILE)
    effect_by_rule_id = dict(zip(
        rules_df["rule_id"], rules_df["normative_effect"],
    ))

    # ======================================================
    # Build objects
    # ======================================================

    print('Building objects')
    analyzer = OrthographyAnalyzer(
        old_index,
        new_index,
    )
    statistics = Statistics()
    calculator = AdoptionCalculator()

    # ======================================================
    # Analyse corpus
    # ======================================================

    print('Analyzing corpus')
    cache_path = PREPROCESSED_CACHE_JSON
    if not cache_path.exists():
        print(f"No cache found at {cache_path}; preprocessing "
              "corpus...")
        preprocess_corpus(CORPUS_DIR, cache_path)
    else:
        print('Using cached preprocessed corpus')
    cached_issues = load_preprocessed_corpus(cache_path)

    # ======================================================
    # Classify corpus (bokmål / nynorsk / unknown)
    # ======================================================

    print('Classifying corpus')
    classified_path = CLASSIFIED_CACHE
    if not classified_path.exists():
        print(f"No cache found at {classified_path}; classifying "
              "corpus...")
        classify_corpus(CORPUS_DIR, classified_path)
    else:
        print('Using cached classification results')
    classified_issues = load_classified_corpus(classified_path)
    results = [
        analyzer.analyze_from_cache(issue) for issue in cached_issues
    ]
    attach_labels(results, classified_issues)

    print(
        f"Label distribution: "
        f"{dict(Counter(r['label'] for r in results))}"
    )
    
    language_dfs = {
        language: statistics.aggregate_by_period(
            [r for r in results if r["label"] in
             LANGUAGE_LABEL_GROUPS[group]],
            months=AGGREGATION_MONTHS,
        )
        for language, group in RULE_LANGUAGE_TO_GROUP.items()
    }

    x_label = GRANULARITY_LABELS[AGGREGATION_MONTHS]

    backtracking_groups = find_backtracking_groups(rules)
    bt_pairs = backtracking_pairs(backtracking_groups)

    print('Generating plots')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for group_name, labels in LANGUAGE_LABEL_GROUPS.items():
        print(f"=== Group: {group_name} ({sorted(labels)}) ===")
        run_group(group_name, labels, results, x_label, rng=rng)

    print("=== Group: same_language ===")
    plot_rules = _deduplicated_plot_rules(
        rules, bt_pairs, "same_language",
    )
    run_same_language(
        rules, backtracking_groups, plot_rules, language_dfs, x_label,
        rng=rng,
    )

    print("Done.")
