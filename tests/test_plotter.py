import matplotlib
matplotlib.use("Agg")

import matplotlib.collections as mcollections  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from orthography_change.plotter import (  # noqa: E402
    _APPROX_LINE_LABEL,
    Plotter,
)


def _curve():
    return pd.DataFrame(
        {
            "period": [1950.0, 1960.0, 1970.0, 1980.0, 1990.0],
            "adoption": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )


def _curve_with_nan_row():
    return pd.DataFrame(
        {
            "period": [1900.0, 1900.5, 1901.0],
            "old_count": [5, 0, 1],
            "new_count": [5, 0, 3],
            "total": [10, 0, 4],
            "adoption": [0.5, float("nan"), 0.75],
            "ci_lower": [0.2365931, float("nan"), 0.3011533],
            "ci_upper": [0.7634069, float("nan"), 0.9540878],
        }
    )


def test_plot_adoption_curve_writes_png(tmp_path):

    plotter = Plotter(output_directory=tmp_path)

    plotter.plot_adoption_curve(
        curve=_curve_with_nan_row(),
        rule_name="skulle_skulde",
        reform_year=1907,
        x_label="Half-year",
    )

    output_file = tmp_path / "skulle_skulde.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_backtracking_curve_two_events(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    plotter.plot_backtracking_curve(
        curve=_curve(),
        label="mel ↔ mjøl",
        events=[(1959, "up"), (1981, "down")],
    )

    assert (tmp_path / "mel ↔ mjøl.png").exists()


def test_plot_backtracking_curve_three_events(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    plotter.plot_backtracking_curve(
        curve=_curve(),
        label="a ↔ b",
        events=[(1959, "up"), (1970, "down"), (1981, "up")],
    )

    assert (tmp_path / "a ↔ b.png").exists()


def test_plot_meta_curve_writes_png(tmp_path):

    plotter = Plotter(output_directory=tmp_path)

    plotter.plot_meta_curve(
        curve=_curve_with_nan_row(),
        x_label="Half-year",
    )

    output_file = tmp_path / "meta_curve.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def _legend_labels_after(plotter, plot_call):
    """Run one plot_* call and read back its legend labels.

    Reads state straight off `plotter._ax` right after the call
    returns, since the Axes is now reused across calls rather than
    torn down (via plt.close) after each one.

    Args:
        plotter: The Plotter instance `plot_call` operates on.
        plot_call: Zero-arg callable that triggers one plot_* call.

    Returns:
        list[str]: Legend labels present on `plotter._ax` right
            after the call.
    """

    plot_call()
    _, labels = plotter._ax.get_legend_handles_labels()
    return labels


def test_plot_adoption_curve_legend_states_ewma_by_default(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_adoption_curve(
            curve=_curve_with_nan_row(),
            rule_name="skulle_skulde",
            reform_year=1907,
            x_label="Half-year",
        ),
    )

    assert any("EWMA" in label for label in labels)
    assert any("25th-75th percentile CI" in label for label in labels)


def test_plot_adoption_curve_legend_omits_ewma_when_disabled(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_adoption_curve(
            curve=_curve_with_nan_row(),
            rule_name="skulle_skulde",
            reform_year=1907,
            x_label="Half-year",
            ewma=False,
        ),
    )

    assert not any("EWMA" in label for label in labels)
    assert any("25th-75th percentile CI" in label for label in labels)


def test_plot_adoption_curve_marker_opacity_reflects_evidence(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    plotter.plot_adoption_curve(
        curve=_curve_with_nan_row(),
        rule_name="skulle_skulde",
        reform_year=1907,
        x_label="Half-year",
    )

    alphas = None
    for collection in plotter._ax.collections:
        if isinstance(collection, mcollections.PathCollection):
            alphas = collection.get_facecolor()[:, 3]

    # Plotted rows are period 1900.0 (total=10, the most evidence)
    # and period 1901.0 (total=4, less evidence); the NaN row
    # (total=0) is masked out entirely.
    assert alphas is not None
    assert len(alphas) == 2
    assert alphas[0] == pytest.approx(1.0)
    assert alphas[0] > alphas[1]


def test_plot_adoption_time_boxplot_writes_png(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    samples_by_reform = {
        "nb_1907": {"reform_year": 1907, "samples": [1.0, 2.0, 3.0]},
        "nn_1917": {"reform_year": 1917, "samples": [0.5, 1.5]},
    }

    plotter.plot_adoption_time_boxplot(samples_by_reform, threshold=0.8)

    output_file = tmp_path / "adoption_time_0.80.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_adoption_time_boxplot_skips_when_all_empty(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    samples_by_reform = {
        "nb_1907": {"reform_year": 1907, "samples": []},
        "nn_1917": {"reform_year": 1917, "samples": []},
    }

    plotter.plot_adoption_time_boxplot(samples_by_reform, threshold=0.9)

    assert list(tmp_path.iterdir()) == []


def test_plot_adoption_curve_legend_shows_approx_line_when_enabled(
    tmp_path,
):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_adoption_curve(
            curve=_curve_with_nan_row(),
            rule_name="skulle_skulde",
            reform_year=1907,
            x_label="Half-year",
            show_approx_line=True,
        ),
    )

    assert _APPROX_LINE_LABEL in labels


def test_plot_adoption_curve_legend_omits_approx_line_by_default(
    tmp_path,
):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_adoption_curve(
            curve=_curve_with_nan_row(),
            rule_name="skulle_skulde",
            reform_year=1907,
            x_label="Half-year",
        ),
    )

    assert _APPROX_LINE_LABEL not in labels


def test_plot_backtracking_curve_legend_shows_approx_line_when_enabled(
    tmp_path,
):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_backtracking_curve(
            curve=_curve(),
            label="mel ↔ mjøl",
            events=[(1959, "up"), (1981, "down")],
            show_approx_line=True,
        ),
    )

    assert _APPROX_LINE_LABEL in labels


def test_plot_backtracking_curve_legend_omits_approx_line_by_default(
    tmp_path,
):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_backtracking_curve(
            curve=_curve(),
            label="mel ↔ mjøl",
            events=[(1959, "up"), (1981, "down")],
        ),
    )

    assert _APPROX_LINE_LABEL not in labels


def test_plot_meta_curve_legend_shows_approx_line_when_enabled(
    tmp_path,
):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_meta_curve(
            curve=_curve_with_nan_row(),
            x_label="Half-year",
            show_approx_line=True,
        ),
    )

    assert _APPROX_LINE_LABEL in labels


def test_plot_meta_curve_legend_omits_approx_line_by_default(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    labels = _legend_labels_after(
        plotter,
        lambda: plotter.plot_meta_curve(
            curve=_curve_with_nan_row(),
            x_label="Half-year",
        ),
    )

    assert _APPROX_LINE_LABEL not in labels


def test_approx_line_smooths_a_sharp_spike(tmp_path):
    plotter = Plotter(output_directory=tmp_path)

    periods = [float(p) for p in range(11)]
    adoption = [0.1] * 11
    adoption[5] = 0.9
    curve = pd.DataFrame({"period": periods, "adoption": adoption})

    plotter.plot_backtracking_curve(
        curve=curve,
        label="spike",
        events=[],
        show_approx_line=True,
        approx_window=9,
    )

    approx_y = None
    for line in plotter._ax.get_lines():
        if line.get_label() == _APPROX_LINE_LABEL:
            approx_y = line.get_ydata()

    assert approx_y is not None
    assert max(approx_y) < max(adoption)
