from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Series color (categorical slot 1, blue) and supporting chrome,
# following this repo's dataviz styling conventions.
_SERIES_COLOR = "#2a78d6"
_REFORM_LINE_COLOR = "#898781"
_GRID_COLOR = "#e1e0d9"

# Rendering DPI for saved PNGs. Kept at 300 (not lowered).
_DPI = 300

# PNG zlib compression level passed to Pillow via savefig's
# pil_kwargs. Matplotlib's default (6) is needlessly slow for these
# simple line/scatter plots; 1 trades slightly larger files for a
# near-free speedup with identical visual content.
_PNG_KWARGS = {"compress_level": 1}

# Label for the shaded confidence band, shown in the legend. Must
# match the confidence level AdoptionCalculator actually computes
# (see wilson_interval's `confidence` argument in adoption.py).
_CI_LABEL = "25th-75th percentile CI"

# Marker opacity range for the evidence cue in _shaded_ci/
# _evidence_alpha: periods with (near-)zero raw occurrences render at
# _MIN_MARKER_ALPHA, the best-evidenced period in the curve at
# _MAX_MARKER_ALPHA.
_MIN_MARKER_ALPHA = 0.25
_MAX_MARKER_ALPHA = 1.0

_EVIDENCE_NOTE = (
    "Marker opacity reflects each period's raw occurrence count"
)

# Color, legend label, and default window for the optional
# moving-average approximation line overlaid on adoption plots.
_APPROX_LINE_COLOR = "#d62728"
_APPROX_LINE_LABEL = "Approximation (moving average)"
_DEFAULT_APPROX_WINDOW = 9


class Plotter:
    """
    Plot adoption curves.
    """

    def __init__(
        self,
        output_directory: str | Path = "plots",
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._fig, self._ax = plt.subplots(figsize=(10, 6))

    def _new_plot(self) -> Axes:
        """Clear and return the Plotter's reusable Axes.

        Clearing via `cla()` is cheaper than creating a new Figure
        per plot call, and removes all artists (lines, text,
        legends) from the previous call.

        Returns:
            matplotlib.axes.Axes: The cleared, reusable Axes.
        """

        self._ax.cla()
        return self._ax

    def _evidence_alpha(self, total: pd.Series) -> np.ndarray:
        """Map raw per-period evidence to marker opacity.

        A sqrt transform is used so opacity rises quickly for small
        counts and levels off for large ones -- most of the signal
        is in flagging "this period had almost no data".

        Args:
            total: Raw per-period occurrence counts, shape (n,).

        Returns:
            numpy.ndarray: Per-point opacity in
                [_MIN_MARKER_ALPHA, _MAX_MARKER_ALPHA], shape (n,).
        """

        peak = total.max()

        if peak <= 0:
            return np.full(len(total), _MAX_MARKER_ALPHA)

        scaled = np.sqrt(total.clip(lower=0) / peak)

        return _MIN_MARKER_ALPHA + (
            _MAX_MARKER_ALPHA - _MIN_MARKER_ALPHA
        ) * scaled

    def _shaded_ci(
        self,
        ax: Axes,
        curve: pd.DataFrame,
        linewidth: float,
        ewma: bool,
    ) -> None:
        """Draw an adoption curve with a shaded confidence band.

        Periods with zero total occurrences (NaN adoption) are
        excluded, since a NaN would break the fill/line continuity.
        Markers are drawn with opacity proportional to each period's
        *raw* occurrence count (regardless of `ewma`), so a period
        whose position is entirely inferred from EWMA history still
        renders faint even though its adoption/CI are not NaN.

        Args:
            ax: Axes to draw on.
            curve: DataFrame with columns "period" (float),
                "adoption" (float), "ci_lower" (float), "ci_upper"
                (float), and "total" (float).
            linewidth: Width of the connecting line.
            ewma: Whether `curve` was computed from an exponentially
                weighted moving average (AdoptionCalculator's
                use_ewma=True); only affects the legend text.
        """

        mask = curve["adoption"].notna()
        plotted = curve.loc[mask]

        suffix = " (EWMA)" if ewma else ""

        ax.fill_between(
            plotted["period"],
            plotted["ci_lower"],
            plotted["ci_upper"],
            color=_SERIES_COLOR,
            alpha=0.2,
            linewidth=0,
            label=f"{_CI_LABEL}{suffix}",
        )

        ax.plot(
            plotted["period"],
            plotted["adoption"],
            "-",
            color=_SERIES_COLOR,
            alpha=0.9,
            linewidth=linewidth,
            label=f"Adoption rate{suffix}",
        )

        rgba = np.tile(mcolors.to_rgba(_SERIES_COLOR), (len(plotted), 1))
        rgba[:, 3] = self._evidence_alpha(plotted["total"])

        ax.scatter(
            plotted["period"],
            plotted["adoption"],
            color=rgba,
            s=36,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )

        self._fig.text(
            0.5, 0.01, _EVIDENCE_NOTE,
            ha="center", va="bottom", fontsize=8, color=_REFORM_LINE_COLOR,
        )

    def _plot_approx_line(
        self,
        ax: Axes,
        curve: pd.DataFrame,
        window: int,
    ) -> None:
        """Overlay a centered moving-average approximation line.

        Uses curve["period"]/curve["adoption"] directly -- i.e.
        whatever was actually plotted (raw or EWMA), not a
        recomputation from counts. NaN-adoption periods (zero
        occurrences) are dropped first, matching `_shaded_ci`, so
        the rolling window only sees real values. `min_periods=1`
        lets the line stay close to the data near the edges / on
        short curves, where a full window isn't available.

        Args:
            ax: Axes to draw on.
            curve: DataFrame with "period" (float) and "adoption"
                (float) columns.
            window: Rolling-average window, in periods.
        """

        mask = curve["adoption"].notna()
        period = curve.loc[mask, "period"]
        smoothed = (
            curve.loc[mask, "adoption"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
        ax.plot(
            period, smoothed, "-", color=_APPROX_LINE_COLOR,
            linewidth=1.75, alpha=0.85, zorder=4,
            label=_APPROX_LINE_LABEL,
        )

    def plot_adoption_curve(
        self,
        curve: pd.DataFrame,
        rule_name: str,
        reform_year: int,
        x_label: str = "Period",
        ewma: bool = True,
        show_approx_line: bool = False,
        approx_window: int = _DEFAULT_APPROX_WINDOW,
    ) -> None:
        """Plot one adoption curve with a shaded Wilson score CI band.

        Args:
            curve: DataFrame with columns "period" (float),
                "adoption" (float), "ci_lower" (float), and
                "ci_upper" (float), as returned by
                AdoptionCalculator.compute_curve().
            rule_name: Name displayed in the title.
            reform_year: Official reform year.
            x_label: Label for the x-axis, describing the
                aggregation granularity (e.g. "Month", "Quarter",
                "Half-year", "Year"). Defaults to "Period".
            ewma: Whether `curve` was computed with
                AdoptionCalculator's use_ewma=True; only affects the
                legend text. Defaults to True to match
                AdoptionCalculator's own default.
            show_approx_line: Whether to overlay a red centered
                moving-average approximation line. Defaults to
                False.
            approx_window: Rolling-average window, in periods, for
                the approximation line. Only used when
                show_approx_line is True.
        """

        ax = self._new_plot()

        self._shaded_ci(ax, curve, linewidth=2, ewma=ewma)

        if show_approx_line:
            self._plot_approx_line(ax, curve, approx_window)

        ax.axvline(
            reform_year,
            linestyle="--",
            linewidth=1.5,
            color=_REFORM_LINE_COLOR,
        )

        ax.text(
            reform_year,
            0.02,
            f"Reform {reform_year}",
            rotation=90,
            va="bottom",
            color=_REFORM_LINE_COLOR,
        )

        ax.set_ylim(0, 1)

        ax.set_xlim(
            curve["period"].min(),
            curve["period"].max(),
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Adoption rate")

        ax.set_title(rule_name)

        ax.grid(True, color=_GRID_COLOR, linewidth=0.8)

        ax.legend(loc="best", frameon=False)

        self._fig.tight_layout()

        self._fig.savefig(
            self.output_directory /
            f"{rule_name}.png",
            dpi=_DPI,
            pil_kwargs=_PNG_KWARGS,
        )

    def plot_backtracking_curve(
        self,
        curve: pd.DataFrame,
        label: str,
        events: list[tuple[int, str]],
        x_label: str = "Period",
        show_approx_line: bool = False,
        approx_window: int = _DEFAULT_APPROX_WINDOW,
    ) -> None:
        """Plot a combined curve for a backtracking form pair.

        Args:
            curve: DataFrame with "period" (float) and "adoption"
                (float) columns -- the single A -> B curve over the
                full corpus range.
            label: Title/filename stem, e.g. "mel ↔ mjøl".
            events: Ordered (reform_year, direction) pairs;
                direction is "up" (toward B) or "down" (toward A).
            x_label: x-axis label for the aggregation granularity.
            show_approx_line: Whether to overlay a red centered
                moving-average approximation line. Defaults to
                False.
            approx_window: Rolling-average window, in periods, for
                the approximation line. Only used when
                show_approx_line is True.
        """

        colors = {"up": "#1b9e77", "down": "#d95f02"}
        arrows = {"up": "↑", "down": "↓"}

        ax = self._new_plot()

        ax.plot(
            curve["period"],
            curve["adoption"],
            linewidth=2,
            marker="o",
        )

        if show_approx_line:
            self._plot_approx_line(ax, curve, approx_window)
            ax.legend(loc="best", frameon=False)

        for year, direction in events:
            ax.axvline(
                year,
                linestyle="--",
                linewidth=2,
                color=colors[direction],
            )
            ax.text(
                year,
                0.02,
                f"{arrows[direction]} Reform {year}",
                rotation=90,
                va="bottom",
            )

        ax.set_ylim(0, 1)

        ax.set_xlim(
            curve["period"].min(),
            curve["period"].max(),
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Adoption rate")

        ax.set_title(label)

        ax.grid(True)

        self._fig.tight_layout()

        self._fig.savefig(
            self.output_directory /
            f"{label}.png",
            dpi=_DPI,
            pil_kwargs=_PNG_KWARGS,
        )

    def plot_meta_curve(
        self,
        curve: pd.DataFrame,
        x_label: str = "Period",
        ewma: bool = True,
        show_approx_line: bool = False,
        approx_window: int = _DEFAULT_APPROX_WINDOW,
    ) -> None:
        """Plot the pooled meta adoption curve with a shaded CI band.

        Args:
            curve: DataFrame with columns "period" (float),
                "adoption" (float), "ci_lower" (float), and
                "ci_upper" (float), as returned by
                AdoptionCalculator.compute_meta_curve().
            x_label: Label for the x-axis, describing the
                aggregation granularity (e.g. "Month", "Quarter",
                "Half-year", "Year"). Defaults to "Period".
            ewma: Whether `curve` was computed with
                AdoptionCalculator's use_ewma=True; only affects the
                legend text. Defaults to True to match
                AdoptionCalculator's own default.
            show_approx_line: Whether to overlay a red centered
                moving-average approximation line. Defaults to
                False.
            approx_window: Rolling-average window, in periods, for
                the approximation line. Only used when
                show_approx_line is True.
        """

        ax = self._new_plot()

        self._shaded_ci(ax, curve, linewidth=3, ewma=ewma)

        if show_approx_line:
            self._plot_approx_line(ax, curve, approx_window)

        ax.set_ylim(0, 1)

        ax.set_xlim(
            curve["period"].min(),
            curve["period"].max(),
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Adoption rate (pooled across rules)")

        ax.set_title("Meta adoption curve")

        ax.grid(True, color=_GRID_COLOR, linewidth=0.8)

        ax.legend(loc="best", frameon=False)

        self._fig.tight_layout()

        self._fig.savefig(
            self.output_directory /
            "meta_curve.png",
            dpi=_DPI,
            pil_kwargs=_PNG_KWARGS,
        )

    def plot_adoption_time_boxplot(
        self,
        samples_by_reform: dict[str, dict],
        threshold: float,
    ) -> None:
        """Plot per-reform time-to-adoption boxplots.

        One box per reform, showing the distribution of years from
        that reform's official reform_year until its rules' adoption
        rate crossed `threshold` (across rules and bootstrap
        resamples). Reforms with no samples are dropped.

        Args:
            samples_by_reform: {reform_id: {"reform_year": int,
                "samples": list[float]}}, as returned by
                reform_timing.compute_reform_boxplot_data().
            threshold: Adoption-rate threshold in (0, 1), used in
                the title and output filename.
        """

        items = [
            (reform_id, data)
            for reform_id, data in samples_by_reform.items()
            if data["samples"]
        ]

        if not items:
            print("Skipping adoption-time boxplot for threshold "
                  f"{threshold:.0%} (no occurrences)")
            return

        items.sort(key=lambda kv: (kv[1]["reform_year"], kv[0]))

        labels = [
            f"{data['reform_year']}\n{reform_id}"
            for reform_id, data in items
        ]
        values = [data["samples"] for _, data in items]

        ax = self._new_plot()

        box = ax.boxplot(values, tick_labels=labels, patch_artist=True)

        for patch in box["boxes"]:
            patch.set_facecolor(_SERIES_COLOR)
            patch.set_alpha(0.35)

        for median in box["medians"]:
            median.set_color(_SERIES_COLOR)

        ax.set_xlabel("Reform")
        ax.set_ylabel("Time to adoption (years)")
        ax.set_title(f"Time to {threshold:.0%} adoption by reform")

        ax.grid(True, axis="y", color=_GRID_COLOR, linewidth=0.8)

        self._fig.tight_layout()

        self._fig.savefig(
            self.output_directory /
            f"adoption_time_{threshold:.2f}.png",
            dpi=_DPI,
            pil_kwargs=_PNG_KWARGS,
        )
