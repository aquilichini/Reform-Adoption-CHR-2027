import numpy as np
import pandas as pd
from scipy.signal import lfilter
from orthography_change.uncertainty import wilson_interval

DEFAULT_EWMA_SPAN = 4  # Larger value perform stronger smoothing


class AdoptionCalculator:
    """Compute adoption curves from period counts."""

    def compute_curve(
            self,
            df: pd.DataFrame,
            old_form: str,
            new_form: str,
            use_ewma: bool = True,
            ewma_span: float = DEFAULT_EWMA_SPAN,
    ) -> pd.DataFrame:
        """Compute the adoption curve of one orthographic rule.

        Args:
            df: Output of Statistics.aggregate_by_period(), with a
                "period" column (float) and one column per
                orthographic form (occurrence counts).
            old_form: Historical spelling.
            new_form: Modern spelling.
            use_ewma: If True (default), compute estimate via EWMA.
            ewma_span: smoothing window size, only used when use_ewma is True.

        Returns:
            pandas.DataFrame with columns:
                - "period" (float)
                - "old_count" (float): occurrences of old_form.
                - "new_count" (float): occurrences of new_form.
                - "total" (float): old_count + new_count.
                - "adoption" (float): new_form / (old_form + new_form) for the period (smoothed if use_ewma).
        """
        curve = df.copy()
        if old_form not in curve.columns:
            curve[old_form] = 0
        if new_form not in curve.columns:
            curve[new_form] = 0
        curve["old_count"] = curve[old_form]
        curve["new_count"] = curve[new_form]
        curve["total"] = curve["old_count"] + curve["new_count"]
        curve = self._add_adoption_and_ci(curve, use_ewma, ewma_span)

        return curve[
            [
                "period",
                "old_count",
                "new_count",
                "total",
                "adoption",
                "ci_lower",
                "ci_upper",
            ]
        ]

    def compute_meta_curve(
        self,
        curves: list[pd.DataFrame],
        use_ewma: bool = True,
        ewma_span: float = DEFAULT_EWMA_SPAN,
    ) -> pd.DataFrame:
        """Compute the pooled-count meta adoption curve.

        Args:
            curves: List of adoption curves, each a DataFrame with "period", "old_count", and "new_count" columns.
            use_ewma: if True, use ewma.
            ewma_span: ewma window size.

        Returns:
            pandas.DataFrame with columns:
                - "period" (float)
                - "old_count" (float): summed across curves.
                - "new_count" (float): summed across curves.
                - "total" (float): old_count + new_count.
                - "adoption" (float): pooled new_count / total.
                - "ci_lower" (float): 25th-percentile Wilson score
                  bound on "adoption".
                - "ci_upper" (float): 75th-percentile Wilson score
                  bound on "adoption".
        """
        counts = pd.concat(
            [c[["period", "old_count", "new_count"]] for c in curves]
        )
        pooled = (
            counts
            .groupby("period", as_index=False)
            [["old_count", "new_count"]]
            .sum()
            .sort_values("period")
        )
        pooled["total"] = pooled["old_count"] + pooled["new_count"]
        pooled = self._add_adoption_and_ci(pooled, use_ewma, ewma_span)
        return pooled[
            [
                "period",
                "old_count",
                "new_count",
                "total",
                "adoption",
                "ci_lower",
                "ci_upper",
            ]
        ]

    def curve_from_counts(
        self,
        period,
        old_count,
        new_count,
        use_ewma: bool = True,
        ewma_span: float = DEFAULT_EWMA_SPAN,
    ) -> pd.DataFrame:
        """Compute an adoption curve directly from count arrays.

        Args:
            period: Period values with shape (n,).
            old_count: Occurrence counts of the old form with shape (n,).
            new_count: Occurrence counts of the new form with shape (n,).
            use_ewma: If True (default), smooth counts before
                computing "adoption"/"ci_lower"/"ci_upper" (see
                compute_curve).
            ewma_span: Span, in periods, of the exponential smoothing
                window. Only used when use_ewma is True.

        Returns:
            pandas.DataFrame:
        """
        curve = pd.DataFrame(
            {
                "period": period,
                "old_count": old_count,
                "new_count": new_count,
            }
        )
        curve["total"] = curve["old_count"] + curve["new_count"]
        curve = self._add_adoption_and_ci(curve, use_ewma, ewma_span)
        return curve[
            [
                "period",
                "old_count",
                "new_count",
                "total",
                "adoption",
                "ci_lower",
                "ci_upper",
            ]
        ]

    def first_crossing_period(
        self,
        curve: pd.DataFrame,
        threshold: float,
    ) -> float | None:
        """Return the earliest period where adoption reaches a level.

        Args:
            curve: DataFrame with "period" (float) and "adoption"
                (float) columns, sorted by "period" (true of
                compute_curve/compute_meta_curve/curve_from_counts
                output).
            threshold: Adoption-rate threshold in (0, 1).

        Returns:
            float or None: Smallest "period" value where "adoption" >= threshold, or None if no such period exists.
        """
        crossed = curve.loc[curve["adoption"] >= threshold, "period"]
        if crossed.empty:
            return None
        return crossed.iloc[0]

    def _add_adoption_and_ci(
        self,
        curve: pd.DataFrame,
        use_ewma: bool,
        ewma_span: float,
    ) -> pd.DataFrame:
        """Add "adoption", "ci_lower", and "ci_upper" columns.

        Args:
            curve: DataFrame with "period", "old_count", "new_count", and "total" columns.
            use_ewma: if True, use ewma.
            ewma_span: window size.

        Returns:
            pandas.DataFrame: curve (sorted by "period"), with "adoption", "ci_lower", and "ci_upper" set.
        """
        curve = curve.sort_values("period").reset_index(drop=True)
        if use_ewma:
            new_count = self._ewma_accumulate(curve["new_count"], ewma_span)
            old_count = self._ewma_accumulate(curve["old_count"], ewma_span)
            total = old_count + new_count
        else:
            new_count = curve["new_count"]
            total = curve["total"]

        curve["adoption"] = float("nan")
        mask = total > 0
        curve.loc[mask, "adoption"] = new_count[mask] / total[mask]
        curve["ci_lower"], curve["ci_upper"] = wilson_interval(
            new_count,
            total,
            confidence=0.5,
        )
        return curve

    def _ewma_accumulate(
        self,
        counts,
        span: float,
    ) -> np.ndarray:
        """Evidence-weighted exponential accumulation of counts.

        Args:
            counts: per-period counts (sorted by period) with shape (n,) or (batch, n) to accumulate many rows at once along the last axis.
            span: ewma window size.

        Returns:
            numpy.ndarray: Accumulated counts with the same shape as `counts`, where acc[t] = counts[t] + (1 - alpha) * acc[t-1].
        """

        alpha = 2 / (span + 1)
        rho = 1 - alpha
        counts = np.asarray(counts, dtype=float)
        return lfilter([1.0], [1.0, -rho], counts, axis=-1)
