import pandas as pd


class TransitionMatrix:
    """Compute transition matrices for orthographic reform adoption."""

    @staticmethod
    def compute(
        df: pd.DataFrame,
        old_form: str,
        new_form: str,
        reform_year: int,
    ) -> pd.DataFrame:
        """Compute the per-period adoption rate.

        Args:
            df: Output of Statistics.aggregate_by_period(), with a "period" column (float) and one column per orthographic form (occurrence counts).
            old_form: Historical spelling.
            new_form: Modern spelling.
            reform_year: Official reform year.

        Returns:
            pandas.DataFrame with columns "period" (float), "old_count" (int), "new_count" (int), "adoption" (float, 0 where the period has zero total occurrences of both forms), "reform_year" (int), "old_form" (str), and "new_form" (str).

        Raises:
            ValueError: If old_form or new_form is not a column of df.
        """

        if old_form not in df.columns:
            raise ValueError(f"Unknown column: {old_form}")

        if new_form not in df.columns:
            raise ValueError(f"Unknown column: {new_form}")

        transition = df[["period"]].copy()
        transition["old_count"] = df[old_form]
        transition["new_count"] = df[new_form]
        total = transition["old_count"] + transition["new_count"]
        transition["adoption"] = (
            transition["new_count"] / total
        ).fillna(0)
        transition["reform_year"] = reform_year
        transition["old_form"] = old_form
        transition["new_form"] = new_form

        return transition
