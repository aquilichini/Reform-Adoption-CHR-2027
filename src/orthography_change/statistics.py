from collections import defaultdict
import pandas as pd
from tqdm import tqdm

ALLOWED_GRANULARITIES = (1, 3, 6, 12)
GRANULARITY_LABELS: dict[int, str] = {
    1: "Month",
    3: "Quarter",
    6: "Half-year",
    12: "Year",
}


class Statistics:
    """Aggregate orthographic counts over a corpus.
    """

    def aggregate_by_period(
        self,
        analyses: list[dict],
        months: int = 6,
    ) -> pd.DataFrame:
        """Aggregate counts into fixed-length time periods.

        Args:
            analyses: Output of OrthographyAnalyzer. Each element is a dict with keys "year" (int), "date" (str "YYYYMMDD" or None/falsy), and "counts" (dict mapping form (str) to occurrence count (int)).
            months: bucket size. Must be one of ALLOWED_GRANULARITIES (1, 3, 6, 12). Defaults to 6.

        Returns:
            pandas.DataFrame with one row per period, sorted by
                period, and columns:
                - "period" (float): period identifier, see above.
                - one column per orthographic form (float): summed occurrence counts, 0 where a form did not occur in that period.

        Raises:
            ValueError: If ``months`` is not one of ALLOWED_GRANULARITIES.
        """

        if months not in ALLOWED_GRANULARITIES:
            raise ValueError(
                f"Invalid months={months!r}; must be one of "
                f"{ALLOWED_GRANULARITIES}"
            )

        data = defaultdict(lambda: defaultdict(int))
        undated_fallback_count = 0
        year_mismatch_count = 0
        for analysis in tqdm(analyses, desc="Aggregating"):
            metadata_year = analysis["year"]
            date = analysis.get("date")
            if date:
                year = int(date[:4])
                month = int(date[4:6])
                if year != metadata_year:
                    year_mismatch_count += 1
            else:
                year = metadata_year
                month = 1
                undated_fallback_count += 1

            period_index = (month - 1) // months
            period = year + period_index * months / 12
            for form, count in analysis["counts"].items():
                data[period][form] += count

        if undated_fallback_count > 0 and months != 12:
            print(
                f"Note: {undated_fallback_count} undated issue(s) "
                "were bucketed into the first period of their year"
            )

        if year_mismatch_count > 0:
            print(
                f"Note: {year_mismatch_count} issue(s) had a date "
                "whose year differs from their recorded year; the "
                "date's year was used for bucketing"
            )

        rows = []
        for period in sorted(data):
            row = {"period": period}
            row.update(data[period])
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["period"])

        dataframe = pd.DataFrame(rows)
        dataframe = dataframe.fillna(0)
        return dataframe
