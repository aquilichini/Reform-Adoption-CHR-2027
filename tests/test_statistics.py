import pytest

from orthography_change.statistics import Statistics


# ============================================================================
# Helpers
# ============================================================================


def make_analysis(
    year: int,
    date: str | None,
    counts: dict[str, int],
) -> dict:
    """Build a minimal OrthographyAnalyzer-style result.

    Args:
        year: Publication year.
        date: "YYYYMMDD" date string, or None if unknown.
        counts: Mapping of orthographic form to occurrence count.

    Returns:
        Dict with keys "year", "date", "counts".
    """

    return {"year": year, "date": date, "counts": counts}


# ============================================================================
# aggregate_by_period: 6-month default
# ============================================================================


def test_aggregate_by_period_default_half_year():

    analyses = [
        make_analysis(1920, "19200315", {"skulde": 2}),
        make_analysis(1920, "19200520", {"skulde": 1}),
        make_analysis(1920, "19200815", {"skulle": 3}),
        make_analysis(1921, "19210110", {"skulle": 1}),
    ]

    df = Statistics().aggregate_by_period(analyses)

    periods = df["period"].tolist()

    assert periods == pytest.approx([1920.0, 1920.5, 1921.0])

    row_1920_0 = df[df["period"] == 1920.0].iloc[0]
    row_1920_5 = df[df["period"] == 1920.5].iloc[0]
    row_1921_0 = df[df["period"] == 1921.0].iloc[0]

    assert row_1920_0["skulde"] == 3
    assert row_1920_5["skulle"] == 3
    assert row_1921_0["skulle"] == 1


# ============================================================================
# aggregate_by_period: monthly
# ============================================================================


def test_aggregate_by_period_monthly():

    analyses = [
        make_analysis(1922, "19220110", {"skulde": 1}),
        make_analysis(1922, "19220205", {"skulde": 1}),
        make_analysis(1922, "19221225", {"skulle": 1}),
    ]

    df = Statistics().aggregate_by_period(analyses, months=1)

    periods = df["period"].tolist()

    assert periods == pytest.approx(
        [1922.0, 1922 + 1 / 12, 1922 + 11 / 12]
    )


# ============================================================================
# aggregate_by_period: quarterly and yearly smoke tests
# ============================================================================


def test_aggregate_by_period_quarterly():

    analyses = [
        make_analysis(1930, "19300201", {"skulde": 1}),
        make_analysis(1930, "19300815", {"skulle": 1}),
    ]

    df = Statistics().aggregate_by_period(analyses, months=3)

    periods = df["period"].tolist()

    assert periods == pytest.approx([1930.0, 1930 + 6 / 12])


def test_aggregate_by_period_yearly():

    analyses = [
        make_analysis(1940, "19400301", {"skulde": 1}),
        make_analysis(1941, "19411120", {"skulle": 1}),
    ]

    df = Statistics().aggregate_by_period(analyses, months=12)

    periods = df["period"].tolist()

    assert periods == pytest.approx([1940.0, 1941.0])


# ============================================================================
# aggregate_by_period: undated fallback
# ============================================================================


def test_aggregate_by_period_undated_falls_back_to_first_period(
    capsys,
):

    analyses = [
        make_analysis(1950, None, {"skulde": 5}),
        make_analysis(1950, "19500801", {"skulle": 2}),
    ]

    df = Statistics().aggregate_by_period(analyses, months=6)

    row_first_half = df[df["period"] == 1950.0].iloc[0]

    assert row_first_half["skulde"] == 5

    captured = capsys.readouterr()

    assert "1" in captured.out


def test_aggregate_by_period_no_fallback_note_at_year_granularity(
    capsys,
):

    analyses = [
        make_analysis(1950, None, {"skulde": 5}),
    ]

    Statistics().aggregate_by_period(analyses, months=12)

    captured = capsys.readouterr()

    assert captured.out == ""


# ============================================================================
# aggregate_by_period: year/date mismatch
# ============================================================================


def test_aggregate_by_period_uses_dates_year_over_recorded_year(
    capsys,
):
    """A date's own year must win over a mismatched folder year.

    Regression test: an issue recorded under year=1922 (e.g. from
    its containing folder) but dated 19230101 must be bucketed into
    period 1923.0, not into 1922 + an offset derived from combining
    the wrong year with the date's month.
    """

    analyses = [
        make_analysis(1922, "19230101", {"skulde": 1}),
        make_analysis(1923, "19230801", {"skulle": 1}),
    ]

    df = Statistics().aggregate_by_period(analyses, months=6)

    periods = df["period"].tolist()

    assert periods == pytest.approx([1923.0, 1923.5])

    row_1923_0 = df[df["period"] == 1923.0].iloc[0]
    assert row_1923_0["skulde"] == 1

    captured = capsys.readouterr()
    assert "1" in captured.out


# ============================================================================
# aggregate_by_period: empty input
# ============================================================================


def test_aggregate_by_period_empty_analyses_has_period_column():
    """Regression test: an empty group must still yield "period".

    With zero analyses (e.g. a language group with no matching
    issues), pd.DataFrame([]) would otherwise produce a completely
    columnless frame, breaking the documented contract and crashing
    downstream consumers that index by "period".
    """

    df = Statistics().aggregate_by_period([])

    assert list(df.columns) == ["period"]
    assert len(df) == 0


# ============================================================================
# aggregate_by_period: invalid granularity
# ============================================================================


def test_aggregate_by_period_invalid_months_raises():

    with pytest.raises(ValueError):

        Statistics().aggregate_by_period([], months=2)
