import pandas as pd
import pytest

from orthography_change.adoption import AdoptionCalculator

_CURVE_COLUMNS = [
    "period",
    "old_count",
    "new_count",
    "total",
    "adoption",
    "ci_lower",
    "ci_upper",
]


# ============================================================================
# compute_curve
# ============================================================================


def test_compute_curve_pooled_ratio():

    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5, 1901.0],
            "skulle": [5, 0, 3],
            "skulde": [5, 0, 1],
        }
    )

    # use_ewma=False: exercise the raw per-period math directly.
    curve = AdoptionCalculator().compute_curve(
        df=df,
        old_form="skulde",
        new_form="skulle",
        use_ewma=False,
    )

    assert list(curve.columns) == _CURVE_COLUMNS

    row_1900_0 = curve[curve["period"] == 1900.0].iloc[0]
    row_1900_5 = curve[curve["period"] == 1900.5].iloc[0]
    row_1901_0 = curve[curve["period"] == 1901.0].iloc[0]

    # Pooled ratio: new / (old + new).
    assert row_1900_0["adoption"] == pytest.approx(0.5)
    assert row_1900_0["old_count"] == 5
    assert row_1900_0["new_count"] == 5
    assert row_1900_0["total"] == 10

    # Hand-computed Wilson interval for successes=5, total=10,
    # confidence=0.5 -- i.e. the 25th-75th percentile band
    # (z=0.6744897501960817).
    assert row_1900_0["ci_lower"] == pytest.approx(
        0.3956999, abs=1e-6
    )
    assert row_1900_0["ci_upper"] == pytest.approx(
        0.6043001, abs=1e-6
    )

    # Zero total occurrences of both forms -> NaN.
    assert pd.isna(row_1900_5["adoption"])
    assert pd.isna(row_1900_5["ci_lower"])
    assert pd.isna(row_1900_5["ci_upper"])

    assert row_1901_0["adoption"] == pytest.approx(3 / 4)


def test_compute_curve_missing_old_form_column():

    # "skulde" (old_form) never occurs anywhere in the corpus, so
    # Statistics.aggregate_by_period never created that column.
    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5],
            "skulle": [5, 0],
        }
    )

    curve = AdoptionCalculator().compute_curve(
        df=df,
        old_form="skulde",
        new_form="skulle",
        use_ewma=False,
    )

    assert list(curve.columns) == _CURVE_COLUMNS

    row_1900_0 = curve[curve["period"] == 1900.0].iloc[0]
    row_1900_5 = curve[curve["period"] == 1900.5].iloc[0]

    # All occurrences are the new form -> full adoption.
    assert row_1900_0["adoption"] == pytest.approx(1.0)

    # Zero occurrences of either form -> NaN.
    assert pd.isna(row_1900_5["adoption"])
    assert pd.isna(row_1900_5["ci_lower"])
    assert pd.isna(row_1900_5["ci_upper"])


def test_compute_curve_empty_period_dataframe():
    """Regression test for a language group with zero analyses.

    Statistics.aggregate_by_period([]) returns a 0-row frame with
    only a "period" column (no form columns). compute_curve must
    still produce the full expected schema instead of raising
    KeyError on the final column selection.
    """

    df = pd.DataFrame(columns=["period"])

    curve = AdoptionCalculator().compute_curve(
        df=df,
        old_form="skulde",
        new_form="skulle",
    )

    assert list(curve.columns) == _CURVE_COLUMNS
    assert len(curve) == 0


def test_compute_curve_ewma_is_the_default():

    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5, 1901.0],
            "skulle": [5, 0, 3],
            "skulde": [5, 0, 1],
        }
    )

    default_curve = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle",
    )
    explicit_curve = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", use_ewma=True,
    )

    pd.testing.assert_frame_equal(default_curve, explicit_curve)


def test_compute_curve_ewma_weights_by_evidence_not_fixed_alpha():

    # A long, converged, all-old-form history (adoption=0), then a
    # single anomalous period with just 1 occurrence, all new-form
    # (raw adoption=1.0), then back to the converged baseline. A
    # normalized EWMA (fixed ~40% weight per period regardless of
    # its evidence) would swing the point estimate hard toward 1.0;
    # evidence-weighted accumulation should barely move it, since a
    # single occurrence is negligible next to the accumulated
    # history.
    periods = [1900.0 + i for i in range(16)]
    old_counts = [5] * 15 + [0]
    new_counts = [0] * 15 + [1]

    df = pd.DataFrame(
        {"period": periods, "skulde": old_counts, "skulle": new_counts}
    )

    curve = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", ewma_span=4,
    )

    anomaly = curve.iloc[-1]

    assert anomaly["period"] == 1915.0
    assert anomaly["adoption"] < 0.2


def test_compute_curve_ewma_fills_zero_occurrence_period():

    # Same fixture as test_compute_curve_pooled_ratio, but with
    # smoothing enabled (the default): period 1900.5 has zero raw
    # occurrences of either form, yet borrows enough history from
    # 1900.0 to no longer be NaN.
    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5, 1901.0],
            "skulle": [5, 0, 3],
            "skulde": [5, 0, 1],
        }
    )

    curve = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", ewma_span=4,
    )

    row_1900_5 = curve[curve["period"] == 1900.5].iloc[0]

    assert not pd.isna(row_1900_5["adoption"])
    assert not pd.isna(row_1900_5["ci_lower"])
    assert not pd.isna(row_1900_5["ci_upper"])
    assert row_1900_5["ci_lower"] <= row_1900_5["adoption"]
    assert row_1900_5["adoption"] <= row_1900_5["ci_upper"]

    # Raw counts are unaffected by smoothing.
    assert row_1900_5["old_count"] == 0
    assert row_1900_5["new_count"] == 0
    assert row_1900_5["total"] == 0


def test_compute_curve_ewma_first_period_matches_raw():

    # A period's own smoothed value only borrows from *earlier*
    # periods (causal EWMA), so the very first period is identical
    # whether or not smoothing is enabled.
    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5],
            "skulle": [5, 0],
            "skulde": [5, 0],
        }
    )

    smoothed = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", use_ewma=True,
    )
    raw = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", use_ewma=False,
    )

    first_smoothed = smoothed.iloc[0]
    first_raw = raw.iloc[0]

    assert first_smoothed["adoption"] == pytest.approx(
        first_raw["adoption"]
    )
    assert first_smoothed["ci_lower"] == pytest.approx(
        first_raw["ci_lower"]
    )
    assert first_smoothed["ci_upper"] == pytest.approx(
        first_raw["ci_upper"]
    )


# ============================================================================
# compute_meta_curve
# ============================================================================


def test_compute_meta_curve_pools_counts_on_period():

    # Rule A: 1900.0 has 3 old / 1 new; 1900.5 has 0 old / 2 new.
    curve_a = pd.DataFrame(
        {
            "period": [1900.0, 1900.5],
            "old_count": [3, 0],
            "new_count": [1, 2],
        }
    )

    # Rule B: 1900.0 has 1 old / 5 new; 1900.5 has 4 old / 4 new.
    curve_b = pd.DataFrame(
        {
            "period": [1900.0, 1900.5],
            "old_count": [1, 4],
            "new_count": [5, 4],
        }
    )

    meta_curve = AdoptionCalculator().compute_meta_curve(
        [curve_a, curve_b], use_ewma=False,
    )

    assert list(meta_curve.columns) == _CURVE_COLUMNS

    row_1900_0 = meta_curve[meta_curve["period"] == 1900.0].iloc[0]
    row_1900_5 = meta_curve[meta_curve["period"] == 1900.5].iloc[0]

    # Pooled counts, not an unweighted mean of per-rule ratios:
    # 1900.0 -> old=3+1=4, new=1+5=6 -> adoption=6/10=0.6
    # (unweighted mean would be (0.25 + 0.833) / 2 != 0.6).
    assert row_1900_0["old_count"] == 4
    assert row_1900_0["new_count"] == 6
    assert row_1900_0["total"] == 10
    assert row_1900_0["adoption"] == pytest.approx(0.6)

    # 1900.5 -> old=0+4=4, new=2+4=6 -> adoption=6/10=0.6
    assert row_1900_5["old_count"] == 4
    assert row_1900_5["new_count"] == 6
    assert row_1900_5["adoption"] == pytest.approx(0.6)

    # CI bounds are within [0, 1] and bracket the point estimate.
    assert 0 <= row_1900_0["ci_lower"] <= row_1900_0["adoption"]
    assert row_1900_0["adoption"] <= row_1900_0["ci_upper"] <= 1


def test_compute_meta_curve_ewma_is_the_default_and_smooths():

    # Rule A: 1900.0 has 3 old / 1 new; 1900.5 has 0 old / 0 new.
    curve_a = pd.DataFrame(
        {
            "period": [1900.0, 1900.5],
            "old_count": [3, 0],
            "new_count": [1, 0],
        }
    )

    default_meta = AdoptionCalculator().compute_meta_curve([curve_a])
    raw_meta = AdoptionCalculator().compute_meta_curve(
        [curve_a], use_ewma=False,
    )

    assert list(default_meta.columns) == _CURVE_COLUMNS

    # 1900.5 has zero raw occurrences -> NaN without smoothing, but
    # the default (use_ewma=True) borrows from 1900.0 and fills it.
    raw_1900_5 = raw_meta[raw_meta["period"] == 1900.5].iloc[0]
    smoothed_1900_5 = default_meta[
        default_meta["period"] == 1900.5
    ].iloc[0]

    assert pd.isna(raw_1900_5["adoption"])
    assert not pd.isna(smoothed_1900_5["adoption"])


# ============================================================================
# curve_from_counts
# ============================================================================


def test_curve_from_counts_matches_compute_curve():

    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5, 1901.0],
            "skulle": [5, 0, 3],
            "skulde": [5, 0, 1],
        }
    )

    from_df = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", use_ewma=False,
    )
    from_counts = AdoptionCalculator().curve_from_counts(
        period=df["period"],
        old_count=df["skulde"],
        new_count=df["skulle"],
        use_ewma=False,
    )

    assert list(from_counts.columns) == _CURVE_COLUMNS
    pd.testing.assert_frame_equal(from_df, from_counts)


def test_curve_from_counts_ewma_matches_compute_curve():

    df = pd.DataFrame(
        {
            "period": [1900.0, 1900.5, 1901.0],
            "skulle": [5, 0, 3],
            "skulde": [5, 0, 1],
        }
    )

    from_df = AdoptionCalculator().compute_curve(
        df=df, old_form="skulde", new_form="skulle", ewma_span=4,
    )
    from_counts = AdoptionCalculator().curve_from_counts(
        period=df["period"],
        old_count=df["skulde"],
        new_count=df["skulle"],
        ewma_span=4,
    )

    pd.testing.assert_frame_equal(from_df, from_counts)


# ============================================================================
# first_crossing_period
# ============================================================================


def test_first_crossing_period_finds_correct_period():

    curve = pd.DataFrame(
        {
            "period": [1900.0, 1901.0, 1902.0, 1903.0],
            "adoption": [0.2, 0.5, 0.85, 0.95],
        }
    )

    assert AdoptionCalculator().first_crossing_period(
        curve, threshold=0.8
    ) == 1902.0


def test_first_crossing_period_returns_none_when_never_reached():

    curve = pd.DataFrame(
        {
            "period": [1900.0, 1901.0, 1902.0],
            "adoption": [0.2, 0.5, 0.6],
        }
    )

    assert AdoptionCalculator().first_crossing_period(
        curve, threshold=0.8
    ) is None
