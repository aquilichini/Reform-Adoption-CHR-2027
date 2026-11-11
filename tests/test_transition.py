import pandas as pd
import pytest

from orthography_change.transition import TransitionMatrix


# ============================================================================
# compute
# ============================================================================


def test_compute_returns_period_based_adoption():

    df = pd.DataFrame(
        {
            "period": [1920.0, 1920.5, 1921.0],
            "skulde": [3, 0, 0],
            "skulle": [0, 3, 1],
        }
    )

    result = TransitionMatrix.compute(
        df,
        old_form="skulde",
        new_form="skulle",
        reform_year=1907,
    )

    assert result["period"].tolist() == [1920.0, 1920.5, 1921.0]
    assert result["old_count"].tolist() == [3, 0, 0]
    assert result["new_count"].tolist() == [0, 3, 1]
    assert result["adoption"].tolist() == pytest.approx(
        [0.0, 1.0, 1.0]
    )
    assert (result["reform_year"] == 1907).all()
    assert (result["old_form"] == "skulde").all()
    assert (result["new_form"] == "skulle").all()


def test_compute_zero_total_period_gives_zero_adoption():

    df = pd.DataFrame(
        {
            "period": [1920.0],
            "skulde": [0],
            "skulle": [0],
        }
    )

    result = TransitionMatrix.compute(
        df,
        old_form="skulde",
        new_form="skulle",
        reform_year=1907,
    )

    assert result["adoption"].tolist() == [0.0]


def test_compute_missing_old_form_column_raises():

    df = pd.DataFrame({"period": [1920.0], "skulle": [1]})

    with pytest.raises(ValueError):
        TransitionMatrix.compute(
            df,
            old_form="skulde",
            new_form="skulle",
            reform_year=1907,
        )


def test_compute_missing_new_form_column_raises():

    df = pd.DataFrame({"period": [1920.0], "skulde": [1]})

    with pytest.raises(ValueError):
        TransitionMatrix.compute(
            df,
            old_form="skulde",
            new_form="skulle",
            reform_year=1907,
        )
