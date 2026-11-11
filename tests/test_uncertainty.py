import numpy as np
import pytest

from orthography_change.uncertainty import wilson_interval


def test_wilson_interval_known_value():

    # successes=5, total=10, 95% confidence: hand-computed via the
    # standard Wilson score formula (z=1.959963984540054).
    lower, upper = wilson_interval([5], [10])

    assert lower[0] == pytest.approx(0.2365931, abs=1e-6)
    assert upper[0] == pytest.approx(0.7634069, abs=1e-6)


def test_wilson_interval_zero_total_is_nan():

    lower, upper = wilson_interval([0], [0])

    assert np.isnan(lower[0])
    assert np.isnan(upper[0])


def test_wilson_interval_narrows_as_total_grows():

    # Same proportion (p=0.5), increasing sample size -> narrower
    # interval.
    lower_small, upper_small = wilson_interval([5], [10])
    lower_large, upper_large = wilson_interval([500], [1000])

    width_small = upper_small[0] - lower_small[0]
    width_large = upper_large[0] - lower_large[0]

    assert width_large < width_small


def test_wilson_interval_bounds_within_unit_interval():

    successes = np.array([0, 1, 5, 10, 3])
    total = np.array([1, 1, 10, 10, 3])

    lower, upper = wilson_interval(successes, total)

    valid = ~np.isnan(lower)
    assert np.all(lower[valid] >= 0)
    assert np.all(upper[valid] <= 1)
    assert np.all(lower[valid] <= upper[valid])


def test_wilson_interval_vectorized_over_array():

    successes = np.array([0, 5, 10])
    total = np.array([0, 10, 10])

    lower, upper = wilson_interval(successes, total)

    assert np.isnan(lower[0]) and np.isnan(upper[0])
    assert not np.isnan(lower[1]) and not np.isnan(upper[1])
    assert not np.isnan(lower[2]) and not np.isnan(upper[2])


def test_wilson_interval_contains_point_estimate_at_extremes():

    # Regression test: for p exactly 0 or 1, floating-point rounding
    # in the center/margin arithmetic could push a clipped bound
    # past p by a tiny negative amount (e.g. -2.22e-16), which
    # matplotlib.pyplot.errorbar rejects as a negative yerr.
    n = np.arange(1, 201)

    lower_ones, upper_ones = wilson_interval(n, n)
    assert np.all(1.0 - lower_ones >= 0)
    assert np.all(upper_ones - 1.0 >= 0)

    zeros = np.zeros_like(n)
    lower_zeros, upper_zeros = wilson_interval(zeros, n)
    assert np.all(0.0 - lower_zeros >= 0)
    assert np.all(upper_zeros - 0.0 >= 0)
