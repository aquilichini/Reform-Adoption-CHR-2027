import numpy as np
from scipy.stats import norm


def wilson_interval(
        successes,
        total,
        confidence: float = 0.95,
):
    """Compute the Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes, shape (n,).
        total: Number of trials, shape (n,).
        confidence: Confidence level in (0, 1). Defaults to 0.95.

    Returns:
        Tuple of (lower, upper), each a numpy.ndarray of shape (n,) 
            with values clipped to [0, 1], and NaN where total == 0.
            The interval always contains the point estimate p, even
            after floating-point rounding.
    """

    successes = np.asarray(successes, dtype=float)
    total = np.asarray(total, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = norm.ppf(1 - (1 - confidence) / 2)
        p = successes / total
        denom = 1 + z ** 2 / total
        center = (p + z ** 2 / (2 * total)) / denom
        margin = (
            z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
            / denom
        )
        lower = center - margin
        upper = center + margin

    zero_total = total == 0
    lower = np.where(zero_total, np.nan, lower)
    upper = np.where(zero_total, np.nan, upper)

    lower = np.minimum(lower, p)
    upper = np.maximum(upper, p)

    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)

    return lower, upper
