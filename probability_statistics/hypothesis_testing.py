"""One-sample z-test when population standard deviation is known."""

import math
from scipy.stats import norm


def one_sample_z_test(sample_mean, null_mean, population_std, n, alternative="two-sided"):
    if population_std <= 0 or n <= 0:
        raise ValueError("population_std and n must be positive.")
    z = (sample_mean - null_mean) / (population_std / math.sqrt(n))
    if alternative == "two-sided":
        p = 2.0 * norm.sf(abs(z))
    elif alternative == "greater":
        p = norm.sf(z)
    elif alternative == "less":
        p = norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    return float(z), float(p)


if __name__ == "__main__":
    z, p = one_sample_z_test(102.0, 100.0, 10.0, 100)
    print("z statistic:", z)
    print("p-value:", p)
