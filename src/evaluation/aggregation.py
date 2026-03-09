import numpy as np
import pandas as pd
from scipy.stats import expon, gaussian_kde, laplace, norm


def minimum(dq_results: pd.Series, certainties: pd.Series):
    return dq_results.min()


def maximum(dq_results: pd.Series, certainties: pd.Series):
    return dq_results.max()


def weighted_minimum(dq_results: pd.Series, certainties: pd.Series):
    weighted_results = dq_results * certainties
    return weighted_results.min()


def weighted_maximum(dq_results: pd.Series, certainties: pd.Series):
    weighted_results = dq_results * certainties
    return weighted_results.max()


def mean(dq_results: pd.Series, certainties: pd.Series):
    return dq_results.mean()


def weighted_mean(dq_results: pd.Series, certainties: pd.Series):
    weighted_sums = (dq_results * certainties).sum()
    sum_of_weights = certainties.sum()
    return weighted_sums / sum_of_weights


def median(dq_results: pd.Series, certainties: pd.Series):
    return dq_results.median()


def weighted_median(dq_results: pd.Series, certainties: pd.Series):
    weighted_results = dq_results * certainties
    return weighted_results.median()


def kl_divergence(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    """Compute KL divergence D_KL(P || Q) for discrete approximations."""
    eps = 1e-12  # Add a small constant to avoid log(0) and division by zero
    p += eps
    q += eps
    return np.sum(p * np.log(p / q)) * dx


def js_divergence(p, q, dx):
    """Compute Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, dx) + 0.5 * kl_divergence(q, m, dx)


def evaluate_divergence(dq_results: pd.Series, aggregated_value: float) -> dict:
    agg: float = aggregated_value
    data = np.array(dq_results)

    # ----- 1. Estimate empirical distribution via KDE -----
    if len(np.unique(data)) == 1:
        data[0] += 1e-12  # Add small noise to avoid singularity in KDE
    kde = gaussian_kde(data)

    x_min, x_max = data.min(), data.max()
    x_grid = np.linspace(x_min, x_max, 1000)
    dx = x_grid[1] - x_grid[0]

    p = kde(x_grid)
    p /= np.sum(p) * dx  # Normalize

    # ----- 2. Gaussian model -----
    sigma = np.std(data, ddof=1)
    q_gauss = norm.pdf(x_grid, loc=agg, scale=sigma)
    q_gauss /= np.sum(q_gauss) * dx

    js_gauss = js_divergence(p, q_gauss, dx)

    # ----- 3. Laplace model -----
    b = np.mean(np.abs(data - agg))
    q_laplace = laplace.pdf(x_grid, loc=agg, scale=b)
    q_laplace /= np.sum(q_laplace) * dx

    js_laplace = js_divergence(p, q_laplace, dx)

    # ----- 4. Exponential model -----
    rate = 1 / (agg + 1e-12)  # Avoid division by zero
    q_exp = expon.pdf(x_grid, scale=1 / rate)
    q_exp /= np.sum(q_exp) * dx

    js_exp = js_divergence(p, q_exp, dx)

    return {
        "js_gaussian": js_gauss,
        "js_laplace": js_laplace,
        "js_exponential": js_exp,
    }


def evaluate_aggregation_methods(
    dq_results: pd.Series, certainties: pd.Series, is_polluted_mask: pd.Series
) -> dict:
    aggregation_methods = [
        minimum,
        maximum,
        weighted_minimum,
        weighted_maximum,
        mean,
        weighted_mean,
        median,
        weighted_median,
    ]

    divergence_results = {}
    weighted_divergence_results = {}
    for method in aggregation_methods:
        aggregated_results = method(dq_results, certainties)
        # mse = ((is_polluted_mask - aggregated_results) ** 2).mean()
        # mse_results[method.__name__] = mse.to_dict()
        divergence_results[method.__name__] = evaluate_divergence(
            dq_results, aggregated_results
        )
        weighted_divergence_results[method.__name__] = evaluate_divergence(
            dq_results * certainties, aggregated_results
        )

    return {
        # "mse": mse_results,
        "divergence": divergence_results,
        "weighted_divergence": weighted_divergence_results,
    }
