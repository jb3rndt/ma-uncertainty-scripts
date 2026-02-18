import numpy as np
import pandas as pd
from scipy.stats import expon, gaussian_kde, laplace, norm


def minimum(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    return dq_results.min(axis=0)


def maximum(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    return dq_results.max(axis=0)


def weighted_minimum(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    weighted_results = dq_results * certainties
    return weighted_results.min(axis=0)


def weighted_maximum(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    weighted_results = dq_results * certainties
    return weighted_results.max(axis=0)


def mean(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    return dq_results.mean(axis=0)


def weighted_mean(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    weighted_sums = (dq_results * certainties).sum(axis=0)
    sum_of_weights = certainties.sum(axis=0)
    return weighted_sums / sum_of_weights


def median(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    return dq_results.median(axis=0)


def weighted_median(dq_results: pd.DataFrame, certainties: pd.DataFrame) -> pd.Series:
    weighted_results = dq_results * certainties
    return weighted_results.median(axis=0)


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


def evaluate_kl_divergence(
    dq_results: pd.DataFrame, aggregated_values: pd.Series
) -> dict:
    results = {"kl_gaussian": {}, "kl_laplace": {}, "kl_exponential": {}}
    for col in dq_results.columns:
        agg: float = aggregated_values[col]
        data = dq_results[col]

        # ----- 1. Estimate empirical distribution via KDE -----
        if data.nunique() == 1:
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

        kl_gauss = kl_divergence(p, q_gauss, dx)

        # ----- 3. Laplace model -----
        b = np.mean(np.abs(data - agg))
        q_laplace = laplace.pdf(x_grid, loc=agg, scale=b)
        q_laplace /= np.sum(q_laplace) * dx

        kl_laplace = kl_divergence(p, q_laplace, dx)

        # ----- 4. Exponential model -----
        rate = 1 / (agg + 1e-12)  # Avoid division by zero
        q_exp = expon.pdf(x_grid, scale=1 / rate)
        q_exp /= np.sum(q_exp) * dx

        kl_exp = kl_divergence(p, q_exp, dx)

        results["kl_gaussian"][col] = kl_gauss
        results["kl_laplace"][col] = kl_laplace
        results["kl_exponential"][col] = kl_exp

    return results


def evaluate_aggregation_methods(
    dq_results: pd.DataFrame, certainties: pd.DataFrame, is_polluted_mask: pd.DataFrame
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

    mse_results = {}
    kl_results = {}
    weighted_kl_results = {}
    for method in aggregation_methods:
        aggregated_results = method(dq_results, certainties)
        mse = ((is_polluted_mask - aggregated_results) ** 2).mean()
        mse_results[method.__name__] = mse.to_dict()
        kl_results[method.__name__] = evaluate_kl_divergence(
            dq_results, aggregated_results
        )
        weighted_kl_results[method.__name__] = evaluate_kl_divergence(
            dq_results * certainties, aggregated_results
        )

    return {
        "mse": mse_results,
        "kl_divergence": kl_results,
        "weighted_kl_divergence": weighted_kl_results,
    }
