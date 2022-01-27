import numpy as np


def portfolios_vars(weights: np.ndarray, SIGMA: np.ndarray) -> np.ndarray:
    return np.einsum("...i,ij,...j", weights, SIGMA, weights)


def portfolios_neg_returns(weights: np.ndarray, R: np.ndarray) -> np.ndarray:
    return -weights @ R


def portfolios_semivariances(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    weights_returns = np.einsum("ij,kj", weights, returns)
    avg_returns = weights_returns.mean(axis=-1, keepdims=True)
    semivariances = np.mean(
        np.square(weights_returns - avg_returns),
        where=(weights_returns < avg_returns),
        axis=-1,
    )
    return semivariances


def portfolios_empirical_VaR(
    weights: np.ndarray, returns: np.ndarray, alpha=0.95
) -> np.ndarray:
    weights_returns = np.einsum("ij,kj", weights, returns)
    return -np.quantile(weights_returns, 1 - alpha, axis=-1)


def get_objective(*objectives):
    def objective(weights: np.ndarray) -> np.ndarray:
        return np.stack([obj(weights) for obj in objectives], axis=-1)

    return objective
