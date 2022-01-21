import numpy as np


def portfolios_vars(weights: np.ndarray, SIGMA: np.ndarray) -> np.ndarray:
    return np.einsum("...i,ij,...j", weights, SIGMA, weights)

def portfolios_neg_returns(weights: np.ndarray, R: np.ndarray) -> np.ndarray:
    return - weights @ R

def get_objective(*objectives):
    def objective(weights: np.ndarray) -> np.ndarray:
        return np.stack([obj(weights) for obj in objectives], axis=-1)
    return objective
