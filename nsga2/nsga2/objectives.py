import numpy as np
import pandas as pd

from nsga2.defaults import MERGED_DATASET_FILEPATH

DEFAULT_RETURNS = np.asarray(pd.read_csv(MERGED_DATASET_FILEPATH, parse_dates=["Date"], index_col="Date").dropna())

def portfolios_vars(weights: np.ndarray, SIGMA: np.ndarray) -> np.ndarray:
    return np.einsum("...i,ij,...j", weights, SIGMA, weights)

def portfolios_neg_returns(weights: np.ndarray, R: np.ndarray) -> np.ndarray:
    return - weights @ R

def portfolios_semivariances(weights: np.ndarray, returns=DEFAULT_RETURNS):
    weights_returns = np.einsum("ij,kj", weights, returns)
    avg_returns = weights_returns.mean(axis=-1, keepdims=True)
    semivariances = np.mean(
        np.square(weights_returns - avg_returns),
        where=(weights_returns < avg_returns),
        axis=-1,
    )
    return semivariances

def get_objective(*objectives):
    def objective(weights: np.ndarray) -> np.ndarray:
        return np.stack([obj(weights) for obj in objectives], axis=-1)
    return objective
