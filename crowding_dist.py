import numpy as np


def get_crowding_distances(objectives: np.ndarray) -> np.ndarray:
    result = np.zeros(len(objectives))
    order = np.argsort(objectives)
    f_min, f_max = objectives[order[[0, -1]]]
    result[order[[0, -1]]] = np.inf
    result[order[1:-1]] = (objectives[order[2:]] - objectives[order[:-2]]) / (
        f_max - f_min
    )
    return result


def select_by_crowding_dist(objectives, n: int) -> np.ndarray:
    crowding_distances = np.zeros(len(objectives))
    for m in range(np.shape(objectives)[-1]):
        crowding_distances += get_crowding_distances(objectives[:, m])
    return np.argsort(crowding_distances)[::-1][:n]
