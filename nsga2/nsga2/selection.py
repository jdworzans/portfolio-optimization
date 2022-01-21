import numpy as np

from nsga2.crowding_dist import select_by_crowding_dist
from nsga2.domination import get_domination_info


def get_pareto_front(domination_count: np.ndarray) -> np.ndarray:
    return (domination_count == 0).nonzero()[0]


def selection(objectives: np.ndarray, n: int) -> np.ndarray:
    domination_count, dominated = get_domination_info(objectives)
    selected = []
    while (n_selected := len(selected)) != n:
        pareto_front = get_pareto_front(domination_count)
        if n_selected + len(pareto_front) <= n:
            new_selected = pareto_front
        else:
            crowding_dist_selected = select_by_crowding_dist(
                objectives[pareto_front], n - n_selected
            )
            new_selected = pareto_front[crowding_dist_selected]
        selected.extend(list(new_selected))
        for idx in new_selected:
            domination_count[idx] = -1
            for dominated_idx in dominated[idx]:
                domination_count[dominated_idx] -= 1
    return selected
