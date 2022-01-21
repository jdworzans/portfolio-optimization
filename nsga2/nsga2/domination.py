from collections import defaultdict

import numpy as np


def get_domination_info(objectives):
    domination_count = np.zeros(len(objectives), dtype=int)
    dominated = defaultdict(set)
    for idx, x in enumerate(objectives):
        dominators_idxs = get_dominators_idxs(x, objectives)
        domination_count[idx] = len(dominators_idxs)
        for dominator_idx in dominators_idxs:
            dominated[dominator_idx].add(idx)
    return domination_count, dominated


def get_dominators_idxs(objective: np.ndarray, objectives: np.ndarray) -> np.ndarray:
    _objective = np.atleast_2d(objective)
    not_greater = (objectives <= _objective).all(axis=-1).nonzero()[0]
    is_any_lower = (objectives[not_greater] < _objective).any(axis=-1)
    return not_greater[is_any_lower]


def get_domination_count(objective: np.ndarray, objectives: np.ndarray) -> int:
    return len(get_dominators_idxs(objective, objectives))
