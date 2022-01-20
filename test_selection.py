import numpy as np
from collections import defaultdict

def selection(objectives: np.ndarray, n: int) -> np.ndarray:

    return np.zeros(n)

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

def test_domination_count():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    assert get_domination_count(example[0], example) == 0
    assert get_domination_count(example[1], example) == 0
    assert get_domination_count(example[2], example) == 2

def test_dominators_idxs():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    np.testing.assert_equal(get_dominators_idxs(example[0], example), np.array([]))
    np.testing.assert_equal(get_dominators_idxs(example[1], example), np.array([]))
    np.testing.assert_equal(get_dominators_idxs(example[2], example), np.array([0, 1]))


def test_domination_info():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    domination_count, dominated = get_domination_info(example)
    np.testing.assert_equal(domination_count, np.array([0, 0, 2]))
    assert dominated[0] == {2}
    assert dominated[1] == {2}
    assert dominated[2] == set()


def test_selection():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    np.testing.assert_equal(selection(example, 2), np.array([0, 1]))



