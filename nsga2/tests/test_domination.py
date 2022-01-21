import numpy as np

from nsga2.domination import get_domination_count, get_domination_info, get_dominators_idxs


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
