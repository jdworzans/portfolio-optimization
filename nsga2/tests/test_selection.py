import numpy as np

from nsga2.selection import selection


def test_selection_with_2_simple_fronts():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    np.testing.assert_equal(selection(example, 2), np.array([0, 1]))
    np.testing.assert_equal(selection(example, 3), np.array([0, 1, 2]))

def test_selection_with_3_simple_fronts():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 2],
        ]
    )
    np.testing.assert_equal(selection(example, 2), np.array([0, 1]))
    np.testing.assert_equal(selection(example, 3), np.array([0, 1, 2]))

def test_selection_with_nontrivial_front():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [0.5, 0.5],
        ]
    )
    assert set(selection(example, 2)) == {0, 1}
    assert set(selection(example, 3)) == {0, 1, 2}
