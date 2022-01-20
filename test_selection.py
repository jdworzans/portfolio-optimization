import numpy as np

from selection import selection


def test_selection():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    np.testing.assert_equal(selection(example, 2), np.array([0, 1]))
