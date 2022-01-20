import numpy as np

from crowding_dist import get_crowding_distances


def test_crowding_distance_for_two_values():
    example = np.array([0, 1])
    crowding_distances = get_crowding_distances(example)
    np.testing.assert_equal(crowding_distances, np.array([np.inf, np.inf]))


def test_crowding_distance_for_three_values():
    example = np.array([0, 2, 1])
    ordered = np.array([0, 1, 2])
    crowding_distances = get_crowding_distances(example)
    np.testing.assert_equal(crowding_distances[[0, 1]], np.array([np.inf, np.inf]))
    assert crowding_distances[2] == (ordered[2] - ordered[0]) / (
        max(example) - min(example)
    )
