import numpy as np

from nsga2.crowding_dist import get_crowding_distances, select_by_crowding_dist


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

def test_select_by_crowding_dist():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [0.5, 0.5],
        ]
    )

    selected2 = select_by_crowding_dist(example, 2)
    assert set(selected2) == {0, 1}

    selected3 = select_by_crowding_dist(example, 3)
    assert set(selected3) == {0, 1, 2}

def test_select_by_crowding_dist_wtih_5_elements():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [0.5, 0.5],
            [0.55, 0.45],
            [0.45, 0.55],
        ]
    )

    selected2 = select_by_crowding_dist(example, 2)
    assert set(selected2) == {0, 1}

    selected4 = select_by_crowding_dist(example, 4)
    assert set(selected4) == {0, 1, 3, 4}

    selected5 = select_by_crowding_dist(example, 5)
    assert set(selected5) == {0, 1, 2, 3, 4}
    