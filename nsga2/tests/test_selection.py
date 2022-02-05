import numpy as np

from nsga2.selection import selection, get_fitness_values


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

def test_fitness_values():
    example = np.array([0, 1, 2])
    adjusted = np.array([2 - 0, 2 - 1, 2 - 2])
    fitness_values = np.array([2/3, 1/3, 0/3])
    np.testing.assert_equal(fitness_values, get_fitness_values(example))

def test_fitness_values_with_inf():
    example = np.array([0, 1, 2, np.inf])
    adjusted = np.array([2 - 0, 2 - 1, 2 - 2, 0])
    fitness_values = np.array([2/3, 1/3, 0/3, 0])
    np.testing.assert_equal(fitness_values, get_fitness_values(example))


def test_fitness_equal_values():
    example = np.array([0, 0, 0])
    fitness_values = np.array([1/3, 1/3, 1/3])
    np.testing.assert_equal(fitness_values, get_fitness_values(example))

def test_fitness_equal_values_with_inf():
    example = np.array([0, 0, 0, np.inf])
    fitness_values = np.array([1/3, 1/3, 1/3, 0])
    np.testing.assert_equal(fitness_values, get_fitness_values(example))


def test_multiple_fitness_values():
    example = np.array(
        [
            [1, 0],
            [0, 1],
            [0.5, 0.5],
        ]
    )
    