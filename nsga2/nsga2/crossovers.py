from itertools import chain

import numpy as np

from nsga2.defaults import DEFAULT_RNG


class Crossover:
    def __init__(self, crossover_probability: float = 0.95, rng=DEFAULT_RNG):
        self.crossover_probability = crossover_probability
        self.rng = rng

    def cross(self, parent1, parent2):
        raise NotImplementedError

    def crossover(self, population):
        children = np.empty_like(population)
        for idx in range(len(population) // 2):
            parent1 = population[2 * idx, :]
            parent2 = population[2 * idx + 1, :]

            if self.rng.random() < self.crossover_probability:
                children[2 * idx, :], children[2 * idx + 1, :] = self.cross(
                    parent1, parent2,
                )
            else:
                children[2 * idx, :], children[2 * idx + 1, :] = parent1, parent2
        if len(population) % 2 == 1:
            children[-1, :] = population[-1]
        return children
    
    def __call__(self, population):
        return self.crossover(population)


class CX(Crossover):
    def cross(self, parent1, parent2):
        idxs = set(range(len(parent1)))
        cycles = [[idxs.pop()]]
        while idxs:
            current_idx = cycles[-1][-1]
            cycle_next_element = parent2[current_idx]
            cycle_next_idx = int(np.where(parent1 == cycle_next_element)[0].squeeze())
            if cycle_next_idx == cycles[-1][0]:
                cycles.append([idxs.pop()])
            else:
                cycles[-1].append(cycle_next_idx)
                idxs.remove(cycle_next_idx)
        child1, child2 = parent1.copy(), parent2.copy()
        for cycle in cycles[1::2]:
            child1[cycle] = parent2[cycle]
            child2[cycle] = parent1[cycle]
        return child1, child2

class PMX(Crossover):
    def _cross(self, parent1: np.ndarray, parent2: np.ndarray, left: int, right: int):
        child1 = parent1.copy()
        child2 = parent2.copy()
        N = len(child1)
        for idx in chain(range(left), range(right, N)):
            while child1[idx] in parent2[left:right]:
                child1[idx] = parent1[left:right][parent2[left:right] == child1[idx]]
            while child2[idx] in parent1[left:right]:
                child2[idx] = parent2[left:right][parent1[left:right] == child2[idx]]
        child1[left:right] = parent2[left:right]
        child2[left:right] = parent1[left:right]
        return child1, child2


    def cross(self, parent1: np.ndarray, parent2: np.ndarray):
        return self._cross(
            parent1, parent2, *sorted(self.rng.choice(len(parent1), 2, replace=False))
        )

class EdgeRecombination(Crossover):
    def _cross(self, parent1, parent2):
        neighbors = {el: set() for el in parent1}
        for i, el in enumerate(parent1):
            neighbors[el].add(parent1[(i - 1) % len(parent1)])
            neighbors[el].add(parent1[(i + 1) % len(parent1)])
        for i, el in enumerate(parent2):
            neighbors[el].add(parent2[(i - 1) % len(parent2)])
            neighbors[el].add(parent2[(i + 1) % len(parent2)])
        child_el = self.rng.choice(np.array(list(neighbors.keys())))
        child = [child_el]
        while len(child) < len(parent1):
            child_el_neighbors = neighbors.pop(child_el)
            for el, el_neighbors in neighbors.items():
                el_neighbors.discard(child_el)
            if child_el_neighbors:
                best_neighbors, best_n_neighbors = [], 0
                for neighbor in child_el_neighbors:
                    n_neighbors = len(neighbors[neighbor])
                    if n_neighbors > best_n_neighbors:
                        best_neighbors, best_n_neighbors = [neighbor], n_neighbors
                    elif n_neighbors == best_n_neighbors:
                        best_neighbors.append(neighbor)
                child_el = self.rng.choice(best_neighbors)
            else:
                child_el = self.rng.choice(np.array(list(neighbors.keys())))
            child.append(child_el)
        return child

    def cross(self, parent1, parent2):
        return self._cross(parent1, parent2), self._cross(parent1, parent2)
