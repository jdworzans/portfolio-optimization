import numpy as np

from nsga2.defaults import DEFAULT_RNG


class Mutation:
    def __init__(self, mutation_probability: float = 0.25, rng=DEFAULT_RNG):
        self.mutation_probability = mutation_probability
        self.rng = rng

    def mutate(self, element: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, population):
        for idx, _ in enumerate(population):
            if self.rng.random() < self.mutation_probability:
                population[idx, :] = self.mutate(population[idx, :])


class PolynomialMutation(Mutation):
    def __init__(
        self, a, b, eta=20.0, mutation_probability: float = 0.25, rng=DEFAULT_RNG
    ):
        super().__init__(mutation_probability, rng)
        self.a = a
        self.b = b
        self.eta_exp = 1 / (1 + eta)

    def mutate(self, element: np.ndarray) -> np.ndarray:
        """Based on 'A Combined Genetic Adaptive Search (GeneAS) for Engineering Design'"""
        u = self.rng.uniform(0, 1, len(element))
        delta = np.where(
            u < 0.5,
            np.power(2 * u, self.eta_exp) - 1,
            1 - np.power(2 * (1 - u), self.eta_exp),
        )
        diff = np.where(
            u < 0.5,
            element - self.a,
            self.b - element,
        )
        return element + delta * diff

class ReverseSequence(Mutation):
    def mutate(self, element: np.ndarray) -> np.ndarray:
        a = self.rng.choice(len(element), 2, False)
        i, j = a.min(), a.max()
        q = element.copy()
        q[i : j + 1] = q[i : j + 1][::-1]
        return q


class TransposeMutation(Mutation):
    def mutate(self, element: np.ndarray) -> np.ndarray:
        a = self.rng.choice(len(element), 2, False)
        q = element.copy()
        q[a] = q[a[::-1]]
        return q
