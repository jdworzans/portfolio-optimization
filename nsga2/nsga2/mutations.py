from itertools import combinations, permutations

from scipy.special import comb, factorial

from nsga2.defaults import DEFAULT_RNG


class Mutation:
    def __init__(self, mutation_probability: float = 0.25, rng=DEFAULT_RNG):
        self.mutation_probability = mutation_probability
        self.rng = rng

    def mutate(self, element):
        raise NotImplementedError

    def __call__(self, population):
        for idx, _ in enumerate(population):
            if self.rng.random() < self.mutation_probability:
                population[idx, :] = self.mutate(population[idx, :])


class ReverseSequence(Mutation):
    def mutate(self, element):
        a = self.rng.choice(len(element), 2, False)
        i, j = a.min(), a.max()
        q = element.copy()
        q[i : j + 1] = q[i : j + 1][::-1]
        return q


class TransposeMutation(Mutation):
    def mutate(self, element):
        a = self.rng.choice(len(element), 2, False)
        q = element.copy()
        q[a] = q[a[::-1]]
        return q


class LocalSearchMutation(Mutation):
    def __init__(self, objective, K: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objective = objective
        self.K = K

    def mutate(self, element):
        assert factorial(self.K) * comb(len(element), self.K) < 10e9
        best, best_objective = element, self.objective(element)
        q = element.copy()
        for present_idxs in map(list, combinations(range(len(element)), self.K)):
            for target_idxs in map(list, permutations(present_idxs)):
                q[present_idxs] = element[target_idxs]
                val = self.objective(q)
                if val < best_objective:
                    best, best_objective = q.copy(), val
                q[present_idxs] = element[present_idxs]
        return best


class IteratedLocalSearchMutation(LocalSearchMutation):
    def mutate(self, element):
        best, best_objective = element, self.objective(element)
        candidate = super().mutate(element)
        candidate_objective = self.objective(candidate)
        while candidate_objective < best_objective:
            best, best_objective = candidate, candidate_objective
            candidate = super().mutate(best)
            candidate_objective = self.objective(candidate)
        return best
