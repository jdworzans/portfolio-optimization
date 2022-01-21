from collections import Counter

import numpy as np
from tqdm import tqdm

from nsga2 import crossovers, defaults, mutations
from nsga2.selection import selection

DEFAULT_POPULATION_SIZE = 500
DEGAULT_N_ITERATIONS = 250
DEFAULT_SGA_ITERATIONS = 3

def get_portfolios_vars(weights, SIGMA):
    return np.einsum("...i,ij,...j", weights, SIGMA, weights)

def get_portfolios_returns(weights, R):
    return weights @ R
    return np.einsum("i,i...", R, weights)

SIGMA = np.eye(18)
R = np.ones(18)


def get_iter(iterable, use_tqdm: bool = False):
    if use_tqdm:
        return tqdm(iterable)
    else:
        return iterable

class NSGA2:
    def __init__(
        self,
        chromosome_length: int,
        population_size: int = DEFAULT_POPULATION_SIZE,
        n_iterations: int = DEGAULT_N_ITERATIONS,
        crossover: crossovers.Crossover = None,
        mutation: mutations.Mutation = None,
        rng = defaults.DEFAULT_RNG,
    ):
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.crossover = crossovers.PMX() if crossover is None else crossover
        self.mutation = mutations.ReverseSequence() if mutation is None else mutation
        self.rng = rng

    def get_initial_population(self, chromosome_length):
        population = self.rng.random((self.population_size, chromosome_length))
        normalized_population = population / population.sum(axis=-1, keepdims=True)
        return normalized_population

    def evaluate(self, population):
        vars = get_portfolios_vars(population, SIGMA)
        returns = get_portfolios_returns(population, R)
        return np.column_stack([returns, vars])

    def parent_selection(self, objective_values):
        fitness_values = objective_values.max(axis=0) - objective_values
        fitness_sum = fitness_values.sum(axis=0)
        if (fitness_sum > 0).any():
            fitness_values = (fitness_values / fitness_sum).mean(axis=-1)
            return self.rng.choice(
                self.population_size, self.population_size, True, fitness_values
            )
        else:
            return self.rng.choice(self.population_size, self.population_size, True)


    def select_new_population(self, objective_values):
        return np.array(selection(objective_values, self.population_size))

    def simulate(self, progress=True):
        objective_history = []
        current_population = self.get_initial_population(self.chromosome_length)
        objective_values = self.evaluate(current_population)

        for t in get_iter(range(self.n_iterations), progress):
            parent_indices = self.parent_selection(objective_values)
            parent_population = current_population[parent_indices, :]
            children_population = self.crossover(parent_population)
            self.mutation(children_population)
            children_objective_values = self.evaluate(children_population)

            objective_values = np.concatenate([objective_values, children_objective_values])
            current_population = np.concatenate([current_population, children_population])

            I = self.select_new_population(objective_values)
            current_population = current_population[I]
            objective_values = objective_values[I]

            objective_history.append(objective_values)

        return np.array(objective_history)

    # def repeat(self, N: int = DEFAULT_SGA_ITERATIONS, progress=True):
    #     objectives = np.zeros((N, self.n_iterations, self.population_size))
    #     it = tqdm(range(N)) if progress else range(N)
    #     for n in it:
    #         new_objectives = self.simulate(progress=False)
    #         objectives[n, :] = new_objectives
    #     return objectives