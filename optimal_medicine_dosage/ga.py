"""
ga.py
Implements the Genetic Algorithm core logic:
population init, fitness evaluation, selection,
crossover, mutation, and iteration loop.
"""

import numpy as np
import random
from fitness_functions import effectiveness, side_effects

DOSE_MIN, DOSE_MAX = 0.0, 100.0

def initialize_population(pop_size, rng):
    """Randomly initialize population of shape (pop_size, 3)."""
    return rng.uniform(DOSE_MIN, DOSE_MAX, size=(pop_size, 3))

def compute_fitness(population):
    """Compute fitness = effectiveness - side_effects."""
    eff = effectiveness(population)
    se = side_effects(population)
    return eff - se

def select_elite(population, fitness, elite_size):
    """Return the top elite individuals."""
    idx = np.argsort(fitness)[-elite_size:][::-1]
    return population[idx].copy()

def tournament_selection(population, fitness, k, rng):
    """Pick k random individuals, return the fittest."""
    idx = rng.integers(0, len(population), size=k)
    best = idx[np.argmax(fitness[idx])]
    return population[best].copy()

def uniform_crossover(parent_a, parent_b, rng):
    """Uniform crossover: gene-wise 50% swap."""
    mask = rng.random(3) < 0.5
    return np.where(mask, parent_a, parent_b)

def mutate(individual, mutation_std, rng):
    """Add Gaussian noise and clip within range."""
    mutated = individual + rng.normal(0, mutation_std, 3)
    return np.clip(mutated, DOSE_MIN, DOSE_MAX)

def run_ga(
    pop_size=50, generations=150, elite_frac=0.1,
    tournament_k=3, crossover_rate=0.9, mutation_std=2.0,
    rng_seed=42
):
    """Main GA loop."""
    rng = np.random.default_rng(rng_seed)
    random.seed(rng_seed)

    population = initialize_population(pop_size, rng)
    elite_size = max(1, int(elite_frac * pop_size))

    best_fit, best_ind = -np.inf, None
    fitness_history = []

    for gen in range(generations):
        fitness = compute_fitness(population)

        # Update global best
        idx = np.argmax(fitness)
        if fitness[idx] > best_fit:
            best_fit, best_ind = fitness[idx], population[idx].copy()
        fitness_history.append(best_fit)

        # Elitism
        elites = select_elite(population, fitness, elite_size)

        # Create next gen
        new_pop = [el for el in elites]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness, tournament_k, rng)
            p2 = tournament_selection(population, fitness, tournament_k, rng)

            child = (
                uniform_crossover(p1, p2, rng)
                if rng.random() < crossover_rate else p1.copy()
            )
            child = mutate(child, mutation_std, rng)
            new_pop.append(child)

        population = np.array(new_pop)

    return best_ind, best_fit, fitness_history, population
