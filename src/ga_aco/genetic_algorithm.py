"""
genetic_algorithm.py
--------------------
Genetic Algorithm for global routing path exploration.
Implements Section 5.2 of the paper.

Objective function (Eq. 1):
    F = α₁·T + α₂·PDR − α₃·D − α₄·L − α₅·E
"""

import numpy as np
import random
from typing import List, Tuple
import networkx as nx
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.network_graph import get_all_paths, compute_path_qos


class GeneticAlgorithm:
    """
    GA-based global path optimizer.

    Parameters
    ----------
    G : nx.DiGraph
        Network graph with QoS edge attributes
    source : int
        Source node
    target : int
        Destination node
    population_size : int
    generations : int
    p_crossover : float
    p_mutation : float
    alpha : list[float]
        [α₁, α₂, α₃, α₄, α₅] — objective function weights (Eq. 1)
    seed : int
    """

    def __init__(self,
                 G: nx.DiGraph,
                 source: int,
                 target: int,
                 population_size: int = 50,
                 generations: int = 100,
                 p_crossover: float = 0.7,
                 p_mutation: float = 0.1,
                 alpha: list = None,
                 seed: int = 42):

        self.G              = G
        self.source         = source
        self.target         = target
        self.population_size = population_size
        self.generations    = generations
        self.p_crossover    = p_crossover
        self.p_mutation     = p_mutation
        self.alpha          = alpha or [0.3, 0.25, 0.2, 0.15, 0.1]
        self.seed           = seed
        self.rng            = np.random.RandomState(seed)
        random.seed(seed)

        # Candidate paths — chromosome pool
        self._all_paths = get_all_paths(G, source, target, cutoff=6)
        if not self._all_paths:
            raise ValueError(f"No paths found from {source} to {target}")

    def fitness(self, path: list) -> float:
        """
        Evaluate fitness using multi-objective function (Eq. 1).

        F = α₁·T + α₂·PDR − α₃·D − α₄·L − α₅·E

        QoS values are normalized to [0, 1].
        """
        qos = compute_path_qos(self.G, path)
        if not qos:
            return -float('inf')

        # Normalize (higher bandwidth → higher T, higher delay → worse)
        T   = min(qos['bandwidth'] / 100.0, 1.0)
        PDR = 1.0 - qos['loss_prob']
        D   = min(qos['delay'] / 500.0, 1.0)
        L   = qos['loss_prob']
        E   = min(qos['energy'] / 10.0, 1.0)

        a = self.alpha
        return a[0]*T + a[1]*PDR - a[2]*D - a[3]*L - a[4]*E

    def _initialize_population(self) -> list:
        """Random selection of paths as initial population."""
        population = []
        for _ in range(self.population_size):
            path = random.choice(self._all_paths)
            population.append(path)
        return population

    def _select(self, population: list, fitnesses: list) -> list:
        """Tournament selection."""
        selected = []
        k = 3  # tournament size
        for _ in range(len(population)):
            contenders = random.sample(list(enumerate(fitnesses)), k)
            winner = max(contenders, key=lambda x: x[1])[0]
            selected.append(population[winner])
        return selected

    def _crossover(self, p1: list, p2: list) -> Tuple[list, list]:
        """
        Single-point crossover on shared intermediate nodes.
        Falls back to returning parents if no shared nodes.
        """
        shared = set(p1[1:-1]) & set(p2[1:-1])
        if not shared or self.rng.random() > self.p_crossover:
            return p1, p2

        point = random.choice(list(shared))
        idx1 = p1.index(point)
        idx2 = p2.index(point)

        child1 = p1[:idx1] + p2[idx2:]
        child2 = p2[:idx2] + p1[idx1:]
        return child1, child2

    def _mutate(self, path: list, p_cong: float = 0.0) -> list:
        """
        Mutation: replace a random subpath with an alternative route.

        Mutation probability increases with predicted congestion (Section 5.2):
            p_m = p_m + 0.05 if P_cong >= 0.7
        """
        pm = self.p_mutation
        if p_cong >= 0.7:
            pm = min(self.p_mutation + 0.05, 0.25)

        if self.rng.random() > pm or len(path) <= 2:
            return path

        # Pick a random intermediate node and reroute
        if len(path) > 2:
            cut = random.randint(1, len(path) - 2)
            sub_paths = get_all_paths(self.G, path[cut], self.target, cutoff=4)
            if sub_paths:
                alt = random.choice(sub_paths)
                return path[:cut] + alt
        return path

    def evolve(self, p_cong: float = 0.0) -> List[list]:
        """
        Run GA evolution and return top candidate paths.

        Parameters
        ----------
        p_cong : float
            Current CatBoost-predicted congestion probability.
            Used to dynamically adjust mutation rate (Section 5.2).

        Returns
        -------
        List of best candidate paths (C_GA)
        """
        population = self._initialize_population()

        for gen in range(self.generations):
            fitnesses  = [self.fitness(p) for p in population]
            selected   = self._select(population, fitnesses)
            next_gen   = []

            for i in range(0, len(selected) - 1, 2):
                c1, c2 = self._crossover(selected[i], selected[i+1])
                next_gen.append(self._mutate(c1, p_cong))
                next_gen.append(self._mutate(c2, p_cong))

            # Elitism: keep best individual
            best_idx = int(np.argmax(fitnesses))
            next_gen[0] = population[best_idx]
            population = next_gen

        # Return top-5 unique paths sorted by fitness
        fitnesses = [self.fitness(p) for p in population]
        ranked    = sorted(zip(fitnesses, population), reverse=True,
                           key=lambda x: x[0])
        seen, candidates = set(), []
        for fit, path in ranked:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)
            if len(candidates) == 5:
                break

        return candidates
