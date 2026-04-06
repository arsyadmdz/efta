"""
efta.model.ga
=============
Minimal genetic algorithm for mixed discrete/continuous parameter optimisation.

Pure numpy — no external dependencies beyond the efta stack.

Gene spec
---------
``gene_specs`` is a dict ``{name: spec}`` or list of ``(name, spec)`` pairs:

- ``list``  → discrete gene, e.g. ``'a': [1, 1.5, 2, 2.5, 3]``
- ``tuple`` → continuous gene (lo, hi), e.g. ``'K': (1e-5, 1e5)``

The fitness function receives a ``dict {name: value}`` and should return a
float.  The GA **maximises** fitness.

Examples
--------
>>> ga = GA(
...     fitness_func = lambda p: -(p['x'] - 3)**2 - (p['K'] - 2)**2,
...     gene_specs   = {'x': [1, 2, 3, 4, 5], 'K': (0.1, 10.0)},
...     generations  = 50,
...     seed         = 42,
... )
>>> sol = ga.run()
>>> sol.params   # {'x': 3.0, 'K': 2.01}
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union
import numpy as np

__all__ = ['GA', 'GASolution']


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class GASolution:
    """
    Result of a :class:`GA` run.

    Attributes
    ----------
    params : dict
        Best solution as ``{name: value}`` in direct (decoded) space.
    fitness : float
        Fitness value of the best solution (higher = better).
    generation : int
        Generation at which the best solution was found.
    n_evals : int
        Total number of fitness evaluations.
    """

    def __init__(self, params: dict, fitness: float,
                 generation: int, n_evals: int):
        self.params     = dict(params)
        self.fitness    = float(fitness)
        self.generation = int(generation)
        self.n_evals    = int(n_evals)

    def __repr__(self) -> str:
        p = ', '.join(f'{k}={v:.4g}' for k, v in self.params.items())
        return (f"GASolution({p}, fitness={self.fitness:.4e}, "
                f"gen={self.generation}, n_evals={self.n_evals})")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_specs(gene_specs) -> List[tuple]:
    """
    Normalise gene_specs to a list of ``(name, kind, spec)`` triples.

    ``kind`` is ``'discrete'`` or ``'continuous'``, inferred from spec type:
    - list  → discrete
    - tuple → continuous (lo, hi)
    """
    if isinstance(gene_specs, dict):
        items = list(gene_specs.items())
    else:
        items = list(gene_specs)   # list of (name, spec) pairs

    parsed = []
    for name, spec in items:
        if isinstance(spec, list):
            if not spec:
                raise ValueError(f"Gene '{name}': discrete list must not be empty.")
            parsed.append((name, 'discrete', list(spec)))
        elif isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = float(spec[0]), float(spec[1])
            if lo < 0:
                raise ValueError(
                    f"Gene '{name}': continuous range must be positive, got ({lo}, {hi}).")
            if lo == 0:
                lo = 1e-300
            if lo >= hi:
                raise ValueError(
                    f"Gene '{name}': lo={lo} >= hi={hi}.")
            parsed.append((name, 'continuous', (lo, hi)))
        else:
            raise ValueError(
                f"Gene '{name}': spec must be a list (discrete) or "
                f"(lo, hi) tuple (continuous), got {type(spec).__name__!r}.")
    return parsed


def _log(lo: float, hi: float):
    return np.log(max(lo, 1e-300)), np.log(max(hi, lo + 1e-300))


def _random_individual(parsed: list, rng: np.random.Generator) -> np.ndarray:
    """Sample a random individual in encoded space."""
    ind = np.empty(len(parsed))
    for i, (_, kind, spec) in enumerate(parsed):
        if kind == 'continuous':
            lo, hi  = spec
            ind[i]  = rng.uniform(*_log(lo, hi))
        else:
            ind[i]  = float(rng.integers(0, len(spec)))
    return ind


def _decode_individual(ind: np.ndarray, parsed: list) -> dict:
    """Decode encoded individual → {name: value} dict."""
    result = {}
    for i, (name, kind, spec) in enumerate(parsed):
        if kind == 'continuous':
            result[name] = float(np.exp(ind[i]))
        else:
            idx          = max(0, min(int(round(ind[i])), len(spec) - 1))
            result[name] = float(spec[idx])
    return result


def _clip_individual(ind: np.ndarray, parsed: list) -> np.ndarray:
    """Clip encoded individual to valid bounds."""
    clipped = ind.copy()
    for i, (_, kind, spec) in enumerate(parsed):
        if kind == 'continuous':
            lo, hi       = _log(*spec)
            clipped[i]   = np.clip(ind[i], lo, hi)
        else:
            clipped[i]   = np.clip(round(ind[i]), 0, len(spec) - 1)
    return clipped


# ---------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------

class GA:
    """
    Genetic algorithm for mixed discrete/continuous optimisation.

    Parameters
    ----------
    fitness_func : callable
        ``fitness_func(params: dict) -> float``.
        Receives ``{name: value}`` in direct space.  GA **maximises** this.
    gene_specs : dict or list of (name, spec)
        Per-gene specification.  Type inferred from spec:

        - ``list``  → discrete, e.g. ``'a': [1, 1.5, 2, 2.5, 3]``
        - ``tuple`` → continuous ``(lo, hi)``, e.g. ``'K': (1e-5, 1e5)``

        Both lo and hi must be positive (genes are log-transformed internally).
    generations : int
        Number of generations (default 100).
    pop_size : int
        Population size (default 20).
    n_parents : int
        Parents selected each generation (default 10).
    mutation_prob : float
        Per-gene mutation probability (default 0.15).
    early_stop : float, optional
        Stop early when fitness ≥ this value.
    seed : int, optional
        RNG seed.

    Examples
    --------
    >>> ga = GA(
    ...     fitness_func = lambda p: -(p['x'] - 3)**2 - (p['K'] - 0.25)**2,
    ...     gene_specs   = {'x': [1, 2, 3], 'K': (0.01, 10.0)},
    ...     generations  = 50,
    ...     seed         = 42,
    ... )
    >>> sol = ga.run()
    >>> sol.params    # {'x': 3.0, 'K': 0.25...}
    >>> sol.fitness
    >>> sol.n_evals
    """

    def __init__(self,
                 fitness_func:  Callable,
                 gene_specs,
                 generations:   int   = 100,
                 pop_size:      int   = 20,
                 n_parents:     int   = 10,
                 mutation_prob: float = 0.15,
                 early_stop:    Optional[float] = None,
                 seed:          Optional[int]   = None):

        self.fitness_func  = fitness_func
        self._parsed       = _parse_specs(gene_specs)
        self.generations   = generations
        self.pop_size      = pop_size
        self.n_parents     = min(n_parents, pop_size)
        self.mutation_prob = mutation_prob
        self.early_stop    = early_stop
        self.seed          = seed
        self._n_genes      = len(self._parsed)

    # ── public ───────────────────────────────────────────────────────────────

    def run(self) -> GASolution:
        """Run the GA and return the best :class:`GASolution`."""
        rng     = np.random.default_rng(self.seed)
        n_evals = 0

        # initialise population
        pop = np.array([_random_individual(self._parsed, rng)
                        for _ in range(self.pop_size)])

        def _eval_pop(population):
            nonlocal n_evals
            scores = np.empty(len(population))
            for j, ind in enumerate(population):
                params    = _decode_individual(ind, self._parsed)
                scores[j] = float(self.fitness_func(params))
                n_evals  += 1
            return scores

        fitness  = _eval_pop(pop)
        best_idx = int(np.argmax(fitness))
        best_ind = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])
        best_gen = 0

        for gen in range(self.generations):
            if self.early_stop is not None and best_fit >= self.early_stop:
                break

            parents  = self._select(pop, fitness, rng)
            offspring = self._crossover(parents, rng)
            offspring = self._mutate(offspring, rng)

            off_fit  = _eval_pop(offspring)

            # elitism: preserve best solution found so far
            worst    = int(np.argmin(off_fit))
            if best_fit > off_fit[worst]:
                offspring[worst] = best_ind.copy()
                off_fit[worst]   = best_fit

            pop     = offspring
            fitness = off_fit

            gen_best = int(np.argmax(fitness))
            if fitness[gen_best] > best_fit:
                best_fit = float(fitness[gen_best])
                best_ind = pop[gen_best].copy()
                best_gen = gen + 1

        return GASolution(
            params     = _decode_individual(best_ind, self._parsed),
            fitness    = best_fit,
            generation = best_gen,
            n_evals    = n_evals,
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _select(self, pop, fitness, rng) -> np.ndarray:
        """Tournament selection."""
        n      = len(pop)
        t_size = max(2, n // self.n_parents)
        parents = []
        for _ in range(self.n_parents):
            idx    = rng.choice(n, size=t_size, replace=False)
            winner = idx[int(np.argmax(fitness[idx]))]
            parents.append(pop[winner].copy())
        return np.array(parents)

    def _crossover(self, parents, rng) -> np.ndarray:
        """Uniform crossover → pop_size offspring."""
        n        = len(parents)
        offspring = np.empty((self.pop_size, self._n_genes))
        for i in range(self.pop_size):
            p1            = parents[i % n]
            p2            = parents[(i + 1) % n]
            mask          = rng.random(self._n_genes) < 0.5
            offspring[i]  = np.where(mask, p1, p2)
        return offspring

    def _mutate(self, offspring, rng) -> np.ndarray:
        """Per-gene adaptive mutation."""
        mutated = offspring.copy()
        for i in range(len(mutated)):
            for j, (_, kind, spec) in enumerate(self._parsed):
                if rng.random() < self.mutation_prob:
                    if kind == 'continuous':
                        log_lo, log_hi = _log(*spec)
                        sigma          = (log_hi - log_lo) * 0.1
                        mutated[i, j] += rng.normal(0, sigma)
                        mutated[i, j]  = np.clip(mutated[i, j], log_lo, log_hi)
                    else:
                        mutated[i, j] = float(rng.integers(0, len(spec)))
        return mutated
