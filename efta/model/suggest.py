"""
efta.model.suggest
==================
Bootstrap model selection via repeated random train/test splits.

:func:`suggest`
    Repeated ``model()`` runs with randomised train/test splits.

:func:`suggest_chem`
    Repeated ``model_chem()`` runs with randomised train/test splits.

Each iteration:
1. Randomly splits data into train (fraction = *training_points*) and test.
2. Fits model on train data.
3. Evaluates MSE on train and MTE (mean test error) on test data.
4. Computes t-test (parameter equality train vs test) and Levene test
   (equal error variance train vs test).

Returns all iterations sorted/filtered by *based_on* criterion.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ttest_ind, levene

from .fitting import (_model, _model_chem,
                      Model, _build_objective,
                      _validate_data, _param_names)
from ..errors import InputError

__all__ = ['_suggest', '_suggest_chem', 'Analyzed']


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

class Analyzed:
    """
    Result of one bootstrap iteration from :func:`suggest` / :func:`suggest_chem`.

    Attributes
    ----------
    fit : Model
        The fitted model for this iteration.
    mse : float
        Mean squared error on training data.
    mte : float
        Mean test error (MSE on held-out test data).
    t_stat : float
        t-statistic from Welch's t-test comparing train vs test residuals.
    t_pval : float
        p-value of the t-test.  High p → train and test parameters agree.
    lev_stat : float
        Levene test statistic comparing error variance of train vs test.
    lev_pval : float
        p-value of Levene test.  High p → equal variance.
    train_idx : np.ndarray
        Indices of data points used for training.
    test_idx : np.ndarray
        Indices of data points used for testing.
    distributions : list of list
        Parameter distributions across all n iterations as
        ``[[p1_iter1, p1_iter2, ...], [p2_iter1, p2_iter2, ...], ...]``.
        One inner list per parameter, in the same order as
        ``fit.param_names``.
    mse_dist : list of float
        MSE on training data for every iteration.
    mte_dist : list of float
        MTE on test data for every iteration.
    t_dist : list of float
        t-test p-values for every iteration.
    lev_dist : list of float
        Levene test p-values for every iteration.
    """

    def __init__(self, fit, mse, mte, t_stat, t_pval,
                 lev_stat, lev_pval, train_idx, test_idx,
                 distributions=None,
                 mse_dist=None, mte_dist=None,
                 t_dist=None, lev_dist=None,
                 param_std=None, n_matched=None):
        self.fit           = fit
        self.mse           = float(mse)
        self.mte           = float(mte)
        self.t_stat        = float(t_stat)
        self.t_pval        = float(t_pval)
        self.lev_stat      = float(lev_stat)
        self.lev_pval      = float(lev_pval)
        self.train_idx     = np.asarray(train_idx)
        self.test_idx      = np.asarray(test_idx)
        self.distributions = distributions or []
        self.mse_dist      = mse_dist  or []
        self.mte_dist      = mte_dist  or []
        self.t_dist        = t_dist    or []
        self.lev_dist      = lev_dist  or []
        self.param_std     = dict(param_std)  if param_std  is not None else {}
        self.n_matched     = int(n_matched)   if n_matched  is not None else None

    def __repr__(self) -> str:
        return (f"Analyzed(popt={tuple(f'{v:.4g}' for v in self.fit.popt)}, "
                f"mse={self.mse:.3e}, mte={self.mte:.3e}, "
                f"t_pval={self.t_pval:.3f}, lev_pval={self.lev_pval:.3f})")

    def __str__(self) -> str:
        lines = ['Analyzed']
        if self.n_matched is not None:
            lines.append(f"  n_matched : {self.n_matched}")
        lines.append(f"  mse      : {self.mse:.4e}  (training)")
        lines.append(f"  mte      : {self.mte:.4e}  (test)")
        lines.append(f"  t-test   : stat={self.t_stat:.4f}  p={self.t_pval:.4f}")
        lines.append(f"  levene   : stat={self.lev_stat:.4f}  p={self.lev_pval:.4f}")
        if self.n_matched is None:
            lines.append(f"  n_train  : {len(self.train_idx)}")
            lines.append(f"  n_test   : {len(self.test_idx)}")
        lines.append("  parameters:")
        for n, v in zip(self.fit.param_names, self.fit.popt):
            std_str = (f"  ± {self.param_std[n]:.3e}"
                       if self.param_std and n in self.param_std else "")
            lines.append(f"    {n:<12s} = {v:.6g}{std_str}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split(n_points: int, training_points: float,
           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for a random split."""
    idx = rng.permutation(n_points)
    n_train = max(1, int(round(n_points * training_points)))
    n_train = min(n_train, n_points - 1)   # at least 1 test point
    return idx[:n_train], idx[n_train:]


def _subset(data_list: List[dict], idx: np.ndarray) -> List[dict]:
    """Subset a list of dicts by index array."""
    return [data_list[i] for i in idx]


def _eval_residuals(func, params, data_points, initials,
                    eq_species, maintain):
    """Evaluate normalised residuals for a set of data points."""
    from .mass_action import mass_action_residuals, is_data_sufficient
    from .fitting import _resolve_parametric_keys
    res = []
    for i, (ceq_i, c0_i) in enumerate(zip(data_points, initials)):
        try:
            rxn_sys = func(*params)
            ceq_r   = _resolve_parametric_keys(ceq_i, params)
            c0_r    = _resolve_parametric_keys(c0_i,  params)
            if is_data_sufficient(rxn_sys, ceq_r, c0_r):
                r = mass_action_residuals(rxn_sys, ceq_r, c0_r)
            else:
                ceq = rxn_sys.equilibrium(
                    c0_r if c0_r else ceq_r, maintain=maintain)
                r = np.array([
                    (ceq.get(sp, 0.0) - ceq_r.get(sp, 0.0)) / max(ceq_r.get(sp, 1e-300), 1e-300)
                    for sp in eq_species
                ])
            res.extend(r.tolist())
        except Exception:
            res.extend([1e6] * max(len(eq_species), 1))
    return np.array(res, dtype=float)


def _stat_tests(train_res: np.ndarray,
                test_res: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute t-test and Levene test between train and test residuals.
    Returns (t_stat, t_pval, lev_stat, lev_pval).
    """
    if len(train_res) < 2 or len(test_res) < 2:
        return 0.0, 1.0, 0.0, 1.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t_res   = ttest_ind(train_res, test_res, equal_var=False)
        lev_res = levene(train_res, test_res)

    return (float(t_res.statistic),  float(t_res.pvalue),
            float(lev_res.statistic), float(lev_res.pvalue))


def _sort_key(r: Analyzed, based_on: str) -> float:
    """Return the sort key for a Analyzed (lower = better)."""
    if based_on == 'mse':
        return r.mse
    if based_on == 'mte':
        return r.mte
    if based_on == 't':
        return -r.t_pval    # high p-value = good (parameters agree)
    if based_on == 'lev':
        return -r.lev_pval  # high p-value = good (equal variance)
    raise InputError(f"based_on must be 'mse', 'mte', 't', or 'lev', got {based_on!r}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _suggest(func,
            equilibrium: dict,
            initial: dict = None,
            maintain: list = None,
            *,
            n: int = 100,
            training_points: float = 0.8,
            best: int = 1,
            based_on: str = 'mse',
            x0: dict = None,
            seed: int = 42,
            v_oa: float = 1.0,
            tolerance: float = 1e-8,
            max_iter: int = 5000,
            **kw) -> List[Analyzed]:
    """
    Repeated :func:`model` runs with randomised train/test splits.

    Parameters
    ----------
    func : callable
        ``func(*params) -> reaction | reactions``
    equilibrium : dict
        Observed equilibrium concentrations (dict of lists).
    initial : dict, optional
        Initial concentrations. If None, uses equilibrium as c0.
    maintain : list, optional
        Species held fixed during solving.
    n : int, optional
        Number of bootstrap iterations (default 100).
    training_points : float, optional
        Fraction of data used for training, 0 < training_points < 1
        (default 0.8).
    best : int, optional
        Number of best results to return (default 1).
    based_on : str, optional
        Criterion for ranking: ``'mse'``, ``'mte'``, ``'t'``, ``'lev'``.
    x0 : dict, optional
        Initial parameter guess.
    seed : int, optional
        Random seed for reproducibility.
    tolerance, max_iter : passed to :func:`model`.

    Returns
    -------
    list of Analyzed
        The *best* top results, sorted by *based_on*.

    Examples
    --------
    >>> results = suggest(f, equilibrium=full_eq, n=50, training_points=0.7)
    >>> print(results[0])   # best model
    """
    _validate_based_on(based_on)
    _validate_training_points(training_points)

    maintain = maintain or []
    initials, equilibria = _validate_data(
        initial if initial is not None else equilibrium, equilibrium)
    n_points   = len(initials)
    eq_species = sorted({sp for eq in equilibria for sp in eq})
    names      = _param_names(func)
    rng        = np.random.default_rng(seed)

    results = []
    for i in range(n):
        train_idx, test_idx = _split(n_points, training_points, rng)

        train_eq  = _subset(equilibria, train_idx)
        train_c0  = _subset(initials,   train_idx)
        test_eq   = _subset(equilibria, test_idx)
        test_c0   = _subset(initials,   test_idx)

        # fit on train
        try:
            fit = _model(func,
                         _lists_to_dict(train_eq),
                         _lists_to_dict(train_c0),
                         maintain=maintain,
                         x0=x0, tolerance=tolerance,
                         max_iter=max_iter, **kw)
        except Exception:
            continue

        # evaluate on train
        train_res = _eval_residuals(
            func, fit.popt, train_eq, train_c0, eq_species, maintain)
        # evaluate on test
        test_res  = _eval_residuals(
            func, fit.popt, test_eq,  test_c0,  eq_species, maintain)

        mse = float(np.mean(train_res ** 2))
        mte = float(np.mean(test_res  ** 2))
        t_stat, t_pval, lev_stat, lev_pval = _stat_tests(train_res, test_res)

        results.append(Analyzed(
            fit=fit, mse=mse, mte=mte,
            t_stat=t_stat, t_pval=t_pval,
            lev_stat=lev_stat, lev_pval=lev_pval,
            train_idx=train_idx, test_idx=test_idx,
        ))

    if not results:
        raise RuntimeError("All fitting iterations failed.")

    results.sort(key=lambda r: _sort_key(r, based_on))

    # build distributions from all iterations
    n_params  = len(names)
    dists     = [[] for _ in range(n_params)]
    mse_dist  = []
    mte_dist  = []
    t_dist    = []
    lev_dist  = []
    for r in results:
        for i, v in enumerate(r.fit.popt):
            dists[i].append(v)
        mse_dist.append(r.mse)
        mte_dist.append(r.mte)
        t_dist.append(r.t_pval)
        lev_dist.append(r.lev_pval)
    for r in results[:best]:
        r.distributions = [list(d) for d in dists]
        r.mse_dist  = list(mse_dist)
        r.mte_dist  = list(mte_dist)
        r.t_dist    = list(t_dist)
        r.lev_dist  = list(lev_dist)

    return results[:best]


def _suggest_chem(func,
                 equilibrium: dict,
                 initial: dict = None,
                 maintain: list = None,
                 *,
                 n: int = 100,
                 training_points: float = 0.8,
                 best: int = 1,
                 based_on: str = 'mse',
                 seed: int = 42,
                 tolerance: float = 1e-8,
                 generations: int = 100,
                 mating: int = 10,
                 v_oa: float = 1.0,
                 **param_constraints) -> List[Analyzed]:
    """
    Repeated :func:`model_chem` runs with randomised train/test splits.

    Same interface as :func:`suggest` but forwards ``acceptable_*``
    keyword arguments to :func:`model_chem`.

    Parameters
    ----------
    func : callable
    equilibrium : dict
    initial : dict, optional
    maintain : list, optional
    n : int, optional
        Number of bootstrap iterations (default 100).
    training_points : float, optional
        Training fraction, 0 < training_points < 1 (default 0.8).
    best : int, optional
        Number of top results to return (default 1).
    based_on : str, optional
        ``'mse'``, ``'mte'``, ``'t'``, or ``'lev'``.
    seed : int, optional
    generations : int, optional
        GA generations per fit (default 100).
    mating : int, optional
        Parents mating per generation (default 10).
    **param_constraints
        ``acceptable_{name}`` constraints forwarded to :func:`model_chem`.

    Returns
    -------
    list of Analyzed
    """
    _validate_based_on(based_on)
    _validate_training_points(training_points)

    maintain = maintain or []
    initials, equilibria = _validate_data(
        initial if initial is not None else equilibrium, equilibrium)
    n_points   = len(initials)
    eq_species = sorted({sp for eq in equilibria for sp in eq})
    rng        = np.random.default_rng(seed)

    results = []
    for i in range(n):
        train_idx, test_idx = _split(n_points, training_points, rng)

        train_eq = _subset(equilibria, train_idx)
        train_c0 = _subset(initials,   train_idx)
        test_eq  = _subset(equilibria, test_idx)
        test_c0  = _subset(initials,   test_idx)

        try:
            fit = _model_chem(func,
                              _lists_to_dict(train_eq),
                              _lists_to_dict(train_c0),
                              maintain=maintain,
                              tolerance=tolerance,
                              generations=generations,
                              mating=mating,
                              random_seed=seed + i,
                              v_oa=v_oa,
                              **param_constraints)
        except Exception:
            continue

        train_res = _eval_residuals(
            func, fit.popt, train_eq, train_c0, eq_species, maintain)
        test_res  = _eval_residuals(
            func, fit.popt, test_eq,  test_c0,  eq_species, maintain)

        mse = float(np.mean(train_res ** 2))
        mte = float(np.mean(test_res  ** 2))
        t_stat, t_pval, lev_stat, lev_pval = _stat_tests(train_res, test_res)

        results.append(Analyzed(
            fit=fit, mse=mse, mte=mte,
            t_stat=t_stat, t_pval=t_pval,
            lev_stat=lev_stat, lev_pval=lev_pval,
            train_idx=train_idx, test_idx=test_idx,
        ))

    if not results:
        raise RuntimeError("All fitting iterations failed.")

    results.sort(key=lambda r: _sort_key(r, based_on))

    # build distributions from all iterations
    n_params  = len(_param_names(func))
    dists     = [[] for _ in range(n_params)]
    mse_dist  = []
    mte_dist  = []
    t_dist    = []
    lev_dist  = []
    for r in results:
        for i, v in enumerate(r.fit.popt):
            dists[i].append(v)
        mse_dist.append(r.mse)
        mte_dist.append(r.mte)
        t_dist.append(r.t_pval)
        lev_dist.append(r.lev_pval)
    for r in results[:best]:
        r.distributions = [list(d) for d in dists]
        r.mse_dist  = list(mse_dist)
        r.mte_dist  = list(mte_dist)
        r.t_dist    = list(t_dist)
        r.lev_dist  = list(lev_dist)

    return results[:best]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lists_to_dict(list_of_dicts: List[dict]) -> dict:
    """Convert list of per-point dicts back to dict-of-lists."""
    if not list_of_dicts:
        return {}
    keys = list_of_dicts[0].keys()
    return {k: [d[k] for d in list_of_dicts] for k in keys}


def _validate_based_on(based_on: str) -> None:
    if based_on not in ('mse', 'mte', 't', 'lev'):
        raise InputError(
            f"based_on must be 'mse', 'mte', 't', or 'lev', got {based_on!r}.")


def _validate_training_points(tp: float) -> None:
    if not (0 < tp < 1):
        raise InputError(
            f"training_points must be strictly between 0 and 1, got {tp}.")


# ---------------------------------------------------------------------------
# MonteCarlo
# ---------------------------------------------------------------------------

class MonteCarlo:
    """
    Result of a Monte Carlo uncertainty analysis from :func:`montecarlo`.

    Attributes
    ----------
    fit : Model
        Best-fit parameters (from the iteration with highest Levene p-value).
    param_std : dict
        Standard deviation of each **continuous** parameter across matched
        iterations.  For ``method='cont'`` all params are included.
        For ``method='ranged'`` only tuple-constrained (continuous) params
        are included; discrete params are used for filtering only.
    distributions : dict
        All n iterations' parameter values: ``{name: [v_iter1, ...]}``.
    lev_dist : list of float
        Levene p-values for all n iterations.
    n_matched : int
        Number of iterations whose discrete params matched the best result.
        For ``method='cont'`` this equals n.
    """

    def __init__(self, fit, param_std, distributions, lev_dist, n_matched):
        self.fit           = fit
        self.param_std     = dict(param_std)
        self.distributions = dict(distributions)
        self.lev_dist      = list(lev_dist)
        self.n_matched     = int(n_matched)

    def __repr__(self) -> str:
        popt = ', '.join(f'{n}={v:.4g}'
                         for n, v in zip(self.fit.param_names, self.fit.popt))
        std  = ', '.join(f'{n}={v:.3e}'
                         for n, v in self.param_std.items())
        return f"MonteCarlo({popt} | std: {std} | n_matched={self.n_matched})"

    def __str__(self) -> str:
        lines = ['MonteCarlo']
        lines.append(f"  n_matched : {self.n_matched}")
        lines.append("  best-fit parameters:")
        for n, v in zip(self.fit.param_names, self.fit.popt):
            std_str = (f"  ± {self.param_std[n]:.3e}"
                       if n in self.param_std else "")
            lines.append(f"    {n:<12s} = {v:.6g}{std_str}")
        lines.append(f"  lev_pval (best) : {max(self.lev_dist):.4f}")
        return '\n'.join(lines)
