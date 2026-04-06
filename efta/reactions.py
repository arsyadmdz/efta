"""
efta.reactions
==============
The :class:`reactions` class holds a collection of :class:`~efta.reaction.reaction`
objects and provides the primary user-facing API for chemical equilibrium calculations.

Usage
-----
::

    from efta import reaction, reactions

    # Define individual equilibrium reactions
    r1 = reaction('H[+] + OH[-] = H2O', 1e14)           # Kw (reverse)
    r2 = reaction('CH3COOH = CH3COO[-] + H[+]', 1.8e-5) # Ka of acetic acid

    # Combine into a coupled system
    sys = reactions(r1, r2)

    # Solve for equilibrium concentrations (mol/L)
    c_eq = sys.equilibrium({'CH3COOH': 0.1, 'H[+]': 1e-7, 'OH[-]': 1e-7})

    # Plot concentration vs swept initial concentration
    fig, ax = sys.plot(
        {'CH3COOH': [1e-4, 1.0], 'H[+]': 1e-7, 'OH[-]': 1e-7},
        logx=True, n_points=50,
    )

    # Find the initial concentration that achieves a target equilibrium
    c0_acid = sys.find(
        unknown='CH3COOH',
        c0={'H[+]': 1e-7, 'OH[-]': 1e-7},
        target={'H[+]': 1e-4},   # target pH 4
    )
"""

from __future__ import annotations

from .errors import InputError, ConcentrationError, ConvergenceError, ConvergenceWarning, warn_convergence

import re
from collections import defaultdict
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .species import species, formula, charge, components, is_nonaqueous, is_organic, is_electron
from .balance import _is_ksp_reaction, _system_is_mixed_phase, _compute_clusters, _compute_decompose
from .system import (
    _build_system, _build_gamma_for_system, _normalize_maintain,
    _conc_from_xi, _equilibrium_concentrations, total,
)
from .solver import _solve_with_ksp
from .reaction import reaction



def _c0_from_input(c0) -> dict:
    """
    Accept either a plain dict or a :class:`~efta.solution` instance as c0.
    Returns a plain dict suitable for passing to the solver.
    """
    from .solution import solution as _solution
    if isinstance(c0, _solution):
        d = dict(c0.concentrations)
        if c0.v_oa is not None:
            d['O/A'] = c0.v_oa
        return d
    return c0


class reactions:
    """
    A system of chemical equilibrium reactions.

    Parameters
    ----------
    *args : reaction objects (or list/tuple constructor args)

    Examples
    --------
    >>> sys = reactions(r1, r2, r3)
    >>> c_eq = sys.equilibrium({'Fe[3+]': 0.01, 'OH[-]': 1e-7, ...})
    """

    def __init__(self, *args):
        self._reactions: List[reaction] = []
        for arg in args:
            if isinstance(arg, reaction):
                self._reactions.append(arg)
            elif isinstance(arg, (list, tuple)):
                self._reactions.append(reaction(*arg))
            else:
                raise InputError(
                    f"Expected reaction or list/tuple, got {type(arg).__name__}")
        if not self._reactions:
            raise InputError("reactions() requires at least one reaction object.\n"
            "Pass reaction instances: reactions(r1, r2, ...).")

    # ------------------------------------------------------------------
    # Activity-coefficient interface (delegates to individual reactions)
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> Dict[str, tuple]:
        """Merged gamma dict from all reactions."""
        merged: Dict[str, tuple] = {}
        for r in self._reactions:
            merged.update(r._gamma)
        return merged

    def set_gamma(self, sp: str, func: Callable, *dep_species: str) -> 'reactions':
        """
        Register an activity-coefficient function for species *sp* in every
        reaction that involves it.
        """
        sp_norm = species(sp)
        stored = False
        for r in self._reactions:
            if sp_norm in r._stoich:
                r.set_gamma(sp, func, *dep_species)
                stored = True
        if not stored:
            self._reactions[0].set_gamma(sp, func, *dep_species)
        return self

    # ------------------------------------------------------------------
    # Convenience species-set properties
    # ------------------------------------------------------------------

    @property
    def species(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.species
        return out

    @property
    def aqueous_species(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.aqueous_species
        return out

    @property
    def organic_species(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.organic_species
        return out

    @property
    def nonaqueous_species(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.nonaqueous_species
        return out

    @property
    def reactants(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.reactants
        return out

    @property
    def products(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.products
        return out

    @property
    def components(self) -> FrozenSet[str]:
        out = frozenset()
        for r in self._reactions: out = out | r.components
        return out

    @property
    def is_mixed_phase(self) -> bool:
        return _system_is_mixed_phase(self._reactions)

    @property
    def type(self) -> str:
        """'speciation', 'precipitation', or 'mix'."""
        from .solver import _reactions_type
        return _reactions_type(self._reactions)

    @property
    def clusters(self) -> FrozenSet[str]:
        return _compute_clusters(self._reactions)

    @property
    def K(self) -> List[float]:
        return [r.K for r in self._reactions]

    @property
    def stoichiometry(self):
        return [r.stoichiometry for r in self._reactions]

    def decompose(self) -> Dict[str, Dict[str, float]]:
        return _compute_decompose(self._reactions)

    def balance(self) -> 'reactions':
        for r in self._reactions: r.balance()
        return self

    def copy(self) -> 'reactions':
        result = reactions.__new__(reactions)
        result._reactions = [r.copy() for r in self._reactions]
        return result

    # ------------------------------------------------------------------
    # Equality and containers
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, reactions): return NotImplemented
        if len(self._reactions) != len(other._reactions): return False
        used = [False] * len(other._reactions)
        for r in self._reactions:
            matched = False
            for j, s in enumerate(other._reactions):
                if not used[j] and r == s:
                    used[j] = True; matched = True; break
            if not matched: return False
        return True

    def __hash__(self) -> int:
        return hash(frozenset(hash(r) for r in self._reactions))

    def __add__(self, other: 'reactions') -> 'reactions':
        if not isinstance(other, reactions): return NotImplemented
        combined = [r.copy() for r in self._reactions]
        for r_new in other._reactions:
            if not any(r_new == r_existing for r_existing in combined):
                combined.append(r_new.copy())
        result = reactions.__new__(reactions)
        result._reactions = combined
        return result

    def __repr__(self) -> str: return '\n'.join(repr(r) for r in self._reactions)
    def __str__(self)  -> str: return '\n'.join(str(r)  for r in self._reactions)
    def __len__(self)  -> int: return len(self._reactions)
    def __iter__(self):        return iter(self._reactions)

    def without(self, indices) -> 'reactions':
        """
        Return a new :class:`reactions` excluding the reactions at *indices*.

        Parameters
        ----------
        indices : int or list of int
            Zero-based index or list of indices to exclude.

        Examples
        --------
        >>> sys2 = sys.without(0)        # drop first reaction
        >>> sys2 = sys.without([1, 3])   # drop reactions 1 and 3
        """
        if isinstance(indices, int):
            indices = [indices]
        exclude = set(indices)
        kept = [r for i, r in enumerate(self._reactions) if i not in exclude]
        if not kept:
            raise InputError(
                f"without({list(indices)}) would remove all reactions.")
        result = reactions.__new__(reactions)
        result._reactions = kept
        return result

    def __getitem__(self, idx):
        """
        Flexible indexing and filtering.

        Parameters
        ----------
        idx : int
            Single reaction by index.
        idx : slice
            Returns a new ``reactions`` object with the selected subset.
        idx : str
            Returns a new ``reactions`` object containing only reactions
            that involve this species name.
        idx : list of int or str
            Mix of the above; returns a new ``reactions`` object.

        Examples
        --------
        >>> sys[0]            # first reaction
        >>> sys[1:]           # all but first, as reactions object
        >>> sys['Fe[3+]']     # all reactions involving Fe[3+]
        """
        if isinstance(idx, int):
            return self._reactions[idx]
        if isinstance(idx, slice):
            result = reactions.__new__(reactions)
            result._reactions = self._reactions[idx]
            return result
        if isinstance(idx, str):
            sp_norm = species(idx)
            matched = [r for r in self._reactions if sp_norm in r._stoich]
            if not matched:
                raise KeyError(
                    f"No reactions involve species {idx!r}.\n"
                    f"Known species: {sorted(self.species)}")
            result = reactions.__new__(reactions)
            result._reactions = matched
            return result
        if isinstance(idx, (list, tuple)):
            rxns = []
            for i in idx:
                item = self[i]
                if isinstance(item, reactions):
                    rxns.extend(item._reactions)
                else:
                    rxns.append(item)
            result = reactions.__new__(reactions)
            result._reactions = rxns
            return result
        raise TypeError(f"reactions indices must be int, slice, or str, got {type(idx).__name__}")

    def __contains__(self, item) -> bool:
        """Support ``reaction_obj in sys`` and ``'species_name' in sys``."""
        if isinstance(item, str):
            try:
                return species(item) in self.species
            except Exception:
                return False
        return item in self._reactions

    def _validate_c0(self, c0: dict, maintain=None) -> None:
        """
        Validate initial concentrations before solving.

        Raises
        ------
        ConcentrationError
            If any concentration is negative.
        InputError
            If ``maintain`` references a species not in the system, or if
            ``c0`` contains a species name that appears in no reaction.
        """
        from .errors import ConcentrationError
        known = self.species | frozenset({'O/A'})
        maintain_set = frozenset(species(s) for s in (maintain or []))

        # Check for species in c0 that aren't in any reaction
        unknown = []
        for sp in c0:
            if sp == 'O/A':
                continue
            try:
                sp_norm = species(sp)
            except Exception:
                sp_norm = sp
            if sp_norm not in known:
                unknown.append(sp)
        if unknown:
            import warnings
            warnings.warn(
                f"c0 contains species not in any reaction: {unknown}.\n"
                f"These concentrations will be ignored by the solver.\n"
                f"Known species: {sorted(known - frozenset(['O/A']))}",
                UserWarning, stacklevel=3)

        # Check for negative concentrations
        negative = {sp: v for sp, v in c0.items()
                    if isinstance(v, (int, float)) and v < 0}
        if negative:
            raise ConcentrationError(
                f"Negative initial concentrations are not physically meaningful: "
                f"{negative}.\n"
                f"All concentrations must be ≥ 0.")

        # Check maintain species are known
        if maintain_set:
            bad_maintain = [s for s in maintain_set if s not in known]
            if bad_maintain:
                from .errors import InputError
                raise InputError(
                    f"maintain= references species not in any reaction: {bad_maintain}.\n"
                    f"Known species: {sorted(known - frozenset(['O/A']))}")

    # ------------------------------------------------------------------
    # Core solve / equilibrium
    # ------------------------------------------------------------------

    def _solve(self, c0, tolerance=1e-6, init_scale=0.1, verbose=False,
               warm_start=None, lc_warm_start=None, presolver_timeout=10.0,
               maintain=None):
        ws = np.asarray(warm_start, dtype=float) if warm_start is not None else None
        all_sp_tmp, _, _, _, _ = _build_system(self._reactions, c0)
        gamma_dict = _build_gamma_for_system(self._reactions, all_sp_tmp)
        xi_arr, info = _solve_with_ksp(
            self._reactions, c0, tolerance=tolerance, init_scale=init_scale,
            verbose=verbose, warm_start=ws, lc_warm_start=lc_warm_start,
            presolver_timeout=presolver_timeout,
            maintain=maintain, gamma_dict=gamma_dict)
        return np.asarray(xi_arr, dtype=float), info

    def solve(self, c0: dict, tolerance: float = 1e-6, init_scale: float = 0.1,
              verbose: bool = False, warm_start=None, lc_warm_start=None,
              presolver_timeout: float = 10.0, maintain=None) -> np.ndarray:
        """
        Compute the equilibrium extent-of-reaction vector ξ (mol/L aqueous).

        This is the low-level solve method.  Most users should call
        :meth:`equilibrium` instead, which returns a more convenient
        ``{species: concentration}`` dict.

        The extent vector ξ has one entry per reaction.  Positive ξ[r] means
        reaction *r* proceeded in the forward direction; negative means it ran
        in reverse.  The equilibrium concentration of species *j* is::

            c[j] = c0[j] + Σ_r  ν[r,j] · ξ[r]

        Parameters
        ----------
        c0 : dict
            Initial concentrations ``{species_name: mol_per_L}``.  Include
            ``'O/A'`` for the organic/aqueous volume ratio in mixed-phase
            systems.
        tolerance : float, optional
            Maximum absolute residual accepted as converged (default 1e-6).
            Smaller values increase accuracy but may slow convergence.
        init_scale : float, optional
            Fraction of the feasible-range width used for the initial guess
            (must be in ``(0, 1]``; default 0.1).
        verbose : bool, optional
            Print per-method solver progress (default False).
        warm_start : array-like, optional
            Initial guess for ξ (length = n_reactions).  Providing a good
            warm start (e.g. from a nearby solved system) can speed up
            convergence significantly.
        lc_warm_start : array-like, optional
            Initial guess in log10(c) space (length = n_species).
        presolver_timeout : float, optional
            Seconds before the fast Method-L pre-solver is abandoned and
            the full solver chain (A → B → DE) is tried (default 10.0).
        maintain : list of str, optional
            Species whose concentrations are held fixed at their ``c0``
            values throughout the solve (e.g. a buffered pH species).

        Returns
        -------
        xi : np.ndarray, shape (n_reactions,)
            Equilibrium extent of each reaction in mol/L (aqueous basis).

        Raises
        ------
        ConvergenceError
            If all solver methods fail to reach ``tolerance``.
        ConcentrationError
            If any value in *c0* is negative.
        InputError
            If *init_scale* is outside ``(0, 1]`` or *maintain* references
            an unknown species.
        """
        c0 = _c0_from_input(c0)
        if not (0 < init_scale <= 1):
            raise InputError("init_scale must be in (0, 1], got {init_scale}.\n"
            "This controls the initial guess scale for the solver.")
        self._validate_c0(c0, maintain=maintain)
        xi_arr, _ = self._solve(c0, tolerance=tolerance, init_scale=init_scale,
                                verbose=verbose, warm_start=warm_start,
                                lc_warm_start=lc_warm_start,
                                presolver_timeout=presolver_timeout,
                                maintain=maintain)
        return xi_arr

    def equilibrium(self, c0: dict, tolerance: float = 1e-6,
                    init_scale: float = 0.1, verbose: bool = False,
                    warm_start=None, lc_warm_start=None,
                    presolver_timeout: float = 10.0,
                    maintain=None) -> Dict[str, float]:
        """
        Compute equilibrium concentrations for the coupled reaction system.

        This is the primary user-facing method for obtaining equilibrium
        concentrations.  It accepts the same parameters as :meth:`solve` and
        returns a plain dict that is easy to work with directly.

        Parameters
        ----------
        c0 : dict
            Initial concentrations ``{species_name: mol_per_L}``.  All species
            that appear in any reaction must be present; unlisted species are
            assumed to be 0 mol/L.  Include ``'O/A'`` for mixed-phase systems.
        tolerance : float, optional
            Convergence criterion (default 1e-6).  Residual is the maximum
            absolute deviation from the mass-action law across all reactions.
        init_scale : float, optional
            Initial guess scale factor in ``(0, 1]`` (default 0.1).
        verbose : bool, optional
            Print solver diagnostics for each method attempted (default False).
        warm_start : array-like, optional
            Warm-start ξ vector for faster convergence on repeated calls.
        lc_warm_start : array-like, optional
            Warm-start log₁₀(c) vector.
        presolver_timeout : float, optional
            Seconds before the fast pre-solver is abandoned (default 10.0).
        maintain : list of str, optional
            Species held at their initial concentration throughout the solve.

        Returns
        -------
        dict
            ``{species_name: equilibrium_concentration_mol_per_L}`` for every
            species in the system.

        Raises
        ------
        ConvergenceError
            If all solvers fail to reach *tolerance* with residual > 1e-2.
        ConvergenceWarning
            If the solver returns a result but residual exceeds *tolerance*
            (concentrations may be slightly inaccurate).

        Examples
        --------
        Solve acetic acid / water system::

            sys = reactions(
                reaction('CH3COOH = CH3COO[-] + H[+]', 1.8e-5),
                reaction('H2O = H[+] + OH[-]', 1e-14),
            )
            c_eq = sys.equilibrium({
                'CH3COOH':   0.1,
                'CH3COO[-]': 0.0,
                'H[+]':      1e-7,
                'OH[-]':     1e-7,
                'H2O':       1.0,
            })
            import numpy as np
            print(f"pH = {-np.log10(c_eq['H[+]']):.2f}")  # pH ≈ 2.87
        """
        c0 = _c0_from_input(c0)
        self._validate_c0(c0, maintain=maintain)
        xi, info = self._solve(c0, tolerance=tolerance, init_scale=init_scale,
                               verbose=verbose, warm_start=warm_start,
                               lc_warm_start=lc_warm_start,
                               presolver_timeout=presolver_timeout,
                               maintain=maintain)

        # --- convergence check ---
        err = info.get('error', float('inf'))
        method = info.get('method', '')
        sys_info = (f"{len(self._reactions)} reaction(s), "
                    f"species: {sorted(c0.keys())[:6]}"
                    f"{'...' if len(c0) > 6 else ''}")
        if err > 1e-2:
            raise ConvergenceError(
                f"Solver failed to converge for this system.\n"
                f"All methods (L, A, B, DE) were tried and the best residual "
                f"({err:.3e}) is far above tolerance ({tolerance:.3e}).\n"
                f"Suggestions:\n"
                f"  • Check that initial concentrations are physically reasonable\n"
                f"  • Ensure all species in c0 appear in at least one reaction\n"
                f"  • Try a different init_scale (current: {init_scale})\n"
                f"  • Use verbose=True to see per-method diagnostics",
                residual=err, method=method, system_info=sys_info)
        elif err > tolerance:
            warn_convergence(err, tolerance, method=method, system_info=sys_info)

        c_direct = info.get('c_eq_direct')
        all_sp, c0_vec, nu, _, v_oa = _build_system(self._reactions, c0)
        org_mask = np.array([is_organic(s) for s in all_sp], dtype=bool)
        maintain_set = frozenset(species(s) for s in (maintain or []))
        maint_mask   = np.array([s in maintain_set for s in all_sp], dtype=bool)

        if c_direct is not None:
            c_eq = np.maximum(np.asarray(c_direct, dtype=float), 0.0)
            for i in np.where(maint_mask)[0]:
                c_eq[i] = float(c0_vec[i])
        else:
            xi_arr = np.asarray(xi, dtype=float)
            c_eq   = np.maximum(_conc_from_xi(xi_arr, c0_vec, nu, org_mask, v_oa,
                                               maintain_mask=maint_mask), 0.0)
        return {s: float(c_eq[i]) for i, s in enumerate(all_sp)
                if not is_electron(s)}


    def find(self,
             unknown: str,
             c0: dict,
             target: dict,
             *,
             bounds: tuple = None,
             tolerance: float = 1e-6,
             target_tolerance: float = 1e-6,
             max_iter: int = 100,
             init_scale: float = 0.1,
             verbose: bool = False,
             presolver_timeout: float = 10.0,
             maintain=None) -> float:
        """
        Find the initial concentration of one unknown species that drives
        the system to a specified set of equilibrium concentrations.

        Uses Brent's method (guaranteed convergence once bracketed).
        See :func:`efta.solver.find._find` for full implementation details.

        Parameters
        ----------
        unknown : str
            Species whose initial concentration is unknown.
        c0 : dict
            Initial concentrations of all other species (mol/L).
        target : dict
            Desired equilibrium concentrations, e.g. ``{'H[+]': 1e-7}``.
        bounds : (float, float), optional
            Search interval for the unknown. Defaults to (0, 10*max(c0)).
        tolerance : float
            Passed to the equilibrium solver.
        target_tolerance : float
            Convergence criterion for the root-finder.
        max_iter : int
            Maximum Brent iterations.
        init_scale : float
            Passed to the equilibrium solver.
        verbose : bool
            Print each objective evaluation.
        presolver_timeout : float
            Passed to the equilibrium solver.
        maintain : list of str, optional
            Species held at fixed concentration throughout.

        Returns
        -------
        float
            Initial concentration of *unknown* (mol/L) that satisfies target.

        Examples
        --------
        >>> c_init = sys.find('CH3COOH',
        ...                   c0={'H2O': 1.0},
        ...                   target={'H[+]': 1e-4})  # pH 4
        """
        c0 = _c0_from_input(c0)
        from .solver.find import _find
        return _find(
            self, unknown, c0, target,
            bounds=bounds,
            tolerance=tolerance,
            target_tolerance=target_tolerance,
            max_iter=max_iter,
            init_scale=init_scale,
            verbose=verbose,
            presolver_timeout=presolver_timeout,
            maintain=maintain,
        )


    def __rshift__(self, sol):
        """
        Return a new solution at equilibrium: ``sys >> sol``.

        Examples
        --------
        >>> eq = sys >> sol
        """
        from .solution import solution as _solution
        if not isinstance(sol, _solution):
            return NotImplemented
        return sol << self

    # ------------------------------------------------------------------
    # Sweep and plotting
    # ------------------------------------------------------------------

    def _sweep_data(self, c0, y='concentration', logx=False, logy=False,
                    init_scale=0.1, n_points=100, recheck=False,
                    recheck_tol=1e-4, recheck_density=5,
                    presolver_timeout=10.0, maintain=None) -> dict:
        """
        Compute equilibrium over a 1-D sweep of one initial concentration.

        One entry in *c0* should be a ``[lo, hi]`` list defining the sweep
        range.  All others are fixed scalars (or callables of the sweep value).
        """
        if y not in ('concentration', 'extent'):
            raise InputError(f"y must be 'concentration' or 'extent', got {y!r}")

        sweep_name = None; sweep_range = None
        fixed_c0   = {}; callable_c0  = {}

        for name, val in c0.items():
            name = species(name) if name != 'O/A' else name
            if name == 'O/A':
                fixed_c0[name] = float(val); continue
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 2:
                if sweep_name is not None:
                    raise InputError("Only one parameter may be a sweep range [lo, hi].\n"
                "All other c0 values must be scalars.")
                sweep_name = name; sweep_range = (float(val[0]), float(val[1]))
            elif callable(val):
                callable_c0[name] = val
            else:
                fixed_c0[name] = float(val)

        if sweep_name is None:
            raise InputError("No sweep range [lo, hi] found in c0.\n"
            "Set one c0 entry to [lo, hi] to define the sweep axis.")

        def _build_cur_c0(x_val):
            cur = {**fixed_c0, sweep_name: float(x_val)}
            for sp_name, fn in callable_c0.items():
                cur[sp_name] = float(fn(x_val))
            return cur

        mid_c0    = _build_cur_c0((sweep_range[0] + sweep_range[1]) / 2)
        sp_order, _, _, _, _ = _build_system(self._reactions, mid_c0)
        n_rxn     = len(self._reactions)

        xs = (np.logspace(np.log10(sweep_range[0]), np.log10(sweep_range[1]), n_points)
              if logx else np.linspace(sweep_range[0], sweep_range[1], n_points))

        sp_results = {s: [np.nan] * len(xs) for s in sp_order}
        c0_results = {s: [np.nan] * len(xs) for s in sp_order}
        xi_results = [[np.nan] * n_rxn for _ in range(len(xs))]
        prev_xi = None; prev_lc_aq = None

        def _solve_one(val, warm_xi, warm_lc):
            cur_c0 = _build_cur_c0(val)
            xi, info = self._solve(cur_c0, init_scale=init_scale,
                                   warm_start=warm_xi, lc_warm_start=warm_lc,
                                   presolver_timeout=presolver_timeout,
                                   maintain=maintain)
            eq = _equilibrium_concentrations(xi, cur_c0, self,
                                             _c_eq_direct=info.get('c_eq_direct'),
                                             maintain=maintain)
            return eq, xi, info.get('lc_aq'), float(info.get('error', np.inf))

        for i, val in enumerate(xs):
            try:
                cur_c0_i = _build_cur_c0(val)
                eq, xi, lc_aq, err = _solve_one(val, prev_xi, prev_lc_aq)

                if recheck and err > recheck_tol:
                    try:
                        eq2, xi2, lc_aq2, err2 = _solve_one(val, None, None)
                        if err2 < err: eq, xi, lc_aq, err = eq2, xi2, lc_aq2, err2
                    except Exception: pass

                    if err > recheck_tol and i > 0:
                        prev_val  = xs[i - 1]
                        dense_xs  = (np.logspace(np.log10(prev_val), np.log10(val),
                                                  recheck_density + 2)[1:-1]
                                     if logx else np.linspace(prev_val, val,
                                                              recheck_density + 2)[1:-1])
                        d_xi, d_lc = prev_xi, prev_lc_aq
                        for dval in dense_xs:
                            try:
                                _, d_xi, d_lc, _ = _solve_one(dval, d_xi, d_lc)
                            except Exception:
                                d_xi, d_lc = None, None
                        try:
                            eq3, xi3, lc_aq3, err3 = _solve_one(val, d_xi, d_lc)
                            if err3 < err: eq, xi, lc_aq, err = eq3, xi3, lc_aq3, err3
                        except Exception: pass

                prev_xi = xi; prev_lc_aq = lc_aq
                for s in sp_order:
                    sp_results[s][i] = eq.get(s, 0.0)
                    c0_results[s][i] = cur_c0_i.get(s, 0.0)
                xi_results[i] = list(np.asarray(xi, dtype=float))
            except Exception:
                prev_xi = None; prev_lc_aq = None

        return {
            'variable':       sweep_name,
            'x':              xs.tolist(),
            'concentrations': {s: list(v) for s, v in sp_results.items()},
            'c0':             {s: list(v) for s, v in c0_results.items()},
            'extents':        xi_results,
            'reactions':      [f'R{i+1}' for i in range(n_rxn)],
            'fixed_c0':       dict(fixed_c0),
            'variable_c0':    dict(callable_c0),
            'log_x':          bool(logx),
            'log_y':          bool(logy),
            'y':              y,
        }

    def plot(self, c0: dict, y: str = 'concentration', maintain=None,
             logx: bool = False,
             n_points: int = 20, select=None, type: str = 'plot',
             color: bool = False, init_scale: float = 0.1,
             recheck: bool = False, recheck_tol: float = 1e-4,
             recheck_density: int = 5, presolver_timeout: float = 2.0):
        """
        Plot equilibrium concentrations (or extents) versus a swept initial
        concentration.

        One entry in *c0* must be a two-element list ``[lo, hi]`` defining
        the sweep axis.  All other entries are fixed scalars or callables
        ``f(x)`` returning a concentration as a function of the sweep value.

        Parameters
        ----------
        c0 : dict
            Initial conditions.  Exactly one entry must be ``[lo, hi]``
            to define the sweep; the others are scalars or callables.
        y : {'concentration', 'extent'}, optional
            What to plot on the y-axis (default ``'concentration'``).
        maintain : list of str, optional
            Species held at their initial concentration throughout.
        logx : bool, optional
            Use a logarithmic x-axis (default False).
        n_points : int, optional
            Number of evaluation points along the sweep (default 20).
        select : list, optional
            Species names (or reaction indices for ``y='extent'``) to plot.
            If ``None``, all are plotted.
        type : str, optional
            Plot type: ``'plot'`` (line), ``'layer'`` (stacked fill),
            ``'bilayer'`` (org above/aq below), ``'log'`` (log y-axis),
            or ``'data'`` (return raw data dict).
        color : bool, optional
            Use the efta colour palette (default False → black lines).
        recheck : bool, optional
            Re-solve high-residual points with a fresh independent guess
            (slower but more robust near sharp transitions, default False).

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes, or a raw data dict if
            ``type='data'``.

        Examples
        --------
        ::

            fig, ax = sys.plot(
                {'CH3COOH': [1e-4, 1.0], 'H[+]': 1e-7, 'OH[-]': 1e-7},
                logx=True, n_points=60,
            )
        """
        c0 = _c0_from_input(c0)
        from .plotting import _plot_reactions
        return _plot_reactions(self, c0, y=y, maintain=maintain, logx=logx,
                               n_points=n_points, select=select, type=type, color=color,
                               init_scale=init_scale, recheck=recheck,
                               recheck_tol=recheck_tol, recheck_density=recheck_density,
                               presolver_timeout=presolver_timeout)

    # ------------------------------------------------------------------
    # Speciation fractions
    # ------------------------------------------------------------------

    def _compute_fractions(self) -> Dict[str, List[str]]:
        """
        Identify element-cluster groups and list the species in each group.

        Returns a dict: {cluster_label: [species, ...]}
        """
        all_sp       = sorted({s for r in self._reactions for s in r._stoich
                                if s != 'O/A'})
        compositions = {s: components(s) for s in all_sp}
        all_atoms    = sorted({a for comp in compositions.values() for a in comp})
        if not all_atoms: return {}

        atom_idx = {a: i for i, a in enumerate(all_atoms)}
        M = np.zeros((len(all_atoms), len(all_sp)), dtype=float)
        for j, sp in enumerate(all_sp):
            for a, cnt in compositions[sp].items():
                M[atom_idx[a], j] = cnt

        parent = list(range(len(all_atoms)))
        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def _union(x, y): parent[_find(x)] = _find(y)

        for i in range(len(all_atoms)):
            for j in range(i + 1, len(all_atoms)):
                row_a, row_b = M[i], M[j]
                sup_a = set(np.where(row_a > 0)[0])
                sup_b = set(np.where(row_b > 0)[0])
                if sup_a != sup_b: continue
                if not sup_a: _union(i, j); continue
                ratios = row_a[list(sup_a)] / row_b[list(sup_a)]
                if np.allclose(ratios, ratios[0], rtol=1e-9):
                    _union(i, j)

        clusters_map = defaultdict(list)
        for i, a in enumerate(all_atoms):
            clusters_map[_find(i)].append(a)

        result: Dict[str, List[str]] = {}
        for _root, atom_list in clusters_map.items():
            cluster_atoms = frozenset(atom_list)
            pure  = [s for s in all_sp
                     if frozenset(compositions[s].keys()) == cluster_atoms]
            raw_label = (min(pure, key=lambda s: sum(compositions[s].values()))
                         if pure else '·'.join(sorted(cluster_atoms)))
            label = re.sub(r'\s*\[[^\]]*[+\-][^\]]*\]\s*$', '', raw_label)
            label = re.sub(r'\s*\([slgorg]+\)\s*$', '', label, flags=re.IGNORECASE)
            label = label.strip() or raw_label
            members = [s for s in all_sp
                       if cluster_atoms & frozenset(compositions[s].keys())]
            result[label] = members

        return result

    def fractions(self, component: str, c0: dict, maintain=None, logx: bool = False,
                  n_points: int = 20, select=None, type: str = 'plot',
                  color: bool = False, init_scale: float = 0.1,
                  recheck: bool = False, recheck_tol: float = 1e-4,
                  recheck_density: int = 5,
                  presolver_timeout: float = 2.0):
        """
        Plot speciation fractions of *component* vs the sweep variable.

        *component* may be a species name, element symbol, or cluster label
        as returned by ``reactions._compute_fractions()``.

        style options: 'plot', 'data', 'layer', 'bilayer', 'norm-bilayer'
        """
        c0 = _c0_from_input(c0)
        from .plotting import _plot_fractions
        return _plot_fractions(self, component, c0, maintain=maintain, logx=logx,
                               n_points=n_points, select=select, type=type,
                               color=color, init_scale=init_scale, recheck=recheck,
                               recheck_tol=recheck_tol,
                               recheck_density=recheck_density,
                               presolver_timeout=presolver_timeout)
