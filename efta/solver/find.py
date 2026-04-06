"""
efta.solver.find
================
Inverse solver: find the initial concentration of one unknown species
that drives the system to a specified set of equilibrium concentrations.

The problem is formulated as a 1-D root-finding problem solved by
Brent's method (scipy.optimize.brentq), which is guaranteed to converge
once the root is bracketed.

Public entry point
------------------
:func:`_find`
    Called by :meth:`efta.reactions.reactions.find` and
    :meth:`efta.reaction.reaction.find`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.optimize import brentq

from ..species import species as _normalise_species
from ..errors import InputError, ConvergenceError


def _find(reactions_obj,
          unknown: str,
          c0: dict,
          target: dict,
          *,
          bounds: Optional[Tuple[float, float]] = None,
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

    The problem is formulated as a 1-D root-finding problem:

        f(c0[unknown]) = 0

    where ``f`` measures the signed normalised difference between the
    computed equilibrium concentrations and the supplied *target* values.
    Brent's method is used, which is guaranteed to converge if the root
    is bracketed.

    Parameters
    ----------
    reactions_obj : efta.reactions
        The reaction system to solve.
    unknown : str
        Name of the species whose initial concentration is unknown.
        Must appear in at least one reaction.
    c0 : dict
        Initial concentrations of all *other* species (mol/L).
        The unknown species should be omitted or set to any placeholder;
        its value will be replaced at each iteration.
    target : dict
        Desired equilibrium concentrations for one or more species, e.g.
        ``{'H[+]': 1e-7}`` to target pH 7.
        All species must appear in the reaction system.
        If multiple species are given, the first entry determines the sign
        for bracketing and the RMS of normalised differences is used for
        convergence.
    bounds : (float, float), optional
        Search interval ``(lo, hi)`` for the unknown initial concentration.
        Both must be ≥ 0.  Defaults to
        ``(0.0, 10 * max(positive c0 values))`` or ``(0.0, 10.0)``.
    tolerance : float, optional
        Convergence tolerance passed to the equilibrium solver.
    target_tolerance : float, optional
        Convergence criterion for the root-finder: stop when the RMS
        normalised residual across all target species is below this value.
    max_iter : int, optional
        Maximum number of Brent iterations.
    init_scale : float, optional
        Passed to the equilibrium solver.
    verbose : bool, optional
        Print each objective evaluation if True.
    presolver_timeout : float, optional
        Passed to the equilibrium solver.
    maintain : list of str, optional
        Species whose concentrations are held fixed throughout.

    Returns
    -------
    float
        The initial concentration of *unknown* (mol/L).

    Raises
    ------
    InputError
        If *unknown* is not in the system, if *target* references unknown
        species, if *bounds* are invalid, or if the root is not bracketed.
    ConvergenceError
        If Brent's method does not converge within *max_iter* iterations.
    """
    # ------------------------------------------------------------------
    # Validate unknown
    # ------------------------------------------------------------------
    try:
        unknown_norm = _normalise_species(unknown)
    except Exception:
        unknown_norm = unknown

    known = reactions_obj.species
    if unknown_norm not in known:
        raise InputError(
            f"unknown species {unknown!r} does not appear in any reaction.\n"
            f"Known species: {sorted(known)}")

    # ------------------------------------------------------------------
    # Validate and normalise target
    # ------------------------------------------------------------------
    bad_target = [sp for sp in target if _normalise_species(sp) not in known]
    if bad_target:
        raise InputError(
            f"target references species not in any reaction: {bad_target}.\n"
            f"Known species: {sorted(known)}")

    target_norm: Dict[str, float] = {
        _normalise_species(sp): float(v) for sp, v in target.items()
    }

    # ------------------------------------------------------------------
    # Validate / derive bounds
    # ------------------------------------------------------------------
    scalar_vals = [v for v in c0.values()
                   if isinstance(v, (int, float)) and v > 0]
    default_hi = 10.0 * max(scalar_vals) if scalar_vals else 10.0

    if bounds is None:
        lo, hi = 0.0, default_hi
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo < 0 or hi < 0:
            raise InputError(
                f"bounds must be non-negative, got ({lo}, {hi}).")
        if lo >= hi:
            raise InputError(
                f"bounds[0] must be < bounds[1], got ({lo}, {hi}).")

    # ------------------------------------------------------------------
    # Objective function (signed)
    # ------------------------------------------------------------------
    # The sign is determined by the first target species so that Brent's
    # method sees a genuine sign change at the root.  The magnitude is the
    # RMS normalised residual across all target species, which handles
    # correlated targets (e.g. H+ and OH- coupled via water equilibrium).
    iteration = [0]
    target_items: List[Tuple[str, float]] = list(target_norm.items())
    primary_sp, primary_tgt = target_items[0]

    def _objective(c_unknown: float) -> float:
        iteration[0] += 1
        c0_trial = dict(c0)
        c0_trial[unknown_norm] = c_unknown
        try:
            ceq = reactions_obj.equilibrium(
                c0_trial,
                tolerance=tolerance,
                init_scale=init_scale,
                presolver_timeout=presolver_timeout,
                maintain=maintain,
            )
        except ConvergenceError:
            return float('nan')

        # sign from primary target
        c_got_primary = ceq.get(primary_sp, 0.0)
        if primary_tgt == 0.0:
            sign_val = c_got_primary
        else:
            sign_val = (c_got_primary - primary_tgt) / primary_tgt

        # RMS across all targets
        sq_sum = 0.0
        for sp, c_tgt in target_items:
            c_got = ceq.get(sp, 0.0)
            sq_sum += (
                ((c_got - c_tgt) / c_tgt) ** 2
                if c_tgt != 0.0
                else c_got ** 2
            )
        rms = (sq_sum / len(target_items)) ** 0.5

        result = math.copysign(rms, sign_val)

        if verbose:
            tgt_str = ', '.join(
                f"{sp}={ceq.get(sp, 0.0):.3e} (want {v:.3e})"
                for sp, v in target_norm.items())
            print(f"  [find] iter {iteration[0]:3d}  "
                  f"{unknown_norm}={c_unknown:.4e}  {tgt_str}  "
                  f"signed_res={result:+.3e}")

        return result

    # ------------------------------------------------------------------
    # Check bracket
    # ------------------------------------------------------------------
    f_lo = _objective(lo)
    f_hi = _objective(hi)

    if abs(f_lo) < target_tolerance:
        return lo
    if abs(f_hi) < target_tolerance:
        return hi

    if not (f_lo * f_hi < 0):
        raise InputError(
            f"Root not bracketed: objective has the same sign at both bounds\n"
            f"  lo={lo:.3e} → f={f_lo:+.3e}\n"
            f"  hi={hi:.3e} → f={f_hi:+.3e}\n"
            f"The target equilibrium concentrations may not be achievable "
            f"within this concentration range for [{unknown}].\n"
            f"Suggestions:\n"
            f"  • Widen bounds= (current: ({lo:.3e}, {hi:.3e}))\n"
            f"  • Check that the target values are physically realistic\n"
            f"  • Use verbose=True to inspect how the equilibrium changes")

    # ------------------------------------------------------------------
    # Brent's method
    # ------------------------------------------------------------------
    try:
        root, info_brent = brentq(
            _objective, lo, hi,
            xtol=tolerance,
            rtol=target_tolerance,
            maxiter=max_iter,
            full_output=True,
        )
    except ValueError as exc:
        raise ConvergenceError(
            f"find(): root-finder failed: {exc}\n"
            f"Check that the target is achievable within "
            f"bounds ({lo:.3e}, {hi:.3e}).",
            method='brent') from exc

    if not info_brent.converged:
        raise ConvergenceError(
            f"find(): Brent's method did not converge after {max_iter} iterations.\n"
            f"Last estimate: [{unknown}]_0 = {root:.4e}\n"
            f"Try increasing max_iter or widening bounds.",
            residual=float('inf'), method='brent')

    return float(root)
