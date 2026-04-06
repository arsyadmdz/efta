"""
efta.model.mass_action
======================
Direct mass-action residuals for fast parameter fitting.

Instead of running the equilibrium solver on each optimizer step, this
module evaluates the mass action law directly from observed (or reconstructed)
equilibrium concentrations:

    r_j = log(K_j) - Σᵢ νᵢⱼ · log(aᵢ)

This is analytic — no iterative solver needed — so each residual evaluation
is O(n_reactions × n_species) arithmetic, orders of magnitude faster than
calling `.equilibrium()`.

Public API
----------
:func:`mass_action_residuals`
    Compute residual vector from a reaction system and concentration data.
:func:`is_data_sufficient`
    Check whether observed data is sufficient for direct fitting.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ['mass_action_residuals', 'is_data_sufficient']


# ---------------------------------------------------------------------------
# Cached species helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _is_nonaqueous_cached(sp: str) -> bool:
    n = sp.strip().lower()
    return n.endswith('(s)') or n.endswith('(l)')


@lru_cache(maxsize=4096)
def _is_electron_cached(sp: str) -> bool:
    from ..species import ELECTRON
    return sp == ELECTRON


@lru_cache(maxsize=4096)
def _is_skip(sp: str) -> bool:
    """Return True if species should be skipped in mass-action calculation."""
    return _is_nonaqueous_cached(sp) or _is_electron_cached(sp) or sp == 'O/A'


# ---------------------------------------------------------------------------
# Pre-compiled reaction data
# ---------------------------------------------------------------------------

def _active_stoich(rxn) -> list:
    """Return list of (species, nu) for active (non-skipped) species."""
    return [(sp, nu) for sp, nu in rxn._stoich.items() if not _is_skip(sp)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_activity(sp: str, c_dict: Dict[str, float],
                  gamma_dict=None) -> float:
    from ..system import _compute_activity
    a = _compute_activity(sp, c_dict, gamma_dict)
    return math.log(max(float(a), 1e-300))


def _is_org(sp: str) -> bool:
    """Return True if species is in the organic phase."""
    return sp.strip().lower().endswith('(org)')


def _reconstruct_missing(active: list,
                          c_obs: Dict[str, float],
                          c_init: Optional[Dict[str, float]],
                          v_oa: float = 1.0) -> Dict[str, float]:
    """
    Reconstruct missing species concentrations from stoichiometry.

    Extent xi (mol/L_aq) is estimated from species with known c0.
    Phase volumes are accounted for via v_oa = V_org / V_aq:
      - Aqueous species:  ceq = c0 + nu * xi
      - Organic species:  ceq = c0 + nu * xi / v_oa
        (xi is in aq units; org concentration = aq extent / v_oa)

    Falls back to all observed species if none have a known c0.
    """
    merged = dict(c_obs)

    xi_known, xi_all = [], []
    for sp, nu in active:
        if sp in c_obs and abs(nu) > 1e-12:
            c0_sp = (c_init.get(sp) if c_init else None)
            # convert observed change back to aq-phase extent
            factor = 1.0 / v_oa if _is_org(sp) else 1.0
            denom  = nu * factor if factor != 1.0 else nu
            xi_val = (c_obs[sp] - (c0_sp if c0_sp is not None else 0.0)) / denom
            xi_all.append(xi_val)
            if c0_sp is not None:
                xi_known.append(xi_val)

    xi_list = xi_known if xi_known else xi_all
    if not xi_list:
        return merged

    xi = sum(xi_list) / len(xi_list)
    for sp, nu in active:
        if sp not in merged:
            c0_sp  = c_init.get(sp, 0.0) if c_init else 0.0
            factor = 1.0 / v_oa if _is_org(sp) else 1.0
            merged[sp] = max(c0_sp + nu * xi * factor, 1e-300)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mass_action_residuals(rxn_sys,
                           ceq_obs: Dict[str, float],
                           c_init:  Optional[Dict[str, float]] = None,
                           gamma_dict=None,
                           v_oa: float = 1.0) -> np.ndarray:
    """
    Compute the mass-action residual vector for a reaction system.

    For each reaction j:

        r_j = log(K_j) - Σᵢ νᵢⱼ · log(aᵢ)

    Parameters
    ----------
    rxn_sys : reaction or reactions
    ceq_obs : dict
        Observed equilibrium concentrations.
    c_init : dict, optional
        Initial concentrations (used to reconstruct missing species).
    gamma_dict : dict, optional
        Activity coefficient functions.

    Returns
    -------
    np.ndarray
        Residual vector, length = n_reactions.
    """
    from ..system import _build_gamma_for_system

    rxns = (rxn_sys._reactions if hasattr(rxn_sys, '_reactions')
            else [rxn_sys])

    # skip gamma build if no gamma registered (common case)
    if gamma_dict is None:
        has_gamma = any(getattr(r, '_gamma', {}) for r in rxns)
        if has_gamma:
            all_sp     = list({sp for r in rxns for sp in r._stoich})
            gamma_dict = _build_gamma_for_system(rxns, all_sp)

    obs_keys  = set(ceq_obs)
    init_keys = set(c_init) if c_init else set()
    known     = obs_keys | init_keys

    residuals = np.empty(len(rxns))
    for i, rxn in enumerate(rxns):
        active = _active_stoich(rxn)

        # only reconstruct if some species are missing
        active_sp = {sp for sp, _ in active}
        if active_sp <= known:
            c_merged = ceq_obs          # fast path: all observed
        else:
            c_merged = _reconstruct_missing(active, ceq_obs, c_init, v_oa=v_oa)

        log_Q = 0.0
        for sp, nu in active:
            log_Q += nu * _log_activity(sp, c_merged, gamma_dict)

        residuals[i] = math.log(max(rxn.K, 1e-300)) - log_Q

    return residuals


def is_data_sufficient(rxn_sys, ceq_obs: Dict[str, float],
                        c_init: Optional[Dict[str, float]] = None) -> bool:
    """
    Check whether observed data is sufficient for direct mass-action fitting.
    """
    rxns  = (rxn_sys._reactions if hasattr(rxn_sys, '_reactions')
             else [rxn_sys])
    known = set(ceq_obs) | (set(c_init) if c_init else set())
    for rxn in rxns:
        active = {sp for sp in rxn._stoich if not _is_skip(sp)}
        if not active & known:
            return False
    return True
