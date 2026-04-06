"""
efta.system
===========
Internal helpers that bridge the high-level :class:`~efta.reaction.reaction` /
:class:`~efta.reactions.reactions` objects and the low-level NumPy-based
solvers.

This module is not part of the public API.  End users should interact with
:class:`~efta.reactions.reactions` (e.g. ``sys.equilibrium(c0)``); the
functions here are called automatically by that layer.

Three main responsibilities
---------------------------
1. **Activity coefficients**
   Evaluate ``gamma(c)`` for a species given the current concentration dict.
   Supports user-supplied non-ideal functions (Davies, Debye-Hückel, etc.)
   as well as the ideal default (gamma = 1).

2. **System assembly**
   Convert a list of :class:`~efta.reaction.reaction` objects plus a ``c0``
   dict into the ``(all_species, c0_vec, nu, K_vec, v_oa)`` tuple consumed
   by every solver.  The stoichiometric matrix ``nu`` has shape
   ``(n_reactions, n_species)`` with positive entries for products and
   negative entries for reactants.

3. **Concentration ↔ extent conversions**
   Translate between the reaction-extent vector ξ (one scalar per reaction,
   in mol/L) and the full concentration vector c (one value per species).
   The key relation is: ``c[j] = c0[j] + Σ_r  ν[r,j] · ξ[r]``.
"""

from __future__ import annotations

from .errors import InputError, ConvergenceError

from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .species import (
    species, formula, is_nonaqueous, is_organic, is_electron,
)
from .balance import _is_ksp_reaction, _system_is_mixed_phase


# ---------------------------------------------------------------------------
# Default (ideal) activity coefficient
# ---------------------------------------------------------------------------

def _one(*args) -> float:
    """Ideal activity coefficient – always 1.0."""
    return 1.0


# ---------------------------------------------------------------------------
# Activity coefficient evaluation
# ---------------------------------------------------------------------------

# Magic dependency token: passing 'I' as a dep_species name causes the
# solver to compute ionic strength on the fly and pass it to gamma func.
_IONIC_STRENGTH_TOKEN = 'I'


def _ionic_strength(c_dict: Dict[str, float]) -> float:
    """
    Compute ionic strength  I = 0.5 · Σ cᵢ zᵢ²  (mol/L).

    Only charged aqueous species contribute; solid, liquid, and organic-phase
    species are skipped.  The result is used as the ``'I'`` dependency when
    evaluating activity-coefficient functions registered via
    :meth:`~efta.reaction.reaction.set_gamma`.

    Parameters
    ----------
    c_dict : dict
        Current concentration dict {species_name: concentration_mol_per_L}.

    Returns
    -------
    float
        Ionic strength in mol/L.
    """
    from .species import charge, is_nonaqueous, is_organic
    I = 0.0
    for sp, c in c_dict.items():
        if is_nonaqueous(sp) or is_organic(sp):
            continue
        z = charge(sp)
        if z != 0:
            I += float(c) * z * z
    return 0.5 * I


def _eval_gamma(entry: tuple, c_dict: Dict[str, float]) -> float:
    """
    Evaluate a single gamma entry ``(func, *dep_names)`` given *c_dict*.

    Centralises the ``'I'`` token logic so all solver methods share one path.
    """
    func = entry[0]
    deps = entry[1:]
    if not deps:
        return float(func())
    _I = None   # computed lazily
    args = []
    for s in deps:
        if s == _IONIC_STRENGTH_TOKEN:
            if _I is None:
                _I = _ionic_strength(c_dict)
            args.append(_I)
        else:
            args.append(float(c_dict.get(s, 0.0)))
    return float(func(*args))



def _compute_activity(sp: str,
                      c_dict: Dict[str, float],
                      gamma_dict: Optional[Dict[str, tuple]]) -> float:
    """
    Return the thermodynamic activity of species *sp*.

    The activity is defined as::

        a(sp) = γ(sp) · c(sp)

    where γ is the activity coefficient (dimensionless) returned by the
    user-supplied gamma function, or 1.0 if none was registered.

    The special dependency token ``'I'`` is replaced at call time by the
    current ionic strength computed from *c_dict*::

        rxn.set_gamma('Fe[3+]', (davies_func, 'I'))

    Parameters
    ----------
    sp : str
        Species name (should already be normalised).
    c_dict : dict
        Current concentration dict.
    gamma_dict : dict or None
        Mapping of species name → ``(func, *dep_names)`` tuples, as built by
        :func:`_build_gamma_for_system`.  Pass ``None`` for ideal behaviour.

    Returns
    -------
    float
        Thermodynamic activity in mol/L (same units as concentration for
        ideal solutions; adjusted by gamma for non-ideal ones).
    """
    c = float(c_dict.get(sp, 0.0))
    if gamma_dict is None:
        return c
    entry = gamma_dict.get(sp)
    if entry is None:
        return c
    func        = entry[0]
    dep_species = entry[1:]
    gamma = _eval_gamma(entry, c_dict)
    return gamma * c


def _build_gamma_for_system(reactions_list: list,
                             all_species: List[str]) -> Dict[str, tuple]:
    """
    Merge the gamma dicts from all reactions into a single system-level dict.

    Later reactions override earlier ones for the same species (consistent
    with how stoichiometry is merged in :func:`_build_system`).
    Missing species receive the ideal default.
    """
    merged: Dict[str, tuple] = {}
    for rxn in reactions_list:
        g = getattr(rxn, '_gamma', {})
        merged.update(g)
    for sp in all_species:
        if sp not in merged:
            merged[sp] = (_one,)
    return merged


# ---------------------------------------------------------------------------
# Maintain (pinned concentration) helpers
# ---------------------------------------------------------------------------

def _normalize_maintain(maintain, all_species: List[str]) -> np.ndarray:
    """
    Convert a list of species names (to maintain) into a boolean mask over
    *all_species*.
    """
    if not maintain:
        return np.zeros(len(all_species), dtype=bool)
    maintain_set = frozenset(species(s) for s in maintain)
    return np.array([s in maintain_set for s in all_species], dtype=bool)


# ---------------------------------------------------------------------------
# Volume-ratio extraction
# ---------------------------------------------------------------------------

def _extract_voa(c0: Dict[str, float], reactions_list: list) -> float:
    """
    Extract the organic/aqueous volume ratio from c0.

    Raises ValueError if a mixed-phase system is detected but 'O/A' is absent
    or non-positive.
    """
    mixed = _system_is_mixed_phase(reactions_list)
    voa   = float(c0.get('O/A', 1.0))
    if mixed and 'O/A' not in c0:
        raise InputError(
            "This reaction system contains both aqueous and organic species. "
            "You must supply 'O/A' in the c0 dictionary.")
    if voa <= 0:
        raise InputError(f"'O/A' must be a positive number, got {voa}.\n"
            "Set c0['O/A'] to the organic/aqueous volume ratio (e.g. 1.0).")
    return voa


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------

def _build_system(reactions_list: list, c0: dict):
    """
    Assemble the numerical arrays needed by all solver methods from a list of
    reactions and a set of initial conditions.

    Parameters
    ----------
    reactions_list : list of reaction
        The coupled equilibrium reactions to solve.
    c0 : dict
        Initial concentrations ``{species_name: mol_per_L}``.  Optionally
        include the key ``'O/A'`` for the organic-to-aqueous volume ratio
        (required for mixed-phase systems).

    Returns
    -------
    all_species : list of str
        All species names, sorted alphabetically.  This defines the column
        ordering of the arrays below.
    c0_vec : np.ndarray, shape (n_species,)
        Initial concentration for each species (0.0 for species not in *c0*).
    nu : np.ndarray, shape (n_reactions, n_species)
        Stoichiometric coefficient matrix.  Positive entries are products;
        negative entries are reactants.
    K_vec : np.ndarray, shape (n_reactions,)
        Equilibrium constants for each reaction.
    v_oa : float
        Organic/aqueous volume ratio (1.0 if not a mixed-phase system).
    """
    c0 = {species(k): v for k, v in c0.items()}

    all_sp  = sorted({s for r in reactions_list for s in r._stoich if s != 'O/A'})
    sp_idx  = {s: i for i, s in enumerate(all_sp)}
    n_sp    = len(all_sp)
    n_rxn   = len(reactions_list)

    c0_vec = np.array([c0.get(s, 0.0) for s in all_sp], dtype=float)
    K_vec  = np.array([r.K for r in reactions_list], dtype=float)
    nu     = np.zeros((n_rxn, n_sp), dtype=float)
    for ri, rxn in enumerate(reactions_list):
        for sp, coeff in rxn._stoich.items():
            if sp == 'O/A':
                continue
            nu[ri, sp_idx[sp]] = coeff

    v_oa = _extract_voa(c0, reactions_list)
    return all_sp, c0_vec, nu, K_vec, v_oa


# ---------------------------------------------------------------------------
# Concentration ↔ extent conversions
# ---------------------------------------------------------------------------

def _conc_from_xi(xi: np.ndarray,
                  c0_vec: np.ndarray,
                  nu: np.ndarray,
                  org_mask: np.ndarray,
                  v_oa: float,
                  maintain_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute equilibrium concentration vector from the extent-of-reaction vector ξ.

    The fundamental relationship is::

        c[j] = c0[j] + Σ_r  ν[r,j] · ξ[r]

    where ξ[r] is the extent of reaction *r* in mol/L (aqueous basis) and
    ν[r,j] is the stoichiometric coefficient of species *j* in reaction *r*.

    Organic-phase species are divided by *v_oa* because organic concentrations
    are reported per litre of organic phase while ξ is on an aqueous-volume
    basis::

        c_org[j] = (c0_org[j] + Σ_r  ν[r,j] · ξ[r]) / v_oa

    Pinned species (``maintain_mask == True``) are not updated — their
    stoichiometry columns are zeroed out before applying the formula.

    Parameters
    ----------
    xi : np.ndarray, shape (n_reactions,)
        Reaction extents in mol/L (aqueous basis).
    c0_vec : np.ndarray, shape (n_species,)
        Initial concentrations.
    nu : np.ndarray, shape (n_reactions, n_species)
        Stoichiometric matrix.
    org_mask : np.ndarray of bool, shape (n_species,)
        True for organic-phase species.
    v_oa : float
        Organic/aqueous volume ratio.
    maintain_mask : np.ndarray of bool or None
        True for pinned (maintained) species; their concentrations are
        left unchanged.

    Returns
    -------
    np.ndarray, shape (n_species,)
        Equilibrium concentrations.
    """
    xi = np.asarray(xi, dtype=float)
    if maintain_mask is not None and np.any(maintain_mask):
        nu_eff = nu.copy()
        nu_eff[:, maintain_mask] = 0.0
    else:
        nu_eff = nu
    delta = nu_eff.T @ xi
    scale = np.where(org_mask, 1.0 / v_oa, 1.0)
    return c0_vec + delta * scale


def _xi_from_ceq(c_eq: np.ndarray,
                  c0_vec: np.ndarray,
                  nu: np.ndarray,
                  org_mask: np.ndarray,
                  v_oa: float,
                  active_mask: np.ndarray) -> np.ndarray:
    """Invert :func:`_conc_from_xi` via least-squares on the active species."""
    dc      = c_eq - c0_vec
    scale   = np.where(org_mask, v_oa, 1.0)
    dc_eff  = (dc * scale)[active_mask]
    nu_act  = nu[:, active_mask]
    xi, _, _, _ = np.linalg.lstsq(nu_act.T, dc_eff, rcond=None)
    return xi


def _lc_to_xi(lc_full: np.ndarray,
               c0_vec: np.ndarray,
               nu: np.ndarray,
               aq_mask: np.ndarray,
               org_mask: np.ndarray,
               v_oa: float,
               maintain_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert a log10-concentration vector to an extent vector."""
    c_eq = 10.0 ** np.clip(lc_full, -300, 300)
    dc   = c_eq - c0_vec
    if maintain_mask is not None and np.any(maintain_mask):
        dc[maintain_mask] = 0.0
    active_mask = (aq_mask | org_mask)
    if maintain_mask is not None:
        active_mask = active_mask & ~maintain_mask
    scale   = np.where(org_mask, v_oa, 1.0)
    dc_eff  = dc * scale
    nu_active = nu[:, active_mask]
    dc_active = dc_eff[active_mask]
    xi, _, _, _ = np.linalg.lstsq(nu_active.T, dc_active, rcond=None)
    return xi


def _equilibrium_concentrations(xi, c0, reactions_obj,
                                  _c_eq_direct=None,
                                  maintain=None) -> Dict[str, float]:
    """
    Convert (ξ, c0, reactions) → {species: equilibrium_concentration}.

    If *_c_eq_direct* is provided (from the solver) it is used directly,
    bypassing the ξ→c conversion.
    """
    rxn_list = (reactions_obj._reactions
                if hasattr(reactions_obj, '_reactions')
                else [reactions_obj])
    all_sp, c0_vec, nu, _, v_oa = _build_system(rxn_list, c0)
    org_mask = np.array([is_organic(s) for s in all_sp], dtype=bool)

    maintain_set = frozenset(species(s) for s in (maintain or []))
    maint_mask   = np.array([s in maintain_set for s in all_sp], dtype=bool)

    if _c_eq_direct is not None:
        c_eq = np.maximum(np.asarray(_c_eq_direct, dtype=float), 0.0)
        if np.any(maint_mask):
            for i in np.where(maint_mask)[0]:
                c_eq[i] = float(c0_vec[i])
    else:
        xi_arr = np.atleast_1d(np.asarray(xi, dtype=float))
        c_eq   = np.maximum(_conc_from_xi(xi_arr, c0_vec, nu, org_mask, v_oa,
                                           maintain_mask=maint_mask), 0.0)

    return {s: float(c_eq[i]) for i, s in enumerate(all_sp)
            if not is_electron(s)}


# ---------------------------------------------------------------------------
# Ksp helpers
# ---------------------------------------------------------------------------

def _ksp_solid_species(rxn) -> Tuple[str, float]:
    """Return (solid_species_name, coefficient) for a Ksp reaction."""
    from .species import is_solid
    for sp, coeff in rxn._stoich.items():
        if is_solid(sp):
            return sp, float(coeff)
    raise InputError(f"No solid species found in Ksp reaction: {rxn!r}.\n"
            "Mark solid species with '(s)' suffix, e.g. 'CaCO3(s)', "
            "and pass ksp=True.")


def _ksp_ion_product(rxn, c_eq: Dict[str, float],
                      gamma_dict: Optional[Dict[str, tuple]] = None) -> float:
    """Compute the ion-activity product Q for a Ksp reaction."""
    Q = 1.0
    for sp, coeff in rxn._stoich.items():
        from .species import is_solid
        if is_solid(sp) or is_electron(sp):
            continue
        a = max(float(_compute_activity(sp, c_eq, gamma_dict)), 1e-300)
        Q *= a ** abs(coeff)
    return Q


def _saturation_index(rxn, c_eq: Dict[str, float],
                       gamma_dict: Optional[Dict[str, tuple]] = None) -> float:
    """
    Return the saturation index  SI = log₁₀(Q / Ksp).

    The saturation index is a standard measure used in precipitation chemistry:

    - SI > 0  →  supersaturated: solution is above the solubility limit;
      precipitation of the solid phase is thermodynamically favoured.
    - SI = 0  →  exactly at equilibrium (solubility limit).
    - SI < 0  →  undersaturated: the solid will dissolve if present.

    Parameters
    ----------
    rxn : reaction
        A Ksp reaction (must have ``ksp=True`` and contain a solid species).
    c_eq : dict
        Current equilibrium concentrations {species_name: mol_per_L}.
    gamma_dict : dict or None
        Activity coefficient functions; ``None`` assumes ideal behaviour.

    Returns
    -------
    float
        log₁₀(Q / Ksp).
    """
    Q = _ksp_ion_product(rxn, c_eq, gamma_dict)
    return np.log10(max(Q, 1e-300) / rxn.K)


def _make_dissolution_rxn(rxn):
    """
    Flip a precipitation reaction into dissolution form if needed.

    Returns (dissolution_rxn, was_flipped).
    """
    sp_solid, nu_solid = _ksp_solid_species(rxn)
    if nu_solid < 0:          # already dissolution form
        return rxn, False
    flipped = object.__new__(type(rxn))
    flipped._stoich = {sp: -c for sp, c in rxn._stoich.items()}
    flipped.K = 1.0 / rxn.K
    flipped._gamma = dict(getattr(rxn, '_gamma', {}))
    return flipped, True


def _reaction_has_complete_side(rxn, c0, tol: float = 0.0) -> bool:
    """
    Return True if at least one side of the reaction has all species present
    at concentration > tol.  Used by the pre-solver to decide readiness.
    """
    reactant_active = [sp for sp, c in rxn._stoich.items()
                       if c < 0 and not is_nonaqueous(sp) and not is_electron(sp)]
    product_active  = [sp for sp, c in rxn._stoich.items()
                       if c > 0 and not is_nonaqueous(sp) and not is_electron(sp)]

    def side_complete(species_list):
        if not species_list:
            return True
        return all(float(c0.get(sp, 0.0)) > tol for sp in species_list)

    return side_complete(reactant_active) or side_complete(product_active)


# ---------------------------------------------------------------------------
# total() – atom-wise mass balance
# ---------------------------------------------------------------------------

def total(c: Dict[str, float]) -> Dict[str, float]:
    """
    Compute the total elemental concentration (mol/L aqueous basis) across all species.

    For each element present in the system, the total concentration is the sum
    of that element's contribution from every species in *c*, weighted by
    stoichiometric count.  Organic-phase species are scaled by ``1/v_oa`` so
    that all totals are expressed on a common aqueous-volume basis.

    This is useful for verifying mass conservation after a solve, or for
    computing the total loading of a metal across aqueous and organic phases.

    Parameters
    ----------
    c : dict
        Concentration dict ``{species_name: mol_per_L}``.  May optionally
        include ``'O/A'`` for the organic/aqueous volume ratio (default 1.0).

    Returns
    -------
    dict
        ``{element_symbol: total_mol_per_L_aqueous}``.

    Examples
    --------
    Check iron mass balance after extraction::

        from efta import total
        c = {'Fe[3+]': 0.001, 'FeA3(org)': 0.0009, 'O/A': 1.0}
        print(total(c))  # {'Fe': 0.0019, ...}
    """
    from .species import components as sp_components
    v_oa = float(c.get('O/A', 1.0))
    if v_oa <= 0:
        raise InputError(
            f"'O/A' (organic/aqueous volume ratio) must be positive, got {v_oa}")

    totals: Dict[str, float] = {}
    for raw_sp, conc in c.items():
        if raw_sp == 'O/A':
            continue
        sp = species(raw_sp)
        if is_electron(sp):
            continue
        conc = float(conc)
        if is_organic(sp):
            conc = conc / v_oa
        atoms = sp_components(sp)
        for elem, n in atoms.items():
            totals[elem] = totals.get(elem, 0.0) + n * conc
    return totals
