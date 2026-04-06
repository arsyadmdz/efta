"""
efta.balance
============
Conservation-law analysis: element balances, charge balances, and the
cluster-decomposition algorithm used internally by the Method-L solver.

This module is not typically used directly by end users.  It is called
automatically during every ``sys.equilibrium()`` call.

Key concepts
------------
Clusters
    Irreducible conservation units (conserved quantities) inferred from
    the reaction network.  For a simple acid-base system the cluster is
    typically ``'H'`` (total-proton conservation); for systems with
    overlapping elements the algorithm factors out minimal building blocks.

    Example — phosphate system::

        H3PO4  ⇌  H[+] + H2PO4[-]   →  clusters: H, H2PO4
        H2PO4[-] ⇌  H[+] + HPO4[2-]  →  cluster H already found
        HPO4[2-] ⇌  H[+] + PO4[3-]   →  cluster H already found

Decompose
    For each bare formula, express it as a linear combination of clusters.
    This is used to write the mass-balance constraints the solver must satisfy.

    Examples::

        SO4   →  {SO4: 1}         (SO4 is its own cluster)
        HSO4  →  {H: 1, SO4: 1}   (splits into H and SO4 clusters)

Excluded clusters
    Clusters derived entirely from pure-phase (solid or liquid) species are
    excluded from the aqueous mass-balance because their concentration is
    fixed by the phase rule (activity = 1 for a pure solid/liquid).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import numpy as np

from .species import (
    species, formula, charge, components,
    is_nonaqueous, is_organic, is_electron,
)

# ---------------------------------------------------------------------------
# Ksp reaction helpers
# ---------------------------------------------------------------------------

def _is_ksp_reaction(rxn) -> bool:
    """
    Classify a reaction as a Ksp (solubility product / precipitation) reaction.

    Heuristic: one side has exactly one solid '(s)' species and the other
    side contains only aqueous species.  This is the standard dissolution form::

        CaCO3(s)  ⇌  Ca[2+] + CO3[2-]   # Ksp, solid on left
        Ca[2+] + CO3[2-]  ⇌  CaCO3(s)   # Ksp, solid on right (same test)

    Returns True if either arrangement matches; False otherwise.
    """
    reactants = {sp for sp, c in rxn._stoich.items() if c < 0}
    products  = {sp for sp, c in rxn._stoich.items() if c > 0}
    from .species import is_solid

    def _check(lone, other):
        lone_solids  = [s for s in lone  if is_solid(s)]
        other_solids = [s for s in other if is_solid(s)]
        return (len(lone_solids) == 1
                and len(other_solids) == 0
                and all(not is_solid(s) for s in other))

    return _check(reactants, products) or _check(products, reactants)


def _reaction_is_mixed_phase(stoich: Dict[str, float]) -> bool:
    """True if a reaction involves both aqueous and organic species."""
    has_aq  = any(not is_nonaqueous(s) and not is_organic(s) and not is_electron(s)
                  for s in stoich)
    has_org = any(is_organic(s) for s in stoich)
    return has_aq and has_org


def _system_is_mixed_phase(reactions_list: list) -> bool:
    return any(_reaction_is_mixed_phase(r._stoich) for r in reactions_list)


def _system_has_electron(reactions_list: list) -> bool:
    return any(is_electron(sp)
               for rxn in reactions_list
               for sp in rxn._stoich)


# ---------------------------------------------------------------------------
# Element / charge balance builder
# ---------------------------------------------------------------------------

def _build_balance(all_species_iter) -> List[List[Tuple[float, str]]]:
    """
    Build element- and charge-balance rows from a collection of species.

    One row is generated per conserved element; an additional row is added
    if any species carries a non-zero charge.  Each row is a list of
    ``(stoichiometric_coefficient, species_name)`` pairs — only species that
    contribute to that element/charge row are included.

    Parameters
    ----------
    all_species_iter : iterable of str
        Species names to include in the balance (electrons are skipped).

    Returns
    -------
    list of list of (float, str)
        Each inner list represents one conservation equation.
        Example for H₂O + H[+] + OH[-]::

            [[(2.0, 'H2O'), (1.0, 'H[+]'), (1.0, 'OH[-]')],   # H balance
             [(1.0, 'H2O'), (1.0, 'OH[-]')],                    # O balance
             [(1.0, 'H[+]'), (-1.0, 'OH[-]')]]                  # charge balance
    """
    sp_list      = [s for s in all_species_iter if not is_electron(s)]
    compositions = {s: components(s) for s in sp_list}
    all_elements = sorted({elem for comp in compositions.values() for elem in comp})

    balances: List[List[Tuple[float, str]]] = []
    for elem in all_elements:
        row = [(compositions[s][elem], s)
               for s in sp_list if compositions[s].get(elem, 0) > 0]
        if row:
            balances.append(row)

    charges_map = {s: charge(s) for s in sp_list}
    if any(c != 0 for c in charges_map.values()):
        charge_row = [(charges_map[s], s) for s in sp_list if charges_map[s] != 0]
        if charge_row:
            balances.append(charge_row)
    return balances


# ---------------------------------------------------------------------------
# Cluster detection algorithm
# ---------------------------------------------------------------------------

def _compute_clusters_with_atoms(reactions_list: list):
    """
    Detect irreducible conservation clusters from the reaction network.

    This is the core of the Method-L mass-balance construction.  Clusters
    are the smallest atomic groupings that are individually conserved across
    all reactions in the system.

    Algorithm
    ---------
    1. Collect every bare formula (formula without charge or phase tag)
       appearing across all reactions.
    2. Mark formulas that appear as the *sole* species on one side of a
       multi-product reaction as "not a cluster candidate" — they must be
       sums of simpler clusters (e.g. ``H2SO4`` splits into ``H`` and ``SO4``
       if the reaction ``H2SO4 ⇌ H[+] + HSO4[-]`` is present).
    3. Iteratively split remaining candidates: if formula A contains formula B
       and B appears on the opposite side of any reaction from A, then A is
       expressed as ``B + remainder`` and the remainder becomes a new candidate.
    4. Repeat until no further splits are possible.

    Returns
    -------
    clusters : frozenset of str
        Cluster name strings (bare formulas of irreducible conserved units).
    atoms : dict
        Mapping of cluster name → element composition dict, including any
        synthetic remainder clusters created during splitting.
    """
    all_sp_in_rxns = [sp for rxn in reactions_list
                      for sp in rxn._stoich
                      if sp != 'O/A' and not is_electron(sp)]

    all_bare = sorted({formula(sp) for sp in all_sp_in_rxns})
    atoms: Dict[str, Dict[str, float]] = {b: dict(components(b)) for b in all_bare}
    bare_of: Dict[str, str] = {sp: formula(sp) for sp in all_sp_in_rxns}

    def _atom_key(d: Dict[str, float]) -> tuple:
        return tuple(sorted(d.items()))

    # formulas that cannot be clusters (they appear as the lone species in a
    # reaction with multiple products → they are sums of simpler clusters)
    not_cluster: set = set()
    for rxn in reactions_list:
        stoich  = rxn._stoich
        sp_list = [sp for sp in stoich if sp != 'O/A' and not is_electron(sp)]
        left  = [sp for sp in sp_list if stoich[sp] < 0]
        right = [sp for sp in sp_list if stoich[sp] > 0]
        for lone, other in [(left, right), (right, left)]:
            if len(lone) == 1 and len(other) >= 2:
                not_cluster.add(bare_of[lone[0]])

    candidates: set = {b for b in all_bare if b not in not_cluster}

    # track which pairs of formulas appear on opposite sides of a reaction
    opposite_pairs: set = set()
    for rxn in reactions_list:
        stoich  = rxn._stoich
        sp_list = [sp for sp in stoich if sp != 'O/A' and not is_electron(sp)]
        left_bare  = {bare_of[sp] for sp in sp_list if stoich[sp] < 0}
        right_bare = {bare_of[sp] for sp in sp_list if stoich[sp] > 0}
        for lb in left_bare:
            for rb in right_bare:
                opposite_pairs.add((lb, rb))
                opposite_pairs.add((rb, lb))

    originals: FrozenSet[str] = frozenset(all_bare)

    # iterative splitting
    changed = True
    while changed:
        changed = False
        to_remove: set = set()
        new_candidates: Dict[str, Dict[str, float]] = {}

        key_to_name: Dict[tuple, str] = {_atom_key(atoms[b]): b for b in candidates}

        for b_big in list(candidates):
            for b_small in list(candidates):
                if b_big == b_small:
                    continue
                a_big   = atoms[b_big]
                a_small = atoms[b_small]
                if not a_small:
                    continue
                if not all(a_big.get(e, 0) >= a_small[e] for e in a_small):
                    continue
                remainder = {e: a_big[e] - a_small.get(e, 0) for e in a_big}
                remainder = {e: v for e, v in remainder.items() if v > 0}
                if not remainder:
                    continue
                # only split originals if they appear on opposite sides
                if b_big in originals and b_small in originals:
                    if (b_small, b_big) not in opposite_pairs:
                        continue
                rkey = _atom_key(remainder)
                if rkey in key_to_name:
                    rem_name = key_to_name[rkey]
                else:
                    rem_name = ''.join(
                        (elem if cnt == 1
                         else f'{elem}{int(cnt) if cnt == int(cnt) else cnt}')
                        for elem, cnt in sorted(remainder.items())
                    )
                to_remove.add(b_big)
                queued_keys = {_atom_key(v) for v in new_candidates.values()}
                if rkey not in key_to_name and rkey not in queued_keys:
                    new_candidates[rem_name] = remainder
                changed = True
                break

        candidates -= to_remove
        for name, rem_atoms in new_candidates.items():
            candidates.add(name)
            atoms[name] = rem_atoms

    return frozenset(candidates), atoms


def _compute_clusters(reactions_list: list) -> FrozenSet[str]:
    """Return the set of conservation-cluster names for the reaction system."""
    clusters, _ = _compute_clusters_with_atoms(reactions_list)
    return clusters


def _compute_decompose(reactions_list: list) -> Dict[str, Dict[str, float]]:
    """
    Express every bare formula in the system as a linear combination of clusters.

    This decomposition is used to build the mass-balance matrix that the
    Method-L solver uses.  Each entry maps a bare formula to the cluster counts
    needed to reconstruct it.

    Returns
    -------
    dict
        ``{bare_formula: {cluster_name: count, ...}, ...}``

    Examples
    --------
    For a carbonate system with clusters ``{'H', 'CO3'}``::

        {
          'H2CO3':  {'H': 2, 'CO3': 1},
          'HCO3':   {'H': 1, 'CO3': 1},
          'CO3':    {'CO3': 1},
          'H':      {'H': 1},
        }
    """
    clusters_set, cluster_atoms_map = _compute_clusters_with_atoms(reactions_list)

    all_sp = [sp for rxn in reactions_list for sp in rxn._stoich
              if sp != 'O/A' and not is_electron(sp)]
    all_bare = sorted({formula(sp) for sp in all_sp})

    cluster_atoms: Dict[str, Dict[str, float]] = {
        c: cluster_atoms_map[c] for c in clusters_set if c in cluster_atoms_map
    }

    def _decompose_bare(b: str) -> Dict[str, float]:
        if b in clusters_set:
            return {b: 1.0}
        b_atoms   = dict(components(b))
        remaining = dict(b_atoms)
        result: Dict[str, float] = {}
        for c in sorted(cluster_atoms,
                        key=lambda x: sum(cluster_atoms[x].values()), reverse=True):
            ca = cluster_atoms[c]
            if not ca: continue
            counts = [remaining.get(e, 0) / ca[e] for e in ca if ca[e] > 0]
            n = int(min(counts)) if counts else 0
            if n > 0:
                for e, v in ca.items():
                    remaining[e] = remaining.get(e, 0) - n * v
                result[c] = result.get(c, 0.0) + n
        remaining = {e: v for e, v in remaining.items() if v > 0}
        if remaining:
            return {b: 1.0}
        return {k: int(v) if v == int(v) else v for k, v in result.items()}

    return {b: _decompose_bare(b) for b in all_bare}


def _compute_excluded_clusters(reactions_list: list) -> FrozenSet[str]:
    """
    Return clusters that arise exclusively from pure-phase (solid or liquid) species.

    By the phase rule, the chemical potential of a pure solid or liquid at a given
    temperature and pressure is fixed — its activity is conventionally set to 1
    and its concentration is not tracked as a free variable.  Any conservation
    cluster derived *only* from such species should therefore be excluded from
    the aqueous mass-balance constraints.

    For example, in the reaction::

        CaCO3(s)  ⇌  Ca[2+] + CO3[2-]

    the clusters ``Ca`` and ``CO3`` come entirely from the solid ``CaCO3(s)``
    and are excluded from the aqueous mass-balance rows.

    Returns
    -------
    frozenset of str
        Cluster names that must be excluded from the mass-balance constraints.
    """
    all_sp = [sp for rxn in reactions_list
              for sp in rxn._stoich
              if sp != 'O/A' and not is_electron(sp)]
    bare: Dict[str, str] = {sp: formula(sp) for sp in all_sp}
    clusters_set = _compute_clusters(reactions_list)

    directly_from_nonaq: set = set()
    for rxn in reactions_list:
        stoich  = rxn._stoich
        sp_list = [sp for sp in stoich if sp != 'O/A' and not is_electron(sp)]
        left  = [sp for sp in sp_list if stoich[sp] < 0]
        right = [sp for sp in sp_list if stoich[sp] > 0]
        for lone_side, other_side in [(left, right), (right, left)]:
            if len(lone_side) == 1 and len(other_side) >= 2:
                if not is_nonaqueous(lone_side[0]):
                    continue
                for sp_other in other_side:
                    directly_from_nonaq.add(bare[sp_other])

    if not directly_from_nonaq:
        return frozenset()

    decomp = _compute_decompose(reactions_list)
    excluded: set = set()
    for b in directly_from_nonaq:
        if b in clusters_set:
            excluded.add(b)
        else:
            for cluster_b in decomp.get(b, {b: 1.0}).keys():
                if cluster_b in clusters_set:
                    excluded.add(cluster_b)
    return frozenset(excluded)


def _compute_maintained_clusters(maintain_mask: np.ndarray,
                                  all_species: List[str],
                                  reactions_list: list) -> FrozenSet[str]:
    """
    Return clusters that should be excluded because the user has *pinned*
    (maintained) one or more species at their initial concentrations.

    When a species is maintained (its concentration is held fixed and not
    allowed to change during the solve), the cluster it belongs to is no
    longer a free mass-balance constraint — the maintained concentration
    effectively acts as an external reservoir.

    Parameters
    ----------
    maintain_mask : np.ndarray of bool, shape (n_species,)
        True at index ``i`` if ``all_species[i]`` is pinned.
    all_species : list of str
        Species list in the same order as ``maintain_mask``.
    reactions_list : list of reaction
        The full reaction system.

    Returns
    -------
    frozenset of str
        Cluster names that must be excluded from the mass-balance constraints.
    """
    if not np.any(maintain_mask):
        return frozenset()

    clusters_set = _compute_clusters(reactions_list)
    decomp       = _compute_decompose(reactions_list)

    maintained_sp = [all_species[i] for i in range(len(all_species))
                     if maintain_mask[i]]
    excluded: set = set()
    for sp in maintained_sp:
        b = formula(sp)
        if b in clusters_set:
            excluded.add(b)
        else:
            for cluster_b in decomp.get(b, {b: 1.0}).keys():
                if cluster_b in clusters_set:
                    excluded.add(cluster_b)
    return frozenset(excluded)
