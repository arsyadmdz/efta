"""
efta.solver._shared
===================
Utilities shared by all solver methods:

- :class:`_TimeoutError` / :class:`_TimedCall` – thread-based timeout guard
- :func:`_reactions_type` – classify a reaction list
- Constraint builders: :func:`_build_ksp_stoich_constraints`,
  :func:`_build_stoich_symmetry_constraints`
- Log-space system assembly: :func:`_build_logspace_system`,
  :func:`_build_B_residual_system`, :func:`_build_logspace_and_constraints`
- :func:`_logspace_initial_guesses` – warm-start strategies A and B
- :func:`_compute_extent_bounds` – per-reaction ξ bounds
- :func:`_clamp_ksp_solids` – post-solve solid-exhaustion clamp
"""

from __future__ import annotations

import threading as _threading
import warnings
from typing import Dict, List, Tuple

import numpy as np

from ..species import formula, is_nonaqueous, is_organic, is_electron, is_solid
from ..balance import (
    _is_ksp_reaction,
    _compute_clusters,
    _compute_decompose,
    _compute_excluded_clusters,
    _compute_maintained_clusters,
)

warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message='.*not making good progress.*')

_EPS_L = 1e-300


# ---------------------------------------------------------------------------
# Timeout guard
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


class _TimedCall:
    """Run *fn(*args, **kwargs)* in a daemon thread with a timeout."""

    def __init__(self, timeout: float):
        self.timeout = timeout

    def __call__(self, fn, *args, **kwargs):
        result   = [None]
        exc      = [None]
        finished = _threading.Event()

        def _worker():
            try:
                result[0] = fn(*args, **kwargs)
            except Exception as e:
                exc[0] = e
            finally:
                finished.set()

        t = _threading.Thread(target=_worker, daemon=True)
        t.start()
        if not finished.wait(timeout=self.timeout):
            raise _TimeoutError(
                f"Solver did not converge within {self.timeout:.1f}s")
        if exc[0] is not None:
            raise exc[0]
        return result[0]


# ---------------------------------------------------------------------------
# Cache for the log-space constraint system
# ---------------------------------------------------------------------------

_logspace_cache: Dict = {}
_LOGSPACE_CACHE_MAX = 64


# ---------------------------------------------------------------------------
# Reaction-type classifier
# ---------------------------------------------------------------------------

def _reactions_type(reactions_list: list) -> str:
    """Return ``'speciation'``, ``'precipitation'``, or ``'mix'``."""
    has_ksp  = any(_is_ksp_reaction(r) for r in reactions_list)
    has_spec = any(not _is_ksp_reaction(r) for r in reactions_list)
    if has_ksp and has_spec:
        return 'mix'
    if has_ksp:
        return 'precipitation'
    return 'speciation'


# ---------------------------------------------------------------------------
# Stoichiometric constraint builders  (used by Methods B and DE)
# ---------------------------------------------------------------------------

def _build_ksp_stoich_constraints(nu, c0_vec, all_species, active_idx, n_act):
    """Generate linear constraints from Ksp reaction stoichiometry ratios."""
    n_rxn, n_sp = nu.shape
    act_set   = set(active_idx.tolist())
    act_local = {int(active_idx[j]): j for j in range(n_act)}

    constraints: List[Tuple[np.ndarray, float]] = []
    seen: List[np.ndarray] = []

    for r in range(n_rxn):
        solid_cols = [s for s in range(n_sp)
                      if is_solid(all_species[s]) and nu[r, s] != 0]
        act_cols   = [s for s in range(n_sp)
                      if s in act_set and nu[r, s] != 0]

        if len(solid_cols) != 1 or len(act_cols) < 2:
            continue

        pivot_s  = act_cols[0]
        pivot_nu = nu[r, pivot_s]
        pivot_j  = act_local[pivot_s]

        for s in act_cols[1:]:
            nu_s  = nu[r, s]
            loc_s = act_local[s]

            coeff          = np.zeros(n_act)
            coeff[pivot_j] =  nu_s
            coeff[loc_s]   = -pivot_nu
            rhs = nu_s * c0_vec[pivot_s] - pivot_nu * c0_vec[s]

            norm = np.linalg.norm(coeff)
            if norm < 1e-14:
                continue
            c_norm = coeff / norm
            duplicate = any(
                np.max(np.abs(sc - c_norm)) < 1e-9 or
                np.max(np.abs(sc + c_norm)) < 1e-9
                for sc in seen)
            if not duplicate:
                seen.append(c_norm)
                constraints.append((coeff, rhs))

    return constraints


def _build_stoich_symmetry_constraints(c0_vec, nu, aq_mask):
    """Generate stoichiometric ratio constraints for reactions involving solids."""
    n_rxn, n_sp = nu.shape
    aq_idx  = np.where(aq_mask)[0]
    n_aq    = len(aq_idx)
    constraints, seen = [], []

    rxns_of = {j: frozenset(r for r in range(n_rxn) if nu[r, aq_idx[j]] != 0)
               for j in range(n_aq)}

    for r in range(n_rxn):
        nonaq_involved = any(nu[r, s] != 0 and not aq_mask[s] for s in range(n_sp))
        if not nonaq_involved:
            continue
        nu_aq_r  = nu[r, aq_idx]
        nonzero  = [(j, nu_aq_r[j]) for j in range(n_aq) if nu_aq_r[j] != 0]
        if len(nonzero) < 2:
            continue
        pivot_j, pivot_v = max(nonzero, key=lambda x: abs(x[1]))
        for j, v in nonzero:
            if j == pivot_j:
                continue
            if rxns_of[j] != rxns_of[pivot_j]:
                continue
            coeff          = np.zeros(n_aq)
            coeff[j]       =  pivot_v
            coeff[pivot_j] = -v
            rhs = pivot_v * c0_vec[aq_idx[j]] - v * c0_vec[aq_idx[pivot_j]]
            norm = np.linalg.norm(coeff)
            if norm < 1e-14:
                continue
            c_norm = coeff / norm
            duplicate = False
            for sc in seen:
                if (np.max(np.abs(sc - c_norm)) < 1e-9 or
                        np.max(np.abs(sc + c_norm)) < 1e-9):
                    duplicate = True
                    break
            if not duplicate:
                seen.append(c_norm)
                constraints.append((coeff, rhs))

    return constraints


# ---------------------------------------------------------------------------
# Log-space system builder
# ---------------------------------------------------------------------------

def _build_logspace_system(c0_vec, nu, K_vec, all_species, aq_mask, org_mask,
                            v_oa, maintain_mask=None):
    """
    Build the log-space equation system:

    - equilibrium rows: ``ν_act · lc + γ_correction = log K``
    - mass-balance rows: ``count_vec · c = T_cluster``
    - charge-balance row (if applicable)

    Returns all arrays/metadata needed by Methods B and DE.
    """
    n_rxn, n_sp = nu.shape

    if maintain_mask is None:
        maintain_mask = np.zeros(n_sp, dtype=bool)

    active_mask = (aq_mask | org_mask) & ~maintain_mask
    active_idx  = np.where(active_mask)[0]
    n_act       = int(len(active_idx))
    act_names   = [all_species[i] for i in active_idx]

    maintain_active_idx = np.where(maintain_mask & (aq_mask | org_mask))[0]
    lc_maintained = np.log10(np.maximum(c0_vec[maintain_active_idx], _EPS_L))

    nu_act  = nu[:, active_idx]
    lK_raw  = np.log10(np.maximum(K_vec, 1e-300))
    if len(maintain_active_idx) > 0:
        lK_adj = lK_raw - nu[:, maintain_active_idx] @ lc_maintained
    else:
        lK_adj = lK_raw.copy()

    rxn_has_act = np.any(nu_act != 0, axis=1)
    nu_act_rows = nu_act[rxn_has_act]
    lK_rows     = lK_adj[rxn_has_act]

    act_name_to_local = {all_species[active_idx[j]]: j for j in range(n_act)}

    class _FakeRxn:
        def __init__(self, stoich): self._stoich = stoich
    fake_rxns = [_FakeRxn({all_species[s]: float(nu[r, s])
                            for s in range(n_sp) if nu[r, s] != 0})
                 for r in range(n_rxn)]

    excluded_clusters = _compute_excluded_clusters(fake_rxns)
    if len(maintain_active_idx) > 0:
        excluded_clusters = excluded_clusters | _compute_maintained_clusters(
            maintain_mask, all_species, fake_rxns)

    decomp       = _compute_decompose(fake_rxns)
    all_clusters = _compute_clusters(fake_rxns)
    bare_of: Dict[str, str] = {sp: formula(sp) for sp in act_names}

    relevant_clusters = frozenset(
        c for c in all_clusters
        if c not in excluded_clusters
        and any(decomp.get(bare_of.get(sp, ''), {}).get(c, 0) != 0
                for sp in act_names)
    )

    best_by_cluster: Dict[str, tuple] = {}
    for cluster in sorted(relevant_clusters):
        count_vec = np.zeros(n_act)
        for j, sp in enumerate(act_names):
            b     = bare_of.get(sp, formula(sp))
            coeff = decomp.get(b, {}).get(cluster, 0.0)
            if coeff == 0.0:
                continue
            weight    = v_oa if is_organic(sp) else 1.0
            count_vec[j] = float(coeff) * weight
        if not np.any(count_vec != 0):
            continue
        T = float(count_vec @ np.array([c0_vec[active_idx[j]] for j in range(n_act)]))
        for gi in maintain_active_idx:
            sp_m = all_species[gi]
            b_m  = formula(sp_m)
            c_m  = decomp.get(b_m, {}).get(cluster, 0.0)
            if c_m:
                weight_m = v_oa if is_organic(sp_m) else 1.0
                T -= float(c_m) * weight_m * float(c0_vec[gi])
        if T <= 0:
            continue
        best_by_cluster[cluster] = (count_vec, T)

    groups = list(best_by_cluster.values())

    e_col = next((j for j, s in enumerate(all_species) if is_electron(s)), None)
    suppress_charge = (e_col is not None
                       and abs(float(np.sum(nu[:, e_col]))) > 1e-12)
    if not suppress_charge and len(maintain_active_idx) > 0:
        from ..species import charge as sp_charge
        if any(float(sp_charge(all_species[gi])) != 0
               for gi in maintain_active_idx):
            suppress_charge = True

    from ..species import charge as sp_charge
    charges = np.array([
        sp_charge(s) if not is_organic(s) else 0.0
        for s in act_names
    ], dtype=float)
    n_charge = 0 if suppress_charge else (1 if bool(np.any(charges != 0)) else 0)

    return (active_idx, n_act, nu_act_rows, lK_rows,
            groups, charges, n_charge, act_name_to_local)


def _build_B_residual_system(c0_vec, nu, aq_mask, nu_act_rows, lK_rows,
                              groups, charges, n_charge, active_idx, n_act,
                              all_species=None):
    """Build extra constraints to make the system square (n_eq == n_active)."""
    n_eq_base = len(lK_rows) + len(groups) + n_charge
    n_extra   = n_act - n_eq_base
    if n_extra <= 0:
        return []

    constraints: List[Tuple[np.ndarray, float]] = []
    seen: List[np.ndarray] = []

    if all_species is not None:
        for coeff, rhs in _build_ksp_stoich_constraints(
                nu, c0_vec, all_species, active_idx, n_act):
            if len(constraints) >= n_extra:
                break
            norm = np.linalg.norm(coeff)
            if norm < 1e-14:
                continue
            c_norm = coeff / norm
            if not any(np.max(np.abs(sc - c_norm)) < 1e-9 or
                       np.max(np.abs(sc + c_norm)) < 1e-9 for sc in seen):
                seen.append(c_norm)
                constraints.append((coeff, rhs))

    if len(constraints) < n_extra:
        aq_idx  = np.where(aq_mask)[0]
        act_set = set(active_idx.tolist())
        for coeff_aq, rhs_aq in _build_stoich_symmetry_constraints(c0_vec, nu, aq_mask):
            if len(constraints) >= n_extra:
                break
            coeff_act = np.zeros(n_act)
            for k, global_k in enumerate(aq_idx):
                if global_k in act_set:
                    local_k = int(np.where(active_idx == global_k)[0][0])
                    coeff_act[local_k] = coeff_aq[k]
            norm = np.linalg.norm(coeff_act)
            if norm < 1e-14:
                continue
            c_norm = coeff_act / norm
            if not any(np.max(np.abs(sc - c_norm)) < 1e-9 or
                       np.max(np.abs(sc + c_norm)) < 1e-9 for sc in seen):
                seen.append(c_norm)
                constraints.append((coeff_act, rhs_aq))

    return constraints


def _build_logspace_and_constraints(c0_vec, nu, K_vec, all_species,
                                     aq_mask, org_mask, v_oa, maintain_mask=None):
    """
    Cached wrapper: build the full log-space system + supplementary constraints.

    Two-level caching strategy
    --------------------------
    During a parameter sweep the reaction network (nu, K_vec, species, masks)
    is fixed while c0_vec changes at every step.  The expensive work —
    cluster decomposition, active-species selection, constraint generation —
    depends only on the *structure*, not on the current concentrations.

    **Structural cache** (keyed on nu, K_vec, all_species, masks):
      Stores count_vecs, maintain_active_idx, nu_act_rows (unadjusted),
      lK_raw, charges, n_charge, active_idx, act_name_to_local,
      sym_constraints.  Built once per unique reaction system.

    **Per-call layer** (runs on every call, cheap):
      Computes lK_rows (adjusted for maintained species) and T totals
      (count_vec · c0_vec) from the cached structural data.

    This means a 100-point sweep over the same system hits the structural
    cache 99 times and only recomputes two dot-products per step.
    """
    global _logspace_cache

    # --- structural key: everything except c0_vec ---
    struct_key = (
        nu.tobytes(), K_vec.tobytes(),
        tuple(all_species), aq_mask.tobytes(), org_mask.tobytes(),
        float(v_oa),
        (maintain_mask.tobytes() if maintain_mask is not None else b''),
    )

    if struct_key not in _logspace_cache:
        # Build the full system once with a dummy c0 of zeros — we only need
        # the structural outputs (active_idx, count_vecs, nu_act_rows, etc.).
        # The concentration-dependent parts (lK_rows, T totals) are computed
        # below from the real c0_vec each call.
        (active_idx, n_act, nu_act_rows_raw, lK_raw_rows,
         _groups_raw, charges, n_charge,
         act_name_to_local) = _build_logspace_system(
            np.zeros_like(c0_vec), nu, K_vec, all_species, aq_mask, org_mask, v_oa,
            maintain_mask=maintain_mask)

        # Recompute count_vecs independent of c0 (T totals stripped out)
        if maintain_mask is None:
            maintain_mask_arr = np.zeros(nu.shape[1], dtype=bool)
        else:
            maintain_mask_arr = maintain_mask

        active_mask         = (aq_mask | org_mask) & ~maintain_mask_arr
        maintain_active_idx = np.where(maintain_mask_arr & (aq_mask | org_mask))[0]

        # Rebuild count_vecs without the T totals (those depend on c0_vec)
        class _FakeRxn:
            def __init__(self, stoich): self._stoich = stoich
        n_rxn, n_sp = nu.shape
        fake_rxns = [_FakeRxn({all_species[s]: float(nu[r, s])
                                for s in range(n_sp) if nu[r, s] != 0})
                     for r in range(n_rxn)]

        excluded  = _compute_excluded_clusters(fake_rxns)
        if len(maintain_active_idx) > 0:
            excluded = excluded | _compute_maintained_clusters(
                maintain_mask_arr, all_species, fake_rxns)
        decomp       = _compute_decompose(fake_rxns)
        all_clusters = _compute_clusters(fake_rxns)
        act_names    = [all_species[i] for i in active_idx]
        bare_of      = {sp: formula(sp) for sp in act_names}

        relevant = frozenset(
            c for c in all_clusters
            if c not in excluded
            and any(decomp.get(bare_of.get(sp, ''), {}).get(c, 0) != 0
                    for sp in act_names))

        count_vecs    = []   # shape: (n_clusters, n_act) — concentration-free
        cluster_order = []
        for cluster in sorted(relevant):
            cv = np.zeros(n_act)
            for j, sp in enumerate(act_names):
                coeff = decomp.get(bare_of.get(sp, formula(sp)), {}).get(cluster, 0.0)
                if coeff:
                    cv[j] = float(coeff) * (v_oa if is_organic(sp) else 1.0)
            if np.any(cv != 0):
                count_vecs.append(cv)
                cluster_order.append(cluster)

        # maintain_count_vecs: contribution to subtract from T for each cluster
        maintain_count_vecs = []
        for ci, cv in enumerate(count_vecs):
            mcv = np.zeros(len(maintain_active_idx))
            for mi, gi in enumerate(maintain_active_idx):
                sp_m    = all_species[gi]
                coeff_m = decomp.get(formula(sp_m), {}).get(cluster_order[ci], 0.0)
                if coeff_m:
                    mcv[mi] = float(coeff_m) * (v_oa if is_organic(sp_m) else 1.0)
            maintain_count_vecs.append(mcv)

        # lK_raw: unadjusted log K (adjustment for maintained species done per-call)
        lK_raw = np.log10(np.maximum(K_vec, 1e-300))
        nu_act = nu[:, active_idx]
        rxn_has_act = np.any(nu_act != 0, axis=1)
        nu_act_rows = nu_act[rxn_has_act]
        lK_raw_active = lK_raw[rxn_has_act]

        # nu for maintained species (for lK adjustment)
        nu_maintain = nu[rxn_has_act][:, maintain_active_idx] \
                      if len(maintain_active_idx) > 0 else None

        sym_constraints = _build_B_residual_system(
            np.zeros_like(c0_vec), nu, aq_mask, nu_act_rows, lK_raw_active,
            [(cv, 1.0) for cv in count_vecs],   # dummy T=1 — coeff structure only
            charges, n_charge, active_idx, n_act, all_species=all_species)

        structural = {
            'active_idx':           active_idx,
            'n_act':                n_act,
            'nu_act_rows':          nu_act_rows,
            'lK_raw_active':        lK_raw_active,
            'nu_maintain':          nu_maintain,
            'maintain_active_idx':  maintain_active_idx,
            'count_vecs':           count_vecs,
            'maintain_count_vecs':  maintain_count_vecs,
            'charges':              charges,
            'n_charge':             n_charge,
            'act_name_to_local':    act_name_to_local,
            'sym_constraints':      sym_constraints,
        }

        if len(_logspace_cache) >= _LOGSPACE_CACHE_MAX:
            _logspace_cache.pop(next(iter(_logspace_cache)))
        _logspace_cache[struct_key] = structural

    s = _logspace_cache[struct_key]

    # --- per-call: compute lK_rows and T totals from current c0_vec ---
    maintain_active_idx = s['maintain_active_idx']
    lK_rows = s['lK_raw_active'].copy()
    if s['nu_maintain'] is not None and len(maintain_active_idx) > 0:
        lc_maint  = np.log10(np.maximum(c0_vec[maintain_active_idx], _EPS_L))
        lK_rows  -= s['nu_maintain'] @ lc_maint

    groups = []
    for cv, mcv in zip(s['count_vecs'], s['maintain_count_vecs']):
        T = float(cv @ c0_vec[s['active_idx']])
        if len(maintain_active_idx) > 0:
            T -= float(mcv @ c0_vec[maintain_active_idx])
        if T > 0:
            groups.append((cv, T))

    return (s['active_idx'], s['n_act'], s['nu_act_rows'], lK_rows,
            groups, s['charges'], s['n_charge'],
            s['act_name_to_local'], s['sym_constraints'])


# ---------------------------------------------------------------------------
# Initial-guess strategies for Methods B and DE
# ---------------------------------------------------------------------------

def _logspace_initial_guesses(c0_vec, active_idx, n_act, nu_act_rows, lK_rows,
                               groups, act_name_to_local, lc_warm=None):
    """Generate a list of initial-guess vectors in log10(c) space."""
    seeds       = [i for i in active_idx if c0_vec[i] > 0]
    c_tot_guess = max((c0_vec[s] for s in seeds), default=1e-3)

    def _propagate(lc, known):
        changed, itr = True, 0
        while changed and itr < n_act * len(nu_act_rows) + 10:
            changed = False; itr += 1
            for ri in range(len(nu_act_rows)):
                unknowns = [j for j in range(n_act)
                            if nu_act_rows[ri, j] != 0 and j not in known]
                if len(unknowns) == 1:
                    j = unknowns[0]; v = nu_act_rows[ri, j]
                    rhs = lK_rows[ri] - sum(
                        nu_act_rows[ri, k] * lc[k]
                        for k in range(n_act)
                        if k != j and nu_act_rows[ri, k] != 0)
                    lc[j] = np.clip(rhs / v, -300, 300)
                    known.add(j); changed = True
        return lc

    def strategy_A(scale=1.0):
        lc    = np.full(n_act, -7.0)
        known = set()
        c_init = np.full(n_act, 1e-10)
        for count_vec, C_total in groups:
            group_seeds = [j for j in range(n_act)
                           if count_vec[j] > 0 and c0_vec[active_idx[j]] > 0]
            if not group_seeds:
                continue
            share = (C_total * scale) / len(group_seeds)
            for j in group_seeds:
                c_init[j] = max(c_init[j], share / count_vec[j])
        for seed in seeds:
            idxs = np.where(active_idx == seed)[0]
            if len(idxs):
                local = int(idxs[0])
                lc[local] = np.log10(max(c_init[local], 1e-30))
                known.add(local)
        if not known and len(lK_rows) > 0:
            lc_est = np.sum(lK_rows) / max(len(lK_rows), 1) / 2
            lc = np.full(n_act, np.clip(lc_est, -14, 0))
        return _propagate(lc, known)

    def strategy_B(frac_product=0.95, pH_guess=5.0):
        lc    = np.full(n_act, -7.0)
        known = set()
        seed_locals = {int(np.where(active_idx == s)[0][0]) for s in seeds
                       if len(np.where(active_idx == s)[0])}
        best_r, best_K, best_prod_local = -1, -np.inf, -1
        for ri in range(len(nu_act_rows)):
            react_l = [j for j in range(n_act) if nu_act_rows[ri, j] < 0]
            prod_l  = [j for j in range(n_act) if nu_act_rows[ri, j] > 0]
            if (react_l and all(j in seed_locals for j in react_l)
                    and len(prod_l) == 1):
                K_r = 10.0 ** lK_rows[ri]
                if K_r > best_K:
                    best_K = K_r; best_r = ri; best_prod_local = prod_l[0]
        if best_r == -1:
            return strategy_A()
        C_total = c_tot_guess
        for count_vec, Ct in groups:
            if count_vec[best_prod_local] > 0:
                C_total = Ct; break
        lc[best_prod_local] = np.log10(max(frac_product * C_total, 1e-30))
        known.add(best_prod_local)
        react_l   = [j for j in range(n_act) if nu_act_rows[best_r, j] < 0]
        nu_prod   = nu_act_rows[best_r, best_prod_local]
        rhs_react = lK_rows[best_r] - nu_prod * lc[best_prod_local]
        sum_nu    = sum(abs(nu_act_rows[best_r, j]) for j in react_l)
        for j in react_l:
            nu_j  = nu_act_rows[best_r, j]
            lc[j] = np.clip(rhs_react * (abs(nu_j) / sum_nu) / nu_j, -300, 0)
            known.add(j)
        h_loc  = act_name_to_local.get('H[+]')
        oh_loc = act_name_to_local.get('OH[-]')
        if h_loc is not None and h_loc not in known:
            lc[h_loc] = -pH_guess; known.add(h_loc)
        if oh_loc is not None and oh_loc not in known:
            lc[oh_loc] = np.log10(1e-14) + pH_guess; known.add(oh_loc)
        return _propagate(lc, known)

    candidates = []
    if lc_warm is not None:
        candidates.append(np.clip(lc_warm, -300, 300))
    for sc in [1.0, 0.1, 0.01, 1e-4]:
        candidates.append(strategy_A(scale=sc))
    for frac in [0.99, 0.80, 0.50]:
        for pH in [3.0, 4.0, 5.0, 6.0, 7.0]:
            candidates.append(strategy_B(frac_product=frac, pH_guess=pH))
    return candidates


# ---------------------------------------------------------------------------
# Extent bounds (for Method A and DE)
# ---------------------------------------------------------------------------

def _compute_extent_bounds(n_rxn, n_sp, nu, c0_vec, active_mask, org_mask,
                            all_species, v_oa, total_mass, maintain_mask=None):
    """Compute per-reaction lower/upper bounds on ξ from stoichiometric constraints."""
    upper = np.full(n_rxn, np.inf)
    lower = np.full(n_rxn, -np.inf)

    if maintain_mask is not None and np.any(maintain_mask):
        nu_eff = nu.copy()
        nu_eff[:, maintain_mask] = 0.0
    else:
        nu_eff = nu

    for r in range(n_rxn):
        for s in range(n_sp):
            v = nu_eff[r, s]
            if v == 0:
                continue

            if is_solid(all_species[s]) and nu[r, s] < 0 and c0_vec[s] > 0:
                upper[r] = min(upper[r], c0_vec[s] / abs(nu[r, s]))

            if not active_mask[s]:
                continue
            scale = (1.0 / v_oa) if org_mask[s] else 1.0
            eff_v = v * scale

            if eff_v < 0:
                if c0_vec[s] > 0:
                    upper[r] = min(upper[r], c0_vec[s] / abs(eff_v))
                elif not any(nu_eff[r2, s] > 0 for r2 in range(n_rxn) if r2 != r):
                    upper[r] = 0.0
            elif eff_v > 0:
                if c0_vec[s] > 0:
                    lower[r] = max(lower[r], -c0_vec[s] / eff_v)
                elif not any(nu_eff[r2, s] < 0 for r2 in range(n_rxn) if r2 != r):
                    lower[r] = 0.0

        if np.isinf(upper[r]): upper[r] = total_mass
        if np.isinf(lower[r]): lower[r] = 0.0
        if upper[r] < lower[r]: upper[r] = lower[r] + total_mass

    return lower, upper


# ---------------------------------------------------------------------------
# Solid-clamping utility
# ---------------------------------------------------------------------------

def _clamp_ksp_solids(c_eq, c0_vec, nu, all_species, aq_mask, org_mask, v_oa):
    """Clamp any solids that would go negative (full dissolution)."""
    c_eq    = np.array(c_eq, dtype=float)
    n_rxn   = nu.shape[0]
    clamped = False

    for r in range(n_rxn):
        solid_cols  = [(s, nu[r, s]) for s in range(nu.shape[1])
                       if is_solid(all_species[s]) and nu[r, s] != 0]
        active_cols = [(s, nu[r, s]) for s in range(nu.shape[1])
                       if (aq_mask[s] or org_mask[s]) and nu[r, s] != 0]

        if len(solid_cols) != 1 or not active_cols:
            continue
        s_solid, nu_solid = solid_cols[0]
        c0_solid = float(c0_vec[s_solid])
        if c0_solid <= 0:
            continue

        s_act, nu_act_v = active_cols[0]
        scale_act  = (1.0 / v_oa) if org_mask[s_act] else 1.0
        xi_implied = (c_eq[s_act] - c0_vec[s_act]) / (nu_act_v * scale_act)

        if c0_solid + nu_solid * xi_implied < -1e-12:
            xi_max = c0_solid / abs(nu_solid)
            for s_a, nu_a in active_cols:
                scale_a   = (1.0 / v_oa) if org_mask[s_a] else 1.0
                c_eq[s_a] = c0_vec[s_a] + nu_a * xi_max * scale_a
            c_eq[s_solid] = 0.0
            clamped = True

    return np.maximum(c_eq, 0.0), clamped
