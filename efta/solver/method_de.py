"""
efta.solver.method_de
=====================
Method DE: differential-evolution global search, followed by local polishing.
Used as a last resort when all local methods fail.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution, fsolve
from ..species import is_organic
from ..system import _one, _conc_from_xi
from ._shared import (_build_logspace_and_constraints, _compute_extent_bounds)


def _solve_method_DE(c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa,
                     tolerance=1e-6, verbose=False, lc_warm_start=None,
                     maintain_mask=None, gamma_dict=None):
    """Differential-evolution global search followed by local polishing."""
    n_rxn, n_sp = nu.shape
    if maintain_mask is None:
        maintain_mask = np.zeros(n_sp, dtype=bool)

    nu_eff      = nu.copy(); nu_eff[:, maintain_mask] = 0.0
    active_mask = aq_mask | org_mask

    def conc(xi):
        return _conc_from_xi(xi, c0_vec, nu_eff, org_mask, v_oa)

    def _build_activity_vec(c_s):
        a = c_s.copy()
        if gamma_dict is not None:
            c_dict_now = {all_species[s]: float(c_s[s]) for s in range(n_sp)}
            for s in range(n_sp):
                if not active_mask[s]: continue
                entry = gamma_dict.get(all_species[s], (_one,))
                func = entry[0]; deps = entry[1:]
                gam = float(func(*tuple(float(c_dict_now.get(d,0.0)) for d in deps)) if deps else func())
                a[s] = max(gam * float(c_s[s]), 1e-300)
        return a

    def equations_xi(xi, form='standard'):
        xi = np.asarray(xi, dtype=float)
        if np.any(np.abs(xi[pinned_zero]) > 1e-10): return [1e10] * n_rxn
        c = conc(xi)
        if np.any(c[active_mask] < -1e-12): return [1e10] * n_rxn
        c_s = np.where(active_mask, np.maximum(c, 1e-30), c)
        a_s = _build_activity_vec(c_s)
        eqs = np.zeros(n_rxn)
        for r in range(n_rxn):
            Q = 1.0
            for s in range(n_sp):
                if not active_mask[s]: continue
                v = nu[r, s]
                if v != 0:
                    if a_s[s] <= 0: return [1e10] * n_rxn
                    Q *= float(a_s[s]) ** v
            if form == 'standard':   eqs[r] = Q / K_vec[r] - 1.0
            elif form == 'linear':   eqs[r] = K_vec[r] - Q
            elif form == 'log':
                if Q <= 0: return [1e10] * n_rxn
                eqs[r] = np.log10(Q / K_vec[r])
        return eqs.tolist()

    def calc_error_xi(xi):
        try:   return float(np.dot(np.array(equations_xi(xi)), np.array(equations_xi(xi))))
        except Exception: return float('inf')

    def constraints_ok_xi(xi):
        xi = np.asarray(xi, dtype=float)
        if np.any(np.abs(xi[pinned_zero]) > 1e-10): return False
        return bool(np.all(conc(xi)[active_mask] >= -1e-12))

    total_mass = max(float(np.sum(c0_vec[active_mask])), 1.0)
    lower, upper = _compute_extent_bounds(
        n_rxn, n_sp, nu, c0_vec, active_mask, org_mask, all_species, v_oa, total_mass,
        maintain_mask=maintain_mask)
    pinned_zero  = (upper == 0.0) & (lower == 0.0)
    de_bounds_xi = list(zip(lower, upper))

    def de_obj_xi(xi, form='standard'):
        try:
            res     = equations_xi(xi, form=form)
            penalty = 0.0 if constraints_ok_xi(xi) else 1e6
            return float(np.sum(np.array(res) ** 2)) + penalty
        except Exception: return 1e12

    def try_fsolve_xi(seed, form):
        try:
            sol   = fsolve(lambda x, f=form: equations_xi(x, form=f), seed, full_output=True)
            x_ref = np.array(sol[0]); res = calc_error_xi(x_ref)
            if res < tolerance and constraints_ok_xi(x_ref): return x_ref, res
        except Exception: pass
        return None, float('inf')

    def refine_xi(seed):
        for form in ('standard', 'linear'):
            x, e = try_fsolve_xi(seed, form)
            if x is not None: return x, e, form
        return None, float('inf'), None

    (active_idx, n_act, nu_act_rows, lK_rows,
     groups, charges, n_charge, act_name_to_local,
     sym_constraints) = _build_logspace_and_constraints(
        c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa,
        maintain_mask=maintain_mask)

    def lc_full_from_act(lc_act):
        lc_full = np.zeros(n_sp)
        for local_j, global_i in enumerate(active_idx):
            lc_full[global_i] = lc_act[local_j]
        if np.any(maintain_mask):
            for gi in np.where(maintain_mask)[0]:
                lc_full[gi] = np.log10(max(float(c0_vec[gi]), 1e-300))
        return lc_full

    def verify_lc(lc_full):
        c_lc = 10.0 ** np.clip(lc_full, -300, 300); sse = 0.0
        for r in range(nu.shape[0]):
            Q = 1.0
            for s in range(n_sp):
                if not active_mask[s]: continue
                v = nu[r, s]
                if v != 0:
                    c_val = max(float(c_lc[s]), 1e-300)
                    if gamma_dict is not None:
                        entry = gamma_dict.get(all_species[s], (_one,))
                        func = entry[0]; deps = entry[1:]
                        c_now = {all_species[k]: float(c_lc[k]) for k in range(n_sp)}
                        gam = float(func(*tuple(float(c_now.get(d,0.0)) for d in deps)) if deps else func())
                    else:
                        gam = 1.0
                    Q *= max(gam * c_val, 1e-300) ** v
            sse += (Q / K_vec[r] - 1.0) ** 2
        return sse

    de_common = dict(maxiter=100, popsize=15, tol=1e-10, mutation=(0.5, 1.5),
                     recombination=0.5, seed=42, polish=False, init='latinhypercube')
    all_results = []

    def _return_xi(xi, method_id, solver_name, error, lc_full=None):
        c_direct = lc_aq_out = None
        if lc_full is not None:
            c_direct  = 10.0 ** np.clip(lc_full, -300, 300)
            lc_aq_out = lc_full[active_idx]
        if verbose: print(f"  [Method DE{method_id}] {solver_name}  SSE={error:.2e}")
        return np.asarray(xi, dtype=float), {
            'method': f'DE{method_id}', 'solver': solver_name,
            'error': error, 'c_eq_direct': c_direct, 'lc_aq': lc_aq_out}

    for de_id, form, label in [
        (1, 'standard', 'DE (extent, standard)'),
        (2, 'linear',   'DE (extent, linear)'),
        (3, 'log',      'DE (extent, log)'),
    ]:
        if verbose: print(f"  [Method DE{de_id}] Trying {label}...")
        try:
            de_result = differential_evolution(
                lambda x, f=form: de_obj_xi(x, form=f), de_bounds_xi, **de_common)
            de_x = de_result.x
            if constraints_ok_xi(de_x):
                x_ref, res, ff = refine_xi(de_x)
                if x_ref is not None:
                    all_results.append((label, x_ref, res, None))
                    return _return_xi(x_ref, de_id, f"{label} + fsolve ({ff})", res)
            all_results.append((label, de_x, calc_error_xi(de_x), None))
        except Exception as exc:
            if verbose: print(f"  [Method DE{de_id}] Exception: {exc}")

    best_result = min(all_results, key=lambda t: t[2], default=None)
    if best_result is not None:
        label, xi, err, lc_full = best_result
        return _return_xi(xi, 6, f"DE6 best ({label})", err, lc_full)

    return np.zeros(n_rxn), {'method': 'DE_failed', 'solver': 'all DE methods failed',
                              'error': np.inf, 'c_eq_direct': None, 'lc_aq': None}
