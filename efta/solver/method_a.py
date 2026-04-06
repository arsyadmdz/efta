"""
efta.solver.method_a
====================
Method A: extent-of-reaction solver using fsolve / least_squares.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve, least_squares
from ..species import is_nonaqueous, is_organic
from ..system import _eval_gamma, _one, _conc_from_xi
from ._shared import _compute_extent_bounds


def _solve_method_A(c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa,
                    tolerance=1e-6, init_scale=0.1, verbose=False,
                    warm_start=None, maintain_mask=None, gamma_dict=None):
    """Solve for ξ directly via fsolve / least_squares in concentration space."""
    n_rxn, n_sp = nu.shape
    if maintain_mask is None:
        maintain_mask = np.zeros(n_sp, dtype=bool)

    nu_eff      = nu.copy(); nu_eff[:, maintain_mask] = 0.0
    active_mask = aq_mask | org_mask

    def _return(xi, method, solver_name, error, c_eq_direct=None):
        if verbose: print(f"  [Method A{method}] {solver_name}  SSE={error:.2e}")
        return np.asarray(xi, dtype=float), {
            'method': f'A{method}', 'solver': solver_name,
            'error': error, 'c_eq_direct': c_eq_direct, 'lc_aq': None}

    if np.all(c0_vec[active_mask] == 0) and not any(is_nonaqueous(s) for s in all_species):
        return _return(np.zeros(n_rxn), 0, "trivial (zero concentrations)", 0.0)

    def conc(xi):
        return _conc_from_xi(xi, c0_vec, nu_eff, org_mask, v_oa)

    def _activity_vec(c_s):
        a = c_s.copy()
        if gamma_dict is not None:
            c_dict_now = {all_species[s]: float(c_s[s]) for s in range(n_sp)}
            for s in range(n_sp):
                if not active_mask[s]: continue
                entry = gamma_dict.get(all_species[s], (_one,))
                gam = _eval_gamma(entry, c_now)
                a[s] = max(gam * float(c_s[s]), 1e-300)
        return a

    def equations(xi, form='standard'):
        xi = np.asarray(xi, dtype=float)
        if np.any(np.abs(xi[pinned_zero]) > 1e-10): return [1e10] * n_rxn
        c = conc(xi)
        if np.any(c[active_mask] < -1e-12): return [1e10] * n_rxn
        c_s = np.where(active_mask, np.maximum(c, 1e-30), c)
        a_s = _activity_vec(c_s)
        eqs = np.zeros(n_rxn)
        for r in range(n_rxn):
            Q = 1.0
            for s in range(n_sp):
                if not active_mask[s]: continue
                v = nu[r, s]
                if v != 0:
                    if a_s[s] <= 0: return [1e10] * n_rxn
                    Q *= float(a_s[s]) ** v
            if form == 'standard': eqs[r] = Q / K_vec[r] - 1.0
            elif form == 'linear': eqs[r] = K_vec[r] - Q
            elif form == 'log':
                if Q <= 0: return [1e10] * n_rxn
                eqs[r] = np.log10(Q / K_vec[r])
        return eqs.tolist()

    def calc_error(xi):
        try:
            r = np.array(equations(xi, form='standard')); return float(np.dot(r, r))
        except Exception: return float('inf')

    def constraints_ok(xi):
        xi = np.asarray(xi, dtype=float)
        if np.any(np.abs(xi[pinned_zero]) > 1e-10): return False
        return bool(np.all(conc(xi)[active_mask] >= -1e-12))

    total_mass = max(float(np.sum(c0_vec[active_mask])), 1.0)
    lower, upper = _compute_extent_bounds(
        n_rxn, n_sp, nu, c0_vec, active_mask, org_mask, all_species, v_oa, total_mass,
        maintain_mask=maintain_mask)
    pinned_zero   = (upper == 0.0) & (lower == 0.0)
    initial_guess = lower + init_scale * (upper - lower)

    def try_fsolve(seed, form):
        try:
            sol   = fsolve(lambda x: equations(x, form=form), seed, full_output=True)
            x_ref = np.array(sol[0]); res = calc_error(x_ref)
            if res < tolerance and constraints_ok(x_ref): return x_ref, res
        except Exception: pass
        return None, float('inf')

    def try_fsolve_refined(seed):
        for form in ('standard', 'linear'):
            x, e = try_fsolve(seed, form)
            if x is not None: return x, e, form
        return None, float('inf'), None

    if warm_start is not None and len(warm_start) == n_rxn:
        ws = np.asarray(warm_start, dtype=float)
        for form in ('standard', 'linear'):
            ws_ref, ws_res = try_fsolve(ws, form)
            if ws_ref is not None:
                return _return(ws_ref, 0, f"warm start + fsolve ({form})", ws_res)
        ws_err = calc_error(ws)
        if ws_err < tolerance and constraints_ok(ws):
            return _return(ws, 0, "warm start (reused)", ws_err)

    for a_id, form_name in [(1, 'standard'), (2, 'linear')]:
        ig          = initial_guess.copy()
        best_res_A  = float('inf')
        stall_count = 0
        for attempt in range(10):
            try:
                sol   = fsolve(lambda x, f=form_name: equations(x, form=f), ig, full_output=True)
                x_sol = np.array(sol[0]); res = calc_error(x_sol)
                if res < tolerance and constraints_ok(x_sol):
                    return _return(x_sol, a_id, f"fsolve ({form_name})", res)
                # early-exit: stop retrying if residual has not improved for 3 attempts
                if res < best_res_A * 0.99:
                    best_res_A  = res
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count >= 3:
                        break
            except Exception:
                stall_count += 1
                if stall_count >= 3:
                    break
            ig = ig * (1 + 0.5 * (np.random.random(n_rxn) - 0.5))

    for a_id, form_name in [(3, 'standard'), (4, 'linear'), (5, 'log')]:
        try:
            ls = least_squares(lambda x, f=form_name: equations(x, form=f),
                               initial_guess, bounds=(lower, upper), method='trf', max_nfev=10000)
            if constraints_ok(ls.x) and calc_error(ls.x) < tolerance:
                return _return(ls.x, a_id, f"LS ({form_name})", calc_error(ls.x))
            if constraints_ok(ls.x):
                x_ref, res, ff = try_fsolve_refined(ls.x)
                if x_ref is not None:
                    return _return(x_ref, a_id, f"LS ({form_name}) + fsolve ({ff})", res)
        except Exception: pass

    return None, {'method': 'A_failed', 'solver': 'A1-A6 all failed',
                  'error': np.inf, 'c_eq_direct': None, 'lc_aq': None}
