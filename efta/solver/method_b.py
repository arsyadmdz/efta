"""
efta.solver.method_b
====================
Method B: fsolve in log-space (B1) and concentration-space (B2).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve
from ..species import is_nonaqueous, is_electron
from ..system import _eval_gamma, _one
from ._shared import (_build_logspace_and_constraints, _logspace_initial_guesses,
                      _clamp_ksp_solids)


def _solve_method_B(c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa,
                    tolerance=1e-6, verbose=False, lc_warm_start=None,
                    maintain_mask=None, gamma_dict=None):
    """Try B1 (log-space fsolve) then B2 (concentration-space fsolve)."""
    n_sp = nu.shape[1]
    (active_idx, n_act, nu_act_rows, lK_rows,
     groups, charges, n_charge, act_name_to_local,
     sym_constraints) = _build_logspace_and_constraints(
        c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa,
        maintain_mask=maintain_mask)

    shared = dict(
        c0_vec=c0_vec, nu=nu, K_vec=K_vec,
        all_species=all_species, aq_mask=aq_mask, org_mask=org_mask,
        v_oa=v_oa, tolerance=tolerance,
        active_idx=active_idx, n_act=n_act,
        nu_act_rows=nu_act_rows, lK_rows=lK_rows,
        groups=groups, charges=charges, n_charge=n_charge,
        act_name_to_local=act_name_to_local, sym_constraints=sym_constraints,
        gamma_dict=gamma_dict,
    )
    lc_warm = lc_warm_start

    for sub_id, label, fn in [
        (1, "B1: log-space fsolve",          _method_B1),
        (2, "B2: concentration-space fsolve", _method_B2),
    ]:
        if verbose: print(f"  [Method B{sub_id}] Trying {label}...")
        try:
            lc_full, res = fn(**shared, lc_warm=lc_warm)
        except Exception as exc:
            if verbose: print(f"  [Method B{sub_id}] Exception: {exc}")
            lc_full, res = None, np.inf

        if lc_full is not None:
            c_lc = 10.0 ** np.clip(lc_full, -300, 300)
            c_lc, was_clamped = _clamp_ksp_solids(
                c_lc, c0_vec, nu, all_species, aq_mask, org_mask, v_oa)
            if was_clamped:
                lc_full = np.log10(np.maximum(c_lc, 1e-300))
                if verbose: print(f"  [Method B{sub_id}] Solid clamped (fully dissolved)")

            lc_sse = 0.0
            for r in range(nu.shape[0]):
                Q = 1.0
                for s in range(n_sp):
                    if is_nonaqueous(all_species[s]) or is_electron(all_species[s]): continue
                    v = nu[r, s]
                    if v != 0:
                        c_s_val = max(float(c_lc[s]), 1e-300)
                        if gamma_dict is not None:
                            entry = gamma_dict.get(all_species[s], (_one,))
                            c_now = {all_species[k]: float(c_lc[k]) for k in range(n_sp)}
                            gam = _eval_gamma(entry, c_now)
                        else:
                            gam = 1.0
                        Q *= max(gam * c_s_val, 1e-300) ** v
                lc_sse += (Q / K_vec[r] - 1.0) ** 2

            if maintain_mask is not None and np.any(maintain_mask):
                for gi in np.where(maintain_mask)[0]:
                    c_lc[gi] = float(c0_vec[gi])

            if verbose: print(f"  [Method B{sub_id}] SUCCESS  SSE={lc_sse:.2e}")
            return lc_full, {
                'method': f'B{sub_id}', 'solver': label,
                'error': lc_sse, 'c_eq_direct': c_lc,
                'lc_aq': lc_full[active_idx],
            }
        if verbose: print(f"  [Method B{sub_id}] Failed (res={res:.2e})")

    return None, {'method': 'B_failed', 'solver': 'B1 and B2 both failed',
                  'error': np.inf, 'c_eq_direct': None, 'lc_aq': None}


def _method_B1(c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa, tolerance,
               active_idx, n_act, nu_act_rows, lK_rows, groups, charges, n_charge,
               act_name_to_local, sym_constraints, lc_warm=None, gamma_dict=None):
    """Log-space fsolve variant of Method B."""
    n_rxn_aq = len(lK_rows)
    n_eq     = n_rxn_aq + len(groups) + n_charge + len(sym_constraints)
    n_sp     = nu.shape[1]
    if n_eq != n_act: return None, np.inf

    sp_act = [all_species[active_idx[j]] for j in range(n_act)]

    def _gamma_corr(lc_act):
        if gamma_dict is None: return np.zeros(n_rxn_aq)
        c_now = {sp_act[j]: 10.0 ** float(lc_act[j]) for j in range(n_act)}
        corr  = np.zeros(n_rxn_aq)
        for i in range(n_rxn_aq):
            for j in range(n_act):
                v = nu_act_rows[i, j]
                if v == 0.0: continue
                entry = gamma_dict.get(sp_act[j], (_one,))
                gam = _eval_gamma(entry, c_now)
                corr[i] += v * np.log10(max(gam, 1e-300))
        return corr

    def residuals(lc_act):
        res = np.empty(n_eq); gc = _gamma_corr(lc_act)
        for i in range(n_rxn_aq):
            res[i] = float(nu_act_rows[i] @ lc_act) + gc[i] - lK_rows[i]
        offset = n_rxn_aq
        c_act  = 10.0 ** np.clip(lc_act, -300, 300)
        for count_vec, C_total in groups:
            res[offset] = float(count_vec @ c_act) - C_total; offset += 1
        if n_charge:
            res[offset] = float(charges @ c_act); offset += 1
        for coeff, rhs in sym_constraints:
            res[offset] = float(coeff @ c_act) - rhs; offset += 1
        return res

    candidates = _logspace_initial_guesses(c0_vec, active_idx, n_act, nu_act_rows,
                                            lK_rows, groups, act_name_to_local, lc_warm=lc_warm)
    best_lc, best_res = None, np.inf
    for ig in candidates:
        try:
            lc_sol = fsolve(residuals, ig, full_output=True)[0]
            res    = float(np.max(np.abs(residuals(lc_sol))))
            if res < best_res: best_res = res; best_lc = lc_sol.copy()
            if best_res < tolerance: break
        except Exception: continue

    if best_lc is None or best_res > 1e-4: return None, best_res if best_lc is not None else np.inf
    lc_full = np.zeros(n_sp)
    for local_j, global_i in enumerate(active_idx):
        lc_full[global_i] = best_lc[local_j]
    return lc_full, best_res


def _method_B2(c0_vec, nu, K_vec, all_species, aq_mask, org_mask, v_oa, tolerance,
               active_idx, n_act, nu_act_rows, lK_rows, groups, charges, n_charge,
               act_name_to_local, sym_constraints, lc_warm=None, gamma_dict=None):
    """Concentration-space fsolve variant of Method B."""
    n_rxn_aq = len(lK_rows)
    K_rows   = 10.0 ** lK_rows
    n_eq     = n_rxn_aq + len(groups) + n_charge + len(sym_constraints)
    if n_eq != n_act: return None, np.inf
    n_sp   = nu.shape[1]
    sp_act = [all_species[active_idx[j]] for j in range(n_act)]

    def _activity_j(j, c_safe):
        if gamma_dict is None: return float(c_safe[j])
        entry = gamma_dict.get(sp_act[j], (_one,))
        c_now = {sp_act[k]: float(c_safe[k]) for k in range(n_act)}
        gam = _eval_gamma(entry, c_now)
        return gam * float(c_safe[j])

    def residuals(c_act):
        c_safe = np.maximum(c_act, 1e-300); res = np.empty(n_eq)
        for i in range(n_rxn_aq):
            rp, pp = 1.0, 1.0
            for j in range(n_act):
                v = nu_act_rows[i, j]
                if v == 0: continue
                a = max(_activity_j(j, c_safe), 1e-300)
                if v < 0: rp *= a ** abs(v)
                else:     pp *= a ** v
            res[i] = K_rows[i] * rp - pp
        offset = n_rxn_aq
        for count_vec, C_total in groups:
            res[offset] = float(count_vec @ c_act) - C_total; offset += 1
        if n_charge:
            res[offset] = float(charges @ c_act); offset += 1
        for coeff, rhs in sym_constraints:
            res[offset] = float(coeff @ c_act) - rhs; offset += 1
        return res

    log_cands  = _logspace_initial_guesses(c0_vec, active_idx, n_act, nu_act_rows,
                                            lK_rows, groups, act_name_to_local, lc_warm=lc_warm)
    candidates = [10.0 ** np.clip(lc, -300, 300) for lc in log_cands]
    c0_act     = np.where(c0_vec[active_idx] > 0, c0_vec[active_idx], 1e-7)
    candidates.insert(0 if lc_warm is None else 1, c0_act)

    best_c, best_res = None, np.inf
    for ig in candidates:
        try:
            c_sol = np.maximum(fsolve(residuals, ig, full_output=True)[0], 0.0)
            res   = float(np.max(np.abs(residuals(c_sol))))
            if res < best_res: best_res = res; best_c = c_sol.copy()
            if best_res < tolerance: break
        except Exception: continue

    if best_c is None or best_res > 1e-4: return None, best_res if best_c is not None else np.inf
    lc_full = np.zeros(n_sp)
    for local_j, global_i in enumerate(active_idx):
        lc_full[global_i] = np.log10(max(best_c[local_j], 1e-300))
    return lc_full, best_res
