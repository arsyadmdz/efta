"""
efta.solver.method_l
====================
Method L: Newton iteration in log10(c) space.
"""
from __future__ import annotations
from typing import List
import numpy as np
from scipy.linalg import qr as _scipy_qr
from scipy.optimize import fsolve
from ..species import formula, is_nonaqueous, is_organic, is_electron, charge as sp_charge
from ..balance import (_compute_clusters, _compute_decompose,
                       _compute_excluded_clusters, _compute_maintained_clusters)
from ..system import _eval_gamma, _one, _xi_from_ceq
from ._shared import _EPS_L
from ..errors import ConvergenceError


def _solve_method_L(c0_vec, nu, K_vec, all_species, v_oa=1.0,
                    tolerance=1e-6, lc_warm_start=None, verbose=False,
                    maintain_mask=None, gamma_dict=None):
    n_rxn, n_sp = nu.shape
    if maintain_mask is None:
        maintain_mask = np.zeros(n_sp, dtype=bool)

    aq_mask  = np.array([not is_nonaqueous(s) and not is_organic(s) and not is_electron(s)
                         for s in all_species], dtype=bool)
    org_mask = np.array([is_organic(s) for s in all_species], dtype=bool)
    active_mask = (aq_mask | org_mask) & ~maintain_mask

    active_idx    = np.where(active_mask)[0]
    n_act         = len(active_idx)
    sp_act        = [all_species[i] for i in active_idx]
    maintain_idx  = np.where(maintain_mask & (aq_mask | org_mask))[0]
    lc_maintained = np.log10(np.maximum(c0_vec[maintain_idx], _EPS_L))

    if n_act == 0 or n_rxn == 0:
        return np.zeros(n_rxn), {'method': 'L_empty', 'error': 0.0,
                                  'c_eq_direct': None, 'lc_aq': None}

    nu_act   = nu[:, active_idx]
    logK     = np.log10(np.maximum(K_vec, _EPS_L))
    logK_adj = logK.copy()

    if len(maintain_idx) > 0:
        logK_adj -= nu[:, maintain_idx] @ lc_maintained
        if gamma_dict is not None:
            c_maint = {all_species[gi]: float(c0_vec[gi]) for gi in maintain_idx}
            for r in range(n_rxn):
                for gi in maintain_idx:
                    sp_m = all_species[gi]; v = float(nu[r, gi])
                    if v == 0.0: continue
                    entry = gamma_dict.get(sp_m, (_one,))
                    c_now = {all_species[k]: float(c0_vec[k]) for k in range(len(all_species))}
                    gam = _eval_gamma(entry, c_now)
                    logK_adj[r] -= v * np.log10(max(gam, 1e-300))

    c0_act = np.maximum(c0_vec[active_mask], 0.0)

    try:
        _, s_vals, _ = np.linalg.svd(nu_act, full_matrices=False)
    except np.linalg.LinAlgError:
        raise ConvergenceError(
        "Method L: SVD failed on the stoichiometry matrix.\n"
        "This usually means all species are non-aqueous or the reaction\n"
        "has no active (free) species to solve for.",
        method="L_svd")
    rank   = int(np.sum(s_vals > 1e-10 * max(float(s_vals[0]), 1.0)))
    n_comp = n_act - rank

    if n_comp == 0:
        lc_ls, _, _, _ = np.linalg.lstsq(nu_act, logK_adj, rcond=None)
        c_sol = np.maximum(10.0 ** lc_ls, 0.0)
        c_eq  = np.zeros(n_sp)
        for j, gi in enumerate(active_idx): c_eq[gi] = float(c_sol[j])
        for gi in maintain_idx: c_eq[gi] = float(c0_vec[gi])
        xi_ls = _xi_from_ceq(c_eq, c0_vec, nu, org_mask, v_oa, active_mask)
        return xi_ls, {'method': 'L_ls', 'error': 0.0, 'c_eq_direct': c_eq.tolist(), 'lc_aq': lc_ls}

    class _FakeRxnL:
        def __init__(self, stoich): self._stoich = stoich
    fake_rxns_L = [_FakeRxnL({all_species[s]: float(nu[r, s])
                               for s in range(n_sp) if nu[r, s] != 0}) for r in range(n_rxn)]

    excluded = _compute_excluded_clusters(fake_rxns_L)
    if len(maintain_idx) > 0:
        excluded = excluded | _compute_maintained_clusters(maintain_mask, all_species, fake_rxns_L)

    decomp_L       = _compute_decompose(fake_rxns_L)
    all_clusters_L = _compute_clusters(fake_rxns_L)
    bare_of_L      = {sp: formula(sp) for sp in sp_act}

    relevant_L = frozenset(c for c in all_clusters_L
        if c not in excluded
        and any(decomp_L.get(bare_of_L.get(sp,''), {}).get(c, 0) != 0 for sp in sp_act))

    e_col_L = next((j for j, s in enumerate(all_species) if is_electron(s)), None)
    suppress_charge = (e_col_L is not None and abs(float(np.sum(nu[:, e_col_L]))) > 1e-12)
    if not suppress_charge and len(maintain_idx) > 0:
        if any(sp_charge(all_species[gi]) != 0 for gi in maintain_idx):
            suppress_charge = True

    charges_L = {sp_act[j]: sp_charge(sp_act[j]) for j in range(n_act)}
    cand_rows: List[np.ndarray] = []
    cand_totals: List[float]    = []
    cand_labels: List[str]      = []

    for cluster in sorted(relevant_L):
        a = np.zeros(n_act)
        for j, sp in enumerate(sp_act):
            coeff = decomp_L.get(bare_of_L.get(sp, formula(sp)), {}).get(cluster, 0.0)
            if coeff:
                a[j] = float(coeff) * (v_oa if is_organic(sp) else 1.0)
        if not np.any(a != 0): continue
        T_cluster = float(a @ c0_act)
        if len(maintain_idx) > 0:
            for gi in maintain_idx:
                coeff_m = decomp_L.get(formula(all_species[gi]), {}).get(cluster, 0.0)
                if coeff_m:
                    T_cluster -= float(coeff_m) * (v_oa if is_organic(all_species[gi]) else 1.0) * float(c0_vec[gi])
        cand_rows.append(a); cand_totals.append(T_cluster); cand_labels.append(cluster)

    q = np.array([float(charges_L.get(sp_act[j], 0.0)) for j in range(n_act)])
    if np.any(q != 0) and not suppress_charge:
        T_charge = float(q @ c0_act)
        if len(maintain_idx) > 0:
            for gi in maintain_idx:
                T_charge -= float(sp_charge(all_species[gi])) * float(c0_vec[gi])
        cand_rows.append(q); cand_totals.append(T_charge); cand_labels.append('charge')

    valid_elem = list(zip(cand_rows, cand_totals, cand_labels))

    if len(valid_elem) < n_comp:
        try:
            _, s_null, Vt_null = np.linalg.svd(nu_act, full_matrices=True)
            rank_null  = int(np.sum(s_null > 1e-10 * max(float(s_null[0]), 1.0)))
            null_basis = Vt_null[rank_null:].T
            T_null     = null_basis.T @ c0_act
            for k in range(min(n_comp - len(valid_elem), null_basis.shape[1])):
                valid_elem.append((null_basis[:, k], float(T_null[k]), f'null_{k}'))
        except Exception:
            pass

    if len(valid_elem) < n_comp:
        valid_elem = list(zip(cand_rows, cand_totals, cand_labels))

    cand_rows   = [v[0] for v in valid_elem]
    cand_totals = [v[1] for v in valid_elem]
    cand_labels = [v[2] for v in valid_elem]

    A_cand = np.array(cand_rows)
    try:
        _, _, piv = _scipy_qr(A_cand.T, pivoting=True)
    except Exception:
        piv = np.arange(len(cand_rows))

    sel   = sorted(piv[:n_comp].tolist())
    A_sel = A_cand[sel]
    T_sel = np.array([cand_totals[i] for i in sel])

    if verbose:
        print(f"  [Method L] n_act={n_act} rank={rank} n_comp={n_comp}  "
              f"balances={[cand_labels[i] for i in sel]}  T={T_sel}")

    total_c0 = max(float(np.sum(c0_act)), 1e-20)
    lc_upper = np.log10(total_c0)
    ln10     = np.log(10.0)

    def _gamma_log_correction(lc):
        if gamma_dict is None: return np.zeros(rank)
        c_now = {sp_act[j]: 10.0 ** lc[j] for j in range(n_act)}
        corr  = np.zeros(rank)
        for r in range(rank):
            for j in range(n_act):
                v = nu_act[r, j]
                if v == 0.0: continue
                entry = gamma_dict.get(sp_act[j], (_one,))
                gam = _eval_gamma(entry, c_now)
                corr[r] += v * np.log10(max(gam, 1e-300))
        return corr

    def _residuals(lc):
        c = 10.0 ** lc; res = np.empty(n_act)
        gc = _gamma_log_correction(lc)
        res[:rank] = nu_act[:rank] @ lc + gc - logK_adj[:rank]
        res[rank:] = A_sel @ c - T_sel
        return res

    def _jacobian(lc):
        c = 10.0 ** lc
        J = np.zeros((n_act, n_act))
        J[:rank, :] = nu_act[:rank]
        J[rank:, :] = A_sel * (c * ln10)
        return J

    def _propagate_lc(lc):
        for _ in range(10):
            for r in range(rank):
                prod_idx  = [j for j in range(n_act) if nu_act[r, j] > 0]
                react_idx = [j for j in range(n_act) if nu_act[r, j] < 0]
                if not prod_idx or any(lc[j] <= -18.0 for j in react_idx): continue
                rhs = logK_adj[r] - sum(nu_act[r, j] * lc[j] for j in react_idx)
                if len(prod_idx) == 1:
                    j = prod_idx[0]; lc_new = rhs / nu_act[r, j]
                    if lc[j] <= -18.0 or lc_new > lc[j]:
                        lc[j] = float(np.clip(lc_new, -20, lc_upper))
                else:
                    lc_avg = rhs / sum(nu_act[r, j] for j in prod_idx)
                    for j in prod_idx:
                        lc_new = lc_avg / nu_act[r, j]
                        if lc[j] <= -18.0 or lc_new > lc[j]:
                            lc[j] = float(np.clip(lc_new, -20, lc_upper))
        return lc

    if lc_warm_start is not None and len(lc_warm_start) == n_act:
        lc0 = np.asarray(lc_warm_start, dtype=float)
    else:
        lc0 = _propagate_lc(np.log10(np.where(c0_act > 0, c0_act, 1e-20)))

    lc = lc0.copy(); success = False; rng = np.random.default_rng(42)

    for attempt in range(4):
        if   attempt == 1: lc = np.clip(lc0, -15, 2) + rng.normal(0, 0.3, n_act)
        elif attempt == 2: lc = np.clip(lc0, -15, 2) + rng.normal(0, 1.0, n_act)
        elif attempt == 3:
            try:   lc = fsolve(_residuals, lc0, fprime=_jacobian, full_output=True)[0]
            except Exception: break

        for _ in range(80):
            r_vec = _residuals(lc); err = float(np.max(np.abs(r_vec)))
            if err < tolerance: success = True; break
            try:
                J = _jacobian(lc); dlc = np.linalg.solve(J, -r_vec)
            except np.linalg.LinAlgError: break
            step = 1.0
            for _ in range(15):
                if float(np.max(np.abs(_residuals(lc + step * dlc)))) < err: break
                step *= 0.5
            lc = lc + step * dlc
        if success: break

    final_err = float(np.max(np.abs(_residuals(lc))))
    if verbose: print(f"  [Method L] max_resid={final_err:.3e}  converged={success}")
    if not success and final_err > 1e-3:
        raise ConvergenceError(
        f"Method L: did not converge after 4 restart attempts.\n"
        f"Try increasing tolerance, adjusting initial concentrations,\n"
        f"or check that all species in c0 are present in the reactions.",
        residual=final_err, method="L")

    c_sol = np.maximum(10.0 ** lc, 0.0)
    c_eq  = np.zeros(n_sp)
    for j, gi in enumerate(active_idx): c_eq[gi] = float(c_sol[j])
    for gi in maintain_idx: c_eq[gi] = float(c0_vec[gi])

    xi_L = _xi_from_ceq(c_eq, c0_vec, nu, org_mask, v_oa, active_mask)
    return xi_L, {'method': 'L', 'error': final_err, 'c_eq_direct': c_eq.tolist(), 'lc_aq': lc}
