"""
efta.solver.dispatch
====================
High-level dispatch functions:

- :func:`_solve_extents`         – A → B → DE chain for a single subsystem
- :func:`_pre_solve_speciation`  – warm-start sequential pre-solver
- :func:`_solve_precipitation`   – Ksp-only solver
- :func:`_solve_mix`             – alternating speciation + Ksp outer loop
- :func:`_solve_with_ksp`        – main entry point
"""
from __future__ import annotations
from typing import FrozenSet
import numpy as np
from ..species import species as canon_species, is_nonaqueous, is_organic, is_electron
from ..balance import _is_ksp_reaction
from ..system import (_one, _build_gamma_for_system, _normalize_maintain,
                      _build_system, _conc_from_xi, _lc_to_xi,
                      _ksp_solid_species, _saturation_index, _make_dissolution_rxn,
                      _reaction_has_complete_side)
from ._shared import _TimeoutError, _TimedCall, _reactions_type
from .method_l import _solve_method_L
from .method_a import _solve_method_A
from .method_b import _solve_method_B
from .method_de import _solve_method_DE


def _solve_extents(c0_vec, nu, K_vec, all_species=None, v_oa=1.0,
                   tolerance=1e-6, init_scale=0.1, verbose=False,
                   return_solver_info=False, warm_start=None, lc_warm_start=None,
                   maintain_mask=None, gamma_dict=None):
    """Try A → B → DE in sequence; return the first successful result."""
    n_rxn, n_sp = nu.shape
    if all_species is not None:
        aq_mask  = np.array([not is_nonaqueous(s) and not is_organic(s) and not is_electron(s)
                             for s in all_species], dtype=bool)
        org_mask = np.array([is_organic(s) for s in all_species], dtype=bool)
    else:
        aq_mask  = np.ones(n_sp, dtype=bool)
        org_mask = np.zeros(n_sp, dtype=bool)
        all_species = [f'S{i}' for i in range(n_sp)]

    if maintain_mask is None:
        maintain_mask = np.zeros(n_sp, dtype=bool)

    shared = dict(c0_vec=c0_vec, nu=nu, K_vec=K_vec,
                  all_species=all_species, aq_mask=aq_mask, org_mask=org_mask,
                  v_oa=v_oa, tolerance=tolerance, maintain_mask=maintain_mask,
                  gamma_dict=gamma_dict)

    xi_A, info_A = _solve_method_A(**shared, init_scale=init_scale,
                                    verbose=verbose, warm_start=warm_start)
    if xi_A is not None and info_A['error'] < tolerance:
        return (xi_A, info_A) if return_solver_info else xi_A

    lc_B, info_B = _solve_method_B(**shared, verbose=verbose, lc_warm_start=lc_warm_start)
    if lc_B is not None and info_B.get('error', np.inf) < tolerance:
        xi_B = _lc_to_xi(lc_B, c0_vec, nu, aq_mask, org_mask, v_oa, maintain_mask=maintain_mask)
        return (xi_B, info_B) if return_solver_info else xi_B

    xi_DE, info_DE = _solve_method_DE(**shared, verbose=verbose, lc_warm_start=lc_warm_start)
    return (xi_DE, info_DE) if return_solver_info else xi_DE


def _pre_solve_speciation(reactions_list, c0_orig, v_oa=1.0,
                           tolerance=1e-6, init_scale=0.1, verbose=False,
                           maintain_mask=None, gamma_dict=None):
    """Sequentially solve reactions in dependency order to provide warm-start values."""
    if all(_reaction_has_complete_side(r, c0_orig) for r in reactions_list):
        if verbose: print("  [pre-solve] All reactions ready — skipping.")
        return dict(c0_orig)

    c0_eff = dict(c0_orig)
    n_rxn  = len(reactions_list)
    solved = [False] * n_rxn

    def _update_c0(rxn, xi, c0_work):
        maint_set = set()
        if maintain_mask is not None:
            all_sp_tmp, _, _, _, _ = _build_system([rxn], c0_work)
            maint_set = {all_sp_tmp[i] for i in range(len(all_sp_tmp))
                         if i < len(maintain_mask) and maintain_mask[i]}
        for sp, nu in rxn._stoich.items():
            if sp == 'O/A' or sp in maint_set: continue
            cur = float(c0_work.get(sp, 0.0))
            c0_work[sp] = max(0.0, cur + (nu * xi / v_oa if is_organic(sp) else nu * xi))

    for _outer in range(n_rxn + 2):
        ready_now = [i for i in range(n_rxn)
                     if not solved[i] and _reaction_has_complete_side(reactions_list[i], c0_eff)]
        if not ready_now: break

        for i in ready_now:
            rxn = reactions_list[i]
            all_sp, c0_vec, nu, K_vec, v_oa_sys = _build_system([rxn], c0_eff)
            maint_local = _normalize_maintain([], all_sp)

            try:
                xi_arr, info = _solve_extents(
                    c0_vec, nu, K_vec, all_species=all_sp, v_oa=v_oa_sys,
                    tolerance=tolerance, init_scale=init_scale, verbose=verbose,
                    return_solver_info=True, maintain_mask=maint_local, gamma_dict=gamma_dict)
                xi_val   = float(np.asarray(xi_arr).flat[0])
                c_direct = info.get('c_eq_direct')
                if c_direct is not None:
                    c_eq_vec = np.maximum(np.asarray(c_direct, dtype=float), 0.0)
                    for j, sp in enumerate(all_sp):
                        if sp != 'O/A': c0_eff[sp] = float(c_eq_vec[j])
                else:
                    _update_c0(rxn, xi_val, c0_eff)
            except Exception as exc:
                if verbose: print(f"  [pre-solve] reaction {i} failed: {exc}")

            solved[i] = True
        if all(solved): break

    if verbose: print(f"  [pre-solve] done — {sum(solved)}/{n_rxn} reactions pre-solved.")
    return c0_eff


def _solve_precipitation(ksp_diss_rxns, c0, tolerance=1e-6, init_scale=0.1,
                          verbose=False, warm_start=None, lc_warm_start=None,
                          maintain_mask=None, gamma_dict=None):
    """Solve a system of purely Ksp reactions, handling solid exhaustion."""
    n_rxn     = len(ksp_diss_rxns)
    xi_final  = np.zeros(n_rxn)
    exhausted = {}
    c0_work   = dict(c0)
    active    = list(range(n_rxn))
    info_out  = {'method': 'precip_trivial', 'error': 0.0, 'c_eq_direct': None, 'lc_aq': None}

    for outer in range(n_rxn + 2):
        if not active: break
        act_rxns    = [ksp_diss_rxns[i] for i in active]
        all_sp, c0_vec, nu, K_vec, v_oa = _build_system(act_rxns, c0_work)
        g_local     = _build_gamma_for_system(act_rxns, all_sp) if gamma_dict else gamma_dict
        maint_local = np.zeros(len(all_sp), dtype=bool)

        xi_sub, info_out = _solve_extents(
            c0_vec, nu, K_vec, all_species=all_sp, v_oa=v_oa,
            tolerance=tolerance, init_scale=init_scale, verbose=verbose,
            return_solver_info=True,
            warm_start=(np.array([warm_start[i] for i in active], dtype=float)
                        if warm_start is not None and len(warm_start) == n_rxn else None),
            lc_warm_start=lc_warm_start, maintain_mask=maint_local, gamma_dict=g_local)
        xi_sub = np.asarray(xi_sub, dtype=float)

        newly_exhausted = []
        for sub_i, orig_i in enumerate(active):
            rxn = ksp_diss_rxns[orig_i]
            sp_solid, nu_solid = _ksp_solid_species(rxn)
            solid_rem = float(c0_work.get(sp_solid, 0.0)) + nu_solid * xi_sub[sub_i]
            if solid_rem < -1e-12:
                max_xi = float(c0_work.get(sp_solid, 0.0)) / abs(nu_solid)
                xi_sub[sub_i] = max_xi
                for sp_, c_ in rxn._stoich.items():
                    if sp_ == 'O/A': continue
                    c0_work[sp_] = max(0.0, float(c0_work.get(sp_, 0.0)) + c_ * max_xi)
                exhausted[orig_i] = max_xi; newly_exhausted.append(orig_i)
                if verbose: print(f"  [precip] {sp_solid} exhausted (xi={max_xi:.4g})")

        for sub_i, orig_i in enumerate(active):
            if orig_i not in exhausted: xi_final[orig_i] = xi_sub[sub_i]
        active = [i for i in active if i not in exhausted]
        if not newly_exhausted: break

    for orig_i, max_xi in exhausted.items():
        xi_final[orig_i] = max_xi
    return xi_final, info_out


def _solve_mix(spec_rxns, ksp_pairs, c0_orig, tolerance=1e-6, init_scale=0.1,
               verbose=False, warm_start=None, lc_warm_start=None,
               ksp_tol=0.02, max_outer=15, c0_presolved=None,
               maintain_set=None, gamma_dict=None):
    """Alternating outer loop: solve speciation, then activate/deactivate Ksp reactions."""
    n_spec     = len(spec_rxns)
    n_ksp      = len(ksp_pairs)
    ksp_rxns   = [r for r, _ in ksp_pairs]
    flip_flags = [f for _, f in ksp_pairs]

    c0         = dict(c0_presolved if c0_presolved is not None else c0_orig)
    n_all      = n_spec + n_ksp
    active_ksp: set = set()
    for ki, rxn in enumerate(ksp_rxns):
        sp_solid, _ = _ksp_solid_species(rxn)
        if float(c0.get(sp_solid, 0.0)) > 0:
            active_ksp.add(ki)

    exhausted_xi: dict = {}
    xi_spec = np.zeros(n_spec); xi_ksp = np.zeros(n_ksp)
    info_out = {'method': 'mix_init', 'error': np.inf, 'c_eq_direct': None, 'lc_aq': None}
    prev_active_ksp = None

    for outer in range(max_outer):
        if prev_active_ksp is not None and active_ksp == prev_active_ksp: break
        prev_active_ksp = set(active_ksp)

        active_ksp_sorted = sorted(active_ksp)
        current_rxns = spec_rxns + [ksp_rxns[ki] for ki in active_ksp_sorted]
        if not current_rxns: break

        ws_cur = None
        if warm_start is not None and len(warm_start) == n_all:
            ws_cur = np.concatenate([warm_start[:n_spec],
                                     [warm_start[n_spec + ki] for ki in active_ksp_sorted]])

        all_sp, c0_vec, nu, K_vec, v_oa = _build_system(current_rxns, c0)
        g_local = _build_gamma_for_system(current_rxns, all_sp) if gamma_dict else gamma_dict
        maint_local = (np.array([s in maintain_set for s in all_sp], dtype=bool)
                       if maintain_set else np.zeros(len(all_sp), dtype=bool))

        xi_cur, info_out = _solve_extents(
            c0_vec, nu, K_vec, all_species=all_sp, v_oa=v_oa,
            tolerance=tolerance, init_scale=init_scale, verbose=verbose,
            return_solver_info=True, warm_start=ws_cur, lc_warm_start=lc_warm_start,
            maintain_mask=maint_local, gamma_dict=g_local)
        xi_cur = np.asarray(xi_cur, dtype=float)

        xi_spec_cur = xi_cur[:n_spec]; xi_ksp_cur = xi_cur[n_spec:]

        clamp_fired = False
        for sub_i, ki in enumerate(active_ksp_sorted):
            rxn = ksp_rxns[ki]; sp_solid, nu_solid = _ksp_solid_species(rxn)
            solid_rem = float(c0.get(sp_solid, 0.0)) + nu_solid * xi_ksp_cur[sub_i]
            if solid_rem < -1e-12:
                max_xi = float(c0.get(sp_solid, 0.0)) / abs(nu_solid)
                xi_ksp_cur[sub_i] = max_xi
                for sp_, c_ in rxn._stoich.items():
                    if sp_ == 'O/A': continue
                    c0[sp_] = max(0.0, float(c0.get(sp_, 0.0)) + c_ * max_xi)
                active_ksp.discard(ki); exhausted_xi[ki] = max_xi; clamp_fired = True
                if verbose: print(f"  [mix {outer}] {sp_solid} exhausted, deactivating")

        if clamp_fired:
            prev_active_ksp = None; xi_spec = xi_spec_cur
            for sub_i, ki in enumerate(active_ksp_sorted):
                if ki not in exhausted_xi: xi_ksp[ki] = xi_ksp_cur[sub_i]
            continue

        xi_spec = xi_spec_cur
        for sub_i, ki in enumerate(active_ksp_sorted): xi_ksp[ki] = xi_ksp_cur[sub_i]

        org_mask_tmp = np.array([is_organic(s) for s in all_sp])
        c_eq_vec = np.maximum(_conc_from_xi(xi_cur, c0_vec, nu, org_mask_tmp, v_oa,
                                             maintain_mask=maint_local), 0.0)
        c_eq = {s: float(c_eq_vec[j]) for j, s in enumerate(all_sp)}
        for sp_, cv_ in c0.items():
            if sp_ not in c_eq and sp_ != 'O/A': c_eq[sp_] = float(cv_)

        changed = False
        for ki, rxn in enumerate(ksp_rxns):
            if ki in active_ksp or ki in exhausted_xi: continue
            sp_solid, _ = _ksp_solid_species(rxn)
            if float(c0.get(sp_solid, 0.0)) <= 0: continue
            si = _saturation_index(rxn, c_eq, gamma_dict)
            if verbose: print(f"  [mix {outer}] {sp_solid} SI={si:.3f} (inactive)")
            if si > ksp_tol:
                active_ksp.add(ki); changed = True
                if verbose: print(f"  [mix {outer}] → ACTIVATE {sp_solid}")

        for ki in list(active_ksp):
            rxn = ksp_rxns[ki]; sp_solid, nu_solid = _ksp_solid_species(rxn)
            si = _saturation_index(rxn, c_eq, gamma_dict)
            solid_eq = float(c0.get(sp_solid, 0.0)) + nu_solid * xi_ksp[ki]
            if verbose: print(f"  [mix {outer}] {sp_solid} SI={si:.3f} solid_eq={solid_eq:.3e}")
            if solid_eq <= 1e-12 and si < -ksp_tol:
                active_ksp.discard(ki); changed = True
                if verbose: print(f"  [mix {outer}] → DEACTIVATE {sp_solid}")

        if not changed:
            if verbose: print(f"  [mix {outer}] converged")
            break

    for ki, max_xi in exhausted_xi.items(): xi_ksp[ki] = max_xi

    info_out = dict(info_out)
    info_out['c_eq_direct'] = None
    info_out['mix_iterations'] = outer + 1
    return xi_spec, xi_ksp, info_out


def _solve_with_ksp(reactions_list, c0, tolerance=1e-6, init_scale=0.1,
                    verbose=False, warm_start=None, lc_warm_start=None,
                    ksp_tol=0.02, max_ksp_iter=15,
                    presolver_timeout=2.0, maintain=None, gamma_dict=None):
    """
    Main entry point for the solver.

    Dispatches to speciation-only, precipitation-only, or mixed path.
    Method L is attempted first with a timeout; on failure falls back to
    the pre-solver + A → B → DE chain.
    """
    ksp_idx  = [i for i, r in enumerate(reactions_list) if _is_ksp_reaction(r)]
    spec_idx = [i for i, r in enumerate(reactions_list) if not _is_ksp_reaction(r)]
    stype    = _reactions_type(reactions_list)

    c0 = {(canon_species(k) if k != 'O/A' else k): v for k, v in c0.items()}

    maintain_set: FrozenSet[str] = frozenset()
    if maintain:
        maintain_set = frozenset(canon_species(s) for s in maintain)

    if gamma_dict is None:
        all_sp_tmp, _, _, _, _ = _build_system(reactions_list, c0)
        gamma_dict = _build_gamma_for_system(reactions_list, all_sp_tmp)

    def _get_maintain_mask(reactions_use, c0_use):
        all_sp, _, _, _, _ = _build_system(reactions_use, c0_use)
        if not maintain_set: return np.zeros(len(all_sp), dtype=bool)
        return np.array([s in maintain_set for s in all_sp], dtype=bool)

    def _run_presolver(spec_rxns_list, c0_use):
        v_oa_local = float(c0_use.get('O/A', 1.0))
        maint_mask = _get_maintain_mask(spec_rxns_list, c0_use)
        return _pre_solve_speciation(
            spec_rxns_list, c0_use, v_oa=v_oa_local,
            tolerance=tolerance, init_scale=init_scale, verbose=verbose,
            maintain_mask=maint_mask, gamma_dict=gamma_dict)

    def _try_timed(fn, *args, **kwargs):
        if presolver_timeout == float('inf'):
            return fn(*args, **kwargs), False
        try:
            result = _TimedCall(presolver_timeout)(fn, *args, **kwargs)
            return result, False
        except _TimeoutError:
            if verbose:
                print(f"  [solver] Method L timed out (>{presolver_timeout:.1f}s)"
                      " — falling back to pre-solver + A")
            return None, True

    def _do_speciation_solve(reactions_use, c0_use, spec_rxns_list):
        all_sp, c0_vec, nu, K_vec, v_oa = _build_system(reactions_use, c0_use)
        aq_mask   = np.array([not is_nonaqueous(s) and not is_organic(s) and not is_electron(s)
                              for s in all_sp], dtype=bool)
        org_mask  = np.array([is_organic(s) for s in all_sp], dtype=bool)
        maint_mask = (np.array([s in maintain_set for s in all_sp], dtype=bool)
                      if maintain_set else np.zeros(len(all_sp), dtype=bool))
        g_local   = _build_gamma_for_system(reactions_use, all_sp)

        def _run_L():
            return _solve_method_L(c0_vec, nu, K_vec, all_sp, v_oa=v_oa,
                                    tolerance=tolerance, lc_warm_start=lc_warm_start,
                                    verbose=verbose, maintain_mask=maint_mask,
                                    gamma_dict=g_local)
        try:
            result_L, timed_out_L = _try_timed(_run_L)
        except Exception as exc:
            if verbose: print(f"  [solver] Method L raised: {exc}")
            result_L, timed_out_L = None, False

        if not timed_out_L and result_L is not None:
            xi_L, info_L = result_L
            if info_L['error'] < tolerance:
                if verbose: print(f"  [solver] Method L succeeded err={info_L['error']:.2e}")
                return xi_L, info_L
            if verbose: print(f"  [solver] Method L err={info_L['error']:.2e} — trying A")

        c0_eff = _run_presolver(spec_rxns_list, c0_use)
        all_sp2, c0_vec2, nu2, K_vec2, v_oa2 = _build_system(reactions_use, c0_eff)
        aq_mask2   = np.array([not is_nonaqueous(s) and not is_organic(s) and not is_electron(s)
                               for s in all_sp2], dtype=bool)
        org_mask2  = np.array([is_organic(s) for s in all_sp2], dtype=bool)
        maint_mask2 = (np.array([s in maintain_set for s in all_sp2], dtype=bool)
                       if maintain_set else np.zeros(len(all_sp2), dtype=bool))
        g_local2 = _build_gamma_for_system(reactions_use, all_sp2)

        xi_A, info_A = _solve_method_A(
            c0_vec=c0_vec2, nu=nu2, K_vec=K_vec2,
            all_species=all_sp2, aq_mask=aq_mask2, org_mask=org_mask2,
            v_oa=v_oa2, tolerance=tolerance, init_scale=init_scale,
            verbose=verbose, warm_start=warm_start,
            maintain_mask=maint_mask2, gamma_dict=g_local2)
        if xi_A is not None and info_A['error'] < tolerance: return xi_A, info_A
        if verbose: print(f"  [solver] Method A err={info_A['error']:.2e} — trying B")

        lc_B, info_B = _solve_method_B(
            c0_vec2, nu2, K_vec2, all_sp2, aq_mask2, org_mask2, v_oa2,
            tolerance=tolerance, verbose=verbose, lc_warm_start=lc_warm_start,
            maintain_mask=maint_mask2, gamma_dict=g_local2)
        if lc_B is not None and info_B.get('error', np.inf) < tolerance:
            xi_B = _lc_to_xi(lc_B, c0_vec2, nu2, aq_mask2, org_mask2, v_oa2,
                              maintain_mask=maint_mask2)
            return xi_B, info_B
        if verbose: print("  [solver] Method B failed — trying DE")

        return _solve_method_DE(
            c0_vec2, nu2, K_vec2, all_sp2, aq_mask2, org_mask2, v_oa2,
            tolerance=tolerance, verbose=verbose, lc_warm_start=lc_warm_start,
            maintain_mask=maint_mask2, gamma_dict=g_local2)

    # ---- speciation only ----
    if stype == 'speciation':
        return _do_speciation_solve(reactions_list, c0, reactions_list)

    ksp_diss_rxns  = []
    ksp_flip_flags = []
    for ki in ksp_idx:
        diss_rxn, flipped = _make_dissolution_rxn(reactions_list[ki])
        ksp_diss_rxns.append(diss_rxn); ksp_flip_flags.append(flipped)

    # ---- precipitation only ----
    if stype == 'precipitation':
        xi_ksp, info = _solve_precipitation(
            ksp_diss_rxns, c0, tolerance=tolerance, init_scale=init_scale,
            verbose=verbose, warm_start=warm_start, lc_warm_start=lc_warm_start,
            maintain_mask=(maintain_set if maintain_set else None),
            gamma_dict=gamma_dict)
        xi_full = np.zeros(len(reactions_list))
        for sub_i, orig_i in enumerate(ksp_idx):
            xi_full[orig_i] = -xi_ksp[sub_i] if ksp_flip_flags[sub_i] else xi_ksp[sub_i]
        return xi_full, info

    # ---- mixed ----
    spec_rxns = [reactions_list[i] for i in spec_idx]
    ksp_pairs = list(zip(ksp_diss_rxns, ksp_flip_flags))

    def _do_mix(c0_use, c0_pre=None):
        xi_spec, xi_ksp, info = _solve_mix(
            spec_rxns, ksp_pairs, c0_use,
            tolerance=tolerance, init_scale=init_scale,
            verbose=verbose, warm_start=warm_start, lc_warm_start=lc_warm_start,
            ksp_tol=ksp_tol, max_outer=max_ksp_iter,
            c0_presolved=c0_pre,
            maintain_set=maintain_set if maintain_set else None,
            gamma_dict=gamma_dict)
        xi_full = np.zeros(len(reactions_list))
        for sub_i, orig_i in enumerate(spec_idx):   xi_full[orig_i] = xi_spec[sub_i]
        for sub_i, orig_i in enumerate(ksp_idx):
            xi_full[orig_i] = -xi_ksp[sub_i] if ksp_flip_flags[sub_i] else xi_ksp[sub_i]
        return xi_full, info

    all_sp_s, c0_vec_s, nu_s, K_vec_s, v_oa_s = _build_system(spec_rxns, c0)
    maint_mask_s = (np.array([s in maintain_set for s in all_sp_s], dtype=bool)
                    if maintain_set else np.zeros(len(all_sp_s), dtype=bool))
    g_local_s = _build_gamma_for_system(spec_rxns, all_sp_s)

    def _run_L_spec():
        return _solve_method_L(c0_vec_s, nu_s, K_vec_s, all_sp_s, v_oa=v_oa_s,
                                tolerance=tolerance, lc_warm_start=lc_warm_start,
                                verbose=verbose, maintain_mask=maint_mask_s,
                                gamma_dict=g_local_s)
    try:
        result_L_mix, timed_out_L_mix = _try_timed(_run_L_spec)
    except Exception as exc:
        if verbose: print(f"  [solver/mix] Method L raised: {exc}")
        result_L_mix, timed_out_L_mix = None, False

    if not timed_out_L_mix and result_L_mix is not None:
        xi_L_s, info_L_s = result_L_mix
        if info_L_s['error'] < tolerance:
            c0_from_L = dict(c0)
            if info_L_s.get('c_eq_direct') is not None:
                c_eq_L = np.maximum(np.asarray(info_L_s['c_eq_direct'], dtype=float), 0.0)
                for j, sp in enumerate(all_sp_s):
                    if sp != 'O/A' and sp not in maintain_set:
                        c0_from_L[sp] = float(c_eq_L[j])
            xi_full_mix, info_mix = _do_mix(c0, c0_pre=c0_from_L)
            if info_mix['error'] < tolerance:
                return xi_full_mix, info_mix

    c0_pre = _run_presolver(spec_rxns, c0)
    return _do_mix(c0, c0_pre=c0_pre)
