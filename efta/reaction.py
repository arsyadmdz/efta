"""
efta.reaction
=============
The :class:`reaction` class represents a single equilibrium reaction.

A reaction is defined by:
  - a stoichiometry dict  {species_name: coefficient}  (negative = reactant)
  - an equilibrium constant K  (must be positive)

Construction
------------
Four constructor signatures are supported:

1. String form:
   ``reaction('Fe[3+] + 3OH[-] = Fe(OH)3(s)', 1e3)``

2. Dict form:
   ``reaction({'Fe[3+]': -1, 'OH[-]': -3, 'Fe(OH)3(s)': 1}, 1e3)``

3. Reactant/product dicts:
   ``reaction({'Fe[3+]': 1, 'OH[-]': 3}, {'Fe(OH)3(s)': 1}, 1e3)``

4. (name, coefficient) pairs:
   ``reaction((-1, 'Fe[3+]'), (-3, 'OH[-]'), (1, 'Fe(OH)3(s)'), 1e3)``

Activity coefficients
---------------------
Non-ideal behaviour is handled via :meth:`set_gamma`::

    rxn.set_gamma('Fe[3+]', (my_gamma_func, 'I'))  # gamma depends on ionic strength I

The function signature is ``gamma(c_dep1, c_dep2, ...)`` and returns a float.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .species import (
    species, formula, charge, components,
    is_nonaqueous, is_organic, is_electron, is_solid,
    ELECTRON,
)
from .errors import ReactionError, BalanceError, InputError, SpeciesError

# ---------------------------------------------------------------------------
# Reaction string parser
# ---------------------------------------------------------------------------

def _parse_reaction_string(rxn_str: str) -> Dict[str, float]:
    """Parse 'A + 2B = C + D' style strings into a stoichiometry dict.
    
    Stoichiometric coefficients may be plain numbers (``2``, ``1.5``) or
    ``$(expr)`` expressions (``$(1/2)``, ``$(sqrt(2))``).
    """
    from .species import _convert_caret_charges, _find_dollar, _eval_dollar
    rxn_str = _convert_caret_charges(rxn_str)

    for sep in ('⇌', '->', '→', '='):
        if sep in rxn_str:
            lhs_raw, rhs_raw = rxn_str.split(sep, 1)
            break
    else:
        raise ReactionError(
            f"Reaction string must contain '=', '⇌', '->' or '→': {rxn_str!r}")

    def _parse_side(side_str: str) -> Dict[str, float]:
        # protect '+' inside charge brackets [...] and $(expr) from splitting
        protected = []
        sq_depth  = 0
        paren_depth = 0
        i = 0
        s = side_str.strip()
        while i < len(s):
            ch = s[i]
            if ch == '[':
                sq_depth += 1; protected.append(ch); i += 1
            elif ch == ']':
                sq_depth -= 1; protected.append(ch); i += 1
            elif ch == '$' and i + 1 < len(s) and s[i+1] == '(':
                # copy entire $(expr) verbatim — protect internal +
                hit = _find_dollar(s, i)
                if hit:
                    span = s[hit[0]:hit[1]]
                    protected.append(span.replace('+', '\x00'))
                    i = hit[1]
                else:
                    protected.append(ch); i += 1
            elif ch == '+' and sq_depth > 0:
                protected.append('\x00'); i += 1
            else:
                protected.append(ch); i += 1
        side_str = ''.join(protected)

        # split on '+' at paren depth 0
        tokens: List[str] = []
        depth, buf = 0, []
        for ch in side_str:
            if ch == '(':
                depth += 1; buf.append(ch)
            elif ch == ')':
                depth -= 1; buf.append(ch)
            elif ch == '+' and depth == 0:
                tokens.append(''.join(buf).strip()); buf = []
            else:
                buf.append(ch)
        if buf:
            tokens.append(''.join(buf).strip())

        result: Dict[str, float] = {}
        for tok in tokens:
            tok = tok.replace('\x00', '+').strip()
            if not tok:
                continue
            # try $(expr) coefficient first
            dollar_hit = _find_dollar(tok, 0)
            if dollar_hit and dollar_hit[0] == 0:
                coeff = _eval_dollar(dollar_hit[2])
                name  = tok[dollar_hit[1]:].strip()
            else:
                m = re.match(r'^(\d+(?:\.\d*)?)(?=\s|[A-Za-z\[(\^$])', tok)
                if m:
                    coeff = float(m.group(1))
                    name  = tok[m.end():].strip()
                else:
                    coeff = 1.0
                    name  = tok
            if not name:
                raise SpeciesError(
                    f"Empty species name in token: {tok!r}\n"
                    "Check for stray '+' or whitespace in the reaction string.")
            result[name] = coeff
        return result

    lhs = _parse_side(lhs_raw)
    rhs = _parse_side(rhs_raw)
    stoich: Dict[str, float] = {}
    for sp, c in lhs.items():
        sp = species(sp)
        stoich[sp] = stoich.get(sp, 0.0) - c
    for sp, c in rhs.items():
        sp = species(sp)
        stoich[sp] = stoich.get(sp, 0.0) + c
    return stoich


def _parse_reaction_args(args) -> Tuple[Dict[str, float], float]:
    """Dispatch to the appropriate constructor overload and return (stoich, K)."""
    # --- overload 1: (string, K) ---
    if (len(args) == 2
            and isinstance(args[0], str)
            and isinstance(args[1], (int, float, np.floating))):
        return _parse_reaction_string(args[0]), float(args[1])

    # --- overload 2: (stoich_dict, K) ---
    if (len(args) == 2
            and isinstance(args[0], dict)
            and isinstance(args[1], (int, float, np.floating))):
        return {str(k): float(v) for k, v in args[0].items()}, float(args[1])

    # --- overload 3: (reactants_dict, products_dict, K) ---
    if (len(args) == 3
            and isinstance(args[0], dict)
            and isinstance(args[1], dict)
            and isinstance(args[2], (int, float, np.floating))):
        stoich: Dict[str, float] = {}
        for sp, c in args[0].items():
            stoich[str(sp)] = stoich.get(str(sp), 0.0) - abs(float(c))
        for sp, c in args[1].items():
            stoich[str(sp)] = stoich.get(str(sp), 0.0) + abs(float(c))
        return stoich, float(args[2])

    # --- overload 4: ([names], [coefficients], K) ---
    if (len(args) == 3
            and isinstance(args[0], (list, tuple))
            and isinstance(args[1], (list, tuple))
            and isinstance(args[2], (int, float, np.floating))):
        names, coeffs, K = args
        if len(names) != len(coeffs):
            raise InputError(f"len(names) ({len(names)}) must equal len(coefficients) ({len(coeffs)}).")
        return {str(n): float(c) for n, c in zip(names, coeffs)}, float(K)

    # --- overload 5: (coeff, 'name'), ..., K ---
    if len(args) >= 2 and isinstance(args[-1], (int, float, np.floating)):
        K = float(args[-1])
        stoich = {}
        for pair in args[:-1]:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise InputError(
                    f"Expected (coefficient, 'name') pair, got: {pair!r}\n"
                    "Each argument before K should be (coeff, 'species_name').")
            coeff, name = pair
            stoich[str(name)] = float(coeff)
        return stoich, K

    raise InputError(
        "Unrecognised Reaction constructor arguments.\n"
        "Accepted forms: reaction('A = B', K), reaction({'A':-1,'B':1}, K),\n"
        "reaction([names], [coeffs], K), or reaction((c, name), ..., K).")


# ---------------------------------------------------------------------------
# Default (ideal) activity coefficient
# ---------------------------------------------------------------------------

def _one(*args) -> float:
    """Ideal activity coefficient: always 1.0."""
    return 1.0


# ---------------------------------------------------------------------------
# reaction class
# ---------------------------------------------------------------------------

class reaction:
    """
    A single chemical equilibrium reaction.

    Parameters
    ----------
    *args : see module docstring for the four supported constructor forms.
    ksp   : bool, optional
        Set True to mark this as a solubility-product (precipitation) reaction.
        Requires at least one '(s)' species.

    Examples
    --------
    >>> r1 = reaction('H[+] + OH[-] = H2O', 1e14)
    >>> r2 = reaction({'CaCO3(s)': -1, 'Ca[2+]': 1, 'CO3[2-]': 1}, 3.36e-9, ksp=True)
    """

    def __init__(self, *args, ksp: bool = False):
        self._stoich, self.K = _parse_reaction_args(args)
        if not self._stoich:
            raise ReactionError("A Reaction must involve at least one species.\n"
                "Check that the reaction string has species on both sides.")
        if self.K <= 0:
            raise InputError(f"K must be positive, got K = {self.K}\n"
                "Equilibrium constants must be > 0 (use ksp=True for Ksp reactions).")
        self.ksp: bool = ksp
        # gamma dict: {species_name: (callable, *dep_species)}
        self._gamma: Dict[str, tuple] = {}
        if ksp:
            has_solid = any(s.strip().lower().endswith('(s)') for s in self._stoich)
            if not has_solid:
                raise ReactionError(
                    "ksp=True requires at least one solid '(s)' species.\n"
                    "Add '(s)' suffix to the solid species, e.g. 'CaCO3(s)'.\n"
                    "Got: "
                    f"Got: {list(self._stoich)}")

    # ------------------------------------------------------------------
    # Activity-coefficient interface
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> Dict[str, tuple]:
        """Dict of registered activity-coefficient functions."""
        return dict(self._gamma)

    def set_gamma(self, sp: str, gamma_spec: tuple) -> 'reaction':
        """
        Register an activity-coefficient function for species *sp*.

        Parameters
        ----------
        sp          : species name (will be normalised)
        gamma_spec  : tuple of ``(func, *dep_names)`` where

                      - *func* is a callable returning gamma (float)
                      - *dep_names* are the names of dependencies passed to
                        *func* at each solver iteration, in order.  Each entry
                        is either a species name (its current equilibrium
                        concentration is passed) or the special token ``'I'``
                        (the current ionic strength in mol/L is passed instead).

        Examples
        --------
        Constant gamma (no dependencies)::

            r.set_gamma('Fe[3+]', (lambda: 0.5,))

        Davies equation using ionic strength::

            import math
            def davies(I):
                sqI = math.sqrt(I)
                return 10 ** (-0.509 * (sqI / (1 + sqI) - 0.3 * I))
            r.set_gamma('Fe[3+]', (davies, 'I'))

        Gamma depending on a specific species concentration::

            r.set_gamma('Fe[3+]', (lambda c_cl: 1 - 0.1 * c_cl, 'Cl[-]'))
        """
        from .system import _IONIC_STRENGTH_TOKEN
        from .errors import InputError
        if not isinstance(gamma_spec, tuple) or len(gamma_spec) == 0:
            raise InputError(
                "gamma_spec must be a non-empty tuple (func, *dep_names), "
                f"got {gamma_spec!r}.")
        func      = gamma_spec[0]
        dep_names = gamma_spec[1:]
        if not callable(func):
            raise InputError(
                f"First element of gamma_spec must be callable, got {func!r}.")
        sp_norm  = species(sp)
        dep_norm = tuple(
            s if s == _IONIC_STRENGTH_TOKEN else species(s)
            for s in dep_names
        )
        self._gamma[sp_norm] = (func,) + dep_norm
        return self

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def stoich(self) -> Dict[str, float]:
        """Stoichiometry dict (copy)."""
        return dict(self._stoich)

    @property
    def species(self) -> FrozenSet[str]:
        """All species names involved in this reaction."""
        return frozenset(self._stoich)

    @property
    def aqueous_species(self) -> FrozenSet[str]:
        """Aqueous (non-solid, non-organic, non-electron) species."""
        return frozenset(s for s in self._stoich
                         if not is_nonaqueous(s) and not is_organic(s)
                         and not is_electron(s))

    @property
    def organic_species(self) -> FrozenSet[str]:
        return frozenset(s for s in self._stoich if is_organic(s))

    @property
    def nonaqueous_species(self) -> FrozenSet[str]:
        return frozenset(s for s in self._stoich if is_nonaqueous(s))

    @property
    def reactants(self) -> FrozenSet[str]:
        return frozenset(s for s, c in self._stoich.items() if c < 0)

    @property
    def products(self) -> FrozenSet[str]:
        return frozenset(s for s, c in self._stoich.items() if c > 0)

    @property
    def components(self) -> FrozenSet[str]:
        """All elements present across all species in this reaction."""
        comps: FrozenSet[str] = frozenset()
        for sp in self._stoich:
            comps = comps | frozenset(components(sp).keys())
        return comps

    @property
    def stoichiometry(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (reactants, products) dicts with positive coefficients."""
        reactants = {sp: abs(c) for sp, c in self._stoich.items() if c < 0}
        products  = {sp:     c  for sp, c in self._stoich.items() if c > 0}
        return (reactants, products)

    @property
    def type(self) -> str:
        """'precipitation' for Ksp reactions, 'speciation' otherwise."""
        from .balance import _is_ksp_reaction
        return 'precipitation' if _is_ksp_reaction(self) else 'speciation'

    @property
    def clusters(self) -> FrozenSet[str]:
        """Conservation clusters detected for this reaction."""
        from .balance import _compute_clusters
        return _compute_clusters([self])

    @property
    def is_mixed_phase(self) -> bool:
        """True if this reaction involves both aqueous and organic species."""
        from .balance import _reaction_is_mixed_phase
        return _reaction_is_mixed_phase(self._stoich)

    @property
    def atom_balance(self):
        """Return the element/charge balance rows for this reaction."""
        from .balance import _build_balance
        return _build_balance(self._stoich.keys())

    @property
    def balanced(self) -> bool:
        """True if the reaction is both element- and charge-balanced."""
        sp_list = [sp for sp in self._stoich if not is_electron(sp)]
        elem_totals: Dict[str, float] = {}
        for sp in sp_list:
            coeff = self._stoich[sp]
            for elem, n in components(sp).items():
                elem_totals[elem] = elem_totals.get(elem, 0.0) + coeff * n
        if any(abs(v) > 1e-9 for v in elem_totals.values()):
            return False
        charge_sum = sum(
            self._stoich[sp] * charge(sp)
            for sp in self._stoich if not is_electron(sp)
        )
        for sp, c in self._stoich.items():
            if is_electron(sp):
                charge_sum += c * (-1)
        return abs(charge_sum) < 1e-9

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: 'reaction') -> 'reaction':
        """Combine two reactions by adding their stoichiometries (K values multiply)."""
        if not isinstance(other, reaction):
            return NotImplemented
        new_stoich = dict(self._stoich)
        for sp, c in other._stoich.items():
            new_stoich[sp] = new_stoich.get(sp, 0.0) + c
        new_stoich = {sp: c for sp, c in new_stoich.items() if abs(c) > 1e-14}
        if not new_stoich:
            raise ReactionError("Addition of these reactions cancels all species.\n"
            "The two reactions are exact negatives of each other.")
        r = reaction.__new__(reaction)
        r._stoich = new_stoich
        r.K       = self.K * other.K
        r.ksp     = self.ksp or other.ksp
        r._gamma  = {**self._gamma, **other._gamma}
        return r

    def _scaled(self, n: float) -> 'reaction':
        n = float(n)
        if n == 0:
            raise ReactionError("Cannot scale a reaction by 0.")
        if n == -1:
            new_stoich = {sp: -c for sp, c in self._stoich.items()}
            new_K = 1.0 / self.K
        else:
            new_stoich = {sp: c * n for sp, c in self._stoich.items()}
            new_K = self.K ** n
        r = reaction.__new__(reaction)
        r._stoich = new_stoich
        r.K       = new_K
        r.ksp     = self.ksp
        r._gamma  = dict(self._gamma)
        return r

    def __mul__(self, n):
        if not isinstance(n, (int, float)):
            return NotImplemented
        return self._scaled(n)

    def __rmul__(self, n):
        if not isinstance(n, (int, float)):
            return NotImplemented
        return self._scaled(n)

    def __truediv__(self, n):
        if not isinstance(n, (int, float)):
            return NotImplemented
        if n == 0:
            raise ReactionError("Cannot divide a reaction by 0.")
        return self._scaled(1.0 / n)

    def __getitem__(self, species_key) -> float:
        """Return the stoichiometric coefficient of a species (0 if absent)."""
        key = species(str(species_key))
        return self._stoich.get(key, 0.0)

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, reaction):
            return NotImplemented
        if set(self._stoich) != set(other._stoich):
            return False
        factor = None
        for sp in self._stoich:
            a = self._stoich[sp]
            b = other._stoich[sp]
            if abs(a) < 1e-14 and abs(b) < 1e-14:
                continue
            if abs(a) < 1e-14 or abs(b) < 1e-14:
                return False
            f = b / a
            if factor is None:
                factor = f
            else:
                if abs(f - factor) > 1e-9 * max(1.0, abs(factor)):
                    return False
        if factor is None:
            return True
        if self.K <= 0:
            return False
        expected_K = self.K ** factor
        return abs(expected_K - other.K) < 1e-9 * max(1.0, abs(expected_K))

    def __hash__(self) -> int:
        coeffs = list(self._stoich.values())
        min_abs = min((abs(c) for c in coeffs if abs(c) > 1e-14), default=1.0)
        canon = tuple(sorted(
            (sp, round(c / min_abs, 9))
            for sp, c in self._stoich.items()
        ))
        return hash(canon)

    # ------------------------------------------------------------------
    # Manipulation: simplify, balance, copy
    # ------------------------------------------------------------------

    @staticmethod
    def _coeff_gcf(coeffs):
        from fractions import Fraction
        from math import gcd, lcm
        from functools import reduce
        fracs   = [Fraction(abs(c)).limit_denominator(10_000) for c in coeffs]
        nums    = [f.numerator   for f in fracs]
        dens    = [f.denominator for f in fracs]
        gcf_num = reduce(gcd, nums)
        lcm_den = reduce(lcm, dens)
        return float(Fraction(gcf_num, lcm_den))

    def simplify(self) -> 'reaction':
        """Divide all coefficients by their GCF, adjusting K accordingly."""
        g = self._coeff_gcf(self._stoich.values())
        if abs(g - 1.0) < 1e-12:
            return self
        self._stoich = {sp: c / g for sp, c in self._stoich.items()}
        self.K = self.K ** g
        return self

    def balance(self) -> 'reaction':
        """
        Rebalance the reaction using SVD of the atom/charge matrix.

        Raises ValueError if the stoichiometry is inconsistent or the null
        space is not one-dimensional.
        """
        from fractions import Fraction
        from math import gcd, lcm
        from functools import reduce

        species_list = list(self._stoich.keys())
        n_sp    = len(species_list)
        signs   = [1 if self._stoich[s] > 0 else -1 for s in species_list]

        elements: set = set()
        for s in species_list:
            elements.update(components(s).keys())
        row_labels = sorted(elements)
        if any(charge(s) != 0 for s in species_list):
            row_labels.append('__charge__')

        if not row_labels:
            raise BalanceError(
                "Cannot balance: no element or charge information found.\n"
                "All species may be unknown or have unparseable formulas.")

        A = np.zeros((len(row_labels), n_sp), dtype=float)
        for ei, elem in enumerate(row_labels):
            for si, s in enumerate(species_list):
                val = (float(charge(s))
                       if elem == '__charge__'
                       else float(components(s).get(elem, 0)))
                A[ei, si] = signs[si] * val

        _, sv, Vt = np.linalg.svd(A, full_matrices=True)
        tol       = max(A.shape) * np.finfo(float).eps * (sv[0] if sv.size else 1.0)
        null_mask = np.array([True] * n_sp)
        null_mask[:len(sv)] = sv < tol
        null_dim = int(null_mask.sum())

        if null_dim == 0:
            raise BalanceError(
            "Cannot balance: the stoichiometry is inconsistent.\n"
            "No combination of integer coefficients satisfies both atom and\n"
            "charge conservation for these species.")
        if null_dim > 1:
            raise BalanceError(
                f"Cannot balance: underdetermined — {null_dim} independent solutions exist.\n"
                "Specify stoichiometric coefficients explicitly.")

        null_vec = Vt[-1]
        if null_vec[0] < 0:
            null_vec = -null_vec

        def _to_minimal_ints(vec):
            for v in vec:
                if abs(v) > 1e-12:
                    if v < 0: vec = -vec
                    break
            first = next((v for v in vec if abs(v) > 1e-12), None)
            if first is None: return None
            normed = vec / first
            for denom in range(1, 201):
                trial    = normed * denom
                rounded  = np.round(trial).astype(int)
                if float(np.max(np.abs(trial - rounded))) < 1e-6:
                    g = reduce(gcd, [abs(int(c)) for c in rounded if c != 0])
                    return rounded // g
            return None

        minimal = _to_minimal_ints(null_vec)
        if minimal is None:
            raise BalanceError(
            "Cannot balance: could not find small integer stoichiometric coefficients.\n"
            "Try specifying the balanced reaction string explicitly.")

        old_abs = [abs(self._stoich[s]) for s in species_list]
        new_abs = [abs(minimal[i]) for i in range(n_sp)]
        ratios  = sorted(n / o for n, o in zip(new_abs, old_abs) if o > 0)
        n_r     = len(ratios)
        scale   = (ratios[n_r // 2] if n_r % 2 == 1
                   else (ratios[n_r // 2 - 1] + ratios[n_r // 2]) / 2)

        self._stoich = {s: signs[i] * minimal[i] for i, s in enumerate(species_list)}
        if abs(scale - 1.0) > 1e-12:
            import math
            new_logK = math.log10(max(self.K, 1e-300)) * scale
            if abs(new_logK) > 300:
                raise BalanceError(
                    f"Coefficients balanced, but implied K adjustment overflows "
                    f"(scale={scale:.4g}, new log₁₀K≈{new_logK:.1f}).")
            self.K = self.K ** scale
        return self

    def decompose(self) -> Dict[str, Dict[str, float]]:
        """Return the conservation-cluster decomposition for this reaction."""
        from .balance import _compute_decompose
        return _compute_decompose([self])

    def copy(self) -> 'reaction':
        r = reaction.__new__(reaction)
        r._stoich = dict(self._stoich)
        r.K       = self.K
        r.ksp     = self.ksp
        r._gamma  = dict(self._gamma)
        return r

    # ------------------------------------------------------------------
    # Solve / equilibrium – delegate to the reactions class
    # ------------------------------------------------------------------

    def solve(self, c0: dict, tolerance: float = 1e-6,
              init_scale: float = 0.1, verbose: bool = False,
              maintain=None) -> float:
        """
        Compute the equilibrium extent of reaction ξ.

        Returns a scalar float (mol / L_aq).
        """
        from .reactions import reactions as _reactions
        return float(_reactions(self).solve(
            c0, tolerance=tolerance, init_scale=init_scale,
            verbose=verbose, maintain=maintain)[0])

    def equilibrium(self, c0: dict, **kw) -> Dict[str, float]:
        """
        Compute equilibrium concentrations.

        Returns a dict {species: concentration_M}.
        """
        from .reactions import reactions as _reactions
        return _reactions(self).equilibrium(c0, **kw)

    def find(self, unknown: str, c0: dict, target: dict, **kw) -> float:
        """
        Find the initial concentration of *unknown* that achieves *target*
        equilibrium concentrations.  See reactions.find() for full docs.
        """
        from .reactions import reactions as _reactions
        return _reactions(self).find(unknown, c0, target, **kw)

    def __rshift__(self, sol):
        """
        Return a new solution at equilibrium: ``r >> sol``.

        Examples
        --------
        >>> eq = r1 >> sol
        """
        from .solution import solution as _solution
        if not isinstance(sol, _solution):
            return NotImplemented
        return sol << self

    def plot(self, c0, **kw):
        """Plot equilibrium concentrations vs sweep. See reactions.plot()."""
        from .reactions import reactions as _reactions
        return _reactions(self).plot(c0, **kw)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_coeff(c: float) -> str:
        a = abs(c)
        if a == 1.0: return ''
        return str(int(a)) if a == int(a) else str(a)

    def _side_str(self, side: str) -> str:
        from .model.freaction import _clean_species as _cs
        parts = []
        for sp, c in self._stoich.items():
            sp_clean = _cs(sp)
            if side == 'reactants' and c < 0:
                parts.append(f"{self._fmt_coeff(c)}{sp_clean}")
            elif side == 'products' and c > 0:
                parts.append(f"{self._fmt_coeff(c)}{sp_clean}")
        return ' + '.join(parts)

    def __repr__(self) -> str:
        return (f"{self._side_str('reactants')} ⇌ "
                f"{self._side_str('products')},  K = {self.K:g}")

    def __str__(self) -> str:
        from .plotting import _format_reaction_str
        return _format_reaction_str(self) + f',  K = {self.K:g}'