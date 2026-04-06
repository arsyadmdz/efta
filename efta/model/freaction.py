"""
efta.model.freaction
====================
:class:`freaction` — a parameterised reaction with ``$(xN)`` placeholders.

:class:`freactions` — a collection of :class:`freaction` and/or
:class:`~efta.reaction.reaction` objects sharing a unified parameter space.

Syntax
------
``$(xN)`` or ``$(expr involving xN)`` — free parameter anywhere:
as a stoichiometric coefficient, a species subscript, or the K value.

Constructor overloads
---------------------
1. **String template**::

       freaction("$(x1)A + $(x2)B = A$(x1)B$(x2), $(x3)")

2. **Reactants dict, products dict, K**::

       freaction({'A': '$(x1)', 'B': 1}, {'A$(x1)B': 1}, '$(x2)')

3. **Stoichiometry dict, K** (negative = reactant)::

       freaction({'A': '-$(x1)', 'B': -1, 'A$(x1)B': 1}, '$(x2)')

API
---
>>> f = freaction("$(x1)A + $(x3)B = A$(x1)B$(x3), $(x6)").params(['a','b','K']).trim()
>>> rxn     = f.fit([2, 3, 100])           # reaction with those values
>>> rxn     = f.model('cont', eq)          # reaction with best-fit values
>>> results = f.analyze(eq, n=100) # bootstrap Analyzeds
"""

from __future__ import annotations

import re
from inspect import Parameter, Signature
from typing import Dict, List, Union

__all__ = ['freaction', 'freactions']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dollar_tools():
    from ..species import _find_dollar, _eval_dollar
    return _find_dollar, _eval_dollar


def _extract_indices(template: str) -> List[int]:
    """Return sorted unique xN indices from all $(expr) in *template*."""
    _find_dollar, _ = _get_dollar_tools()
    indices = set()
    pos = 0
    while True:
        hit = _find_dollar(template, pos)
        if hit is None:
            break
        for xm in re.finditer(r'x(\d+)', hit[2]):
            indices.add(int(xm.group(1)))
        pos = hit[1]
    return sorted(indices)


def _substitute(template: str, mapping: Dict[int, float]) -> str:
    """Evaluate all ``$(expr)`` in *template*, substituting xN from *mapping*."""
    _find_dollar, _ = _get_dollar_tools()
    result = []
    pos = 0
    while True:
        hit = _find_dollar(template, pos)
        if hit is None:
            result.append(template[pos:])
            break
        result.append(template[pos:hit[0]])
        expr   = hit[2]
        filled = re.sub(r'x(\d+)',
                        lambda m: str(mapping[int(m.group(1))]),
                        expr)
        try:
            val = float(eval(filled, _EVAL_NS, {}))
        except Exception as e:
            raise ValueError(
                f"Could not evaluate freaction expression $({expr!r}) "
                f"with {filled!r}: {e}")
        result.append(str(int(val)) if val == int(val) else f'$({val:.6g})')
        pos = hit[1]
    return ''.join(result)


def _rewrite_indices(template: str, reindex: dict) -> str:
    """Rewrite xN indices in all $(expr) in *template* using *reindex* map."""
    _find_dollar, _ = _get_dollar_tools()
    result = []
    pos = 0
    while True:
        hit = _find_dollar(template, pos)
        if hit is None:
            result.append(template[pos:])
            break
        result.append(template[pos:hit[0]])
        new_expr = re.sub(
            r'x(\d+)',
            lambda m: f'x{reindex[int(m.group(1))]}' if int(m.group(1)) in reindex
                      else m.group(0),
            hit[2]
        )
        result.append(f'$({new_expr})')
        pos = hit[1]
    return ''.join(result)


def _build_eval_ns() -> dict:
    import math
    ns = {
        '__builtins__': {},
        **{k: getattr(math, k) for k in dir(math) if not k.startswith('_')},
        'abs': abs, 'round': round, 'int': int, 'float': float,
        'min': min, 'max': max,
    }
    try:
        import numpy as np
        ns.update({'np': np, 'log2': np.log2, 'cbrt': np.cbrt})
    except ImportError:
        pass
    return ns

# Build once at module load — constant dict, no need to rebuild each call
_EVAL_NS = _build_eval_ns()


def _build_signature(names: List[str]) -> Signature:
    return Signature([Parameter(n, Parameter.POSITIONAL_OR_KEYWORD) for n in names])


def _infer_method(func) -> str:
    """Infer fitting method from stored constraints.
    'cont' if all constraints are (0, inf), 'ranged' otherwise."""
    constraints = getattr(func, 'constraints', {})
    for spec in constraints.values():
        if isinstance(spec, list):
            return 'ranged'
        if isinstance(spec, tuple) and spec != (0, float('inf')):
            return 'ranged'
    return 'cont'


def _clean_species(name: str) -> str:
    """
    Simplify a species name produced after parameter substitution.

    Rules (applied after $(expr) are evaluated to plain numbers):
    - ``X1Y``  → ``XY``    (integer 1 count omitted)
    - ``X0Y``  → ``XY``    (0 count: drop X and the 0)
    - ``X1.50Y`` stays     (non-integer decimal kept as-is)
    - Any remaining ``$(expr)`` → run through species_std

    Works on: atom counts and charge values inside [...].
    """
    import re
    from ..species import species_std

    # first handle any remaining $(...) with species_std
    name = species_std(name)

    # ── atom counts: replace integer 1 (not inside [...]) ────────────────────
    # scan char by char tracking bracket depth
    result   = []
    in_brack = False
    i        = 0
    s        = name
    n        = len(s)

    while i < n:
        ch = s[i]
        if ch == '[':
            in_brack = True
            result.append(ch); i += 1; continue
        if ch == ']':
            in_brack = False
            result.append(ch); i += 1; continue

        if not in_brack and ch.isdigit():
            # collect full number (int or decimal)
            j = i
            while j < n and (s[j].isdigit() or s[j] == '.'):
                j += 1
            num_str = s[i:j]
            try:
                val = float(num_str)
            except ValueError:
                result.append(ch); i += 1; continue

            if val == 0.0:
                # drop preceding element (uppercase + any following lowercase)
                while result and result[-1].islower():
                    result.pop()
                if result and result[-1].isupper():
                    result.pop()
                i = j
            elif val == 1.0 and '.' not in num_str:
                # omit plain integer 1
                i = j
            else:
                result.append(num_str)
                i = j
        else:
            result.append(ch); i += 1

    return ''.join(result)


def _indices_to_names(indices: List[int]) -> List[str]:
    return [f'x{i}' for i in indices]


# ---------------------------------------------------------------------------
# Template builder from dict overloads
# ---------------------------------------------------------------------------

def _dict_to_template(args) -> str:
    def _fmt(v) -> str:
        if isinstance(v, str):
            return v
        v = float(v)
        return str(int(abs(v))) if abs(v) == int(abs(v)) else f'{abs(v):.6g}'

    def _side(d):
        parts = []
        for sp, c in d.items():
            s = _fmt(c)
            parts.append(f'{s}{sp}' if s != '1' else sp)
        return ' + '.join(parts)

    if len(args) == 3 and isinstance(args[0], dict) and isinstance(args[1], dict):
        lhs = _side(args[0])
        rhs = _side(args[1])
        K   = str(args[2]) if isinstance(args[2], str) else f'{float(args[2]):.6g}'
        return f'{lhs} = {rhs}, {K}'

    if len(args) == 2 and isinstance(args[0], dict):
        reac, prod = {}, {}
        for sp, v in args[0].items():
            if isinstance(v, str):
                s = v.strip()
                if s.startswith('-'):
                    reac[sp] = s[1:].strip() or '1'
                else:
                    prod[sp] = s.lstrip('+').strip() or '1'
            else:
                (reac if float(v) < 0 else prod)[sp] = abs(float(v))
        lhs = _side(reac)
        rhs = _side(prod)
        K   = str(args[1]) if isinstance(args[1], str) else f'{float(args[1]):.6g}'
        return f'{lhs} = {rhs}, {K}'

    raise ValueError(
        "freaction accepts: (template_str), "
        "(reactants_dict, products_dict, K), or (stoich_dict, K).")


# ---------------------------------------------------------------------------
# freaction
# ---------------------------------------------------------------------------

class freaction:
    """
    Parameterised reaction with ``$(xN)`` placeholders.

    See module docstring for constructor overloads and full API.
    """

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            template = args[0]
        elif len(args) >= 2:
            template = _dict_to_template(args)
        else:
            raise ValueError("freaction requires at least one argument.")

        self._template   = template
        self._indices    = _extract_indices(template)
        self._names      = _indices_to_names(self._indices)
        self._gamma: Dict[str, tuple] = {}
        self.constraints: dict = {}   # {name: list | (lo, hi)}
        self.__signature__ = _build_signature(self._names)

    # ── parameter naming ─────────────────────────────────────────────────────

    def params(self, names: List[str]) -> 'freaction':
        """
        Rename free parameters to meaningful names. Returns *self*.

        Examples
        --------
        >>> f = freaction("$(x1)A = B, $(x2)").params(['a', 'K'])
        >>> f(a=2, K=100)
        """
        if len(names) != len(self._indices):
            raise ValueError(
                f"params() expected {len(self._indices)} names "
                f"(indices: {self._indices}), got {len(names)}.")
        for n in names:
            if not n.isidentifier():
                raise ValueError(f"{n!r} is not a valid Python identifier.")
        self._names        = list(names)
        self.__signature__ = _build_signature(self._names)
        return self

    # ── trim ─────────────────────────────────────────────────────────────────


    def constrain(self, **constraints) -> 'freaction':
        """
        Set or update parameter constraints.

        Each keyword argument must match a parameter name.  Values:

        - ``list``  — discrete set, e.g. ``a=[1, 2, 3]``
        - ``tuple`` — continuous range ``(lo, hi)``, e.g. ``K=(1e-5, 1e5)``

        Unconstrained params default to ``(0, inf)``.
        Calling again **merges** — existing constraints are preserved unless
        explicitly overwritten.

        Returns *self* (chainable).

        Examples
        --------
        >>> f = freaction("$(x1)A = B, $(x2)").params(['a', 'K'])
        >>> f.constrain(a=[1, 2, 3], K=(1e-5, 1e5))
        >>> f.constrain(K=(1e-3, 1e3))   # updates K only, a unchanged
        """
        for name, spec in constraints.items():
            if name not in self._names:
                raise ValueError(
                    f"constrain(): {name!r} is not a parameter of this freaction. "
                    f"Known parameters: {self._names}")
            if isinstance(spec, list):
                self.constraints[name] = list(spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                self.constraints[name] = (float(spec[0]), float(spec[1]))
            else:
                raise ValueError(
                    f"constrain(): {name!r} must be a list or (lo, hi) tuple, "
                    f"got {type(spec).__name__!r}.")
        return self

    def trim(self) -> 'freaction':
        """
        Renumber ``$(xN)`` indices to ``x1, x2, x3, ...`` if non-contiguous.

        Custom parameter names are preserved in order. Returns *self*.

        Examples
        --------
        >>> f = freaction("$(x1)A + $(x3)B = C, $(x6)").params(['a','b','K'])
        >>> f.trim()
        >>> f.template    # "$(x1)A + $(x2)B = C, $(x3)"
        >>> f.param_names # ['a', 'b', 'K']
        """
        if self._indices == list(range(1, len(self._indices) + 1)):
            return self
        reindex        = {old: new for new, old in enumerate(self._indices, start=1)}
        self._template = _rewrite_indices(self._template, reindex)
        self._indices  = list(range(1, len(self._indices) + 1))
        self.__signature__ = _build_signature(self._names)
        return self

    # ── activity coefficients ─────────────────────────────────────────────────

    def set_gamma(self, sp: str, gamma_spec: tuple) -> 'freaction':
        """Register an activity-coefficient function. Returns self."""
        self._gamma[sp] = gamma_spec
        return self

    @property
    def gamma(self) -> dict:
        return dict(self._gamma)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def indices(self) -> List[int]:
        return list(self._indices)

    @property
    def param_names(self) -> List[str]:
        return list(self._names)

    @property
    def template(self) -> str:
        return self._template

    # ── internal builder ─────────────────────────────────────────────────────

    def _build_reaction(self, values):
        """Build a reaction from a sequence of parameter values (internal)."""
        from ..reaction import reaction as _reaction, _parse_reaction_string

        # Cache by rounded parameter tuple — the optimizer calls this once per
        # parameter set but _build_objective evaluates residuals across all data
        # points with the same parameters, so we'd re-substitute 17× for free.
        key = tuple(round(float(v), 12) for v in values)
        if hasattr(self, '_rxn_cache') and self._rxn_cache[0] == key:
            return self._rxn_cache[1]

        mapping     = {idx: float(v) for idx, v in zip(self._indices, values)}
        substituted = _substitute(self._template, mapping)
        last_comma  = substituted.rfind(',')
        if last_comma == -1:
            raise ValueError(
                f"freaction template must contain a comma separating "
                f"the reaction from K, got: {substituted!r}")
        rxn_part = substituted[:last_comma].strip()
        k_part   = substituted[last_comma + 1:].strip()
        try:
            K = float(k_part)
        except ValueError:
            _find_dollar, _eval_dollar = _get_dollar_tools()
            hit = _find_dollar(k_part, 0)
            K   = _eval_dollar(hit[2]) if hit else float(k_part)

        # Cache parsed stoichiometry — only re-parse when rxn_part changes
        if not hasattr(self, '_stoich_cache') or self._stoich_cache[0] != rxn_part:
            stoich = _parse_reaction_string(rxn_part)
            stoich = {_clean_species(sp): nu for sp, nu in stoich.items()}
            self._stoich_cache = (rxn_part, stoich)
        else:
            stoich = self._stoich_cache[1]

        rxn = _reaction(stoich, K)
        for sp, spec in self._gamma.items():
            rxn.set_gamma(sp, spec)

        self._rxn_cache = (key, rxn)
        return rxn

    # ── public API ───────────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs):
        """Call like a function — builds a reaction from positional/keyword args."""
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        values = [float(bound.arguments[n]) for n in self._names]
        return self._build_reaction(values)

    def fit(self, values: list):
        """
        Build a :class:`~efta.reaction.reaction` from explicit parameter values.

        Parameters
        ----------
        values : list
            Values in param order (same as ``param_names``).

        Returns
        -------
        reaction

        Examples
        --------
        >>> f = freaction("$(x1)A = B, $(x2)").params(['a','K'])
        >>> rxn = f.fit([2, 100])
        """
        if len(values) != len(self._indices):
            raise ValueError(
                f"fit() expected {len(self._indices)} values "
                f"({self._names}), got {len(values)}.")
        return self._build_reaction(values)

    def model(self, equilibrium: dict,
              initial: dict = None, v_oa: float = 1.0, **kwargs):
        """
        Fit and return a :class:`~efta.reaction.reaction` with best-fit parameters.

        Parameters
        ----------
        equilibrium : dict
        initial : dict, optional
        **kwargs
            Forwarded to :func:`~efta.model.fitting.model`.

        Returns
        -------
        reaction
        """
        from .fitting import model as _model_fn
        result = _model_fn(self, equilibrium,
                           initial=initial, v_oa=v_oa, **kwargs)
        return self._build_reaction(result.popt)

    def analyze(self, equilibrium: dict,
                initial: dict = None, v_oa: float = 1.0, **kwargs):
        """
        Bootstrap model selection. Returns ``list[Analyzed]``.

        Parameters
        ----------
        equilibrium : dict
        initial : dict, optional
        **kwargs
            Forwarded to :func:`~efta.model.fitting.analyze`.
        """
        from .fitting import analyze as _analyze_fn
        return _analyze_fn(self, equilibrium,
                           initial=initial, v_oa=v_oa, **kwargs)


    def montecarlo(self, equilibrium: dict,
                   initial: dict = None,
                   v_oa: float = 1.0,
                   **kwargs) -> 'Analyzed':
        """
        Monte Carlo uncertainty analysis by perturbing concentrations.

        Parameters
        ----------
        equilibrium : dict
        initial : dict, optional
        **kwargs
            ``noise``, ``n``, ``seed``, and fitting kwargs.

        Returns
        -------
        MonteCarlo
        """
        from .fitting import montecarlo as _mc
        return _mc(self, equilibrium, initial=initial,
                   v_oa=v_oa, **kwargs)



    def __repr__(self) -> str:
        renamed = any(n != f'x{i}' for n, i in zip(self._names, self._indices))
        base = f"freaction({self._template!r})"
        if renamed:
            return base + f".params({self._names!r})"
        return base


# ---------------------------------------------------------------------------
# freactions
# ---------------------------------------------------------------------------

class freactions:
    """
    Collection of :class:`freaction` and/or :class:`~efta.reaction.reaction`
    objects sharing a unified parameter space.
    """

    def __init__(self, *members):
        from ..reaction import reaction as _reaction

        self._members = list(members)
        all_indices   = set()
        for m in members:
            if isinstance(m, freaction):
                all_indices.update(m.indices)
            elif not isinstance(m, _reaction):
                raise TypeError(
                    f"freactions members must be freaction or reaction, "
                    f"got {type(m).__name__!r}.")

        self._indices = sorted(all_indices)
        # inherit names already set on members
        name_map = {}
        for m in members:
            if isinstance(m, freaction):
                for idx, name in zip(m.indices, m.param_names):
                    name_map[idx] = name
        self._names        = [name_map.get(i, f'x{i}') for i in self._indices]
        self.__signature__ = _build_signature(self._names)

        # initialise constraints, merging from member freactions
        self.constraints: dict = {}
        for m in members:
            if isinstance(m, freaction):
                for idx, name in zip(m.indices, m.param_names):
                    sys_name = name_map.get(idx, f'x{idx}')
                    if name in m.constraints:
                        self.constraints[sys_name] = m.constraints[name]

    # ── parameter naming ─────────────────────────────────────────────────────

    def params(self, names: List[str]) -> 'freactions':
        """Rename shared parameters. Returns *self*."""
        if len(names) != len(self._indices):
            raise ValueError(
                f"params() expected {len(self._indices)} names, got {len(names)}.")
        for n in names:
            if not n.isidentifier():
                raise ValueError(f"{n!r} is not a valid Python identifier.")
        self._names        = list(names)
        self.__signature__ = _build_signature(self._names)
        for m in self._members:
            if isinstance(m, freaction):
                member_names = [names[self._indices.index(i)] for i in m.indices]
                m.params(member_names)
        return self

    # ── trim ─────────────────────────────────────────────────────────────────


    def constrain(self, **constraints) -> 'freactions':
        """
        Set or update parameter constraints at the system level.

        Merges with any constraints already stored on member freactions.
        Calling again merges — existing constraints are preserved unless
        explicitly overwritten.

        Returns *self* (chainable).

        Examples
        --------
        >>> sys = freactions(fa, fb).params(['a', 'K1', 'K2'])
        >>> sys.constrain(a=[1, 2, 3], K1=(1e-5, 1e5), K2=(1e-5, 1e5))
        """
        for name, spec in constraints.items():
            if name not in self._names:
                raise ValueError(
                    f"constrain(): {name!r} is not a parameter of this freactions. "
                    f"Known parameters: {self._names}")
            if isinstance(spec, list):
                self.constraints[name] = list(spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                self.constraints[name] = (float(spec[0]), float(spec[1]))
            else:
                raise ValueError(
                    f"constrain(): {name!r} must be a list or (lo, hi) tuple.")
        return self

    def trim(self) -> 'freactions':
        """
        Renumber ``$(xN)`` indices to ``x1, x2, ...`` across all members.
        Custom names are preserved. Returns *self*.
        """
        if self._indices == list(range(1, len(self._indices) + 1)):
            return self
        reindex = {old: new for new, old in enumerate(self._indices, start=1)}
        for m in self._members:
            if isinstance(m, freaction):
                m._template = _rewrite_indices(m._template, reindex)
                m._indices  = sorted(reindex[i] for i in m._indices)
                m.__signature__ = _build_signature(m._names)
        self._indices      = list(range(1, len(self._indices) + 1))
        self.__signature__ = _build_signature(self._names)
        return self

    # ── activity coefficients ─────────────────────────────────────────────────

    def set_gamma(self, sp: str, gamma_spec: tuple) -> 'freactions':
        """Broadcast set_gamma to all freaction members. Returns self."""
        for m in self._members:
            if isinstance(m, freaction):
                m.set_gamma(sp, gamma_spec)
        return self

    @property
    def gamma(self) -> dict:
        merged = {}
        for m in self._members:
            if isinstance(m, freaction):
                merged.update(m.gamma)
        return merged

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def indices(self) -> List[int]:
        return list(self._indices)

    @property
    def param_names(self) -> List[str]:
        return list(self._names)

    # ── internal builder ─────────────────────────────────────────────────────

    def _build_reactions(self, values):
        """Build a reactions object from a sequence of parameter values."""
        from ..reactions import reactions as _reactions
        idx_to_val = {idx: float(v) for idx, v in zip(self._indices, values)}
        built = []
        for m in self._members:
            if isinstance(m, freaction):
                member_vals = [idx_to_val[i] for i in m.indices]
                built.append(m._build_reaction(member_vals))
            else:
                built.append(m)
        return _reactions(*built)

    # ── public API ───────────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs):
        """Call like a function — builds a reactions from positional/keyword args."""
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        values = [float(bound.arguments[n]) for n in self._names]
        return self._build_reactions(values)

    def fit(self, values: list):
        """
        Build a :class:`~efta.reactions.reactions` from explicit parameter values.

        Parameters
        ----------
        values : list
            Values in param order (same as ``param_names``).

        Returns
        -------
        reactions
        """
        if len(values) != len(self._indices):
            raise ValueError(
                f"fit() expected {len(self._indices)} values "
                f"({self._names}), got {len(values)}.")
        return self._build_reactions(values)

    def model(self, equilibrium: dict,
              initial: dict = None, v_oa: float = 1.0, **kwargs):
        """
        Fit and return a :class:`~efta.reactions.reactions` with best-fit parameters.

        Parameters
        ----------
        equilibrium : dict
        initial : dict, optional
        **kwargs
            Forwarded to :func:`~efta.model.fitting.model`.

        Returns
        -------
        reactions
        """
        from .fitting import model as _model_fn
        result = _model_fn(self, equilibrium,
                           initial=initial, v_oa=v_oa, **kwargs)
        return self._build_reactions(result.popt)

    def analyze(self, equilibrium: dict,
                initial: dict = None, v_oa: float = 1.0, **kwargs):
        """
        Bootstrap model selection. Returns ``list[Analyzed]``.

        Parameters
        ----------
        equilibrium : dict
        initial : dict, optional
        **kwargs
            Forwarded to :func:`~efta.model.fitting.analyze`.
        """
        from .fitting import analyze as _analyze_fn
        return _analyze_fn(self, equilibrium,
                           initial=initial, v_oa=v_oa, **kwargs)


    def montecarlo(self, equilibrium: dict,
                   initial: dict = None,
                   v_oa: float = 1.0,
                   **kwargs) -> 'Analyzed':
        """
        Monte Carlo uncertainty analysis by perturbing concentrations.

        Parameters
        ----------
        equilibrium : dict
        initial : dict, optional
        **kwargs
            ``noise``, ``n``, ``seed``, and fitting kwargs.

        Returns
        -------
        MonteCarlo
        """
        from .fitting import montecarlo as _mc
        return _mc(self, equilibrium, initial=initial,
                   v_oa=v_oa, **kwargs)



    def __repr__(self) -> str:
        parts = ', '.join(repr(m) for m in self._members)
        return f"freactions({parts})"
