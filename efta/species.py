"""
efta.species
============
Low-level utilities for parsing and normalising chemical species strings.

Species names in efta use a compact bracket notation for charge::

    Fe[3+]        # iron(III)
    OH[-]         # hydroxide
    SO4[2-]       # sulfate
    H[+]          # proton
    Fe(OH)3(s)    # solid ferric hydroxide
    H2A2(org)     # organic-phase di-2-ethylhexylphosphoric acid dimer
    e[-]          # electron (for redox reactions)

Fractional stoichiometric coefficients can be written using ``$(expr)``
syntax, e.g. ``$(1/3)Fe3O4(s)`` or ``Fe[$(3/2)+]`` for mixed-valence iron.

Public functions
----------------
species(s)
    Normalise a species string to its canonical form (e.g. ``'Fe^3+'``
    becomes ``'Fe[3+]'``).

formula(s)
    Bare chemical formula without charge brackets or phase tags
    (e.g. ``'Fe[3+]'`` → ``'Fe'``, ``'CaCO3(s)'`` → ``'CaCO3'``).

charge(s)
    Net ionic charge as a float (e.g. ``'Fe[3+]'`` → ``3.0``).

components(s)
    Element-count dict (e.g. ``'H2SO4'`` → ``{'H': 2, 'S': 1, 'O': 4}``).
    Result is cached via ``lru_cache`` — repeated calls are essentially free.

phase(s)
    Phase as a human-readable string: ``'aqueous'``, ``'solid'``,
    ``'organic'``, ``'liquid'``, or ``'gas'``.

construct(components, charge, phase)
    Build a species string from an element-count dict, charge, and phase.

species_std(s)
    Expand ``$(expr)`` placeholders in a species string to produce a clean
    display name suitable for labels and output.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Tuple

from .errors import SpeciesError

# ---------------------------------------------------------------------------
# Expression evaluation namespace for $(expr) notation
# ---------------------------------------------------------------------------

import math as _math
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

#: Safe namespace for evaluating ``$(expr)`` in species strings.
_EVAL_NS: dict = {
    '__builtins__': {},
    # all public math functions (log, log10, sqrt, exp, sin, cos, ...)
    **{k: getattr(_math, k) for k in dir(_math) if not k.startswith('_')},
    # Python builtins useful for chemistry
    'abs': abs, 'round': round, 'int': int, 'float': float, 'min': min, 'max': max,
}
if _HAS_NUMPY:
    _EVAL_NS.update({
        'np': _np,
        'log2': _np.log2, 'cbrt': _np.cbrt,
        'arcsin': _np.arcsin, 'arccos': _np.arccos, 'arctan': _np.arctan,
    })

def _find_dollar(s: str, start: int = 0):
    """
    Find the next ``$(expr)`` in *s* starting at *start*.
    Returns ``(begin, end, expr)`` or ``None`` if not found.
    *begin* is the index of '$', *end* is the index after the closing ')',
    *expr* is the expression string (without outer ``$()``).
    Handles arbitrarily nested parentheses.
    """
    i = s.find('$(', start)
    if i == -1:
        return None
    depth = 0
    j = i + 1    # points at '('
    while j < len(s):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
            if depth == 0:
                return (i, j + 1, s[i + 2: j])
        j += 1
    return None  # unmatched


def _iter_dollars(s: str):
    """Yield all ``(begin, end, expr)`` tuples for ``$(...)`` in *s``."""
    pos = 0
    while True:
        hit = _find_dollar(s, pos)
        if hit is None:
            break
        yield hit
        pos = hit[1]


def _eval_dollar(expr: str) -> float:
    """Evaluate a ``$(expr)`` expression string safely."""
    try:
        return float(eval(expr, _EVAL_NS, {}))
    except Exception as e:
        raise SpeciesError(
            f"Could not evaluate expression '$({expr})': {e}\n"
            f"Check for syntax errors or undefined functions.")


def _has_dollar(s: str) -> bool:
    """Return True if *s* contains any ``$(...)`` expression."""
    return '$(' in s


# Keep a simple regex for charge bracket detection only (no nested parens needed)
_DOLLAR_CHARGE_RE = re.compile(
    r'^\$\('
)

# ---------------------------------------------------------------------------
# Electron pseudo-species
# ---------------------------------------------------------------------------

#: Canonical name used internally to represent an electron.
ELECTRON = 'e[-]'

#: Regex matching all common ways a user might write an electron.
_ELECTRON_ALIASES_RE = re.compile(
    r'^e\s*(\[\s*-\s*\]|\^\s*-|-)?\s*$',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Phase detection helpers
# ---------------------------------------------------------------------------

def is_nonaqueous(name: str) -> bool:
    """Return True for solid '(s)' and pure-liquid '(l)' species."""
    n = name.strip().lower()
    return n.endswith('(s)') or n.endswith('(l)')


def is_organic(name: str) -> bool:
    """Return True for species tagged with '(org)'."""
    return name.strip().lower().endswith('(org)')


def is_solid(name: str) -> bool:
    """Return True for '(s)' species only."""
    return name.strip().lower().endswith('(s)')


def is_active(name: str) -> bool:
    """Return True for species that participate in concentration balance
    (i.e. not a pure solid or pure liquid)."""
    return not is_nonaqueous(name)


def is_electron(name: str) -> bool:
    """Return True if *name* is the special electron pseudo-species."""
    return name == ELECTRON

# ---------------------------------------------------------------------------
# Charge bracket helpers
# ---------------------------------------------------------------------------

# Charge bracket regex — supports integer and decimal (curly-brace) charges
_CHARGE_BRACKET_RE = re.compile(
    r'^\[(\+|-|[\d.]+[+-]|[+-][\d.]+)\]$'
)
# also matches [$(expr)+], [$(expr)-], [+$(expr)], [-$(expr)]
_CHARGE_BRACKET_DOLLAR_RE = re.compile(
    r'^\[([+-]?\$\([^)]*(?:\([^)]*\)[^)]*)*\)[+-]?)\]$'
)

_CARET_CHARGE_RE = re.compile(r'\^(\$\([^)]*(?:\([^)]*\)[^)]*)*\)[+-]|[+-]\$\([^)]*(?:\([^)]*\)[^)]*)*\)|[\d.]+[+-]|[+-][\d.]+|[+-])')


def _convert_caret_charges(s: str) -> str:
    """Convert caret-notation charges (e.g. Fe^3+, Fe^1.5+, Fe^$(2**2)+) to bracket notation."""
    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        if inner in ('+', '-'):
            return f'[{inner}]'
        if re.fullmatch(r'[+-][\d.]+', inner):
            sign, digits = inner[0], inner[1:]
            return f'[{digits}{sign}]'
        return f'[{inner}]'
    return _CARET_CHARGE_RE.sub(_replace, s)



def _find_close_bracket(s: str, open_pos: int) -> int:
    """Find the matching ']' for '[' at *open_pos*, skipping nested parens."""
    depth = 0
    j = open_pos + 1
    while j < len(s):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
        elif s[j] == ']' and depth == 0:
            return j
        j += 1
    return -1

def _is_charge_bracket(bracket: str) -> bool:
    """Return True if *bracket* is a valid charge descriptor like '[3+]' or '[$(expr)+]'."""
    if _CHARGE_BRACKET_RE.match(bracket):
        return True
    # check for $(expr) charge — contains $( somewhere inside []
    inner = bracket[1:-1] if bracket.startswith('[') and bracket.endswith(']') else ''
    if '$(' in inner:
        # validate: must end with + or - (after the expression)
        stripped = inner.rstrip('+-')
        hit = _find_dollar(stripped, 0)
        if hit and hit[0] == 0 and hit[1] == len(stripped):
            return True
        # or leading sign
        if inner.startswith('+') or inner.startswith('-'):
            stripped2 = inner[1:].rstrip('+-')
            hit2 = _find_dollar(stripped2, 0)
            if hit2 and hit2[0] == 0 and hit2[1] == len(stripped2):
                return True
    return False

# ---------------------------------------------------------------------------
# Public API: species, formula, charge, phase
# ---------------------------------------------------------------------------

def species(species_str: str) -> str:
    """
    Normalise a species string into a canonical form.

    Examples
    --------
    >>> species('Fe^3+')
    'Fe[3+]'
    >>> species('OH-')
    'OH[-]'
    >>> species('e-')
    'e[-]'
    """
    if _ELECTRON_ALIASES_RE.match(species_str.strip()):
        return ELECTRON

    s = species_str.replace(' ', '')
    s = _convert_caret_charges(s)

    def _transform_charge(inner: str) -> str:
        if inner in ('+', '-', '0'):
            return f'[{inner}]'
        if inner in ('1+', '+1'):
            return '[+]'
        if inner in ('1-', '-1'):
            return '[-]'
        if re.fullmatch(r'\d+\+', inner):
            return f'[{inner}]'
        if re.fullmatch(r'\d+\-', inner):
            return f'[{inner}]'
        m = re.fullmatch(r'\+(\d+)', inner)
        if m:
            return f'[{m.group(1)}+]'
        m = re.fullmatch(r'\-(\d+)', inner)
        if m:
            return f'[{m.group(1)}-]'
        return f'[{inner}]'

    out, i = [], 0
    while i < len(s):
        if s[i] == '[':
            j = _find_close_bracket(s, i)
            if j == -1:
                out.append(s[i]); i += 1
            else:
                bracket = s[i:j+1]
                if _is_charge_bracket(bracket):
                    out.append(_transform_charge(bracket[1:-1]))
                else:
                    out.append(bracket)
                i = j + 1
        else:
            out.append(s[i]); i += 1
    return ''.join(out)


def _strip_phase_tag(species_str: str) -> str:
    """Remove a trailing phase tag like '(aq)', '(s)', '(org)' etc."""
    return re.sub(r'\s*\((aq|l|s|g|org)\)\s*$', '', species_str.strip(),
                  flags=re.IGNORECASE)


def formula(species_str: str) -> str:
    """
    Return the bare chemical formula of a species – dropping charge brackets
    and phase tags.

    Examples
    --------
    >>> formula('Fe[3+]')
    'Fe'
    >>> formula('SO4[2-]')
    'SO4'
    """
    s = _strip_phase_tag(species(species_str))
    out, i = [], 0
    while i < len(s):
        if s[i] == '[':
            j = _find_close_bracket(s, i)
            if j == -1:
                out.append(s[i]); i += 1
                continue
            bracket = s[i:j+1]
            if _is_charge_bracket(bracket):
                pass          # drop charge bracket entirely
            else:
                out.append('(')
                out.append(bracket[1:-1])
                out.append(')')
            i = j + 1
        else:
            out.append(s[i]); i += 1
    return ''.join(out)


def phase(species_str: str) -> str:
    """
    Return the phase of a species as a human-readable string.

    Returns one of: 'aqueous', 'liquid', 'solid', 'gas', 'organic'.
    Defaults to 'aqueous' if no phase tag is present.
    """
    m = re.search(r'\((aq|l|s|g|org)\)\s*$', species_str.strip(), re.IGNORECASE)
    if m:
        tag = m.group(1).lower()
        return {'aq': 'aqueous', 'l': 'liquid', 's': 'solid',
                'g': 'gas', 'org': 'organic'}[tag]
    return 'aqueous'


def charges(species_str: str) -> List[float]:
    """
    Return a list of all charge values encoded in a species string.

    Supports integer charges (``[3+]``) and non-integer charges
    (``[{1.5}+]``).  Returns floats; integer charges are returned as
    whole-number floats (e.g. ``3.0``).

    Examples
    --------
    >>> charges('Fe[3+]')
    [3.0]
    >>> charges('Fe[{1.5}+]')
    [1.5]
    >>> charges('H2O')
    [0]
    """
    species_str = _convert_caret_charges(species_str)
    result: List[float] = []
    i = 0
    while i < len(species_str):
        if species_str[i] == '[':
            # find matching ] accounting for $(expr) nested parens
            depth_paren = 0
            j = i + 1
            while j < len(species_str):
                c = species_str[j]
                if c == '(':
                    depth_paren += 1
                elif c == ')':
                    depth_paren -= 1
                elif c == ']' and depth_paren == 0:
                    break
                j += 1
            if j >= len(species_str):
                i += 1
                continue
            bracket = species_str[i:j+1]
            if _is_charge_bracket(bracket):
                inner = bracket[1:-1]
                if inner == '+':
                    result.append(1.0)
                elif inner == '-':
                    result.append(-1.0)
                elif '$(' in inner:
                    # $(expr)[+-] or [+-]$(expr)
                    # trailing sign applies to expression value
                    # [$(expr)+] → +eval(expr), [$(expr)-] → -eval(expr)
                    # [$(-3)+] → +(-3) = -3, [$(-3)-] → -(-3) = +3
                    if inner.endswith('+'):
                        trail_sign, expr_part = 1.0, inner[:-1]
                    elif inner.endswith('-'):
                        trail_sign, expr_part = -1.0, inner[:-1]
                    else:
                        trail_sign, expr_part = 1.0, inner
                    # leading sign before $(
                    if expr_part.startswith('+'):
                        lead_sign, expr_part = 1.0, expr_part[1:]
                    elif expr_part.startswith('-'):
                        lead_sign, expr_part = -1.0, expr_part[1:]
                    else:
                        lead_sign = 1.0
                    hit = _find_dollar(expr_part, 0)
                    if hit:
                        val = _eval_dollar(hit[2])
                        result.append(lead_sign * trail_sign * val)
                    else:
                        result.append(lead_sign * trail_sign)
                else:
                    sign   = 1.0 if '+' in inner else -1.0
                    digits = ''.join(c for c in inner if c.isdigit() or c == '.')
                    result.append(float(digits) * sign if digits else sign)
            i = j + 1
        else:
            i += 1
    return result if result else [0]


def charge(species_str: str) -> float:
    """Return the total (net) charge of a species as a float."""
    return sum(charges(species_str))

# ---------------------------------------------------------------------------
# Formula parsing: components (element composition)
# ---------------------------------------------------------------------------

def _add_implicit_ones(string: str) -> str:
    """Insert implicit '1' counts where the formula grammar expects them.

    ``$(expr)`` spans are treated as atomic count tokens — no implicit
    ones are inserted inside them.
    """
    # Walk character by character; when we hit $(, skip the whole expression.
    string = string + '#'
    result = []
    i = 0
    n = len(string) - 1   # last char is '#' sentinel
    while i < n:
        ch = string[i]
        result.append(ch)
        # if entering a $(expr) span, copy verbatim to closing )
        if ch == '$' and i + 1 < n and string[i + 1] == '(':
            depth = 0
            i += 1
            while i <= n:
                c = string[i]
                result.append(c)
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            # after the $(...), check next char for implicit-1 rule
            nxt = string[i] if i <= n else '#'
            if nxt in (')', '(', '.') or nxt.isupper() or nxt == '#':
                pass  # $(expr) is itself the count; no implicit 1
            continue
        if ch.isdigit() or ch == '(':
            i += 1
            continue
        nxt = string[i + 1]
        if nxt in (')', '(', '.') or nxt.isupper() or nxt == '#':
            result.append('1')
        i += 1
    return ''.join(result)


def _split_by_dot(string: str) -> List[Tuple[str, float]]:
    """Split a formula like '3CuSO4.5H2O' into [(CuSO4, 3), (H2O, 5)].
    
    Dots inside ``$(expr)`` are not treated as subspecies separators.
    Multipliers may be ``$(expr)`` expressions.
    """
    if string.count('(') != string.count(')'):
        raise SpeciesError(f"Unbalanced parentheses in formula: {string!r}\nCheck for mismatched ( ) in the species name.")
    brack, dollar, chars = 0, False, []
    i = 0
    while i < len(string):
        ch = string[i]
        if ch == '$' and i + 1 < len(string) and string[i+1] == '(':
            # dollar expression — copy verbatim, no dot splitting inside
            j = string.find(')', i + 2)
            # handle nested parens
            depth = 1
            k = i + 2
            while k < len(string) and depth:
                if string[k] == '(': depth += 1
                elif string[k] == ')': depth -= 1
                k += 1
            chars.extend(list(string[i:k]))
            i = k
        elif ch == '(':
            brack += 1; chars.append(ch); i += 1
        elif ch == ')':
            brack -= 1; chars.append(ch); i += 1
        elif ch == '.' and brack == 0:
            chars.append(','); i += 1
        else:
            chars.append(ch); i += 1
    parts = ''.join(chars).split(',')
    result = []
    for part in parts:
        # multiplier may be plain integer or $(expr)
        dm = re.match(r'^(\d+)(.*)', part)
        em = _find_dollar(part, 0)
        if em and em[0] == 0:
            mult = _eval_dollar(em[2])
            result.append((part[em[1]:], mult))
        elif dm:
            result.append((dm.group(2), int(dm.group(1))))
        else:
            result.append((part, 1))
    return result


def _atom_split(fml: str, multiplier: float = 1) -> List[Tuple[str, float]]:
    """Recursively tokenise a formula string into (element, count) pairs."""
    fml = _add_implicit_ones(fml)
    pairs, i, n = [], 0, len(fml)
    while i < n:
        ch = fml[i]
        if ch == '(':
            depth, j = 1, i + 1
            while j < n and depth:
                if fml[j] == '(':   depth += 1
                elif fml[j] == ')': depth -= 1
                j += 1
            inner = fml[i+1:j-1]
            k = j
            while k < n and fml[k].isdigit():
                k += 1
            inner_mult = int(fml[j:k]) if k > j else 1
            pairs.extend(_atom_split(inner, multiplier * inner_mult))
            i = k
        elif ch.isupper():
            elem = ch; i += 1
            while i < n and fml[i].islower():
                elem += fml[i]; i += 1
            if i < n and fml[i] == '$' and i + 1 < n and fml[i+1] == '(':
                # $(expr) notation
                dm = _find_dollar(fml, i)
                if dm is None or dm[0] != i:
                    raise SpeciesError(f"Unmatched '$(' in formula: {fml!r}")
                count = _eval_dollar(dm[2])
                i = dm[1]
            else:
                k = i
                while k < n and (fml[k].isdigit() or fml[k] == '.'):
                    k += 1
                if k > i:
                    count = float(fml[i:k])
                    i = k
                else:
                    count = 1
            pairs.append((elem, count * multiplier))
        else:
            i += 1
    return pairs


@lru_cache(maxsize=4096)
def components(species_str: str) -> Dict[str, float]:
    """
    Return a dict mapping element symbol → atom count for a species.

    The result is cached; calling with the same string is O(1).

    Examples
    --------
    >>> components('H2SO4')
    {'H': 2, 'S': 1, 'O': 4}
    >>> components('Fe[3+]')
    {'Fe': 1}
    """
    bare = formula(species_str)
    if not bare:
        return {}
    result: Dict[str, float] = {}
    for fragment, frag_mult in _split_by_dot(bare):
        for elem, cnt in _atom_split(fragment, frag_mult):
            if elem:
                result[elem] = result.get(elem, 0) + cnt
    return result

def construct(components: Dict[str, float],
              charge: float = 0,
              phase: str = None) -> str:
    """
    Build an efta species string from a component dict, optional charge,
    and optional phase.

    Elements are ordered by the Hill system: C first, H second, then all
    remaining elements alphabetically.  Integer counts of 1 are omitted;
    non-integer counts are written with up to 4 significant figures.

    Parameters
    ----------
    components : dict
        Mapping of element symbol → atom count, e.g. ``{'C': 1, 'H': 4}``.
        Counts must be positive.
    charge : int or float, optional
        Net charge of the species (default 0).  Positive = cation,
        negative = anion.  Non-integer charges are encoded as ``[{n}+]``.
    phase : str or None, optional
        Phase tag appended to the formula.  Accepted values:
        ``None`` (aqueous, no tag), ``'org'``, ``'s'``, ``'l'``, ``'g'``.

    Returns
    -------
    str
        Species string in efta notation, e.g. ``'CH4'``, ``'SO4[2-]'``,
        ``'CaCO3(s)'``, ``'HA(org)'``.

    Examples
    --------
    >>> construct({'C': 1, 'H': 4})
    'CH4'
    >>> construct({'S': 1, 'O': 4}, charge=-2)
    'SO4[2-]'
    >>> construct({'Ca': 1, 'C': 1, 'O': 3}, phase='s')
    'CaCO3(s)'
    >>> construct({'Fe': 1}, charge=3)
    'Fe[3+]'
    >>> construct({'Fe': 1}, charge=1.5)
    'Fe[1.5+]'
    >>> construct({'H': 1}, charge=1)
    'H[+]'
    >>> construct({'O': 1, 'H': 1}, charge=-1)
    'OH[-]'
    >>> construct({'H': 1, 'A': 1}, phase='org')
    'HA(org)'
    """
    from .errors import InputError

    if not components:
        raise InputError("components dict must not be empty.")

    bad = {el: n for el, n in components.items() if n <= 0}
    if bad:
        raise InputError(
            f"All atom counts must be positive, got: {bad}.")

    valid_phases = {None, 'org', 's', 'l', 'g'}
    if phase not in valid_phases:
        raise InputError(
            f"phase must be one of {sorted(str(p) for p in valid_phases if p)!r} "
            f"or None, got {phase!r}.")

    # Hill ordering: C first, H second, then rest alphabetically
    # Use exact symbol match — 'Ca', 'Co' etc. are NOT carbon/hydrogen
    def _hill_key(el):
        if el == 'C':  return (0, 'C')
        if el == 'H':  return (1, 'H')
        return (2, el)

    def _fmt_count(n):
        if n == int(n) and n == 1:
            return ''
        if n == int(n):
            return str(int(n))
        # non-integer: dollar-paren notation $(n) parsed by _atom_split
        return '$(' + f'{n:.6g}' + ')'

    formula_str = ''.join(
        el + _fmt_count(n)
        for el, n in sorted(components.items(), key=lambda x: _hill_key(x[0]))
    )

    # charge bracket
    charge = float(charge)

    def _fmt_charge_val(z: float) -> str:
        """Format charge magnitude: whole number → plain digits, else decimal."""
        if z == int(z):
            return str(int(z))
        return f'{z:.6g}'

    if charge == 0.0:
        charge_str = ''
    elif charge == 1.0:
        charge_str = '[+]'
    elif charge == -1.0:
        charge_str = '[-]'
    elif charge > 0:
        charge_str = f'[{_fmt_charge_val(charge)}+]'
    else:
        charge_str = f'[{_fmt_charge_val(abs(charge))}-]'

    # phase tag
    phase_str = f'({phase})' if phase else ''

    return formula_str + charge_str + phase_str


def species_std(species_str: str) -> str:
    """
    Convert a species string to standard display form by evaluating
    ``$(expr)`` placeholders according to context:

    **In species name (atom count):**
    - Integer value → plain digits: ``$(2)`` → ``2``
    - Value = 1 → omit entirely: ``$(1)`` → ``""``
    - Value = 0 → drop preceding element + count: ``AB$(0)C`` → ``AC``
    - Non-integer → keep ``$(n.nn)`` notation unchanged

    **In charge bracket ``[...]``:**
    - Expand value to plain number with sign
    - ``[$(2)+]`` → ``[2+]``,  ``[$(1)+]`` → ``[+]``
    - ``[$(0.5)+]`` → ``[0.50+]``
    - Negative value with ``+`` sign → flip: ``[$(-2)+]`` → ``[2-]``
    - ``[$(-1)+]`` → ``[-]``

    Species with no ``$(...)`` are returned unchanged.

    Examples
    --------
    >>> species_std("LaCl$(2)A$(1)(HA)$(3)(org)")
    'LaCl2A(HA)3(org)'
    >>> species_std("AB[$(2)+]")
    'AB[2+]'
    >>> species_std("AB[$(-1)+]")
    'AB[-]'
    >>> species_std("A$(1.5)B")
    'A$(1.5)B'
    >>> species_std("H2O")
    'H2O'
    """
    if '$(' not in species_str:
        return species_str

    result  = []
    s       = species_str
    pos     = 0
    n       = len(s)
    in_bracket = False   # inside [...]

    while pos < n:
        ch = s[pos]

        # track charge bracket depth
        if ch == '[':
            in_bracket = True
            result.append(ch)
            pos += 1
            continue

        if ch == ']':
            in_bracket = False
            result.append(ch)
            pos += 1
            continue

        # dollar expression
        if ch == '$' and pos + 1 < n and s[pos + 1] == '(':
            hit = _find_dollar(s, pos)
            if hit is None:
                result.append(ch)
                pos += 1
                continue

            val = _eval_dollar(hit[2])
            end = hit[1]   # position after closing )

            if in_bracket:
                # ── charge context ─────────────────────────────────────────
                # peek at sign character after closing )
                sign_ch = s[end] if end < n else '+'
                end_after_sign = end + 1 if end < n and s[end] in '+-' else end

                # determine effective charge value
                if sign_ch == '+':
                    eff = val          # [$(v)+] → charge = +v
                elif sign_ch == '-':
                    eff = -val         # [$(v)-] → charge = -v
                else:
                    eff = val
                    end_after_sign = end

                abs_eff = abs(eff)
                out_sign = '+' if eff >= 0 else '-'

                if abs_eff == 1.0:
                    token = out_sign          # [+] or [-]
                elif abs_eff == float(int(abs_eff)):
                    token = f'{int(abs_eff)}{out_sign}'
                else:
                    token = f'{abs_eff:.2f}{out_sign}'

                # remove what we already appended for '[' and replace
                # with the fully formatted bracket if sign is next
                result.append(token)
                pos = end_after_sign

            else:
                # ── atom count context ─────────────────────────────────────
                if val == 0.0:
                    # drop preceding element (uppercase + any following lowercase)
                    while result and result[-1].islower():
                        result.pop()
                    if result and result[-1].isupper():
                        result.pop()
                    pos = end
                elif val == 1.0:
                    # omit count — element already appended, nothing to add
                    pos = end
                elif val == float(int(val)):
                    result.append(str(int(val)))
                    pos = end
                else:
                    # non-integer: keep $(n.nn) notation
                    result.append(f'$({val:.2f})')
                    pos = end

        else:
            result.append(ch)
            pos += 1

    return ''.join(result)
