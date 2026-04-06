"""
efta.solventextraction.units
============================
Process unit helpers for building solvent-extraction flowsheets.

Classes
-------
:class:`separator`
    Split a solution into fractional volume cuts and return a
    :class:`~efta.mixture.mixture`.  Enables reflux-style circuit design
    by directing fractions of an outlet stream back into earlier stages.

Examples
--------
>>> from efta.solventextraction.units import separator
>>>
>>> # Equal thirds
>>> s = separator(1, 1, 1)
>>> m = s(solution)          # mixture of 3 × V/3 slices
>>>
>>> # Reflux: return 1 part, pass 2 parts forward
>>> splitter = separator(1, 2)
>>> reflux, forward = splitter(extract)[0], splitter(extract)[1]
>>>
>>> # Asymmetric cut
>>> s = separator(1, 2, 1)
>>> m = s(feed)              # volumes: V/4, V/2, V/4
>>> m[0].volume, m[1].volume, m[2].volume
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

from ..solution import solution as _solution
from ..mixture  import mixture  as _mixture
from ..errors   import InputError

__all__ = ['splitter']


class splitter:
    """
    Volumetric flow splitter.

    Splits a :class:`~efta.solution.solution` into *n* fractions according
    to relative weights supplied at construction, then returns a
    :class:`~efta.mixture.mixture` whose members are the resulting
    sub-solutions in the same order.

    Each fraction preserves the original concentration (mol/L) — only the
    volume (and therefore the moles) changes.  This reflects a physical
    stream splitter: the composition of every cut is identical to the feed;
    only the flow rate differs.

    Parameters
    ----------
    *fractions : int or float
        Relative weights of each cut.  Must all be positive.  They do not
        need to sum to any particular value — the class normalises them
        internally.

        Examples:
          ``separator(1, 1)``      → two equal halves
          ``separator(1, 2, 1)``   → quarter, half, quarter
          ``separator(3, 1)``      → 75 % and 25 %
          ``separator(1)``         → trivial single cut (returns mixture of 1)

    Attributes
    ----------
    fractions : tuple of float
        The raw weights as supplied.
    weights : tuple of float
        Normalised fractional volumes (sum = 1).
    n : int
        Number of cuts.

    Examples
    --------
    Basic split:

    >>> s = separator(1, 2, 1)
    >>> m = s(sol)
    >>> m[0].volume   # V/4
    >>> m[1].volume   # V/2
    >>> m[2].volume   # V/4

    Reflux design — send 1/3 of the extract back, pass 2/3 forward:

    >>> splitter = separator(1, 2)
    >>> m = splitter(extract)
    >>> reflux  = m[0]   # 1/3 of extract → back to stage
    >>> forward = m[1]   # 2/3 of extract → next unit

    Callable multiple times with different solutions of any volume:

    >>> m1 = s(feed_a)
    >>> m2 = s(feed_b)   # same proportions, different total volume
    """

    def __init__(self, *fractions: Union[int, float]):
        if not fractions:
            raise InputError(
                "separator requires at least one fraction weight.")

        parsed: list[float] = []
        for i, f in enumerate(fractions):
            f = float(f)
            if f <= 0:
                raise InputError(
                    f"separator: fraction[{i}] must be positive, got {f}.")
            parsed.append(f)

        total = sum(parsed)
        self.fractions: Tuple[float, ...] = tuple(parsed)
        self.weights:   Tuple[float, ...] = tuple(f / total for f in parsed)
        self.n:         int               = len(parsed)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def __call__(self, sol: _solution) -> _mixture:
        """
        Split *sol* into fractional cuts and return a
        :class:`~efta.mixture.mixture`.

        Each cut has the same concentration as *sol* but a volume equal to
        ``weight[i] * sol.volume``.  Moles in cut *i* are therefore
        ``weight[i] * total_moles``.

        Parameters
        ----------
        sol : solution
            The solution to split.  Its volume is divided proportionally;
            concentrations are unchanged.

        Returns
        -------
        mixture
            Members are in the same order as the fractions given to the
            constructor.  ``m[0]`` corresponds to ``fractions[0]``, etc.

        Raises
        ------
        InputError
            If *sol* is not a :class:`~efta.solution.solution`.

        Examples
        --------
        >>> s = separator(1, 2, 1)
        >>> m = s(sol)
        >>> len(m)          # 3
        >>> m[0].volume     # sol.volume * 0.25
        >>> m[1].volume     # sol.volume * 0.50
        >>> m[2].volume     # sol.volume * 0.25
        >>> m[0]['H[+]']    # same concentration as sol['H[+]']
        """
        if not isinstance(sol, _solution):
            raise InputError(
                f"separator expects a solution, got {type(sol).__name__!r}.")

        v_total = sol.volume
        cuts = [sol(w * v_total) for w in self.weights]
        return _mixture(*cuts)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def split(self, sol: _solution) -> _mixture:
        """Alias for ``self(sol)``."""
        return self(sol)

    def __repr__(self) -> str:
        frac_str = ', '.join(str(f) for f in self.fractions)
        wt_str   = ', '.join(f'{w:.4g}' for w in self.weights)
        return f"separator({frac_str})  →  [{wt_str}]"

    def __str__(self) -> str:
        lines = [f"separator  ({self.n} cuts)"]
        for i, (f, w) in enumerate(zip(self.fractions, self.weights)):
            lines.append(f"  [{i}]  weight={f:.4g}  →  {w * 100:.2f}% of feed volume")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> float:
        """Return the normalised weight at index *i*."""
        return self.weights[i]

    def __iter__(self):
        return iter(self.weights)
