"""
efta.mixture
============
The :class:`mixture` class holds an ordered list of :class:`~efta.solution`
objects and provides a unified view over them.

Construction
------------
::

    from efta import mixture, solution
    m = mixture(sol1, sol2, sol3)

Repr / print
------------
Each solution is shown with its index, separated by ``======``.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Union

from .solution import solution as _solution
from .errors   import InputError

__all__ = ['mixture']

_SEP = "=" * 38


class mixture:
    """
    An ordered collection of :class:`~efta.solution` objects.

    Parameters
    ----------
    *solutions : solution
        One or more solution instances.

    Examples
    --------
    >>> m = mixture(sol1, sol2, sol3)
    >>> m[0]              # first solution
    >>> m[1:3]            # slice → new mixture
    >>> len(m)
    >>> for sol in m: ...
    >>> m << sys          # equilibrate all → new mixture
    >>> m * 2             # scale all concentrations → new mixture
    >>> m.mix()           # combine all into one solution
    >>> m.pH              # list of pH values
    >>> m.total('Fe')     # total iron across all solutions (mol/L per solution)
    """

    def __init__(self, *solutions: _solution):
        for i, sol in enumerate(solutions):
            if not isinstance(sol, _solution):
                raise InputError(
                    f"All arguments must be solution instances; "
                    f"argument {i} is {type(sol).__name__!r}.")
        self._solutions: List[_solution] = list(solutions)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._solutions)

    def __getitem__(self, idx) -> Union['mixture', _solution]:
        """Index or slice.  A slice returns a new :class:`mixture`."""
        if isinstance(idx, slice):
            return mixture(*self._solutions[idx])
        return self._solutions[idx]

    def __iter__(self) -> Iterator[_solution]:
        return iter(self._solutions)

    def __contains__(self, sol: _solution) -> bool:
        return sol in self._solutions

    def without(self, indices) -> 'mixture':
        """
        Return a new :class:`mixture` excluding the solutions at *indices*.

        Parameters
        ----------
        indices : int or list of int
            Zero-based index or list of indices to exclude.

        Examples
        --------
        >>> m2 = m.without(0)        # drop first solution
        >>> m2 = m.without([1, 3])   # drop solutions 1 and 3
        """
        if isinstance(indices, int):
            indices = [indices]
        exclude = set(indices)
        kept = [s for i, s in enumerate(self._solutions) if i not in exclude]
        if not kept:
            raise InputError(
                f"without({list(indices)}) would remove all solutions.")
        return mixture(*kept)

    def append(self, sol: _solution) -> 'mixture':
        """Append a solution and return *self*."""
        if not isinstance(sol, _solution):
            raise InputError(
                f"Expected a solution instance, got {type(sol).__name__!r}.")
        self._solutions.append(sol)
        return self

    def __iadd__(self, sol: _solution) -> 'mixture':
        """``m += sol`` appends a solution in-place."""
        return self.append(sol)

    # ------------------------------------------------------------------
    # Bulk operations mirroring solution
    # ------------------------------------------------------------------

    def __lshift__(self, rxn_sys) -> 'mixture':
        """
        Equilibrate all solutions and return a new :class:`mixture`.

        ``m << sys``  →  ``mixture(sol << sys for sol in m)``

        Examples
        --------
        >>> eq_mixture = m << sys
        >>> eq_mixture = m << r1
        """
        return mixture(*[sol << rxn_sys for sol in self._solutions])

    def __rshift__(self, other) -> 'mixture':
        """
        ``sys >> m`` — same as ``m << sys``.
        Defined on :class:`mixture` to handle the right-hand case gracefully.
        """
        return self.__lshift__(other)

    def __mul__(self, factor: float) -> 'mixture':
        """
        Return a new mixture with all concentrations scaled by *factor*.

        Examples
        --------
        >>> m2 = m * 2
        """
        return mixture(*[sol * factor for sol in self._solutions])

    def __rmul__(self, factor: float) -> 'mixture':
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> 'mixture':
        """
        Return a new mixture with all concentrations divided by *factor*.

        Examples
        --------
        >>> m2 = m / 2
        """
        return mixture(*[sol / factor for sol in self._solutions])

    def strip(self, phase: str) -> 'mixture':
        """
        Return a new mixture with the given phase stripped from all solutions.

        Parameters
        ----------
        phase : str
            One of ``'organic'``, ``'aqueous'``, ``'solid'``, ``'liquid'``,
            ``'gas'``.

        Examples
        --------
        >>> aq_only = m.strip('organic')
        """
        return mixture(*[sol.strip(phase) for sol in self._solutions])

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def mix(self) -> _solution:
        """
        Combine all solutions into a single :class:`~efta.solution` by
        summing moles and volumes (same as chaining ``+``).

        Returns
        -------
        solution

        Examples
        --------
        >>> combined = m.mix()
        >>> combined = m.mix().equilibrate(sys)
        """
        if not self._solutions:
            raise InputError("Cannot mix an empty mixture.")
        result = self._solutions[0]
        for sol in self._solutions[1:]:
            result = result + sol
        return result

    def total(self, query: str) -> List[float]:
        """
        Total concentration of a species or element in each solution (mol/L).

        Delegates to :meth:`~efta.solution.solution.total` for each member.
        Two modes of *query*:

        1. **Exact species** — phase-tagged name like ``'H2A2(org)'``.
        2. **Element symbol** — stoichiometry-weighted sum, e.g. ``'Fe'``.

        Parameters
        ----------
        query : str
            Species name or element symbol.

        Returns
        -------
        list of float
            One value per solution in the mixture.

        Examples
        --------
        >>> m.total('La')         # [c_La_in_sol0, c_La_in_sol1, ...]
        >>> m.total('H2A2(org)')  # exact species concentration per solution
        """
        return [sol.total(query) for sol in self._solutions]

    # ------------------------------------------------------------------
    # Inspection properties
    # ------------------------------------------------------------------

    @property
    def pH(self) -> List[Optional[float]]:
        """List of pH values, one per solution (None if H+ absent)."""
        return [sol.pH for sol in self._solutions]

    @property
    def ionic_strength(self) -> List[float]:
        """List of ionic strength values (mol/L), one per solution."""
        return [sol.ionic_strength for sol in self._solutions]

    @property
    def I(self) -> List[float]:
        """Alias for :attr:`ionic_strength`."""
        return self.ionic_strength

    @property
    def volume(self) -> List[float]:
        """List of volumes (L), one per solution."""
        return [sol.volume for sol in self._solutions]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def _format(self) -> str:
        if not self._solutions:
            return "mixture(empty)"
        blocks = []
        for i, sol in enumerate(self._solutions):
            blocks.append(f"[{i}]\n{sol}")
        return f"\n{_SEP}\n".join(blocks)

    def __repr__(self) -> str:
        return self._format()

    def __str__(self) -> str:
        return self._format()
