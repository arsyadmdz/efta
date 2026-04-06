"""
efta.solventextraction.sx
=========================
Single-stage solvent extraction (batch equilibrium).

The :class:`sx` class accepts any number of :class:`~efta.solution.solution`
objects as inlets (regardless of phase), runs equilibrium via :meth:`run`,
and exposes the two outlet phases through :attr:`outlets`.

Module-level functions
----------------------
:func:`multiphase`          — blend solutions then split by phase.
:func:`distribution_coef`   — D value from a ``mixture(organic, aqueous)``.
:func:`separation_factor`   — β from a ``mixture(organic, aqueous)``.

Parameters
----------
reactions : reaction | reactions
    Equilibrium reaction(s) describing the extraction chemistry.
*solutions : solution
    One or more feed solutions (any mix of aqueous and organic phase).

Examples
--------
>>> from efta import reaction, solution
>>> from efta.solventextraction import sx
>>> from efta.solventextraction.sx import distribution_coef, separation_factor
>>>
>>> rxn     = reaction('LaCl[2+] + 3H2A2(org) = LaClA2(HA)4(org) + 2H[+]', 10.6)
>>> feed    = solution({'LaCl[2+]': 0.003, 'H[+]': 0.3}, volume=1.0)
>>> organic = solution({'H2A2(org)': 0.25}, volume=1.0)
>>>
>>> stage = sx(rxn, feed, organic)
>>> stage.run()                           # default efficiency=1.0
>>> extract, raffinate = stage.outlets[0], stage.outlets[1]
>>>
>>> # Distribution metrics as sx methods (use stage.outlets internally)
>>> D    = stage.distribution_coef('La')
>>> beta = stage.separation_factor('La', 'Nd')
>>>
>>> # Or as standalone functions taking any mixture(organic, aqueous):
>>> D    = distribution_coef('La',       stage.outlets)
>>> beta = separation_factor('La', 'Nd', stage.outlets)
>>>
>>> # outlets before run() = phase separation of blended inlets only
>>> stage2 = sx(rxn, feed, organic)
>>> pre_org, pre_aq = stage2.outlets[0], stage2.outlets[1]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from ..solution  import solution  as _solution
from ..mixture   import mixture   as _mixture
from ..reactions import reactions as _reactions_cls
from ..reaction  import reaction  as _reaction_cls
from ..species   import is_organic as _is_org
from ..errors    import InputError


__all__ = ['sx', 'multiphase', 'distribution_coef', 'separation_factor']


def multiphase(*solutions: _solution) -> _mixture:
    """
    Blend multiple solutions then separate into organic and aqueous phases.

    Combines all *solutions* by summing moles and volumes (the same as
    successive ``+`` operations), then calls :meth:`~efta.solution.solution.separate`
    on the blended result to split by phase.

    This is a convenience function for constructing the initial
    ``mixture(organic, aqueous)`` from any set of input solutions before
    passing them into an :class:`sx` stage.

    Parameters
    ----------
    *solutions : solution
        Two or more solution objects to blend.

    Returns
    -------
    mixture
        ``mixture(organic_solution, aqueous_solution)``
        Index 0 = organic outlet, index 1 = aqueous outlet.

    Examples
    --------
    >>> from efta.solventextraction.sx import multiphase
    >>> m = multiphase(feed_aq, scrub_aq, organic)
    >>> org, aq = m[0], m[1]
    """
    if not solutions:
        raise InputError("multiphase() requires at least one solution.")
    for i, sol in enumerate(solutions):
        if not isinstance(sol, _solution):
            raise InputError(
                f"multiphase() argument {i}: expected a solution, "
                f"got {type(sol).__name__!r}.")
    blended = _blend_solutions(list(solutions))
    return blended.separate()


def _blend_solutions(solutions: List[_solution]) -> _solution:
    """
    Combine a list of solutions into one by summing moles and volumes.
    """
    if len(solutions) == 1:
        return solutions[0]
    result = solutions[0]
    for sol in solutions[1:]:
        result = result + sol
    return result


class sx:
    """
    Single-stage solvent extraction unit.

    Inlets are a mutable ordered list of :class:`~efta.solution.solution`
    objects.  Any phase mix is accepted.  :meth:`run` blends all inlets,
    solves the equilibrium, and updates :attr:`outlets`.

    Before :meth:`run` is called, :attr:`outlets` reflects the raw phase
    separation of the blended inlets (no reaction).

    Attributes
    ----------
    reactions : reactions
    inlets : mixture
        Current input solutions (mutable via :meth:`input` / :meth:`stop`).
    outlets : mixture
        ``mixture(extract, raffinate)`` — organic first, aqueous second.
        Updated by :meth:`run`.
    ran : bool
        True after :meth:`run` has been called at least once.
    efficiency : float or None
        Efficiency used in the last :meth:`run` call; None before first run.
    """

    def __init__(self,
                 reactions: Union[_reaction_cls, _reactions_cls],
                 *solutions: _solution):

        # ── normalise reactions ───────────────────────────────────────────
        if isinstance(reactions, _reaction_cls):
            reactions = _reactions_cls(reactions)
        elif not isinstance(reactions, _reactions_cls):
            raise InputError("reactions must be a reaction or reactions object.")
        self.reactions = reactions

        # ── validate and store inlets ─────────────────────────────────────
        if not solutions:
            raise InputError(
                "At least one solution must be provided as an inlet.")
        for i, sol in enumerate(solutions):
            if not isinstance(sol, _solution):
                raise InputError(
                    f"Inlet {i}: expected a solution object, "
                    f"got {type(sol).__name__!r}.")

        self._inlets: List[_solution] = list(solutions)
        self.ran:       bool          = False
        self.efficiency: Optional[float] = None
        self._managed:  bool          = False   # True when owned by a multistage

        # Compute initial outlets (phase separation only, no reaction)
        self._outlets: _mixture = self._phase_separate()

    # ------------------------------------------------------------------
    # Inlet management
    # ------------------------------------------------------------------

    @property
    def inlets(self) -> _mixture:
        """
        Current inlet solutions as a :class:`~efta.mixture.mixture`.

        Returns
        -------
        mixture
            Live view of inlets; index matches :meth:`stop` index.
        """
        return _mixture(*self._inlets)

    def input(self, sol: _solution) -> 'sx':
        """
        Add *sol* to the inlets and update :attr:`outlets` (phase-separated).

        Parameters
        ----------
        sol : solution

        Returns
        -------
        sx
            *self*, for chaining.

        Examples
        --------
        >>> stage.input(scrub_solution)
        """
        if self._managed:
            raise InputError(
                "This sx stage is part of a multistage system.  "
                "Use multistage.input(i, solution) instead.")
        if not isinstance(sol, _solution):
            raise InputError(
                f"Expected a solution object, got {type(sol).__name__!r}.")
        self._inlets.append(sol)
        # Recompute outlets if not yet run, keep run results if already run
        if not self.ran:
            self._outlets = self._phase_separate()
        return self

    def stop(self, i: int) -> 'sx':
        """
        Remove the inlet at index *i* and shift subsequent indices down.

        Parameters
        ----------
        i : int
            Zero-based index into the current inlets.

        Returns
        -------
        sx
            *self*, for chaining.

        Examples
        --------
        >>> stage.stop(0)   # remove first inlet
        >>> stage.stop(-1)  # remove last inlet
        """
        if self._managed:
            raise InputError(
                "This sx stage is part of a multistage system.  "
                "Use multistage.stop(i, inlet_index) instead.")
        n = len(self._inlets)
        if not -n <= i < n:
            raise InputError(
                f"Inlet index {i} out of range for {n} inlets.")
        self._inlets.pop(i)
        if not self._inlets:
            raise InputError(
                "Removing inlet would leave sx with no inlets.")
        if not self.ran:
            self._outlets = self._phase_separate()
        return self

    def empty(self) -> 'sx':
        """
        Clear all inlets (sets the inlet list to ``[]``).

        .. note::
            This leaves the stage with no inlets; calling :meth:`run`
            afterwards will raise an error until inlets are restored via
            :meth:`input`.

        Returns
        -------
        sx
            *self*, for chaining.

        Examples
        --------
        >>> stage.empty()
        """
        if self._managed:
            raise InputError(
                "This sx stage is part of a multistage system.  "
                "Use multistage.empty(i) instead.")
        self._inlets = []
        return self

    # ------------------------------------------------------------------
    # Outlets
    # ------------------------------------------------------------------

    @property
    def outlets(self) -> _mixture:
        """
        ``mixture(extract, raffinate)`` — organic outlet at index 0,
        aqueous outlet at index 1.

        Before :meth:`run`, returns the phase separation of the blended
        inlets (no reaction applied).  After :meth:`run`, returns the
        equilibrium result.
        """
        return self._outlets

    # ------------------------------------------------------------------
    # Phase separation helper (pre-run outlets)
    # ------------------------------------------------------------------

    def _phase_separate(self) -> _mixture:
        """
        Blend all inlets and split by phase without running any reaction.
        Returns mixture(organic_sol, aqueous_sol).
        Delegates to solution.separate() which uses v_oa from the blended solution.
        """
        blended = _blend_solutions(self._inlets)
        return blended.separate()

    # ------------------------------------------------------------------
    # Core solver
    # ------------------------------------------------------------------

    def run(self, efficiency: float = 1.0) -> 'sx':
        """
        Blend all inlets, solve the extraction equilibrium, and update
        :attr:`outlets`.  Returns *self* so calls can be chained.

        Parameters
        ----------
        efficiency : float, optional
            Extraction efficiency in [0, 1].  At ``efficiency < 1`` the
            computed extent of reaction is scaled back toward the initial
            composition, modelling non-ideal contact.  Default ``1.0``.

        Returns
        -------
        sx
            *self* (mutated: :attr:`outlets` updated, :attr:`ran` set True).

        Examples
        --------
        >>> stage.run()
        >>> stage.run(efficiency=0.85)
        >>> extract = stage.run(0.9).outlets[0]
        """
        if not 0.0 <= float(efficiency) <= 1.0:
            raise InputError(
                f"efficiency must be in [0, 1], got {efficiency}.")
        self.efficiency = float(efficiency)

        # ── blend all inlets ──────────────────────────────────────────────
        blended = _blend_solutions(self._inlets)

        # Determine v_oa from blended solution (or fall back to 1.0)
        v_oa = blended.v_oa
        if v_oa is None:
            # All inlets are same phase — still need a ratio for the solver
            has_org = any(_is_org(sp) for sp in blended)
            v_oa = 1.0      # arbitrary; solver result will reflect composition

        v_total = blended.volume
        v_aq  = v_total / (1.0 + v_oa)
        v_org = v_total * v_oa / (1.0 + v_oa)

        # Build c0 for the solver
        c0: Dict[str, float] = dict(blended.concentrations)
        c0['O/A'] = v_oa

        ceq = self.reactions.equilibrium(c0)

        ceq_sol = _solution(ceq, volume=blended.volume, v_oa=v_oa)
        mix = ceq_sol.separate()

        # Apply efficiency by blending equilibrium result back toward initial
        if self.efficiency < 1.0:
            c0_plain = {k: v for k, v in c0.items() if k != 'O/A'}
            ceq_eff = {
                sp: (c0_plain.get(sp, 0.0)
                     + self.efficiency * (ceq[sp] - c0_plain.get(sp, 0.0)))
                for sp in ceq
            }
            ceq_sol_eff = _solution(ceq_eff, volume=blended.volume, v_oa=v_oa)
            mix = ceq_sol_eff.separate()

        self._outlets = mix
        self.ran = True
        return self

    # ------------------------------------------------------------------
    # Distribution and separation metrics
    # ------------------------------------------------------------------

    def distribution_coef(self, component: str) -> float:
        """
        Distribution coefficient D for *component* across this stage's outlets.

        Uses the converged (or pre-run phase-separated) outlets:
        organic extract at index 0, aqueous raffinate at index 1.

            D = org_concentration(component) / aq_concentration(component)

        Parameters
        ----------
        component : str
            Species name or element symbol (passed to
            :meth:`~efta.solution.solution.org_concentration` and
            :meth:`~efta.solution.solution.aq_concentration`).

        Returns
        -------
        float
            D value.  ``float('inf')`` if aqueous concentration is zero.

        Examples
        --------
        >>> stage.run()
        >>> D = stage.distribution_coef('La')
        """
        return distribution_coef(component, self._outlets)

    def separation_factor(self, component1: str, component2: str) -> float:
        """
        Separation factor β(component1 / component2) across this stage:

            β = D(component1) / D(component2)

        Parameters
        ----------
        component1 : str
            Numerator species or element symbol.
        component2 : str
            Denominator species or element symbol.

        Returns
        -------
        float

        Examples
        --------
        >>> stage.run()
        >>> beta = stage.separation_factor('La', 'Nd')
        """
        return separation_factor(component1, component2, self._outlets)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        rxn_count = (len(self.reactions._reactions)
                     if hasattr(self.reactions, '_reactions') else 1)
        status = f"efficiency={self.efficiency:.3g}" if self.ran else "not run"
        return (f"sx(inlets={len(self._inlets)}, "
                f"reactions={rxn_count}, "
                f"{status})")

    def __str__(self) -> str:
        lines = [
            "sx — single-stage solvent extraction",
            f"  Inlets     : {len(self._inlets)} solution(s)",
            f"  Ran        : {self.ran}",
        ]
        if self.ran:
            lines.append(f"  Efficiency : {self.efficiency:.4g}")
        lines += [
            f"  Outlets [0] (extract)   : V={self._outlets[0].volume:.4g} L",
            f"  Outlets [1] (raffinate) : V={self._outlets[1].volume:.4g} L",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level standalone functions
# ---------------------------------------------------------------------------

def _resolve_phases(mix: _mixture):
    """
    Identify which member of *mix* is the organic phase and which is aqueous.

    Scans each solution's species to find the one containing organic species.
    Returns ``(organic_sol, aqueous_sol)``.  If both or neither contain organic
    species, falls back to index 0 = organic, index 1 = aqueous (the
    ``mixture(extract, raffinate)`` convention used throughout efta).
    """
    if len(mix) < 2:
        raise InputError(
            f"distribution_coef / separation_factor require a mixture with "
            f"at least 2 solutions, got {len(mix)}.")

    org_idx = None
    aq_idx  = None
    for i, sol in enumerate(mix):
        has_org = any(_is_org(sp) for sp in sol)
        if has_org and org_idx is None:
            org_idx = i
        elif not has_org and aq_idx is None:
            aq_idx = i

    # Fall back to positional convention when auto-detect is ambiguous
    if org_idx is None:
        org_idx = 0
    if aq_idx is None:
        aq_idx = 1
    if org_idx == aq_idx:
        aq_idx = 1 if org_idx == 0 else 0

    return mix[org_idx], mix[aq_idx]


def distribution_coef(component: str, mix: _mixture) -> float:
    """
    Distribution coefficient D for *component* from a two-phase mixture.

    The organic and aqueous solutions are identified automatically by
    inspecting species phase tags.  Falls back to the positional convention
    ``mix[0]`` = organic, ``mix[1]`` = aqueous when phase detection is
    ambiguous.

        D = org_concentration(component) / aq_concentration(component)

    Parameters
    ----------
    component : str
        Species name or element symbol.
    mix : mixture
        A ``mixture`` containing at least one organic and one aqueous solution
        (e.g. the return value of :attr:`sx.outlets` or
        :func:`multiphase`).

    Returns
    -------
    float
        D value.  ``float('inf')`` if the aqueous concentration is zero.

    Examples
    --------
    >>> from efta.solventextraction.sx import distribution_coef
    >>> D = distribution_coef('La', stage.outlets)
    >>> D = distribution_coef('La', multiphase(feed, organic))
    """
    if not isinstance(mix, _mixture):
        raise InputError(
            f"distribution_coef: mix must be a mixture, got {type(mix).__name__!r}.")

    org_sol, aq_sol = _resolve_phases(mix)

    c_org = org_sol.org_concentration(component)
    c_aq  = aq_sol.aq_concentration(component)

    if c_aq == 0.0:
        return float('inf')
    return c_org / c_aq


def separation_factor(component1: str,
                      component2: str,
                      mix: _mixture) -> float:
    """
    Separation factor β(component1 / component2) from a two-phase mixture.

        β = D(component1) / D(component2)

    Phase resolution follows the same rules as :func:`distribution_coef`.

    Parameters
    ----------
    component1 : str
        Numerator species or element symbol.
    component2 : str
        Denominator species or element symbol.
    mix : mixture
        A ``mixture`` containing at least one organic and one aqueous solution.

    Returns
    -------
    float
        β value.  ``float('inf')`` if D2 == 0.
        ``float('nan')`` if both D values are infinite.

    Examples
    --------
    >>> from efta.solventextraction.sx import separation_factor
    >>> beta = separation_factor('La', 'Nd', stage.outlets)
    """
    if not isinstance(mix, _mixture):
        raise InputError(
            f"separation_factor: mix must be a mixture, got {type(mix).__name__!r}.")

    D1 = distribution_coef(component1, mix)
    D2 = distribution_coef(component2, mix)

    if D2 == 0.0:
        return float('inf')
    if D1 == float('inf') and D2 == float('inf'):
        return float('nan')
    return D1 / D2
