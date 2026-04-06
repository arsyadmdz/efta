"""
efta.solution
=============
The :class:`solution` class represents a chemical solution at equilibrium
(or any defined composition) with a known volume.

It wraps a concentration dict and a volume, and provides:

- Attribute-style and dict-style access to concentrations
- Moles, mass, and activity calculations
- Phase-filtered views (aqueous, organic, solid)
- Ionic strength and charge balance
- Mixing two solutions (``+`` operator)
- Pretty printing
- Direct construction from :meth:`efta.reactions.reactions.equilibrium`
  via :meth:`reactions.solution`
- ``sol(V)`` — clone with a different volume (moles preserved)
- ``sol.separate()`` — split into mixture(organic_sol, aqueous_sol)
- ``sol.total(species)`` — total element or species concentration (mol/L)
- ``sol.aq_concentration(query)`` — aqueous-phase concentration only
- ``sol.org_concentration(query)`` — organic-phase concentration only
- Use as dict key (identity-based hashing)
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

from .species import (
    species as _norm, formula, charge, components,
    phase, is_organic, is_solid, is_nonaqueous, is_electron,
)
from .errors import InputError
from .periodic_table import periodic_table as _PT

__all__ = ['solution']


def _molar_mass(sp: str) -> Optional[float]:
    """Return molar mass (g/mol) of *sp* using the periodic_table singleton."""
    comp = components(sp)
    if not comp:
        return None
    return _PT.molar_mass(comp)


class solution:
    """
    A chemical solution with known composition and volume.

    Parameters
    ----------
    concentrations : dict
        Mapping of species name → concentration in mol/L.
        Species names are normalised on construction.
    volume : float, optional
        Volume of the solution in litres.  Default is 1.0 L.

    Examples
    --------
    Construct directly:

    >>> from efta import Solution
    >>> sol = Solution({'H[+]': 1e-4, 'OH[-]': 1e-10, 'H2O': 1.0}, volume=0.5)

    Construct from equilibrium:

    >>> sys = reactions(r1, r2)
    >>> sol = sys.solution({'CH3COOH': 0.1, 'H2O': 1.0}, volume=1.0)

    Access concentrations:

    >>> sol['H[+]']          # by species name
    >>> sol.pH               # convenience property
    >>> sol.ionic_strength   # mol/L
    >>> sol.aqueous          # dict of aqueous species only
    >>> sol.moles('CH3COOH') # mol
    >>> sol.mass('CH3COOH')  # g

    Clone with a different volume (moles preserved):

    >>> sol_2L = sol(2.0)    # same moles, new volume

    Separate into organic and aqueous phases:

    >>> m = sol.separate()   # mixture(organic, aqueous) — volumes from v_oa

    Total and phase-specific concentrations:

    >>> sol.total('Fe')              # total iron by element (mol/L)
    >>> sol.aq_concentration('La')   # aqueous lanthanum only
    >>> sol.org_concentration('La')  # organic lanthanum only

    Mix two solutions:

    >>> mixed = sol1 + sol2
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self,
                 concentrations: Dict[str, float],
                 volume: float = 1.0,
                 v_oa: float = None):
        if volume < 0:
            raise InputError(
                f"volume must be >= 0, got {volume}.\n"
                f"Volume is in litres.")

        # normalise species names and store
        self._conc: Dict[str, float] = {}
        for sp, c in concentrations.items():
            sp_norm = _norm(sp)
            if is_electron(sp_norm):
                continue
            c = float(c)
            if c < 0:
                from .errors import ConcentrationError
                raise ConcentrationError(
                    f"Negative concentration for {sp!r}: {c}.\n"
                    f"All concentrations must be >= 0.")
            self._conc[sp_norm] = c

        self._volume: float = float(volume)
        self._v_oa: Optional[float] = self._init_voa(v_oa)
        self._gamma: dict = {}

    @classmethod
    def _from_eq(cls, ceq: Dict[str, float], volume: float = 1.0,
                 v_oa: float = None) -> 'solution':
        """Internal: construct from a raw equilibrium dict (already normalised)."""
        obj = cls.__new__(cls)
        obj._conc   = {sp: float(c) for sp, c in ceq.items()
                       if not is_electron(sp)}
        obj._volume = float(volume)
        obj._v_oa   = obj._init_voa(v_oa)
        obj._gamma  = {}
        return obj

    # ------------------------------------------------------------------
    # Volume cloning  —  sol(V)
    # ------------------------------------------------------------------

    def __call__(self, new_volume: float) -> 'solution':
        """
        Return a new solution with the same moles but a different volume.

        Concentrations are rescaled so that moles = c * V are preserved:
            c_new = c_old * V_old / V_new

        Parameters
        ----------
        new_volume : float
            Target volume in litres (must be positive).

        Returns
        -------
        solution

        Examples
        --------
        >>> sol_2L = sol(2.0)     # same moles, volume doubled → c halved
        >>> sol_pt5 = sol(0.5)    # same moles, volume halved  → c doubled
        """
        new_volume = float(new_volume)
        if new_volume <= 0:
            raise InputError(
                f"Volume must be positive, got {new_volume}.")
        factor = self._volume / new_volume
        new_conc = {sp: c * factor for sp, c in self._conc.items()}
        result = solution._from_eq(new_conc, volume=new_volume, v_oa=self._v_oa)
        result._gamma = dict(self._gamma)
        return result

    # ------------------------------------------------------------------
    # Phase separation  —  sol.separate(v_aq, v_org)
    # ------------------------------------------------------------------

    def separate(self) -> 'mixture':
        """
        Separate the solution's composition into two pure-phase solutions and
        return a :class:`~efta.mixture.mixture` of ``(organic, aqueous)``.

        Volumes for each phase are derived from the solution's own
        :attr:`v_oa` and :attr:`volume`:

        - Mixed-phase (``v_oa`` is set):
          ``V_aq = V / (1 + v_oa)``,  ``V_org = V * v_oa / (1 + v_oa)``
        - Pure-aqueous (no organic species): ``V_aq = V``, ``V_org = V``
          (organic slot will be empty)
        - Pure-organic (only organic species): ``V_aq = V``, ``V_org = V``
          (aqueous slot will be empty)

        Aqueous species (including ``H2O(l)`` and solids) go to the aqueous
        solution.  Organic-phase species go to the organic solution.

        Returns
        -------
        mixture
            ``mixture(organic_solution, aqueous_solution)``
            Index 0 = organic outlet, index 1 = aqueous outlet.
            A phase with no species still appears as a zero-concentration
            solution with its assigned volume.

        Examples
        --------
        >>> m = sol.separate()
        >>> org, aq = m[0], m[1]
        """
        from .mixture import mixture as _mixture
        from .species import phase as _phase_fn

        r = self._v_oa
        v_total = self._volume

        if r is not None:
            _v_aq  = v_total / (1.0 + r)
            _v_org = v_total * r / (1.0 + r)
        else:
            _v_aq  = v_total
            _v_org = v_total

        # Guard against zero volumes (degenerate v_oa edge cases)
        if _v_aq  <= 0: _v_aq  = v_total
        if _v_org <= 0: _v_org = v_total

        org_conc: Dict[str, float] = {}
        aq_conc:  Dict[str, float] = {}

        # Concentrations in self are mol / v_total.  Each phase solution is
        # stored at its own smaller volume, so we must rescale so that moles
        # are conserved:  n = c_blend × v_total  →  c_phase = n / v_phase
        for sp, c in self._conc.items():
            if _phase_fn(sp) == 'organic':
                org_conc[sp] = c * v_total / _v_org
            else:
                aq_conc[sp] = c * v_total / _v_aq

        org_sol = solution._from_eq(org_conc, volume=_v_org)
        aq_sol  = solution._from_eq(aq_conc,  volume=_v_aq)

        return _mixture(org_sol, aq_sol)

    # ------------------------------------------------------------------
    # Hashability — identity-based, so solutions can be used as dict keys
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        # Identity equality keeps dict-key semantics predictable.
        # Use .concentrations == other.concentrations for value equality.
        return self is other

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def _init_voa(self, v_oa) -> Optional[float]:
        """Validate and return the stored v_oa, or None for single-phase solutions."""
        has_aq  = any(not is_organic(sp) and not is_solid(sp) for sp in self._conc)
        has_org = any(is_organic(sp) for sp in self._conc)
        mixed   = has_aq and has_org

        if v_oa is not None:
            v_oa = float(v_oa)
            if v_oa <= 0:
                raise InputError(
                    f"v_oa must be positive, got {v_oa}.\n"
                    f"v_oa is the organic/aqueous volume ratio.")
            return v_oa

        if not mixed:
            return None
        return 1.0

    @property
    def v_oa(self) -> Optional[float]:
        """
        Organic-to-aqueous volume ratio (V_org / V_aq).

        Returns None if the solution contains only aqueous or only organic
        species.  Defaults to 1.0 for mixed-phase solutions.

        Setting v_oa automatically updates v_ao = 1 / v_oa.
        """
        return self._v_oa

    @v_oa.setter
    def v_oa(self, value: float) -> None:
        if self._v_oa is None:
            raise InputError(
                "Cannot set v_oa on a single-phase solution.\n"
                "v_oa is only defined when both aqueous and organic species are present.")
        value = float(value)
        if value <= 0:
            raise InputError(
                f"v_oa must be positive, got {value}.\n"
                f"v_oa is the organic/aqueous volume ratio.")
        self._v_oa = value

    @property
    def v_ao(self) -> Optional[float]:
        """
        Aqueous-to-organic volume ratio (V_aq / V_org) = 1 / v_oa.

        Returns None if the solution contains only aqueous or only organic
        species.

        Setting v_ao automatically updates v_oa = 1 / v_ao.
        """
        if self._v_oa is None:
            return None
        return 1.0 / self._v_oa

    @v_ao.setter
    def v_ao(self, value: float) -> None:
        if self._v_oa is None:
            raise InputError(
                "Cannot set v_ao on a single-phase solution.\n"
                "v_ao is only defined when both aqueous and organic species are present.")
        value = float(value)
        if value <= 0:
            raise InputError(
                f"v_ao must be positive, got {value}.\n"
                f"v_ao is the aqueous/organic volume ratio.")
        self._v_oa = 1.0 / value

    @property
    def volume(self) -> float:
        """Volume of the solution (L)."""
        return self._volume

    @property
    def concentrations(self) -> Dict[str, float]:
        """Full concentration dict {species: mol/L}."""
        return dict(self._conc)

    # Phase keys recognised by __getitem__
    _PHASE_KEYS = frozenset({'aq', 'org', 's', 'l'})

    def __getitem__(self, sp: str):
        """
        Return concentration of *sp* in mol/L (0.0 if absent).

        Special phase-key shortcuts return a new :class:`solution` containing
        only the species of that phase, with the corresponding phase volume:

        ``sol['aq']``  — aqueous species; V_aq = V / (1 + v_oa)
        ``sol['org']`` — organic species; V_org = V * v_oa / (1 + v_oa)
        ``sol['s']``   — solid species;   volume = 0 (convention)
        ``sol['l']``   — liquid species;  volume = 0 (convention)

        For single-phase solutions ``v_oa`` is None, so ``'aq'`` returns
        volume = V and ``'org'`` returns an empty solution with volume = 0.
        """
        from .species import phase as _phase
        if sp in self._PHASE_KEYS:
            phase_map = {'aq': 'aqueous', 'org': 'organic',
                         's': 'solid',    'l': 'liquid'}
            target = phase_map[sp]
            conc   = {s: c for s, c in self._conc.items()
                      if _phase(s) == target}
            # compute phase volume
            r = self._v_oa  # None for single-phase
            if sp == 'aq':
                vol = self._volume / (1 + r) if r is not None else self._volume
            elif sp == 'org':
                vol = self._volume * r / (1 + r) if r is not None else 0.0
            else:  # 's' or 'l'
                vol = 0.0
            return solution._from_eq(conc, volume=vol)
        return self._conc.get(_norm(sp), 0.0)

    def __contains__(self, sp: str) -> bool:
        try:
            return _norm(sp) in self._conc
        except Exception:
            return sp in self._conc

    def __iter__(self) -> Iterator[str]:
        return iter(self._conc)

    def __len__(self) -> int:
        return len(self._conc)

    def get(self, sp: str, default: float = 0.0) -> float:
        """Return concentration of *sp*, or *default* if absent."""
        try:
            return self._conc.get(_norm(sp), default)
        except Exception:
            return default

    def keys(self):
        return self._conc.keys()

    def values(self):
        return self._conc.values()

    def items(self):
        return self._conc.items()

    # ------------------------------------------------------------------
    # Per-species calculations
    # ------------------------------------------------------------------

    def moles(self, sp: str) -> float:
        """Return moles of *sp* in the solution (mol = c * V)."""
        return self[sp] * self._volume

    def mass(self, sp: str) -> Optional[float]:
        """
        Return mass of *sp* in the solution (g), or None if the molar
        mass cannot be determined from the species formula.
        """
        mm = _molar_mass(sp)
        if mm is None:
            return None
        return self.moles(sp) * mm

    def molar_mass(self, sp: str) -> Optional[float]:
        """Return molar mass of *sp* in g/mol, or None if unknown."""
        return _molar_mass(sp)

    def activity(self, sp: str) -> float:
        """
        Return the ideal activity of *sp* (concentration / 1 mol/L).
        For non-aqueous species (solids, pure liquids) the activity is 1.0.
        """
        sp_norm = _norm(sp)
        if is_nonaqueous(sp_norm):
            return 1.0
        return self[sp_norm]

    # ------------------------------------------------------------------
    # Phase-filtered views
    # ------------------------------------------------------------------

    @property
    def aqueous(self) -> Dict[str, float]:
        """Concentrations of aqueous species only."""
        return {sp: c for sp, c in self._conc.items()
                if not is_nonaqueous(sp) and not is_organic(sp)}

    @property
    def organic(self) -> Dict[str, float]:
        """Concentrations of organic-phase species only."""
        return {sp: c for sp, c in self._conc.items() if is_organic(sp)}

    @property
    def solid(self) -> Dict[str, float]:
        """Concentrations of solid-phase species only."""
        return {sp: c for sp, c in self._conc.items() if is_solid(sp)}

    @property
    def dissolved(self) -> Dict[str, float]:
        """Concentrations of dissolved (aqueous + organic) species."""
        return {sp: c for sp, c in self._conc.items()
                if not is_solid(sp)}

    # ------------------------------------------------------------------
    # Solution properties
    # ------------------------------------------------------------------

    @property
    def pH(self) -> Optional[float]:
        """
        pH of the solution (-log10([H+])), or None if H+ is absent.
        """
        h = self._conc.get('H[+]', 0.0)
        if h <= 0:
            return None
        return -math.log10(h)

    @property
    def pOH(self) -> Optional[float]:
        """pOH of the solution, or None if OH- is absent."""
        oh = self._conc.get('OH[-]', 0.0)
        if oh <= 0:
            return None
        return -math.log10(oh)

    @property
    def I(self) -> float:
        """Ionic strength (mol/L). Alias for :attr:`ionic_strength`."""
        return self.ionic_strength

    @property
    def ionic_strength(self) -> float:
        """
        Ionic strength I = 0.5 * sum(c_i * z_i^2) in mol/L.
        Only aqueous charged species contribute.
        """
        I = 0.0
        for sp, c in self._conc.items():
            if is_nonaqueous(sp) or is_organic(sp):
                continue
            z = charge(sp)
            if z != 0:
                I += c * z * z
        return 0.5 * I

    @property
    def charge_balance(self) -> float:
        """
        Charge balance = sum(c_i * z_i) in mol/L.
        Should be ~0 for a physically consistent solution.
        Positive means excess cations, negative means excess anions.
        """
        bal = 0.0
        for sp, c in self._conc.items():
            if is_nonaqueous(sp) or is_organic(sp):
                continue
            bal += c * charge(sp)
        return bal

    @property
    def total_dissolved_solids(self) -> Optional[float]:
        """
        Total dissolved solids (g/L) — sum of (c * M) for all dissolved
        species whose molar mass is known.
        Returns None if any dissolved species has an unknown molar mass.
        """
        tds = 0.0
        for sp, c in self._conc.items():
            if is_solid(sp):
                continue
            mm = _molar_mass(sp)
            if mm is None:
                return None
            tds += c * mm
        return tds

    def total(self, query: str) -> float:
        """
        Total concentration of a species or element across all species (mol/L).

        Two modes:

        1. **Exact species match** — if *query* (after normalisation) matches
           a species name in this solution, returns that species' concentration.
           This handles phase-tagged species like ``'H2A2(org)'``.

        2. **Element sum** — if *query* looks like an element symbol (e.g.
           ``'Fe'``, ``'La'``), returns the stoichiometry-weighted sum of all
           species that contain that element.

        Parameters
        ----------
        query : str
            A species name (exact match, normalised) or an element symbol.

        Returns
        -------
        float
            Concentration in mol/L.

        Examples
        --------
        >>> sol.total('H2A2(org)')   # exact species → its concentration
        >>> sol.total('Fe')          # total iron across FeOH[+], Fe[3+], etc.
        >>> sol.total('La')          # total lanthanum
        """
        # Try exact species match first
        try:
            sp_norm = _norm(query)
            if sp_norm in self._conc:
                return self._conc[sp_norm]
        except Exception:
            pass

        # Element sum fallback
        tot = 0.0
        for sp, c in self._conc.items():
            comp = components(sp)
            tot += c * comp.get(query, 0.0)
        return tot

    def aq_concentration(self, query: str) -> float:
        """
        Aqueous-phase concentration of a species or element (mol/L).

        Sums only species that are **not** organic-phase.  Uses the same
        dual-mode resolution as :meth:`total`:

        1. **Exact species match** — normalised species name present in this
           solution and not organic → its concentration.
        2. **Element sum** — stoichiometry-weighted sum over all *non-organic*
           species that contain *query* as an element symbol.

        Parameters
        ----------
        query : str
            Species name or element symbol.

        Returns
        -------
        float
            Concentration in mol/L of *query* in the aqueous phase.

        Examples
        --------
        >>> sol.aq_concentration('La')        # total aqueous lanthanum
        >>> sol.aq_concentration('LaCl[2+]')  # exact aqueous species
        """
        try:
            sp_norm = _norm(query)
            if sp_norm in self._conc and not is_organic(sp_norm):
                return self._conc[sp_norm]
        except Exception:
            pass

        tot = 0.0
        for sp, c in self._conc.items():
            if is_organic(sp):
                continue
            comp = components(sp)
            tot += c * comp.get(query, 0.0)
        return tot

    def org_concentration(self, query: str) -> float:
        """
        Organic-phase concentration of a species or element (mol/L).

        Sums only species that **are** organic-phase.  Uses the same
        dual-mode resolution as :meth:`total`:

        1. **Exact species match** — normalised species name present in this
           solution and is organic → its concentration.
        2. **Element sum** — stoichiometry-weighted sum over all *organic*
           species that contain *query* as an element symbol.

        Parameters
        ----------
        query : str
            Species name or element symbol.

        Returns
        -------
        float
            Concentration in mol/L of *query* in the organic phase.

        Examples
        --------
        >>> sol.org_concentration('La')           # total organic lanthanum
        >>> sol.org_concentration('LaA3(org)')    # exact organic species
        >>> sol.org_concentration('H2A2(org)')    # extractant concentration
        """
        try:
            sp_norm = _norm(query)
            if sp_norm in self._conc and is_organic(sp_norm):
                return self._conc[sp_norm]
        except Exception:
            pass

        tot = 0.0
        for sp, c in self._conc.items():
            if not is_organic(sp):
                continue
            comp = components(sp)
            tot += c * comp.get(query, 0.0)
        return tot

    # ------------------------------------------------------------------
    # Mixing
    # ------------------------------------------------------------------

    def __add__(self, other: 'solution') -> 'solution':
        """
        Mix two solutions by combining moles and summing volumes.

        Concentrations in the mixed solution are:
            c_mix[sp] = (n1[sp] + n2[sp]) / (V1 + V2)

        The mixed v_oa is computed from the actual organic and aqueous volumes
        contributed by each solution:

        - Mixed-phase solution (v_oa set):
            V_aq  = V / (1 + v_oa),  V_org = V * v_oa / (1 + v_oa)
        - Single-phase organic solution (v_oa None, only organic species):
            V_org = V,  V_aq = 0
        - Single-phase aqueous solution (v_oa None, only aqueous species):
            V_aq = V,  V_org = 0

        v_oa_mix = total_V_org / total_V_aq  (None if total_V_aq == 0)

        Parameters
        ----------
        other : solution

        Returns
        -------
        solution
            A new solution with the mixed composition and combined volume.

        Examples
        --------
        >>> mixed = acidic_sol + basic_sol
        """
        if not isinstance(other, solution):
            return NotImplemented
        v_total = self._volume + other._volume
        all_sp  = set(self._conc) | set(other._conc)
        mixed   = {
            sp: (self.moles(sp) + other.moles(sp)) / v_total
            for sp in all_sp
        }

        def _phase_volumes(sol):
            """Return (V_org, V_aq) for a solution based on v_oa and phase content."""
            v = sol._volume
            r = sol._v_oa
            if r is not None:
                # Mixed-phase solution: split by v_oa
                return v * r / (1.0 + r), v / (1.0 + r)
            # Single-phase: classify by species content
            has_org = any(is_organic(sp) for sp in sol._conc)
            has_aq  = any(not is_organic(sp) and not is_solid(sp)
                          for sp in sol._conc)
            if has_org and not has_aq:
                return v, 0.0   # pure organic
            if has_aq and not has_org:
                return 0.0, v   # pure aqueous
            # Empty or ambiguous single-phase: treat as aqueous
            return 0.0, v

        v_org1, v_aq1 = _phase_volumes(self)
        v_org2, v_aq2 = _phase_volumes(other)
        v_org_total   = v_org1 + v_org2
        v_aq_total    = v_aq1  + v_aq2

        if v_aq_total > 0 and v_org_total > 0:
            v_oa_mix = v_org_total / v_aq_total
        elif v_aq_total > 0:
            v_oa_mix = None   # no organic phase present
        elif v_org_total > 0:
            v_oa_mix = None   # no aqueous phase present
        else:
            v_oa_mix = None

        return solution._from_eq(mixed, volume=v_total, v_oa=v_oa_mix)

    def dilute(self, factor: float) -> 'solution':
        """
        Return a new Solution diluted by *factor* (volume multiplied by factor,
        concentrations divided by factor).

        Parameters
        ----------
        factor : float
            Dilution factor > 1 means more dilute (e.g. 2 = half concentration).

        Examples
        --------
        >>> diluted = sol.dilute(10)   # 10× dilution
        """
        if factor <= 0:
            raise InputError(
                f"Dilution factor must be positive, got {factor}.")
        return solution._from_eq(
            {sp: c / factor for sp, c in self._conc.items()},
            volume=self._volume * factor,
        )

    def scale_volume(self, new_volume: float) -> 'solution':
        """
        Return a new Solution with the same composition but a different volume.
        Moles are preserved; concentrations are recalculated.

        Parameters
        ----------
        new_volume : float
            New volume in litres.
        """
        if new_volume <= 0:
            raise InputError(
                f"new_volume must be positive, got {new_volume}.")
        factor = new_volume / self._volume
        return solution._from_eq(
            {sp: c / factor for sp, c in self._conc.items()},
            volume=new_volume,
        )

    def __mul__(self, factor: float) -> 'solution':
        """
        Return a new solution with all concentrations scaled by *factor*.
        Volume is unchanged.

        Examples
        --------
        >>> concentrated = sol * 2   # double all concentrations
        >>> diluted      = sol * 0.5
        """
        factor = float(factor)
        if factor <= 0:
            raise InputError(f"Scaling factor must be positive, got {factor}.")
        return solution._from_eq(
            {sp: c * factor for sp, c in self._conc.items()},
            volume=self._volume, v_oa=self._v_oa,
        )

    def __rmul__(self, factor: float) -> 'solution':
        """Support ``2 * sol``."""
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> 'solution':
        """
        Return a new solution with all concentrations divided by *factor*.
        Volume is unchanged.

        Examples
        --------
        >>> diluted = sol / 2
        """
        factor = float(factor)
        if factor <= 0:
            raise InputError(f"Divisor must be positive, got {factor}.")
        return self.__mul__(1.0 / factor)

    def add(self, sp: str, moles: float) -> 'solution':
        """
        Return a new solution with *moles* of species *sp* added.

        The moles are converted to a concentration increase using the
        current volume: Δc = moles / volume.  Volume is unchanged.

        Parameters
        ----------
        sp    : str   — species name (will be normalised)
        moles : float — moles to add (must be > 0)

        Examples
        --------
        >>> sol2 = sol.add('NaOH', 0.001)   # add 1 mmol NaOH
        """
        moles = float(moles)
        if moles <= 0:
            raise InputError(f"moles must be positive, got {moles}.")
        sp_norm = _norm(sp)
        new_conc = dict(self._conc)
        new_conc[sp_norm] = new_conc.get(sp_norm, 0.0) + moles / self._volume
        return solution._from_eq(new_conc, volume=self._volume, v_oa=self._v_oa)

    def remove(self, sp: str, moles: float) -> 'solution':
        """
        Return a new solution with *moles* of species *sp* removed.

        Raises :class:`~efta.errors.InputError` if removal would result in
        a negative concentration.

        Parameters
        ----------
        sp    : str   — species name (will be normalised)
        moles : float — moles to remove (must be > 0)

        Examples
        --------
        >>> sol2 = sol.remove('H[+]', 1e-5)
        """
        from .errors import ConcentrationError
        moles = float(moles)
        if moles <= 0:
            raise InputError(f"moles must be positive, got {moles}.")
        sp_norm = _norm(sp)
        current = self._conc.get(sp_norm, 0.0) * self._volume
        if moles > current + 1e-20:
            raise ConcentrationError(
                f"Cannot remove {moles:.4e} mol of {sp!r}: "
                f"only {current:.4e} mol present.")
        new_conc = dict(self._conc)
        new_conc[sp_norm] = max(0.0, (current - moles) / self._volume)
        return solution._from_eq(new_conc, volume=self._volume, v_oa=self._v_oa)

    def to_c0(self) -> dict:
        """
        Return a plain dict suitable for passing directly to the solver.

        Includes ``'O/A'`` if :attr:`v_oa` is set.

        Examples
        --------
        >>> sys.equilibrium(sol.to_c0())   # equivalent to sys.equilibrium(sol)
        """
        d = dict(self._conc)
        if self._v_oa is not None:
            d['O/A'] = self._v_oa
        return d

    def strip(self, phase: str) -> 'solution':
        """
        Return a new solution with all species of the given phase removed.

        Parameters
        ----------
        phase : str
            One of ``'organic'``, ``'aqueous'``, ``'solid'``, ``'liquid'``,
            ``'gas'``.

        Examples
        --------
        >>> aq_only = sol.strip('organic')   # remove organic phase
        >>> sol_no_solid = sol.strip('solid')
        """
        from .species import phase as _phase
        valid = {'organic', 'aqueous', 'solid', 'liquid', 'gas'}
        phase = phase.strip().lower()
        if phase not in valid:
            raise InputError(
                f"phase must be one of {sorted(valid)}, got {phase!r}.")
        new_conc = {sp: c for sp, c in self._conc.items()
                    if _phase(sp) != phase}
        new_v_oa = self._v_oa
        if phase == 'organic':
            new_v_oa = None
        elif phase == 'aqueous':
            new_v_oa = None
        return solution._from_eq(new_conc, volume=self._volume, v_oa=new_v_oa)

    # ------------------------------------------------------------------
    # Gamma / activity coefficients
    # ------------------------------------------------------------------

    def _prepare_rxnsys(self, rxn_sys):
        """
        Normalise *rxn_sys* to a :class:`~efta.reactions.reactions` instance
        and inject any solution-level gamma functions into its reaction objects.
        Returns the (possibly wrapped) reactions object.
        """
        from .reactions import reactions as _reactions
        from .reaction  import reaction  as _reaction
        import copy
        if isinstance(rxn_sys, _reaction):
            rxn_sys = _reactions(rxn_sys)
        if self._gamma:
            new_rxns = []
            for rxn in rxn_sys._reactions:
                r = copy.copy(rxn)
                r._gamma = {**getattr(rxn, '_gamma', {}), **self._gamma}
                new_rxns.append(r)
            rxn_sys = _reactions(*new_rxns)
        return rxn_sys

    def equilibrate(self, rxn_sys, **kw) -> 'solution':
        """
        Equilibrate this solution in-place and return *self*.

        Parameters
        ----------
        rxn_sys : reaction or reactions
            The reaction system to equilibrate against.
        **kw
            Forwarded to :meth:`~efta.reactions.reactions.equilibrium`.

        Returns
        -------
        solution
            *self*, mutated to equilibrium composition.

        Examples
        --------
        >>> sol.equilibrate(sys)
        >>> print(sol.pH)
        """
        rxn_sys = self._prepare_rxnsys(rxn_sys)
        ceq = rxn_sys.equilibrium(self, **kw)
        self._conc = {sp: float(c) for sp, c in ceq.items()
                      if not is_electron(sp)}
        return self

    def __lshift__(self, rxn_sys) -> 'solution':
        """
        Return a new solution at equilibrium: ``sol << sys``.

        Returns a new object; *self* is not modified.

        Examples
        --------
        >>> eq = sol << sys
        >>> eq = sol << r1
        """
        rxn_sys = self._prepare_rxnsys(rxn_sys)
        ceq = rxn_sys.equilibrium(self)
        result = solution._from_eq(ceq, volume=self._volume, v_oa=self._v_oa)
        result._gamma = dict(self._gamma)
        return result

    # ------------------------------------------------------------------
    # Activity coefficients
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> dict:
        """
        Dict of registered activity-coefficient functions
        ``{species: (func, *dep_names)}``.
        """
        return dict(self._gamma)

    def set_gamma(self, sp: str, gamma_spec: tuple) -> 'solution':
        """
        Register an activity-coefficient function for species *sp*.

        Uses the same interface as :meth:`~efta.reaction.reaction.set_gamma`:
        a tuple ``(func, *dep_names)`` where dep_names may include ``'I'``
        for ionic strength.

        The registered gamma functions are used in
        :meth:`saturation_index` and are passed through to
        :meth:`equilibrate` / ``<<`` / ``>>`` via the solver.

        Parameters
        ----------
        sp         : species name
        gamma_spec : tuple ``(callable, *dep_names)``

        Examples
        --------
        >>> sol.set_gamma('Fe[3+]', (davies, 'I'))
        """
        from .system import _IONIC_STRENGTH_TOKEN
        from .errors import InputError
        from .species import species as _norm
        if not isinstance(gamma_spec, tuple) or len(gamma_spec) == 0:
            raise InputError(
                f"gamma_spec must be a non-empty tuple (func, *dep_names), "
                f"got {gamma_spec!r}.")
        func = gamma_spec[0]
        if not callable(func):
            raise InputError(
                f"First element of gamma_spec must be callable, got {func!r}.")
        dep_names = gamma_spec[1:]
        sp_norm  = _norm(sp)
        dep_norm = tuple(
            s if s == _IONIC_STRENGTH_TOKEN else _norm(s)
            for s in dep_names
        )
        self._gamma[sp_norm] = (func,) + dep_norm
        return self

    def pC(self, sp: str) -> Optional[float]:
        """
        Return ``-log10(c)`` for species *sp*, or None if the concentration
        is zero or the species is absent.

        Examples
        --------
        >>> sol.pC('H[+]')   # same as sol.pH
        >>> sol.pC('Fe[3+]')
        """
        c = self[sp]
        if c <= 0:
            return None
        return -math.log10(c)

    def saturation_index(self, rxn) -> float:
        """
        Saturation index SI = log10(Q / K) for a reaction.

        SI > 0  → supersaturated (precipitation favoured)
        SI = 0  → at equilibrium
        SI < 0  → undersaturated (dissolution favoured)

        Parameters
        ----------
        rxn : reaction
            A precipitation/dissolution reaction with a solid species.

        Examples
        --------
        >>> r_calcite = reaction('CaCO3(s) = Ca[2+] + CO3[2-]', 3.3e-9, ksp=True)
        >>> sol.saturation_index(r_calcite)
        """
        from .system import _saturation_index, _build_gamma_for_system
        gamma_dict = _build_gamma_for_system([rxn], list(self._conc.keys()))
        for sp, entry in self._gamma.items():
            gamma_dict[sp] = entry
        return _saturation_index(rxn, self._conc, gamma_dict)

    def _table_lines(self) -> list:
        """
        Build table rows: species sorted by phase order (org, aq, liq, solid),
        then by descending concentration within each phase.
        Each row: species name, concentration, moles, mass.
        Footer: volume, and v_oa/v_ao if mixed-phase.
        """
        from .species import phase as _phase

        phase_order = {'organic': 0, 'aqueous': 1, 'liquid': 2,
                       'solid': 3, 'gas': 4}

        rows = sorted(
            self._conc.items(),
            key=lambda x: (phase_order.get(_phase(x[0]), 9), -x[1])
        )

        w_sp = max((len(sp) for sp in self._conc), default=8)
        w_sp = max(w_sp, 8)

        lines = []
        for sp, c in rows:
            lines.append(f"  {sp:<{w_sp}s}  {c:.4e} mol/L")

        lines.append("")
        lines.append(f"  V = {self._volume:.4g} L")
        if self._v_oa is not None:
            lines.append(f"  V_org/V_aq = {self._v_oa:.4g}"
                         f"  (V_aq/V_org = {1/self._v_oa:.4g})")

        return lines

    def __repr__(self) -> str:
        return "\n".join(self._table_lines())

    def __str__(self) -> str:
        return "\n".join(self._table_lines())
