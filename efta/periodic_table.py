"""
efta.periodic_table
===================
Singleton :data:`periodic_table` containing atomic number and standard
atomic mass (IUPAC 2021) for all 118 elements, plus support for
user-defined custom components (e.g. isotopes, coarse-grained beads).

Access
------
::

    from efta import periodic_table

    periodic_table['Fe']            # -> {'Z': 26, 'mass': 55.845}
    periodic_table.mass('Fe')       # -> 55.845  (g/mol)
    periodic_table.atomic_number('Fe')  # -> 26
    periodic_table.symbol(26)       # -> 'Fe'

    # add a custom component
    periodic_table.update_components('D', 2.014)   # deuterium as custom
    periodic_table.remove_components('D')

    # list all
    list(periodic_table)            # all symbols in Z order
    len(periodic_table)             # 118 (+ custom)

Notes
-----
- All 118 standard elements are **default** components and cannot be removed.
- Custom components are assigned Z values starting from 1000, auto-incremented.
  If a custom component is removed, all higher custom Z values shift down by 1
  so the ID space stays contiguous within the custom range.
- ``update_component`` on an existing custom symbol updates its mass only
  (Z is unchanged); on an existing default symbol, only the mass is updated
  (Z cannot change for defaults).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from .errors import InputError

__all__ = ['periodic_table', 'PeriodicTable']

# ---------------------------------------------------------------------------
# IUPAC 2021 standard atomic weights (abridged to 5 sig. figs.)
# Source: https://iupac.org/what-we-do/periodic-table-of-elements/
# ---------------------------------------------------------------------------

_DEFAULT_ELEMENTS: List[Tuple[int, str, float]] = [
    # (Z, symbol, mass g/mol)
    (1,   'H',   1.008),
    (2,   'He',  4.0026),
    (3,   'Li',  6.941),
    (4,   'Be',  9.0122),
    (5,   'B',   10.81),
    (6,   'C',   12.011),
    (7,   'N',   14.007),
    (8,   'O',   15.999),
    (9,   'F',   18.998),
    (10,  'Ne',  20.180),
    (11,  'Na',  22.990),
    (12,  'Mg',  24.305),
    (13,  'Al',  26.982),
    (14,  'Si',  28.085),
    (15,  'P',   30.974),
    (16,  'S',   32.06),
    (17,  'Cl',  35.45),
    (18,  'Ar',  39.948),
    (19,  'K',   39.098),
    (20,  'Ca',  40.078),
    (21,  'Sc',  44.956),
    (22,  'Ti',  47.867),
    (23,  'V',   50.942),
    (24,  'Cr',  51.996),
    (25,  'Mn',  54.938),
    (26,  'Fe',  55.845),
    (27,  'Co',  58.933),
    (28,  'Ni',  58.693),
    (29,  'Cu',  63.546),
    (30,  'Zn',  65.38),
    (31,  'Ga',  69.723),
    (32,  'Ge',  72.630),
    (33,  'As',  74.922),
    (34,  'Se',  78.971),
    (35,  'Br',  79.904),
    (36,  'Kr',  83.798),
    (37,  'Rb',  85.468),
    (38,  'Sr',  87.62),
    (39,  'Y',   88.906),
    (40,  'Zr',  91.224),
    (41,  'Nb',  92.906),
    (42,  'Mo',  95.95),
    (43,  'Tc',  97.0),      # no stable isotope; mass of most stable isotope
    (44,  'Ru',  101.07),
    (45,  'Rh',  102.91),
    (46,  'Pd',  106.42),
    (47,  'Ag',  107.87),
    (48,  'Cd',  112.41),
    (49,  'In',  114.82),
    (50,  'Sn',  118.71),
    (51,  'Sb',  121.76),
    (52,  'Te',  127.60),
    (53,  'I',   126.90),
    (54,  'Xe',  131.29),
    (55,  'Cs',  132.91),
    (56,  'Ba',  137.33),
    (57,  'La',  138.91),
    (58,  'Ce',  140.12),
    (59,  'Pr',  140.91),
    (60,  'Nd',  144.24),
    (61,  'Pm',  145.0),     # no stable isotope
    (62,  'Sm',  150.36),
    (63,  'Eu',  151.96),
    (64,  'Gd',  157.25),
    (65,  'Tb',  158.93),
    (66,  'Dy',  162.50),
    (67,  'Ho',  164.93),
    (68,  'Er',  167.26),
    (69,  'Tm',  168.93),
    (70,  'Yb',  173.05),
    (71,  'Lu',  174.97),
    (72,  'Hf',  178.49),
    (73,  'Ta',  180.95),
    (74,  'W',   183.84),
    (75,  'Re',  186.21),
    (76,  'Os',  190.23),
    (77,  'Ir',  192.22),
    (78,  'Pt',  195.08),
    (79,  'Au',  196.97),
    (80,  'Hg',  200.59),
    (81,  'Tl',  204.38),
    (82,  'Pb',  207.2),
    (83,  'Bi',  208.98),
    (84,  'Po',  209.0),     # no stable isotope
    (85,  'At',  210.0),     # no stable isotope
    (86,  'Rn',  222.0),     # no stable isotope
    (87,  'Fr',  223.0),     # no stable isotope
    (88,  'Ra',  226.0),     # no stable isotope
    (89,  'Ac',  227.0),     # no stable isotope
    (90,  'Th',  232.04),
    (91,  'Pa',  231.04),
    (92,  'U',   238.03),
    (93,  'Np',  237.0),     # no stable isotope
    (94,  'Pu',  244.0),     # no stable isotope
    (95,  'Am',  243.0),     # no stable isotope
    (96,  'Cm',  247.0),     # no stable isotope
    (97,  'Bk',  247.0),     # no stable isotope
    (98,  'Cf',  251.0),     # no stable isotope
    (99,  'Es',  252.0),     # no stable isotope
    (100, 'Fm',  257.0),     # no stable isotope
    (101, 'Md',  258.0),     # no stable isotope
    (102, 'No',  259.0),     # no stable isotope
    (103, 'Lr',  262.0),     # no stable isotope
    (104, 'Rf',  267.0),     # no stable isotope
    (105, 'Db',  268.0),     # no stable isotope
    (106, 'Sg',  269.0),     # no stable isotope
    (107, 'Bh',  270.0),     # no stable isotope
    (108, 'Hs',  277.0),     # no stable isotope
    (109, 'Mt',  278.0),     # no stable isotope
    (110, 'Ds',  281.0),     # no stable isotope
    (111, 'Rg',  282.0),     # no stable isotope
    (112, 'Cn',  285.0),     # no stable isotope
    (113, 'Nh',  286.0),     # no stable isotope
    (114, 'Fl',  289.0),     # no stable isotope
    (115, 'Mc',  290.0),     # no stable isotope
    (116, 'Lv',  293.0),     # no stable isotope
    (117, 'Ts',  294.0),     # no stable isotope
    (118, 'Og',  294.0),     # no stable isotope
]

_CUSTOM_Z_START = 1000  # custom components start at Z=1000


class PeriodicTable:
    """
    Singleton periodic table with atomic number (Z) and atomic mass for all
    118 elements, plus user-defined custom components.

    Do not instantiate directly — use the module-level :data:`periodic_table`.

    Structure of each entry
    -----------------------
    ``{'Z': int, 'mass': float, 'custom': bool}``

    - ``Z``      : atomic number (1–118 for elements; ≥1000 for custom)
    - ``mass``   : standard atomic weight in g/mol
    - ``custom`` : False for the 118 standard elements, True for user additions
    """

    _instance: Optional['PeriodicTable'] = None

    def __new__(cls) -> 'PeriodicTable':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # symbol -> {'Z': int, 'mass': float, 'custom': bool}
        self._data: Dict[str, Dict] = {}
        # Z -> symbol  (reverse lookup)
        self._z_to_sym: Dict[int, str] = {}
        # set of default symbols (cannot be removed)
        self._defaults: frozenset

        for Z, sym, mass in _DEFAULT_ELEMENTS:
            entry = {'Z': Z, 'mass': float(mass), 'custom': False}
            self._data[sym] = entry
            self._z_to_sym[Z] = sym

        self._defaults = frozenset(self._data.keys())
        self._next_custom_Z: int = _CUSTOM_Z_START

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def __getitem__(self, symbol: str) -> Dict:
        """Return entry dict for *symbol*: ``{'Z': int, 'mass': float, 'custom': bool}``."""
        if symbol not in self._data:
            raise KeyError(
                f"Element or component {symbol!r} not in periodic table.\n"
                f"Use periodic_table.update_components({symbol!r}, mass) to add it.")
        return dict(self._data[symbol])  # return a copy

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate symbols in Z order (defaults first, then custom by Z)."""
        return iter(sorted(self._data, key=lambda s: self._data[s]['Z']))

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        n_custom = sum(1 for e in self._data.values() if e['custom'])
        return (f"PeriodicTable({len(self._defaults)} elements"
                f"{f', {n_custom} custom' if n_custom else ''})")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def mass(self, symbol: str) -> float:
        """Return atomic/component mass in g/mol."""
        return self[symbol]['mass']

    def atomic_number(self, symbol: str) -> int:
        """Return atomic number (Z) for *symbol*."""
        return self[symbol]['Z']

    def symbol(self, Z: int) -> str:
        """Return symbol for atomic number *Z*."""
        if Z not in self._z_to_sym:
            raise KeyError(f"No element with Z={Z} in periodic table.")
        return self._z_to_sym[Z]

    def is_default(self, symbol: str) -> bool:
        """Return True if *symbol* is one of the 118 standard elements."""
        return symbol in self._defaults

    def is_custom(self, symbol: str) -> bool:
        """Return True if *symbol* was added via :meth:`update_component`."""
        return symbol in self._data and self._data[symbol]['custom']

    # ------------------------------------------------------------------
    # Custom component management
    # ------------------------------------------------------------------

    def update_components(self, symbol: str, mass: float) -> None:
        """
        Add a custom component or update its mass.

        - If *symbol* is a new component: it is added with an auto-assigned Z
          (starting at 1000, incrementing for each custom addition).
        - If *symbol* already exists as a **custom** component: its mass is
          updated; Z is unchanged.
        - If *symbol* is a **default** element: only its mass is updated;
          Z cannot be changed for defaults.

        Parameters
        ----------
        symbol : str
            Component symbol, e.g. ``'D'``, ``'Lig'``, ``'Me'``.
            Must be a non-empty string.
        mass : float
            Molar mass in g/mol.  Must be positive.

        Examples
        --------
        >>> periodic_table.update_components('D', 2.014)   # deuterium
        >>> periodic_table.update_components('Lig', 320.0) # custom ligand
        >>> periodic_table.update_components('H', 1.0079)  # refine H mass
        """
        symbol = str(symbol).strip()
        if not symbol:
            raise InputError("Component symbol must be a non-empty string.")
        mass = float(mass)
        if mass <= 0:
            raise InputError(
                f"Mass must be positive, got {mass} for {symbol!r}.")

        if symbol in self._data:
            # update mass only
            self._data[symbol]['mass'] = mass
        else:
            # new custom component
            Z = self._next_custom_Z
            self._next_custom_Z += 1
            entry = {'Z': Z, 'mass': mass, 'custom': True}
            self._data[symbol] = entry
            self._z_to_sym[Z] = symbol

    def remove_components(self, symbol: str) -> None:
        """
        Remove a custom component.

        Default elements (Z 1–118) cannot be removed.  After removal, the Z
        values of all higher custom components are decremented by 1 so the
        custom Z range stays contiguous.

        Parameters
        ----------
        symbol : str
            Symbol of the custom component to remove.

        Raises
        ------
        InputError
            If *symbol* does not exist or is a default element.

        Examples
        --------
        >>> periodic_table.remove_components('D')
        """
        symbol = str(symbol).strip()
        if symbol not in self._data:
            raise InputError(
                f"{symbol!r} is not in the periodic table.")
        if symbol in self._defaults:
            raise InputError(
                f"{symbol!r} is a default element and cannot be removed.\n"
                f"You can update its mass with update_components({symbol!r}, new_mass).")

        removed_Z = self._data[symbol]['Z']
        del self._data[symbol]
        del self._z_to_sym[removed_Z]

        # shift down all custom Z values above the removed one
        to_shift = sorted(
            [(Z, sym) for Z, sym in self._z_to_sym.items()
             if Z > removed_Z and self._data[sym]['custom']],
            key=lambda x: x[0]
        )
        for old_Z, sym in to_shift:
            new_Z = old_Z - 1
            self._data[sym]['Z'] = new_Z
            del self._z_to_sym[old_Z]
            self._z_to_sym[new_Z] = sym

        self._next_custom_Z -= 1

    # ------------------------------------------------------------------
    # Bulk views
    # ------------------------------------------------------------------

    @property
    def elements(self) -> Dict[str, Dict]:
        """All 118 default elements as {symbol: entry}."""
        return {sym: dict(e) for sym, e in self._data.items()
                if not e['custom']}

    @property
    def custom_components(self) -> Dict[str, Dict]:
        """All user-defined components as {symbol: entry}."""
        return {sym: dict(e) for sym, e in self._data.items()
                if e['custom']}

    def molar_mass(self, species_components: Dict[str, float]) -> Optional[float]:
        """
        Compute molar mass (g/mol) from a components dict
        ``{element_symbol: count}``.

        Returns None if any component symbol is not in the table.

        Parameters
        ----------
        species_components : dict
            As returned by :func:`efta.species.components`.

        Examples
        --------
        >>> from efta.species import components
        >>> periodic_table.molar_mass(components('H2SO4'))
        98.079
        """
        total = 0.0
        for sym, count in species_components.items():
            if sym not in self._data:
                return None
            total += self._data[sym]['mass'] * count
        return total


#: Module-level singleton.  Import and use directly::
#:
#:     from efta import periodic_table
#:     periodic_table.mass('Fe')   # 55.845
periodic_table = PeriodicTable()
