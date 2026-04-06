"""
efta – Equilibrium Formulation and Thermodynamic Analysis
==========================================================

A Python library for solving chemical equilibrium problems.

Quick start
-----------
::

    from efta import reaction, reactions

    # Acetic acid dissociation
    r1 = reaction('CH3COOH = CH3COO[-] + H[+]', 1.8e-5)

    # Water autoionisation
    r2 = reaction('H2O = H[+] + OH[-]', 1e-14)

    sys = reactions(r1, r2)

    # Equilibrium concentrations for 0.1 M acetic acid solution
    c_eq = sys.equilibrium({
        'CH3COOH': 0.1,
        'CH3COO[-]': 0.0,
        'H[+]': 1e-7,
        'OH[-]': 1e-7,
        'H2O': 1.0,
    })

    print(f"pH = {-np.log10(c_eq['H[+]']):.2f}")

Module overview
---------------
+------------------------+--------------------------------------------------+
| Module                 | Contents                                         |
+========================+==================================================+
| species                | Name normalisation, formula, charge, components |
+------------------------+--------------------------------------------------+
| reaction               | The ``reaction`` class                           |
+------------------------+--------------------------------------------------+
| reactions              | The ``reactions`` class (collection / solver)   |
+------------------------+--------------------------------------------------+
| balance                | Cluster detection, conservation-law analysis    |
+------------------------+--------------------------------------------------+
| system                 | System assembly helpers, activity coefficients  |
+------------------------+--------------------------------------------------+
| solver                 | Numerical solvers (L, A, B, DE)                 |
+------------------------+--------------------------------------------------+
| plotting               | Visualisation; the ``style`` singleton          |
+------------------------+--------------------------------------------------+
| solution               | The ``solution`` class                          |
+------------------------+--------------------------------------------------+
| mixture                | The ``mixture`` class                           |
+------------------------+--------------------------------------------------+
| solventextraction.sx   | ``sx``, ``multiphase``, D/beta functions        |
+------------------------+--------------------------------------------------+
| solventextraction.     | ``multistage``, ``countercurrent``,             |
| multistage             | ``crosscurrent``, ``strip_countercurrent``,     |
|                        | ``strip_crosscurrent``                          |
+------------------------+--------------------------------------------------+
| solventextraction.     | ``splitter``                                   |
| units                  |                                                 |
+------------------------+--------------------------------------------------+

Public API
----------
The following names are exported at the top level for convenience:

``reaction``, ``reactions`` -- the two main classes.

``r`` / ``R`` -- backward-compatible short aliases.

``species``, ``formula``, ``charge``, ``components``, ``phase``, ``total``
  -- species utility functions.

``style`` -- the :class:`~efta.plotting.PlotStyle` singleton.

``solution``, ``mixture`` -- solution and mixture classes.

``sx``, ``multistage`` -- single-stage and multistage solvent extraction.

``countercurrent``, ``crosscurrent``,
``strip_countercurrent``, ``strip_crosscurrent``
  -- multistage topology constructors.

``distribution_coef``, ``separation_factor``, ``multiphase``
  -- standalone SX metric and utility functions.

``splitter`` -- volumetric flow splitter for reflux circuit design.
"""

# -- core chemistry ------------------------------------------------------------
from .species       import species, formula, charge, components, phase
from .species       import construct, species_std
from .reaction      import reaction
from .reactions     import reactions
from .system        import total

# -- fitting / modelling -------------------------------------------------------
from .model         import (
    model, analyze, montecarlo, Model, Analyzed,
    distribution, freaction, freactions,
)

# -- solution and mixture ------------------------------------------------------
from .solution      import solution
from .mixture       import mixture

# -- solvent extraction --------------------------------------------------------
from .solventextraction.sx         import (
    sx,
    multiphase,
    distribution_coef,
    separation_factor,
)
from .solventextraction.multistage import (
    multistage,
    countercurrent,
    crosscurrent,
    strip_countercurrent,
    strip_crosscurrent,
)
from .solventextraction.units      import splitter

# -- utilities -----------------------------------------------------------------
from .periodic_table import periodic_table, PeriodicTable

# -- visualisation and styling -------------------------------------------------
from .plotting  import style
from .styling   import (
    randomize_linestyle, randomize_color, randomize_pattern,
    coloring, PRESETS,
    fontsize, legend_fontsize,
    x_fontsize, x_tick_fontsize, x_title_fontsize,
    y_fontsize, y_tick_fontsize, y_title_fontsize,
)

# -- errors --------------------------------------------------------------------
from .errors import (
    EftaError, SpeciesError, ReactionError, BalanceError,
    InputError, ConcentrationError, ConvergenceError, ConvergenceWarning,
)

# -- backward-compatible short aliases -----------------------------------------
r = reaction
R = reactions

__all__ = [
    # Core chemistry
    "reaction",
    "reactions",
    "r",
    "R",
    # Species utilities
    "species",
    "formula",
    "charge",
    "components",
    "phase",
    "construct",
    "species_std",
    # Mass balance
    "total",
    # Fitting / modelling
    "model",
    "analyze",
    "montecarlo",
    "Model",
    "Analyzed",
    "distribution",
    "freaction",
    "freactions",
    # Solution and mixture
    "solution",
    "mixture",
    # Solvent extraction -- single stage
    "sx",
    "multiphase",
    "distribution_coef",
    "separation_factor",
    # Solvent extraction -- multistage
    "multistage",
    "countercurrent",
    "crosscurrent",
    "strip_countercurrent",
    "strip_crosscurrent",
    # Solvent extraction -- units
    "splitter",
    # Utilities
    "periodic_table",
    "PeriodicTable",
    # Visualisation
    "style",
    # Styling helpers
    "randomize_linestyle",
    "randomize_color",
    "randomize_pattern",
    "coloring",
    "PRESETS",
    "fontsize",
    "legend_fontsize",
    "x_fontsize",
    "x_tick_fontsize",
    "x_title_fontsize",
    "y_fontsize",
    "y_tick_fontsize",
    "y_title_fontsize",
    # Errors
    "EftaError",
    "SpeciesError",
    "ReactionError",
    "BalanceError",
    "InputError",
    "ConcentrationError",
    "ConvergenceError",
    "ConvergenceWarning",
]

try:
    from importlib.metadata import version as _meta_version
    __version__: str = _meta_version("efta")
except Exception:
    __version__ = "1.0.0"  # fallback when not installed via pip
