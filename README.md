<p align="center">
  <img src="https://github.com/arsyadmdz/efta/blob/main/docs/_static/logo.png" width="200" alt="efta logo">
</p>

<h1 align="center">efta — Equilibrium Formulation API</h1>

<p align="center">
  <a href="https://pypi.org/project/efta/"><img src="https://badge.fury.io/py/efta.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://efta.readthedocs.io"><img src="https://readthedocs.org/projects/efta/badge/?version=latest" alt="Documentation"></a>
</p>

**efta** is a Python library for solving chemical equilibrium problems — from simple acid-base dissociation to complex multi-phase solvent extraction systems.

---

## Features

- **Flexible reaction input** — define reactions as strings (`'Fe[3+] + 3OH[-] = Fe(OH)3(s)'`), dicts, or coefficient–name pairs
- **Multi-reaction systems** — solve coupled equilibria simultaneously using four built-in numerical methods
- **Speciation and precipitation** — handles both aqueous speciation (Ka, Kb, Kf) and solubility products (Ksp)
- **Solvent extraction** — single-stage and multistage counter-current / cross-current extraction circuits
- **Activity coefficients** — plug in Davies, Debye-Hückel, or any custom gamma function
- **Parameter fitting** — fit equilibrium constants to measured concentration data, with Monte Carlo uncertainty analysis
- **Plotting** — concentration profiles, speciation fraction diagrams, and extraction stage profiles via matplotlib

---

## Installation

```bash
pip install efta
```

Requires Python 3.10+ and NumPy, SciPy, matplotlib (installed automatically).

---

## Quick Start

### Acid-Base Equilibrium

```python
from efta import reaction, reactions

# Define individual equilibrium reactions with their equilibrium constants
acetic_acid  = reaction('CH3COOH = CH3COO[-] + H[+]', 1.8e-5)   # Ka of acetic acid
water_autoion = reaction('H2O = H[+] + OH[-]',          1e-14)    # Kw of water

# Combine reactions into a system and solve
sys = reactions(acetic_acid, water_autoion)

# Provide initial concentrations (mol/L) for all species
c_eq = sys.equilibrium({
    'CH3COOH':    0.1,    # 0.1 M acetic acid
    'CH3COO[-]':  0.0,
    'H[+]':       1e-7,   # neutral pH starting guess
    'OH[-]':      1e-7,
    'H2O':        1.0,
})

import numpy as np
print(f"pH = {-np.log10(c_eq['H[+]']):.2f}")   # → pH ≈ 2.87
```

### Precipitation / Solubility

```python
from efta import reaction, reactions

# ksp=True tells efta this is a solubility product reaction
calcite = reaction('CaCO3(s) = Ca[2+] + CO3[2-]', 3.36e-9, ksp=True)

sys = reactions(calcite)
c_eq = sys.equilibrium({'CaCO3(s)': 1.0, 'Ca[2+]': 0.0, 'CO3[2-]': 0.0})

print(f"[Ca²⁺] = {c_eq['Ca[2+]']:.2e} mol/L")
```

### Concentration Sweeps and Plotting

```python
import numpy as np

# Sweep initial [CH3COOH] from 1e-4 to 1.0 M on a log scale
fig, ax = sys.plot(
    {'CH3COOH': [1e-4, 1.0], 'H[+]': 1e-7, 'OH[-]': 1e-7, 'H2O': 1.0},
    sweep='CH3COOH',
    logx=True,
    logy=True,
    n_points=60,
)
```

---

## Species Notation

efta uses a compact notation for chemical species:

| Notation | Meaning | Example |
|---|---|---|
| `Fe[3+]` | Fe with charge 3+ | ferric iron |
| `OH[-]` | hydroxide | |
| `Fe(OH)3(s)` | solid phase | ferric hydroxide precipitate |
| `H2A2(org)` | organic phase | di-2-ethylhexylphosphoric acid |
| `e[-]` | electron | for redox reactions |
| `$(1/3)` | fractional coefficient | `$(1/3)Fe[3+]` |

---

## Reaction Construction

Four equivalent ways to define the same reaction:

```python
from efta import reaction

# 1. String (most readable)
r = reaction('Fe[3+] + 3OH[-] = Fe(OH)3(s)', 1e3)

# 2. Stoichiometry dict  (negative = reactant, positive = product)
r = reaction({'Fe[3+]': -1, 'OH[-]': -3, 'Fe(OH)3(s)': 1}, 1e3)

# 3. Separate reactant and product dicts
r = reaction({'Fe[3+]': 1, 'OH[-]': 3}, {'Fe(OH)3(s)': 1}, 1e3)

# 4. (coefficient, name) pairs — last argument is K
r = reaction((-1, 'Fe[3+]'), (-3, 'OH[-]'), (1, 'Fe(OH)3(s)'), 1e3)
```

### Combining Reactions

Reactions can be added and scaled. K values update automatically:

```python
# Adding two reactions combines their stoichiometries; K values multiply
r_combined = r1 + r2

# Scaling multiplies all coefficients; K is raised to that power
r_half = r1 / 2       # divide all coefficients by 2  →  K becomes √K
r_rev  = r1 * -1      # reverse the reaction          →  K becomes 1/K
```

---

## The `reactions` System

A `reactions` object holds multiple coupled reactions and provides the solver interface:

```python
from efta import reaction, reactions

sys = reactions(r1, r2, r3)

# --- Solve for equilibrium concentrations ---
c_eq = sys.equilibrium({'Fe[3+]': 0.01, 'OH[-]': 1e-7, ...})

# --- Inspect which species are in the system ---
print(sys.species)           # frozenset of all species names
print(sys.aqueous_species)   # aqueous species only
print(sys.organic_species)   # organic-phase species only

# --- Plot a concentration sweep ---
fig, ax = sys.plot({'Fe[3+]': [1e-5, 0.1], ...}, sweep='Fe[3+]', logx=True)

# --- Inverse solve: find initial [X] that gives a target equilibrium ---
c_target = sys.find(
    unknown='NaOH',
    c0={'NaOH': 0.0, 'H[+]': 1e-7, ...},
    target={'H[+]': 1e-8},    # target pH 8
)
```

---

## The `solution` Class

A `solution` pairs a concentration dict with a volume, and provides convenient access methods:

```python
from efta import solution

sol = solution({'H[+]': 1e-4, 'OH[-]': 1e-10, 'H2O': 1.0}, volume=0.5)

sol['H[+]']          # 1e-4  — species concentration in mol/L
sol.pH               # 4.0   — convenience property
sol.ionic_strength   # mol/L
sol.moles('H[+]')    # mol = concentration × volume
sol.mass('H2O')      # g    = moles × molar mass

sol.aqueous          # dict of aqueous-phase species only
sol.organic          # dict of organic-phase species only
sol.solid            # dict of solid-phase species only

sol_2L = sol(2.0)    # clone with 2 L volume — moles are preserved
mixed  = sol1 + sol2  # mix two solutions (moles add, volumes add)
```

### Creating a `solution` directly from `reactions`

```python
sol = sys.solution({'CH3COOH': 0.1, 'H2O': 1.0}, volume=1.0)
# Returns a solution object at equilibrium
```

---

## Activity Coefficients (Non-Ideal Systems)

Register a gamma function for any species in a reaction. The solver calls it at each iteration and adjusts K accordingly:

```python
import math

# Davies equation activity coefficient (depends on ionic strength I)
def davies(I):
    sqI = math.sqrt(I)
    return 10 ** (-0.509 * 3**2 * (sqI / (1 + sqI) - 0.3 * I))

# 'I' is a special token — efta computes ionic strength and passes it to davies()
rxn.set_gamma('Fe[3+]', (davies, 'I'))

# Gamma depending on another species' concentration
rxn.set_gamma('Fe[3+]', (lambda c_cl: 1 - 0.1 * c_cl, 'Cl[-]'))

# Constant gamma (no dependencies)
rxn.set_gamma('Fe[3+]', (lambda: 0.5,))
```

---

## Solvent Extraction

efta includes a full solvent extraction module for modelling liquid–liquid extraction processes.

```python
from efta import reaction, solution
from efta.solventextraction import (
    sx, countercurrent, distribution_coef, separation_factor, splitter
)

# Extraction reaction: metal transfers from aqueous to organic phase
rxn  = reaction('LaCl[2+] + 3H2A2(org) = LaClA2(HA)4(org) + 2H[+]', 10.6)

feed = solution({'LaCl[2+]': 0.003, 'H[+]': 0.3}, volume=1.0)  # aqueous feed
org  = solution({'H2A2(org)': 0.25},               volume=1.0)  # organic phase

# Single-stage extraction
stage = sx(rxn, feed, org)
stage.run()
extract, raffinate = stage.outlets[0], stage.outlets[1]

D    = stage.distribution_coef('La')       # D = [La]_org / [La]_aq
beta = stage.separation_factor('La', 'Ce') # β = D(La) / D(Ce)

# 5-stage counter-current extraction circuit
ms = countercurrent(rxn, stages=5, feed=feed, organic=org)
ms.run()                    # solve all stages at equilibrium
ms.run(efficiency=0.85)     # with stage efficiency < 1

raffinate = ms.outlets[5]   # aqueous exit after stage 5
extract   = ms.outlets[1]   # organic exit after stage 1

# Plot concentration profile across stages
ms.plot(['La', 'Ce'], phase='aq')

# Reflux design with a flow splitter
split = splitter(1, 2)      # splits flow: 1/3 reflux, 2/3 forward
reflux, forward = split(extract)
```

Available multistage topologies:

| Function | Description |
|---|---|
| `countercurrent` | Aqueous feeds stage 1→n, organic feeds stage n→1 |
| `crosscurrent` | Aqueous feeds stage 1→n, fresh organic at every stage |
| `strip_countercurrent` | Organic 1→n, aqueous n→1 (stripping mode) |
| `strip_crosscurrent` | Organic 1→n, fresh aqueous at every stage |

---

## Parameter Fitting

Fit unknown equilibrium constants to experimental data:

```python
from efta import freaction, freactions, model, analyze

# $(x1) is a free parameter — efta will optimise it
r_fit = freaction('Fe[3+] + 3OH[-] = Fe(OH)3(s)', '$(x1)', ksp=True)

# Experimental data: list of {species: measured_concentration} dicts
data = [
    {'Fe[3+]': 1e-5, 'OH[-]': 1e-3},
    {'Fe[3+]': 2e-5, 'OH[-]': 5e-4},
    # ...
]

best_fit = model(r_fit, data)
print(f"Best log K = {best_fit.logK:.2f}")

# Bootstrap uncertainty analysis
result = analyze(r_fit, data, n_bootstrap=200)
print(f"log K = {result.logK_mean:.2f} ± {result.logK_std:.2f}")
```

---

## Module Reference

| Module | Description |
|---|---|
| `efta.reaction` | `reaction` class — single equilibrium reaction |
| `efta.reactions` | `reactions` class — coupled system and solver |
| `efta.species` | Species name parsing: `species()`, `formula()`, `charge()`, `components()` |
| `efta.solution` | `solution` class — composition + volume |
| `efta.mixture` | `mixture` class — ordered collection of solutions |
| `efta.balance` | Cluster detection and conservation-law analysis |
| `efta.system` | System assembly, activity coefficients, extent↔concentration |
| `efta.solver` | Numerical solvers (Method L, A, B, DE) |
| `efta.model` | Parameter fitting, Monte Carlo analysis |
| `efta.plotting` | Concentration plots, speciation diagrams, `style` singleton |
| `efta.styling` | Runtime palette and font-size helpers |
| `efta.periodic_table` | Atomic masses and element lookup |
| `efta.solventextraction.sx` | Single-stage extraction |
| `efta.solventextraction.multistage` | Multistage extraction circuits |
| `efta.solventextraction.units` | `splitter` flow-splitter unit |

---

## Error Handling

All efta exceptions inherit from `EftaError`, so you can catch the entire family with a single clause:

```python
from efta import EftaError, ConvergenceError, ConvergenceWarning
import warnings

# Turn convergence warnings into hard errors (useful during debugging)
warnings.filterwarnings('error', category=ConvergenceWarning)

try:
    c_eq = sys.equilibrium(c0)
except ConvergenceError as e:
    print(f"Solver failed — best residual: {e.residual:.2e}")
except EftaError as e:
    print(f"efta error: {e}")
```

| Exception | Raised when |
|---|---|
| `SpeciesError` | Species name cannot be parsed |
| `ReactionError` | Reaction is malformed (bad K, empty side, …) |
| `BalanceError` | Automatic balancing fails |
| `InputError` | Invalid argument passed to a function |
| `ConcentrationError` | Negative or missing initial concentration |
| `ConvergenceError` | All solver methods fail to converge |
| `ConvergenceWarning` | Solver returns a result but residual exceeds tolerance |

---

## License

MIT — see [LICENSE](LICENSE) for details.
