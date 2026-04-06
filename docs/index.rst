efta — Equilibrium Formulation and Thermodynamic Analysis
==========================================================

**efta** is a Python library for solving chemical equilibrium problems —
from simple acid-base dissociation to complex multi-phase solvent extraction systems.

.. code-block:: bash

   pip install efta

----

Quick Start
-----------

.. code-block:: python

   from efta import reaction, reactions
   import numpy as np

   # Define equilibrium reactions
   r1 = reaction('CH3COOH = CH3COO[-] + H[+]', 1.8e-5)   # acetic acid Ka
   r2 = reaction('H2O = H[+] + OH[-]',          1e-14)    # water Kw

   # Build a coupled system and solve
   sys = reactions(r1, r2)
   c_eq = sys.equilibrium({
       'CH3COOH':    0.1,
       'CH3COO[-]':  0.0,
       'H[+]':       1e-7,
       'OH[-]':      1e-7,
       'H2O':        1.0,
   })

   print(f"pH = {-np.log10(c_eq['H[+]']):.2f}")   # pH ≈ 2.87

----

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

----

Features
--------

- **Flexible reaction input** — strings, dicts, or coefficient–name pairs
- **Multi-reaction systems** — solve coupled equilibria simultaneously
- **Speciation and precipitation** — Ka, Kb, Kf, and Ksp reactions
- **Solvent extraction** — single-stage and multistage circuits
- **Activity coefficients** — Davies, Debye-Hückel, or custom gamma functions
- **Parameter fitting** — fit K values to experimental data with Monte Carlo uncertainty
- **Plotting** — concentration profiles and speciation fraction diagrams

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
