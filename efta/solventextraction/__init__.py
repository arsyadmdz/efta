"""
efta.solventextraction
======================
Solvent extraction modelling — single-stage and multistage.

Submodules
----------
sx
    Single-stage batch equilibrium extraction unit.
multistage
    Multistage extraction with arbitrary flow topology.
units
    Process unit helpers (``separator`` flow splitter).

Classes / functions exported at this level
------------------------------------------
sx
    Single-stage extraction.  ``sx(reactions, *solutions).run(efficiency)``
    → returns *self*; outlets at ``stage.outlets``.

multistage
    Multistage system.  ``multistage(reactions, aq_streams, org_streams)``
    ``.run(efficiency, iter)`` → returns *self*.

countercurrent(reactions, stages, feed, organic)
    Standard counter-current topology: aq 1 → n, org n → 1.

crosscurrent(reactions, stages, feed, organic)
    Cross-current topology: aq 1 → n, fresh org at every stage.

strip_countercurrent(reactions, stages, organic, feed)
    Counter-current stripping: org 1 → n, aq n → 1.

strip_crosscurrent(reactions, stages, organic, feed)
    Cross-current stripping: org 1 → n, fresh aq at every stage.

multiphase(*solutions)
    Blend solutions then split into mixture(organic, aqueous).

distribution_coef(component, mixture)
    D = org_concentration / aq_concentration from a two-phase mixture.

separation_factor(component1, component2, mixture)
    β = D(component1) / D(component2) from a two-phase mixture.

separator(*fractions)
    Volumetric flow splitter; enables reflux circuit design.

Quick start
-----------
::

    from efta import reaction, solution
    from efta.solventextraction import (
        sx, countercurrent, distribution_coef, separation_factor, separator,
    )

    rxn  = reaction('LaCl[2+] + 3H2A2(org) = LaClA2(HA)4(org) + 2H[+]', 10.6)
    feed = solution({'LaCl[2+]': 0.003, 'H[+]': 0.3}, volume=1.0)
    org  = solution({'H2A2(org)': 0.25},               volume=1.0)

    # -- single stage ---------------------------------------------------------
    stage = sx(rxn, feed, org)
    stage.run()
    extract, raffinate = stage.outlets[0], stage.outlets[1]
    D    = stage.distribution_coef('La')
    beta = stage.separation_factor('La', 'Ce')

    # -- 5-stage counter-current ----------------------------------------------
    ms = countercurrent(rxn, stages=5, feed=feed, organic=org)
    ms.run()                          # default efficiency=1.0
    ms.run(0.85)                      # with efficiency
    raffinate = ms.outlets[5]         # aqueous exit at stage 5
    extract   = ms.outlets[1]         # organic exit at stage 1
    ms.plot(['La', 'Ce'], phase='aq') # concentration profile

    # -- reflux with separator ------------------------------------------------
    splitter = separator(1, 2)        # 1/3 reflux, 2/3 forward
    m = splitter(extract)
    reflux, forward = m[0], m[1]
"""

from .sx         import sx, multiphase, distribution_coef, separation_factor
from .multistage import (
    multistage,
    countercurrent,
    crosscurrent,
    strip_countercurrent,
    strip_crosscurrent,
)
from .units      import splitter

__all__ = [
    # Single-stage
    'sx',
    'multiphase',
    'distribution_coef',
    'separation_factor',
    # Multistage
    'multistage',
    'countercurrent',
    'crosscurrent',
    'strip_countercurrent',
    'strip_crosscurrent',
    # Units
    'splitter',
]
