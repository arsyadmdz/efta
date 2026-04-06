"""
efta.model
==========
Reaction modelling and parameter fitting.

User-facing API lives on :class:`freaction` and :class:`freactions`:

- ``.fit(equilibrium, method='cont'|'ranged', ...)``  — single best fit
- ``.analyze(equilibrium, method='cont'|'ranged', ...)``  — bootstrap distribution

Classes
-------
:class:`Model`
    Result of a single fit run.
:class:`Analyzed`
    Result of one bootstrap iteration from ``.analyze()``.
:class:`freaction`
    Parameterised reaction with ``$(xN)`` placeholders.
:class:`freactions`
    Collection of freaction/reaction sharing a parameter space.
"""

from .fitting     import model, analyze, montecarlo, Model
from .freaction   import freaction, freactions
from .suggest      import _suggest, _suggest_chem, Analyzed
from .distribution import distribution
from .mass_action import mass_action_residuals, is_data_sufficient
from .ga           import GA, GASolution

__all__ = ['model', 'analyze', 'montecarlo', 'Model', '', 'GA', 'GASolution',
           'freaction', 'freactions',
           'Analyzed',
           'mass_action_residuals', 'is_data_sufficient',
           'distribution']
