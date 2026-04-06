"""
efta.solver
===========
Numerical equilibrium solvers.  This subpackage is internal — end users
should call :meth:`~efta.reactions.reactions.equilibrium` instead.

Four methods are tried in sequence, from fastest to most robust:

Method L (log-space Newton)
    Newton's method operating in log₁₀(c) space.  Fastest for well-posed
    systems with a reasonable initial guess.  Uses the cluster mass-balance
    as constraints.  Applied first via a fast pre-solver that solves reactions
    sequentially, then globally.

Method A (extent-of-reaction)
    ``scipy.optimize.fsolve`` / ``least_squares`` on the extent-of-reaction
    vector ξ.  Reliable for most speciation problems.

Method B (log-space fsolve)
    B1: ``fsolve`` in log₁₀(c) space.
    B2: ``fsolve`` in concentration space as fallback.

Method DE (differential evolution)
    Global stochastic search in log₁₀(c) space using
    ``scipy.optimize.differential_evolution``.  Slow but can find solutions
    where the local methods all fail.

Implementation layout
---------------------
- ``_shared``    – timeout guard, constraint builders, log-space system assembly
- ``method_l``   – Method L: Newton in log₁₀(c) space
- ``method_a``   – Method A: extent-of-reaction (fsolve / least_squares)
- ``method_b``   – Method B: fsolve in log-space (B1) and concentration-space (B2)
- ``method_de``  – Method DE: differential evolution global search
- ``dispatch``   – high-level dispatch (_solve_extents, _solve_with_ksp, …)
- ``find``       – inverse solver (_find: Brent root-finding on equilibrium)
"""

from ._shared import (
    _TimeoutError,
    _TimedCall,
    _reactions_type,
    _build_ksp_stoich_constraints,
    _build_stoich_symmetry_constraints,
    _build_logspace_system,
    _build_B_residual_system,
    _build_logspace_and_constraints,
    _logspace_initial_guesses,
    _compute_extent_bounds,
    _clamp_ksp_solids,
)
from .method_l  import _solve_method_L
from .method_a  import _solve_method_A
from .method_b  import _solve_method_B, _method_B1, _method_B2
from .method_de import _solve_method_DE
from .dispatch  import (
    _solve_extents,
    _pre_solve_speciation,
    _solve_precipitation,
    _solve_mix,
    _solve_with_ksp,
)

__all__ = [
    '_TimeoutError', '_TimedCall', '_reactions_type',
    '_build_ksp_stoich_constraints', '_build_stoich_symmetry_constraints',
    '_build_logspace_system', '_build_B_residual_system',
    '_build_logspace_and_constraints', '_logspace_initial_guesses',
    '_compute_extent_bounds', '_clamp_ksp_solids',
    '_solve_method_L',
    '_solve_method_A',
    '_solve_method_B', '_method_B1', '_method_B2',
    '_solve_method_DE',
    '_solve_extents', '_pre_solve_speciation',
    '_solve_precipitation', '_solve_mix', '_solve_with_ksp',
]

from .find import _find