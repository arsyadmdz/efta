"""
efta.errors
===========
Exception hierarchy for the efta package.

All efta-specific exceptions inherit from :exc:`EftaError` so callers can
catch the entire family with a single ``except EftaError`` clause.

Hierarchy
---------
::

    EftaError
    ├── SpeciesError          – species name parsing / unknown species
    ├── ReactionError         – malformed or inconsistent reaction definition
    │   └── BalanceError      – stoichiometry cannot be balanced / is inconsistent
    ├── InputError            – bad values passed to a public function
    │   └── ConcentrationError– negative, missing, or physically impossible concentration
    └── ConvergenceError      – solver failed to converge
        └── ConvergenceWarning– solver converged but residual is above threshold
                                (raised as a warning, not an exception, by default)
"""

from __future__ import annotations

import warnings


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class EftaError(Exception):
    """Base class for all efta exceptions."""


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

class SpeciesError(EftaError):
    """
    Raised when a species name cannot be parsed or is used inconsistently.

    Examples
    --------
    - Unbalanced parentheses in a formula string
    - A species referenced in ``c0`` that does not appear in any reaction
    - A species name that contains illegal characters
    """


# ---------------------------------------------------------------------------
# Reaction
# ---------------------------------------------------------------------------

class ReactionError(EftaError):
    """
    Raised when a reaction is malformed or internally inconsistent.

    Examples
    --------
    - No species on one side of the arrow
    - K ≤ 0
    - Duplicate species on the same side
    - Adding two reactions that cancel all species
    """


class BalanceError(ReactionError):
    """
    Raised when automatic balancing of a reaction fails.

    Examples
    --------
    - The stoichiometry matrix is rank-deficient (under-determined)
    - No integer solution exists within the search bounds
    - The null-space has dimension > 1 (ambiguous balance)
    """


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class InputError(EftaError):
    """
    Raised when a public function receives an invalid argument.

    Examples
    --------
    - ``init_scale`` outside (0, 1]
    - ``type`` argument not in the allowed set
    - ``maintain`` references a species not in the system
    - ``O/A`` ratio ≤ 0 in a mixed-phase system
    """


class ConcentrationError(InputError):
    """
    Raised when an initial concentration is physically impossible.

    Examples
    --------
    - A negative initial concentration
    - A required species missing from ``c0`` with no sensible default
    - Total mass in a cluster is zero or negative before solving
    """


# ---------------------------------------------------------------------------
# Solver convergence
# ---------------------------------------------------------------------------

class ConvergenceError(EftaError):
    """
    Raised when all solver methods (L, A, B, DE) fail to converge.

    Attributes
    ----------
    residual : float
        The best max-absolute-residual achieved before giving up.
    method : str
        The last solver method attempted (e.g. ``'DE3'``).
    system_info : str
        A human-readable description of the reaction system that failed.
    """

    def __init__(self, message: str, residual: float = float('inf'),
                 method: str = '', system_info: str = ''):
        super().__init__(message)
        self.residual    = residual
        self.method      = method
        self.system_info = system_info

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.residual < float('inf'):
            parts.append(f"best residual: {self.residual:.3e}")
        if self.method:
            parts.append(f"last method: {self.method}")
        if self.system_info:
            parts.append(f"system: {self.system_info}")
        return "\n  ".join(parts)


class ConvergenceWarning(UserWarning):
    """
    Issued when the solver returns a result but the residual exceeds
    ``tolerance``.  The returned concentrations may be inaccurate.

    Use ``warnings.filterwarnings('error', category=efta.ConvergenceWarning)``
    to turn this into a hard error.
    """


def warn_convergence(residual: float, tolerance: float,
                     method: str = '', system_info: str = '') -> None:
    """Issue a :exc:`ConvergenceWarning` with a standardised message."""
    msg = (
        f"Solver returned a result but residual {residual:.3e} exceeds "
        f"tolerance {tolerance:.3e}"
    )
    if method:
        msg += f" (method: {method})"
    if system_info:
        msg += f"\n  system: {system_info}"
    warnings.warn(msg, ConvergenceWarning, stacklevel=3)
