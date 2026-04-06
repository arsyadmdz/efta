"""
efta.solventextraction.multistage
==================================
Counter-current and cross-current multistage solvent extraction.

Stage indices are **1-based**: the first stage is stage 1, the last is
stage *n*.  All flow-series lists and ``ms[i]`` indexing use 1-based
integers.

Convenience constructors
------------------------
:func:`countercurrent`       — aq 1→n, org n→1  (classic extraction)
:func:`crosscurrent`         — aq 1→n, fresh org at every stage
:func:`strip_countercurrent` — org 1→n, aq n→1  (classic stripping)
:func:`strip_crosscurrent`   — org 1→n, fresh aq at every stage

Parameters
----------
reactions : reaction | reactions
    Equilibrium reaction(s) shared by all stages.
aq_streams : list of (solution, list[int])
    Aqueous feed streams, 1-based stage indices.
org_streams : list of (solution, list[int])
    Organic feed streams, 1-based stage indices.

Examples
--------
Classic counter-current 3-stage extraction:

>>> from efta import reaction, solution
>>> from efta.solventextraction import multistage, countercurrent
>>>
>>> rxn  = reaction('LaCl[2+] + 3H2A2(org) = LaClA2(HA)4(org) + 2H[+]', 10.6)
>>> feed = solution({'LaCl[2+]': 0.003, 'H[+]': 0.3}, volume=1.0)
>>> org  = solution({'H2A2(org)': 0.25},               volume=1.0)
>>>
>>> # via countercurrent helper
>>> ms = countercurrent(rxn, stages=3, feed=feed, organic=org)
>>>
>>> # or manually (1-based indices)
>>> ms = multistage(rxn,
...                 [(feed, [1, 2, 3])],   # aqueous: feed → 1 → 2 → 3
...                 [(org,  [3, 2, 1])])   # organic: feed → 3 → 2 → 1
>>>
>>> ms.run()                   # default efficiency=1.0, iter=n²=9
>>> ms.run(0.85)               # re-run at lower efficiency
>>> ms.run(0.9, iter=50)       # explicit iteration count
>>>
>>> ms[1]                      # sx object at stage 1 (read-only)
>>> ms[3]                      # sx object at stage 3 (read-only)
>>> ms.inlets                  # {1: [feed], 3: [org]}
>>> ms.outlets                 # {3: raffinate_sol, 1: extract_sol}
>>> ms.inflow                  # [1, 3]  — inflow stage indices
>>> ms.outflow                 # [1, 3]  — outflow stage indices
>>> ms.instages                # [sx at stage 1, sx at stage 3]
>>> ms.outstages               # [sx at stage 1, sx at stage 3]
>>> ms.stages                  # [sx, sx, sx]  — all stages in order
>>>
>>> ms.plot(['La', 'H'], phase='aq')   # aqueous concentration profile
>>> ms.plot(['La'], phase='org')       # organic concentration profile
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

from ..solution   import solution  as _solution
from ..mixture    import mixture   as _mixture
from ..reactions  import reactions as _reactions_cls
from ..reaction   import reaction  as _reaction_cls
from ..species    import is_organic as _is_org
from ..errors     import InputError
from .sx          import sx as _sx


__all__ = ['multistage', 'countercurrent', 'crosscurrent',
           'strip_countercurrent', 'strip_crosscurrent']

_StreamList = List[Tuple[_solution, List[int]]]


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_stream_list(streams: _StreamList,
                          label: str,
                          n_stages: int) -> None:
    """Raise InputError if *streams* is malformed (1-based indices)."""
    if not streams:
        raise InputError(f"{label} must be a non-empty list.")
    for i, item in enumerate(streams):
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise InputError(
                f"{label}[{i}]: expected (solution, list[int]), "
                f"got {type(item).__name__!r}.")
        sol, flow = item
        if not isinstance(sol, _solution):
            raise InputError(
                f"{label}[{i}]: first element must be a solution object.")
        flow = list(flow)
        if not flow:
            raise InputError(
                f"{label}[{i}]: flow_series must be a non-empty list.")
        if len(flow) != len(set(flow)):
            raise InputError(
                f"{label}[{i}]: flow_series contains duplicate stage indices.")
        bad = [idx for idx in flow if idx < 1 or idx > n_stages]
        if bad:
            raise InputError(
                f"{label}[{i}]: flow_series indices {bad} out of range "
                f"(valid: 1–{n_stages}, stages are 1-based).")


def _fmt_sol(sol) -> str:
    conc = sol.concentrations if hasattr(sol, 'concentrations') else {}
    parts = [f"{sp}={c:.3e}" for sp, c in conc.items() if c > 1e-15]
    return "{" + ", ".join(parts[:5]) + ("..." if len(parts) > 5 else "") + "}"


# ── main class ────────────────────────────────────────────────────────────────

class multistage:
    """
    Arbitrary-flow multistage solvent extraction system.

    Stage indices are **1-based**.  Stage count is inferred from the maximum
    stage index across all streams.  One :class:`~efta.solventextraction.sx.sx`
    object is constructed per stage on :meth:`__init__`.

    :meth:`run` wires the stages together, iterates the equilibrium loop,
    and returns *self*.

    Attributes
    ----------
    n_stages : int
    reactions : reactions
    aq_streams : list of (solution, list[int])
    org_streams : list of (solution, list[int])
    converged : bool
    n_iter : int
    efficiency : float or None
    """

    def __init__(self,
                 reactions:   Union[_reaction_cls, _reactions_cls],
                 aq_streams:  _StreamList,
                 org_streams: _StreamList):

        # ── normalise reactions ───────────────────────────────────────────
        if isinstance(reactions, _reaction_cls):
            reactions = _reactions_cls(reactions)
        elif not isinstance(reactions, _reactions_cls):
            raise InputError("reactions must be a reaction or reactions object.")
        self.reactions = reactions

        # ── infer stage count from 1-based indices ────────────────────────
        all_idx = (
            [idx for _, flow in aq_streams  for idx in flow] +
            [idx for _, flow in org_streams for idx in flow]
        )
        if not all_idx:
            raise InputError(
                "No stage indices found in aq_streams or org_streams.")
        if min(all_idx) < 1:
            raise InputError(
                f"Stage indices are 1-based; found index {min(all_idx)}.")
        n_stages = max(all_idx)

        # ── validate streams ──────────────────────────────────────────────
        _validate_stream_list(aq_streams,  'aq_streams',  n_stages)
        _validate_stream_list(org_streams, 'org_streams', n_stages)

        self.aq_streams:  _StreamList = [(sol, list(flow))
                                         for sol, flow in aq_streams]
        self.org_streams: _StreamList = [(sol, list(flow))
                                         for sol, flow in org_streams]
        self.n_stages  = n_stages

        # Run-state
        self.converged:  bool            = False
        self.n_iter:     int             = 0
        self.efficiency: Optional[float] = None

        # ── topology maps (1-based stage keys) ───────────────────────────
        # _aq_next[i]  = stage that receives sx[i]'s raffinate, or None
        # _org_next[i] = stage that receives sx[i]'s extract,   or None
        self._aq_next:  Dict[int, Optional[int]] = {i: None for i in range(1, n_stages + 1)}
        self._org_next: Dict[int, Optional[int]] = {i: None for i in range(1, n_stages + 1)}
        for _, flow in self.aq_streams:
            for pos in range(len(flow) - 1):
                self._aq_next[flow[pos]] = flow[pos + 1]
        for _, flow in self.org_streams:
            for pos in range(len(flow) - 1):
                self._org_next[flow[pos]] = flow[pos + 1]

        # ── external feed map (1-based) ───────────────────────────────────
        self._entry_feeds: Dict[int, List[_solution]] = {
            i: [] for i in range(1, n_stages + 1)
        }
        for sol, flow in self.aq_streams:
            self._entry_feeds[flow[0]].append(sol)
        for sol, flow in self.org_streams:
            self._entry_feeds[flow[0]].append(sol)

        # ── instantiate sx objects (stored internally 0-based list) ───────
        # Public access is via 1-based ms[i]; internally _stages[i-1].
        # Non-entry stages start as None; they are constructed in run()
        # the first time a transfer wires an inlet into them, so no
        # phantom volume corrupts the O/A ratio on the first iteration.
        self._stages: List[Optional[_sx]] = []
        for i in range(1, n_stages + 1):
            feeds = self._entry_feeds[i]
            if feeds:
                stage = _sx(self.reactions, *feeds)
                stage._managed = True
                self._stages.append(stage)
            else:
                self._stages.append(None)

    # ── item access ───────────────────────────────────────────────────────────

    def __getitem__(self, i: int) -> _sx:
        """
        Return the :class:`sx` object at stage *i*.

        Accepts **1-based positive indices** (``ms[1]`` = stage 1) as well as
        standard **0-based / negative** Python indices (``ms[0]`` = stage 1,
        ``ms[-1]`` = last stage).

        Parameters
        ----------
        i : int

        Returns
        -------
        sx

        Examples
        --------
        >>> ms[1]    # stage 1  (1-based)
        >>> ms[0]    # stage 1  (0-based)
        >>> ms[-1]   # last stage
        """
        n = self.n_stages
        if i < 1:                          # 0-based or negative
            if i < -n:
                raise IndexError(
                    f"Stage index {i} out of range ({-n} to {n}).")
            return self._stages[i]
        if i > n:                          # 1-based positive out of range
            raise IndexError(
                f"Stage index {i} out of range (1–{n}).")
        return self._stages[i - 1]

    def __len__(self) -> int:
        return self.n_stages

    @property
    def stages(self) -> List[_sx]:
        """
        All :class:`~efta.solventextraction.sx.sx` stages in order,
        as a plain list (index 0 = stage 1, index n-1 = stage n).

        Returns
        -------
        list of sx

        Examples
        --------
        >>> ms.stages           # [sx(...), sx(...), ...]
        >>> ms.stages[0]        # sx at stage 1
        >>> ms.stages[-1]       # sx at stage n
        """
        return list(self._stages)

    @property
    def inflow(self) -> List[int]:
        """
        1-based indices of stages that receive external feed (inflow stages).

        A stage is an inflow stage if at least one stream begins there and
        that stream spans more than one stage (i.e. not a degenerate
        single-stage loop).

        Returns
        -------
        list of int

        Examples
        --------
        >>> ms.inflow    # [1, 3]  — stages that receive external feeds
        """
        aq_entry_terminals  = {flow[0] for _, flow in self.aq_streams  if flow[0] == flow[-1]}
        org_entry_terminals = {flow[0] for _, flow in self.org_streams if flow[0] == flow[-1]}
        aq_entries  = {flow[0] for _, flow in self.aq_streams}
        org_entries = {flow[0] for _, flow in self.org_streams}
        idxs = set()
        for idx in aq_entries:
            if idx not in aq_entry_terminals:
                idxs.add(idx)
        for idx in org_entries:
            if idx not in org_entry_terminals:
                idxs.add(idx)
        return sorted(idxs)

    @property
    def outflow(self) -> List[int]:
        """
        1-based indices of stages that produce a system outlet (outflow stages).

        Returns
        -------
        list of int

        Examples
        --------
        >>> ms.outflow   # [1, 3]  — stages whose outlets leave the system
        """
        aq_entry_stages  = {flow[0] for _, flow in self.aq_streams}
        org_entry_stages = {flow[0] for _, flow in self.org_streams}
        aq_terminals     = {flow[-1] for _, flow in self.aq_streams}
        org_terminals    = {flow[-1] for _, flow in self.org_streams}
        idxs = set()
        for idx in aq_terminals:
            if idx not in aq_entry_stages:
                idxs.add(idx)
        for idx in org_terminals:
            if idx not in org_entry_stages:
                idxs.add(idx)
        return sorted(idxs)

    @property
    def instages(self) -> List[_sx]:
        """
        :class:`sx` objects at inflow stages, in ascending stage-index order.

        Returns
        -------
        list of sx

        Examples
        --------
        >>> ms.instages    # [sx at stage 1, sx at stage 3]
        """
        return [self._stages[i - 1] for i in self.inflow]

    @property
    def outstages(self) -> List[_sx]:
        """
        :class:`sx` objects at outflow stages, in ascending stage-index order.

        Returns
        -------
        list of sx

        Examples
        --------
        >>> ms.outstages   # [sx at stage 1, sx at stage 3]
        """
        return [self._stages[i - 1] for i in self.outflow]

    # ── managed inlet interface ───────────────────────────────────────────────

    def _resolve_inflow_stage(self, i: int) -> int:
        """Resolve *i* (1-based inflow index) to a 1-based stage index."""
        inflow = self.inflow
        n = len(inflow)
        if n == 0:
            raise InputError("This multistage system has no inflow stages.")
        if not -n <= i <= n or i == 0:
            raise InputError(
                f"Inflow index {i} out of range; "
                f"{n} inflow stage(s) available (1-based, or negative).")
        return inflow[i - 1] if i > 0 else inflow[i]

    def input(self, i: int, sol: _solution) -> 'multistage':
        """
        Add *sol* as an inlet to inflow stage *i*.

        *i* is a **1-based index into the list of inflow stages**
        (i.e. ``ms.inflow[i-1]``), not a stage number.  Negative indices are
        accepted.

        Parameters
        ----------
        i : int
            1-based inflow index.
        sol : solution

        Returns
        -------
        multistage
            *self*, for chaining.

        Examples
        --------
        >>> ms.input(1, extra_feed)   # add to the first inflow stage
        """
        if not isinstance(sol, _solution):
            raise InputError(
                f"Expected a solution object, got {type(sol).__name__!r}.")
        stage_idx = self._resolve_inflow_stage(i)
        stage = self._stages[stage_idx - 1]
        if stage is None:
            stage = _sx(self.reactions, sol)
            stage._managed = True
            self._stages[stage_idx - 1] = stage
        else:
            stage._inlets.append(sol)
        self._entry_feeds[stage_idx].append(sol)
        return self

    def stop(self, i: int, inlet: int) -> 'multistage':
        """
        Remove inlet at position *inlet* from inflow stage *i*.

        *i* is a **1-based index into the list of inflow stages**.
        *inlet* is a zero-based index into that stage's current ``_inlets``
        list (negative indices accepted).

        Parameters
        ----------
        i : int
            1-based inflow index.
        inlet : int
            Zero-based index of the inlet to remove.

        Returns
        -------
        multistage
            *self*, for chaining.

        Examples
        --------
        >>> ms.stop(1, 0)    # remove first inlet of the first inflow stage
        >>> ms.stop(1, -1)   # remove last inlet of the first inflow stage
        """
        stage_idx = self._resolve_inflow_stage(i)
        stage = self._stages[stage_idx - 1]
        if stage is None:
            raise InputError(
                f"Inflow stage {stage_idx} has not been initialised yet.")
        n = len(stage._inlets)
        if not -n <= inlet < n:
            raise InputError(
                f"Inlet index {inlet} out of range for {n} inlet(s) "
                f"at inflow stage {stage_idx}.")
        removed = stage._inlets.pop(inlet)
        # Mirror removal in _entry_feeds if present
        try:
            self._entry_feeds[stage_idx].remove(removed)
        except ValueError:
            pass
        return self

    def empty(self, i: int) -> 'multistage':
        """
        Clear all inlets of inflow stage *i*.

        *i* is a **1-based index into the list of inflow stages**.

        Parameters
        ----------
        i : int
            1-based inflow index.

        Returns
        -------
        multistage
            *self*, for chaining.

        Examples
        --------
        >>> ms.empty(1)    # clear all inlets of the first inflow stage
        """
        stage_idx = self._resolve_inflow_stage(i)
        stage = self._stages[stage_idx - 1]
        if stage is None:
            raise InputError(
                f"Inflow stage {stage_idx} has not been initialised yet.")
        stage._inlets = []
        self._entry_feeds[stage_idx] = []
        return self

    # ── inlets property ───────────────────────────────────────────────────────

    @property
    def inlets(self) -> Dict[int, _solution]:
        """
        External feed solutions keyed by their entry stage (1-based).

        If a stage receives more than one external feed (e.g. both an aqueous
        and an organic stream enter the same stage), their solutions are blended
        into a single solution via volume-weighted mixing.

        A stage is included if it is the entry point for at least one stream
        whose phase does NOT also exit at that same stage.  In a standard
        counter-current system this gives ``{1: feed_aq, n: feed_org}``.

        Returns
        -------
        dict[int, solution]
            ``{stage_idx: solution}``

        Examples
        --------
        >>> ms.inlets        # {1: feed_aq, 20: feed_org}
        """
        aq_entry_terminals: Set[int] = set()
        org_entry_terminals: Set[int] = set()
        for _, flow in self.aq_streams:
            if flow[0] == flow[-1]:
                aq_entry_terminals.add(flow[0])
        for _, flow in self.org_streams:
            if flow[0] == flow[-1]:
                org_entry_terminals.add(flow[0])

        aq_entries  = {flow[0] for _, flow in self.aq_streams}
        org_entries = {flow[0] for _, flow in self.org_streams}

        from .sx import _blend_solutions as _blend
        result: Dict[int, _solution] = {}
        for idx, sols in self._entry_feeds.items():
            if not sols:
                continue
            has_real_aq  = idx in aq_entries  and idx not in aq_entry_terminals
            has_real_org = idx in org_entries and idx not in org_entry_terminals
            if has_real_aq or has_real_org:
                result[idx] = _blend(sols) if len(sols) > 1 else sols[0]
        return result

    # ── outlets property ──────────────────────────────────────────────────────

    @property
    def outlets(self) -> Dict[int, Union[_solution, _mixture]]:
        """
        Terminal output solutions keyed by stage index (1-based).

        For each stream the terminal stage is the last stage in its flow.
        A terminal stage contributes an outlet *for each phase whose stream
        ends there*, regardless of whether it also receives an external feed
        of another phase:

        - Aqueous-stream terminal → raffinate (``stage.outlets[1]``)
        - Organic-stream terminal → extract   (``stage.outlets[0]``)
        - Terminal for both phases → ``mixture(extract, raffinate)``

        In a standard counter-current system stage 1 is both the organic
        terminal AND the aqueous feed entry: it still appears in outlets as
        the organic extract exit.  Stage n is both the aqueous terminal AND
        the organic feed entry: it still appears as the aqueous raffinate
        exit.

        Returns
        -------
        dict[int, solution | mixture]

        Examples
        --------
        >>> ms.outlets           # {20: raffinate, 1: extract}
        """
        # Per-phase feed entry stages
        aq_entry_stages:  Set[int] = {flow[0] for _, flow in self.aq_streams}
        org_entry_stages: Set[int] = {flow[0] for _, flow in self.org_streams}

        aq_terminals:  Set[int] = set()
        org_terminals: Set[int] = set()
        for _, flow in self.aq_streams:
            aq_terminals.add(flow[-1])
        for _, flow in self.org_streams:
            org_terminals.add(flow[-1])

        all_terminals = aq_terminals | org_terminals
        result: Dict[int, Union[_solution, _mixture]] = {}

        for idx in all_terminals:
            stage = self._stages[idx - 1]

            try:
                outs = stage.outlets
                # Include organic outlet only if this stage is an org terminal
                # and NOT also the org feed entry (degenerate 1-stage stream).
                org_out = (outs[0]
                           if idx in org_terminals and idx not in org_entry_stages
                           else None)
                # Include aqueous outlet only if this stage is an aq terminal
                # and NOT also the aq feed entry (degenerate 1-stage stream).
                aq_out  = (outs[1]
                           if idx in aq_terminals and idx not in aq_entry_stages
                           else None)
            except (IndexError, AttributeError):
                continue

            if org_out is not None and aq_out is not None:
                result[idx] = _mixture(org_out, aq_out)
            elif aq_out is not None:
                result[idx] = aq_out
            elif org_out is not None:
                result[idx] = org_out

        return result

    # ── core solver ───────────────────────────────────────────────────────────

    def transfer(self) -> 'multistage':
        """
        Reset all stage inlets to their external feeds, then propagate current
        outlets through the flow topology — without reacting.

        Specifically:

        1. Every stage's ``_inlets`` is reset to its external feed(s) only
           (``[]`` for non-entry stages).
        2. For each aqueous stream, the raffinate of stage ``flow[pos]`` is
           wired as an inlet to stage ``flow[pos+1]``.
        3. For each organic stream, the extract of stage ``flow[pos]`` is
           wired as an inlet to stage ``flow[pos+1]``.

        Stages that have not been reacted yet (``ran=False``) are skipped
        during propagation — their outlets are phase-separated inlets, not
        equilibrium results.

        This is the Jacobi transfer step.  Calling :meth:`react` afterwards
        will react all stages with the wired inlets.

        Returns
        -------
        multistage
            *self*, for chaining.

        Examples
        --------
        >>> ms.transfer()
        >>> ms.react()
        """
        n = self.n_stages

        # Step 1: reset all inlets to external feeds only
        for i in range(1, n + 1):
            stage = self._stages[i - 1]
            if stage is None:
                continue
            stage._inlets = list(self._entry_feeds[i])
            stage.ran     = False

        # Step 2: wire aqueous raffinate transfers
        for _sol, flow in self.aq_streams:
            for pos in range(len(flow) - 1):
                src_idx   = flow[pos]
                dst_idx   = flow[pos + 1]
                src_stage = self._stages[src_idx - 1]
                dst_stage = self._stages[dst_idx - 1]

                if src_stage is None or not src_stage.ran:
                    continue

                raffinate = src_stage.outlets[1]
                ext_feeds = list(self._entry_feeds[dst_idx])

                if dst_stage is None:
                    dst_stage = _sx(self.reactions, raffinate)
                    dst_stage._managed = True
                    self._stages[dst_idx - 1] = dst_stage
                else:
                    org_inlets = [
                        s for s in dst_stage._inlets
                        if s not in ext_feeds
                        and any(_is_org(sp) for sp in s.concentrations)
                    ]
                    dst_stage._inlets = ext_feeds + [raffinate] + org_inlets

        # Step 3: wire organic extract transfers
        for _sol, flow in self.org_streams:
            for pos in range(len(flow) - 1):
                src_idx   = flow[pos]
                dst_idx   = flow[pos + 1]
                src_stage = self._stages[src_idx - 1]
                dst_stage = self._stages[dst_idx - 1]

                if src_stage is None or not src_stage.ran:
                    continue

                extract   = src_stage.outlets[0]
                ext_feeds = list(self._entry_feeds[dst_idx])

                if dst_stage is None:
                    dst_stage = _sx(self.reactions, extract)
                    dst_stage._managed = True
                    self._stages[dst_idx - 1] = dst_stage
                else:
                    aq_inlets = [
                        s for s in dst_stage._inlets
                        if s not in ext_feeds
                        and not any(_is_org(sp) for sp in s.concentrations)
                    ]
                    dst_stage._inlets = ext_feeds + aq_inlets + [extract]

        return self

    def react(self, efficiency: float = 1.0) -> 'multistage':
        """
        Run equilibrium on every stage once using its current inlets, and
        update each stage's outlets.  Stages with no inlets are skipped.

        This is the Jacobi reaction step — it operates on whatever inlets are
        currently wired, so it is most useful after :meth:`transfer`.

        .. note::
            This method does **not** update :attr:`converged`, :attr:`n_iter`,
            or :attr:`efficiency` on the ``multistage`` object — those are
            managed by :meth:`run`.  If you are iterating manually, track
            convergence yourself by comparing successive :attr:`outlets`.

        Parameters
        ----------
        efficiency : float, optional
            Extraction efficiency in [0, 1].  Default ``1.0``.

        Returns
        -------
        multistage
            *self*, for chaining.

        Examples
        --------
        >>> ms.transfer()
        >>> ms.react(0.9)
        """
        if not 0.0 <= float(efficiency) <= 1.0:
            raise InputError(f"efficiency must be in [0, 1], got {efficiency}.")
        efficiency = float(efficiency)

        for i in range(1, self.n_stages + 1):
            stage = self._stages[i - 1]
            if stage is None or not stage._inlets:
                continue
            stage.ran = False
            stage.run(efficiency=efficiency)

        return self

    def run(self, efficiency: float = 1.0, iter=None,
            tolerance: float = 1e-6) -> 'multistage':
        if not 0.0 <= float(efficiency) <= 1.0:
            raise InputError(f"efficiency must be in [0, 1], got {efficiency}.")
        efficiency      = float(efficiency)
        tolerance       = float(tolerance)
        self.efficiency = efficiency

        n        = self.n_stages
        max_iter = int(iter) if iter is not None else n * n

        # Step 1: reset all stages to their external feeds only.
        # Entry stages get their feed inlets restored.
        # Non-entry stages get their inlets cleared to [] so they don't run
        # with stale inlets from the previous iteration — transfers will
        # re-wire them before they are used.  A None stage stays None.
        for i in range(1, n + 1):
            stage = self._stages[i - 1]
            if stage is None:
                continue
            stage._inlets = list(self._entry_feeds[i])   # [] for non-entry stages
            stage.ran     = False

        prev_aq_conc  = {i: {} for i in range(1, n + 1)}
        prev_org_conc = {i: {} for i in range(1, n + 1)}
        self.converged = False
        self.n_iter    = 0

        for iteration in range(max_iter):

            # ── Run every stage once with current inlets ──────────────────────
            for i in range(1, n + 1):
                stage = self._stages[i - 1]
                if stage is None or not stage._inlets:
                    continue
                stage.ran = False
                stage.run(efficiency=efficiency)

            # ── Transfer aqueous: raffinate of flow[pos] → inlet of flow[pos+1] ─
            for _sol, flow in self.aq_streams:
                for pos in range(len(flow) - 1):
                    src_idx   = flow[pos]
                    dst_idx   = flow[pos + 1]
                    src_stage = self._stages[src_idx - 1]
                    dst_stage = self._stages[dst_idx - 1]

                    if src_stage is None or not src_stage.ran:
                        continue

                    raffinate = src_stage.outlets[1]
                    ext_feeds = list(self._entry_feeds[dst_idx])

                    if dst_stage is None:
                        # First time this stage receives any inlet: construct it.
                        dst_stage = _sx(self.reactions, raffinate)
                        dst_stage._managed = True
                        self._stages[dst_idx - 1] = dst_stage
                    else:
                        # Retain only ext_feeds + any already-wired organic inlets;
                        # the previous aq inter-stage inlet is replaced, not appended.
                        org_inlets = [
                            s for s in dst_stage._inlets
                            if s not in ext_feeds
                            and any(_is_org(sp) for sp in s.concentrations)
                        ]
                        dst_stage._inlets = ext_feeds + [raffinate] + org_inlets

            # ── Transfer organic: extract of flow[pos] → inlet of flow[pos+1] ──
            for _sol, flow in self.org_streams:
                for pos in range(len(flow) - 1):
                    src_idx   = flow[pos]
                    dst_idx   = flow[pos + 1]
                    src_stage = self._stages[src_idx - 1]
                    dst_stage = self._stages[dst_idx - 1]

                    if src_stage is None or not src_stage.ran:
                        continue

                    extract   = src_stage.outlets[0]
                    ext_feeds = list(self._entry_feeds[dst_idx])

                    if dst_stage is None:
                        # First time this stage receives any inlet: construct it.
                        dst_stage = _sx(self.reactions, extract)
                        dst_stage._managed = True
                        self._stages[dst_idx - 1] = dst_stage
                    else:
                        # Retain only ext_feeds + any already-wired aqueous inlets;
                        # the previous org inter-stage inlet is replaced, not appended.
                        aq_inlets = [
                            s for s in dst_stage._inlets
                            if s not in ext_feeds
                            and not any(_is_org(sp) for sp in s.concentrations)
                        ]
                        dst_stage._inlets = ext_feeds + aq_inlets + [extract]

            self.n_iter = iteration + 1

            # ── Convergence check ─────────────────────────────────────────────
            max_rel = 0.0
            for i in range(1, n + 1):
                stage   = self._stages[i - 1]
                cur_aq  = stage.outlets[1].concentrations if (stage is not None and stage.ran) else {}
                cur_org = stage.outlets[0].concentrations if (stage is not None and stage.ran) else {}
                for d_new, d_old in [(cur_aq, prev_aq_conc[i]),
                                    (cur_org, prev_org_conc[i])]:
                    for sp, c_new in d_new.items():
                        c_old   = d_old.get(sp, 0.0)
                        denom   = max(abs(c_new), abs(c_old), 1e-30)
                        max_rel = max(max_rel, abs(c_new - c_old) / denom)

            prev_aq_conc  = {i: (self._stages[i-1].outlets[1].concentrations
                                if (self._stages[i-1] is not None and self._stages[i-1].ran) else {})
                            for i in range(1, n + 1)}
            prev_org_conc = {i: (self._stages[i-1].outlets[0].concentrations
                                if (self._stages[i-1] is not None and self._stages[i-1].ran) else {})
                            for i in range(1, n + 1)}

            if max_rel < tolerance:
                self.converged = True
                break

        return self

    # ── concentration profile plot ────────────────────────────────────────────

    def plot(self,
             components: List[str],
             phase: str = 'aq',
             color: bool = True,
             title: Optional[str] = None):
        """
        Plot the concentration profile of *components* across all stages.

        Parameters
        ----------
        components : list of str
            Species names or element symbols to plot.
        phase : str, optional
            ``'aq'`` (default), ``'org'``, or ``'both'``.
            ``'both'`` draws aqueous and organic series on the **same** axes.
            Legend labels include ``(aq)`` / ``(org)`` suffixes to distinguish
            the two phases.
        color : bool, optional
            If True (default) use ``style.colors``; if False draw in black.
        title : str or None, optional
            Figure title.  Auto-generated if None.

        Style rules
        -----------
        - Different **component** → different marker, same linestyle per phase.
        - Different **phase**     → different linestyle, same marker per component.

        Returns
        -------
        (fig, ax)
            Matplotlib Figure and Axes objects.  Always a single Axes, even
            for ``phase='both'``.

        Examples
        --------
        >>> ms.plot(['La', 'Ce'], phase='aq')
        >>> ms.plot(['La', 'Ce'], phase='org', color=False)
        >>> fig, ax = ms.plot(['La', 'Ce'], phase='both')
        """
        import matplotlib.pyplot as plt
        from ..plotting import style as _style, _format_species

        if phase not in ('aq', 'org', 'both'):
            raise InputError(f"phase must be 'aq', 'org', or 'both', got {phase!r}.")

        if self.efficiency is None:
            raise InputError(
                "multistage has not been run yet.  Call ms.run() first.")

        n          = self.n_stages
        stage_nums = list(range(1, n + 1))

        # ── auto title ────────────────────────────────────────────────────
        eff_str    = f'η={self.efficiency:.3g}' if self.efficiency != 1.0 else ''
        conv_str   = (f'{"converged" if self.converged else "not converged"}'
                      f' ({self.n_iter} iters)')
        extra      = (', ' + eff_str) if eff_str else ''
        phase_word = {'aq': 'Aqueous', 'org': 'Organic', 'both': 'Concentration'}[phase]
        auto_title = (f'{phase_word} concentration profile  '
                      f'({n} stages{extra}, {conv_str})')

        # ── phases to draw ────────────────────────────────────────────────
        phases = ['aq', 'org'] if phase == 'both' else [phase]

        # linestyle cycles with phase (index 0 = aq, index 1 = org)
        # marker    cycles with component index
        phase_ls = {ph: _style.line_styles[i % len(_style.line_styles)]
                    for i, ph in enumerate(phases)}

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        ax.grid(False)

        n_shown = len(components) * len(phases)
        legend_fs = (_style._fs('legend') if _style.legend_fontsize is not None
                     else (_style.legend_fontsize_many if n_shown > 10
                           else _style.legend_fontsize_few))

        for ph in phases:
            ls          = phase_ls[ph]
            ph_suffix   = f' ({"aq" if ph == "aq" else "org"})' if phase == 'both' else ''

            for idx, comp in enumerate(components):
                ys = []
                for i in stage_nums:
                    stage = self._stages[i - 1]
                    try:
                        sol = stage.outlets[1] if ph == 'aq' else stage.outlets[0]
                        c   = (sol.aq_concentration(comp) if ph == 'aq'
                               else sol.org_concentration(comp))
                    except (AttributeError, IndexError):
                        c = 0.0
                    ys.append(c)

                lc = _style.colors[idx % len(_style.colors)] if color else 'black'
                mk = _style.marker_styles[idx % len(_style.marker_styles)]
                ax.plot(stage_nums, ys,
                        color=lc, linewidth=_style.linewidth,
                        linestyle=ls, marker=mk, markersize=7,
                        label=_format_species(comp) + ph_suffix)

        ax.set_xticks(stage_nums)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_xlabel('Stage', fontsize=_style._fs('x_title'))
        ylabel = 'Concentration (M)' if phase == 'both' else f'{phase_word} concentration (M)'
        ax.set_ylabel(ylabel, fontsize=_style._fs('y_title'))
        ax.tick_params(axis='x', labelsize=_style._fs('x_tick'))
        ax.tick_params(axis='y', labelsize=_style._fs('y_tick'))
        ax.set_title(title if title is not None else auto_title,
                     fontsize=_style._fs('x_title') + 1)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0, frameon=False, fontsize=legend_fs)
        plt.tight_layout()
        return fig, ax

    # ── representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        aq_flows  = [flow for _, flow in self.aq_streams]
        org_flows = [flow for _, flow in self.org_streams]
        status = (f"converged={self.converged}, n_iter={self.n_iter}"
                  if self.efficiency is not None else "not run")
        return (f"multistage(n={self.n_stages}, "
                f"aq={aq_flows}, org={org_flows}, {status})")

    def __str__(self) -> str:
        lines = [
            f"multistage — {self.n_stages}-stage solvent extraction",
            f"  Default iter: {self.n_stages ** 2}  (n²)",
        ]
        if self.efficiency is not None:
            lines += [
                f"  Efficiency : {self.efficiency:.4g}",
                f"  Converged  : {self.converged}  ({self.n_iter} passes)",
            ]
        else:
            lines.append("  Status     : not run")

        lines.append(f"  Aq  streams ({len(self.aq_streams)}):")
        for i, (sol, flow) in enumerate(self.aq_streams):
            lines.append(f"    [{i}] vol={sol.volume:.4g} L  flow={flow}")

        lines.append(f"  Org streams ({len(self.org_streams)}):")
        for i, (sol, flow) in enumerate(self.org_streams):
            lines.append(f"    [{i}] vol={sol.volume:.4g} L  flow={flow}")

        if self.efficiency is not None and self.outlets:
            lines.append("  Outlets:")
            org_terminal_set = {flow[-1] for _, flow in self.org_streams}
            for idx, sol in self.outlets.items():
                label = "aq+org" if isinstance(sol, _mixture) else (
                    "org" if idx in org_terminal_set else "aq"
                )
                lines.append(f"    stage[{idx}] ({label}): {_fmt_sol(sol)}")

        return "\n".join(lines)


# ── convenience constructors ──────────────────────────────────────────────────

def _check_args(stages: int,
                feed:    _solution,
                organic: _solution) -> int:
    """Shared validation for all convenience constructors."""
    stages = int(stages)
    if stages < 1:
        raise InputError(f"stages must be >= 1, got {stages}.")
    if not isinstance(feed, _solution):
        raise InputError("feed must be a solution object.")
    if not isinstance(organic, _solution):
        raise InputError("organic must be a solution object.")
    return stages


def countercurrent(reactions: Union[_reaction_cls, _reactions_cls],
                   stages:    int,
                   feed:      _solution,
                   organic:   _solution) -> multistage:
    """
    Standard counter-current extraction.

    Aqueous *feed* flows forward through all stages (1 → 2 → … → n).
    Fresh *organic* enters at stage *n* and flows backward (n → … → 1).

    ::

        feed(aq) →[1]→[2]→ … →[n]→ raffinate
                   ↕    ↕       ↕
        extract ←[1]←[2]← … ←[n]← organic

    Inlets  : {1: feed,    n: organic}
    Outlets : {n: raffinate, 1: extract}

    Parameters
    ----------
    reactions : reaction | reactions
    stages : int
        Number of stages (≥ 1).
    feed : solution
        Aqueous feed.
    organic : solution
        Organic solvent feed.

    Returns
    -------
    multistage

    Examples
    --------
    >>> ms = countercurrent(rxn, stages=5, feed=feed_aq, organic=org)
    >>> ms.run()
    >>> ms.run(tolerance=1e-8)
    >>> raffinate = ms.outlets[5]
    >>> extract   = ms.outlets[1]
    """
    stages = _check_args(stages, feed, organic)
    return multistage(
        reactions,
        aq_streams  = [(feed,    list(range(1, stages + 1)))],
        org_streams = [(organic, list(range(stages, 0, -1)))],
    )


def crosscurrent(reactions: Union[_reaction_cls, _reactions_cls],
                 stages:    int,
                 feed:      _solution,
                 organic:   _solution) -> multistage:
    """
    Cross-current extraction.

    Aqueous *feed* flows forward through all stages (1 → 2 → … → n).
    A **fresh portion** of *organic* contacts each stage independently and
    exits that stage directly — there is no inter-stage organic flow.

    ::

        feed(aq) →[1]→[2]→ … →[n]→ raffinate
                   ↕    ↕       ↕
                  org  org     org   (fresh at every stage)

    Each stage has its own organic outlet.

    Inlets  : {1: feed,  1..n: organic (one per stage)}
    Outlets : {n: raffinate,  1..n: extract per stage}

    Parameters
    ----------
    reactions : reaction | reactions
    stages : int
        Number of stages (≥ 1).
    feed : solution
        Aqueous feed.
    organic : solution
        Fresh organic fed to every stage.

    Returns
    -------
    multistage

    Examples
    --------
    >>> ms = crosscurrent(rxn, stages=5, feed=feed_aq, organic=org)
    >>> ms.run()
    >>> raffinate = ms.outlets[5]
    >>> extract_s1 = ms[1].outlets[0]
    """
    stages = _check_args(stages, feed, organic)
    org_streams = [(organic, [i]) for i in range(1, stages + 1)]
    return multistage(
        reactions,
        aq_streams  = [(feed, list(range(1, stages + 1)))],
        org_streams = org_streams,
    )


def strip_countercurrent(reactions: Union[_reaction_cls, _reactions_cls],
                         stages:    int,
                         organic:   _solution,
                         feed:      _solution) -> multistage:
    """
    Counter-current stripping (back-extraction).

    The inverse of :func:`countercurrent`.  The loaded *organic* phase enters
    at stage 1 and flows forward (1 → 2 → … → n).  The aqueous stripping
    agent (*feed*) enters at stage *n* and flows backward (n → … → 1).

    ::

        organic →[1]→[2]→ … →[n]→ stripped organic (raffinate-side)
                  ↕    ↕       ↕
        strip aq←[1]←[2]← … ←[n]← feed(aq)

    Inlets  : {1: organic,  n: feed(aq)}
    Outlets : {n: loaded extract,  1: strip liquor (aqueous)}

    Parameters
    ----------
    reactions : reaction | reactions
    stages : int
        Number of stages (≥ 1).
    organic : solution
        Loaded organic feed (enters at stage 1).
    feed : solution
        Aqueous stripping agent (enters at stage n).

    Returns
    -------
    multistage

    Examples
    --------
    >>> ms = strip_countercurrent(rxn, stages=5, organic=loaded_org, feed=strip_aq)
    >>> ms.run()
    >>> stripped_organic = ms.outlets[5]
    >>> strip_liquor     = ms.outlets[1]
    """
    stages = _check_args(stages, feed, organic)
    return multistage(
        reactions,
        aq_streams  = [(feed,    list(range(stages, 0, -1)))],
        org_streams = [(organic, list(range(1, stages + 1)))],
    )


def strip_crosscurrent(reactions: Union[_reaction_cls, _reactions_cls],
                       stages:    int,
                       organic:   _solution,
                       feed:      _solution) -> multistage:
    """
    Cross-current stripping (back-extraction).

    The inverse of :func:`crosscurrent`.  The loaded *organic* flows forward
    through all stages (1 → 2 → … → n).  A **fresh portion** of aqueous
    stripping agent (*feed*) contacts each stage independently and exits
    that stage directly.

    ::

        organic →[1]→[2]→ … →[n]→ stripped organic
                  ↕    ↕       ↕
                 aq   aq      aq    (fresh stripping agent at every stage)

    Each stage has its own aqueous outlet (strip liquor).

    Inlets  : {1: organic,  1..n: feed(aq) one per stage}
    Outlets : {n: stripped extract,  1..n: strip liquor per stage}

    Parameters
    ----------
    reactions : reaction | reactions
    stages : int
        Number of stages (≥ 1).
    organic : solution
        Loaded organic feed.
    feed : solution
        Fresh aqueous stripping agent fed to every stage.

    Returns
    -------
    multistage

    Examples
    --------
    >>> ms = strip_crosscurrent(rxn, stages=5, organic=loaded_org, feed=strip_aq)
    >>> ms.run()
    >>> stripped_organic  = ms.outlets[5]
    >>> strip_liquor_s1   = ms[1].outlets[1]
    """
    stages = _check_args(stages, feed, organic)
    aq_streams = [(feed, [i]) for i in range(1, stages + 1)]
    return multistage(
        reactions,
        aq_streams  = aq_streams,
        org_streams = [(organic, list(range(1, stages + 1)))],
    )
