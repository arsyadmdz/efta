"""
efta.plotting
=============
All visualisation code for efta.

The :class:`PlotStyle` singleton (accessible as ``efta.style``) controls
colours, line styles, hatch patterns, and line width used throughout.

Public helpers
--------------
_format_species()   – convert 'Fe[3+]' to 'Fe³⁺' for axis labels
_format_reaction_str() – human-readable reaction string
_plot_reactions()   – concentration / extent line-plots
_plot_fractions()   – speciation-fraction plots (line, layer, bilayer)
"""

from __future__ import annotations

from .errors import InputError

import re
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .species import species, is_nonaqueous, is_organic, is_electron

# ---------------------------------------------------------------------------
# Unicode subscript / superscript maps
# ---------------------------------------------------------------------------

_SUBSCRIPT_MAP   = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
_SUPERSCRIPT_MAP = str.maketrans('0123456789+-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻')


# ---------------------------------------------------------------------------
# PlotStyle singleton
# ---------------------------------------------------------------------------

class PlotStyle:
    """
    Singleton that stores visual styling for all efta plots.

    Attributes
    ----------
    colors          : list of hex colour strings (cycled over species/reactions)
    line_styles     : list of matplotlib line-style specs
    hatch_styles    : list of hatch patterns for filled plots
    linewidth       : default line width
    fill_alpha      : alpha for fill_between patches
    fill_linewidth  : edge linewidth for patches
    legend_fontsize_many, legend_fontsize_few : legend text sizes
    fontsize            : global fallback font size (default 11)
    legend_fontsize     : legend text size (overrides many/few when set)
    x_tick_fontsize     : x-axis tick label size
    x_title_fontsize    : x-axis title (label) size
    y_tick_fontsize     : y-axis tick label size
    y_title_fontsize    : y-axis title (label) size
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.colors: List[str] = [
            '#000000', '#F8CD5A', '#364C86', '#B3514E', '#61A363',
            '#D7D0BD', '#C49A1A', '#1E2F5C', '#7A2926', '#3A6B3C',
            '#A89880', '#FAEA9A', '#7085B8', '#D98C8A', '#97C999', '#EDE8DF',
        ]
        self.line_styles: list = [
            '-', '--', ':', '-.',
            (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)),
            (0, (5, 2)), (0, (5, 2, 1, 2)), (0, (2, 1)), (0, (10, 3)),
            (0, (1, 2, 5, 2)), (0, (7, 2, 1, 2)), (0, (4, 1, 1, 1, 1, 1)), (0, (6, 2, 2, 2)),
            (0, (3, 3)),
        ]
        self.hatch_styles: list = [
            '', '/', '//', '///', '\\', '\\\\', '|', '-', '+', 'x', 'xx',
            'o', 'O', '.', '..', '*', '**', 'x/', '|-', 'o/', '\\|', '//-',
            'x|', '\\.', 'O.', '+/', '*-', '///.', '\\\\|', 'xx/',
        ]
        self.linewidth:            float = 2.0
        self.fill_alpha:           float = 0.9
        self.fill_linewidth:       float = 0.5
        self.marker_styles: list = [
            'o', 's', '^', 'D', 'v', 'p', 'h', '*',
            'P', 'X', '<', '>', 'H', 'd', '8', '+',
        ]
        self.legend_fontsize_many: int   = 8
        self.legend_fontsize_few:  int   = 9
        # font sizes — None means "use fontsize fallback"
        self.fontsize:             float = 11.0
        self.legend_fontsize:      float = None
        self.x_tick_fontsize:      float = None
        self.x_title_fontsize:     float = None
        self.y_tick_fontsize:      float = None
        self.y_title_fontsize:     float = None

    def _fs(self, kind: str) -> float:
        """
        Resolve a font size by kind:
        'x_tick', 'x_title', 'y_tick', 'y_title', 'legend'
        Falls back to self.fontsize if the specific attribute is None.
        """
        val = getattr(self, f'{kind}_fontsize', None)
        return val if val is not None else self.fontsize

    def reset(self) -> 'PlotStyle':
        """Reset all style settings to defaults."""
        self._init()
        return self

    def __repr__(self) -> str:
        return (f"PlotStyle(colors={len(self.colors)}, "
                f"line_styles={len(self.line_styles)}, "
                f"hatch_styles={len(self.hatch_styles)}, "
                f"marker_styles={len(self.marker_styles)}, "
                f"linewidth={self.linewidth}, "
                f"fill_alpha={self.fill_alpha})")


#: Module-level style singleton.  Modify attributes to customise all plots.
style = PlotStyle()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_species(species_str: str) -> str:
    """
    Format a normalised species name for use in plot labels.

    Converts numeric subscripts to Unicode subscripts and charge brackets
    to Unicode superscripts::

        'Fe[3+]'  →  'Fe³⁺'
        'H2SO4'   →  'H₂SO₄'
        'Ca[2+](aq)' → 'Ca²⁺(aq)'
    """
    s = species_str.strip()
    phase_tag = ''
    m_phase = re.search(r'\((aq|org|s|l)\)\s*$', s, re.IGNORECASE)
    if m_phase:
        phase_tag = f'({m_phase.group(1)})'
        s = s[:m_phase.start()].rstrip()
    charge_super = ''
    m_charge = re.search(r'\[([+\-]?\d*[+\-]?)\]\s*$', s)
    if m_charge:
        raw = m_charge.group(1)
        charge_super = ''.join(ch.translate(_SUPERSCRIPT_MAP) for ch in raw)
        s = s[:m_charge.start()].rstrip()
    parts = s.split('.')
    formatted_parts = []
    for part_idx, part in enumerate(parts):
        if part_idx == 0:
            out = []
            for i, ch in enumerate(part):
                if ch.isdigit():
                    prev = part[i - 1] if i > 0 else ''
                    if prev and (prev.isalpha() or prev in ')_' or prev.isdigit()):
                        out.append(ch.translate(_SUBSCRIPT_MAP))
                    else:
                        out.append(ch)
                else:
                    out.append(ch)
            formatted_parts.append(''.join(out))
        else:
            out = []
            letter_seen = False
            for ch in part:
                if ch.isdigit() and not letter_seen:
                    out.append(ch)
                elif ch.isdigit():
                    out.append(ch.translate(_SUBSCRIPT_MAP))
                else:
                    if ch.isalpha(): letter_seen = True
                    out.append(ch)
            formatted_parts.append('.' + ''.join(out))
    return ''.join(formatted_parts) + charge_super + phase_tag


def _format_reaction_str(rxn) -> str:
    """Return a pretty-printed reaction string with Unicode subscripts/superscripts."""
    from .model.freaction import _clean_species as _cs
    def _side(items):
        parts = []
        for sp, coeff in items:
            a = abs(coeff)
            coeff_str = '' if a == 1.0 else (str(int(a)) if a == int(a) else str(a))
            parts.append(coeff_str + _format_species(_cs(sp)))
        return ' + '.join(parts)
    reactants = [(sp, c) for sp, c in rxn._stoich.items() if c < 0]
    products  = [(sp, c) for sp, c in rxn._stoich.items() if c > 0]
    return _side(reactants) + ' ⇌ ' + _side(products)


def _rxn_label(rxn, idx: int) -> str:
    """Short reaction label suitable for legends."""
    return f'R{idx + 1}: {_format_reaction_str(rxn)}'


# ---------------------------------------------------------------------------
# Axis setup helper
# ---------------------------------------------------------------------------

def _setup_ax(ax, logx: bool, xs):
    """Apply common axis formatting."""
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(0.8)
    ax.grid(False)
    if logx:
        ax.set_xscale('log')
    ax.set_xlim(xs[0], xs[-1])


# ---------------------------------------------------------------------------
# Main line-plot (concentration / extent)
# ---------------------------------------------------------------------------

def _plot_reactions(reactions_obj, c0, y='concentration', maintain=None,
                    logx=False, n_points=20, select=None,
                    type='plot', color=False, init_scale=0.1,
                    recheck=False, recheck_tol=1e-4, recheck_density=5,
                    presolver_timeout=2.0):
    """Implement reactions.plot()."""
    style_obj = globals()['style']   # singleton; avoid shadowing by 'type' param
    VALID = ('plot', 'data', 'log', 'layer', 'bilayer')
    if type not in VALID:
        raise InputError(f"type must be one of {VALID}, got {type!r}")

    if select is None: select = []
    n_rxn = len(reactions_obj._reactions)
    if select:
        if y == 'extent':
            bad = [i for i in select if not isinstance(i, int) or not (0 <= i < n_rxn)]
            if bad: raise InputError(f"select indices out of range: {bad}.")
        else:
            select = [species(str(s)) for s in select]

    data = reactions_obj._sweep_data(
        c0, y=y, logx=logx, logy=False, init_scale=init_scale, n_points=n_points,
        recheck=recheck, recheck_tol=recheck_tol, recheck_density=recheck_density,
        presolver_timeout=presolver_timeout, maintain=maintain)

    if type == 'data':
        return data

    sweep_name = data['variable']
    xs         = np.array(data['x'])
    sp_order   = list(data['concentrations'].keys())
    xi_results = data['extents']
    sp_results = data['concentrations']
    fixed_c0   = data['fixed_c0']

    # build curves dict: {label: ydata}
    curves: Dict[str, np.ndarray] = {}
    if y == 'concentration':
        visible = [s for s in sp_order if not is_nonaqueous(s) and not is_electron(s)]
        if select:
            unknown = [s for s in select if s not in sp_order]
            if unknown:
                import warnings; warnings.warn(f"select: species not found: {unknown}")
            visible = [s for s in visible if s in select]
        for s in visible:
            fs    = _format_species(s)
            label = f'[{fs}](org)' if is_organic(s) else f'[{fs}]'
            curves[label] = np.array(sp_results[s], dtype=float)
        ylabel    = 'Concentration (M)'
        ylabel_abs = ylabel   # concentrations are always non-negative
        plot_count = len(visible)
    else:
        xi_arr     = np.array(xi_results, dtype=float)
        rxn_idx    = list(range(n_rxn)) if not select else select
        for pi, ri in enumerate(rxn_idx):
            label = _rxn_label(reactions_obj._reactions[ri], ri)
            curves[label] = xi_arr[:, ri]
        ylabel     = 'Extent of reaction ξ  (mol / L$_{aq}$)'
        ylabel_abs = '|ξ|  (mol / L$_{aq}$)'
        plot_count = len(rxn_idx)

    sweep_fmt = _format_species(sweep_name)
    n_org     = sum(1 for s in sp_order if is_organic(s))
    org_note  = f',  v_oa={fixed_c0.get("O/A","?")}' if n_org > 0 else ''
    rc_note   = f',  recheck={recheck_tol:.0e}' if recheck else ''
    nonaq     = [s for s in sp_order if is_nonaqueous(s)]
    mode_note = 'extent of reaction' if y == 'extent' else 'speciation'
    subtitle  = (f'({len(reactions_obj._reactions)} reaction(s),  '
                 f'{plot_count} {"reactions" if y == "extent" else "active species"}'
                 + org_note + rc_note
                 + (f',  {len(nonaq)} pure phase(s) not shown'
                    if nonaq and y == 'concentration' else '') + ')')

    def _legend(ax, n):
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  frameon=False,
                  fontsize=style_obj._fs('legend') if style_obj.legend_fontsize is not None
                           else (style_obj.legend_fontsize_many if n > 10
                           else style_obj.legend_fontsize_few))
    def _ax_labels(ax, ylab):
        ax.set_xlabel(f'[{sweep_fmt}]₀  (M)', fontsize=style_obj._fs('x_title'))
        ax.set_ylabel(ylab, fontsize=style_obj._fs('y_title'))
        ax.set_title(f'Equilibrium {mode_note}  vs  [{sweep_fmt}]₀\n{subtitle}',
                     fontsize=style_obj._fs('x_title') + 1)
        ax.tick_params(axis='x', labelsize=style_obj._fs('x_tick'))
        ax.tick_params(axis='y', labelsize=style_obj._fs('y_tick'))

    # ── plot / log ──────────────────────────────────────────────────────────
    if type in ('plot', 'log'):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        _setup_ax(ax, logx, xs)
        all_y = []
        for idx, (label, ydata) in enumerate(curves.items()):
            ls = style_obj.line_styles[idx % len(style_obj.line_styles)]
            lc = style_obj.colors[idx % len(style_obj.colors)] if color else 'black'
            if type == 'log':
                yplot = np.log10(np.abs(np.where(ydata == 0, np.nan, ydata)))
            else:
                yplot = ydata
            ax.plot(xs, yplot, color=lc, linewidth=style_obj.linewidth,
                    linestyle=ls, label=label)
            valid = yplot[np.isfinite(yplot)]
            if len(valid): all_y.extend(valid.tolist())
        if type == 'log':
            _log_ylabel = ('log₁₀|ξ|  (mol / L$_{aq}$)' if y == 'extent'
                          else 'log₁₀|c|  (M)')
            ax.set_ylabel(_log_ylabel, fontsize=style_obj._fs('y_title'))
            if all_y:
                pad = (max(all_y) - min(all_y)) * 0.05 or 0.5
                ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        else:
            if all_y:
                y_min, y_max = min(all_y), max(all_y)
                pad = (y_max - y_min) * 0.03 if y_max > y_min else abs(y_max) * 0.03
                ax.set_ylim(y_min - pad, y_max + pad)
        _log_ylabel = ('log₁₀|ξ|  (mol / L$_{aq}$)' if y == 'extent'
                      else 'log₁₀|c|  (M)')
        _ax_labels(ax, ylabel if type == 'plot' else _log_ylabel)
        _legend(ax, plot_count)
        plt.tight_layout()
        return fig, ax

    # ── layer ───────────────────────────────────────────────────────────────
    if type == 'layer':
        # concentration: stack directly; extent: use absolute values
        layer_ylabel = ylabel_abs if y == 'extent' else ylabel
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        _setup_ax(ax, logx, xs)
        bottom  = np.zeros(len(xs))
        handles = []
        for idx, (label, ydata) in enumerate(curves.items()):
            yabs = np.abs(ydata)
            lc   = style_obj.colors[idx % len(style_obj.colors)] if color else 'white'
            ht   = style_obj.hatch_styles[idx % len(style_obj.hatch_styles)]
            patch = ax.fill_between(xs, bottom, bottom + yabs,
                                    facecolor=lc, edgecolor='black',
                                    linewidth=style_obj.fill_linewidth, hatch=ht,
                                    label=label, alpha=style_obj.fill_alpha)
            handles.append(patch)
            bottom = bottom + yabs
        _ax_labels(ax, layer_ylabel)
        ax.legend(handles[::-1], [h.get_label() for h in handles[::-1]],
                  loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  frameon=False,
                  fontsize=style_obj._fs('legend') if style_obj.legend_fontsize is not None
                  else (style_obj.legend_fontsize_many if plot_count > 10
                  else style_obj.legend_fontsize_few))
        plt.tight_layout()
        return fig, ax

    # ── bilayer ─────────────────────────────────────────────────────────────
    # Concentration: organic species stack above x-axis, aqueous below
    #   (both positive values, divided by the x-axis baseline).
    # Extent: positive extents stack upward, negative downward (signed).
    if type == 'bilayer':
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('white')
        _setup_ax(ax, logx, xs)
        handles = []
        bot_pos = np.zeros(len(xs))   # upward baseline
        bot_neg = np.zeros(len(xs))   # downward baseline

        if y == 'concentration':
            # split curves by phase: organic → above (positive), aqueous → below (negative)
            # labels encode phase via '(org)' suffix added earlier
            org_curves = {lb: yd for lb, yd in curves.items() if lb.endswith('(org)')}
            aq_curves  = {lb: yd for lb, yd in curves.items() if not lb.endswith('(org)')}

            all_sp_order = list(org_curves.keys()) + list(aq_curves.keys())
            color_map = {lb: style_obj.colors[i % len(style_obj.colors)]
                         for i, lb in enumerate(all_sp_order)}
            hatch_map = {lb: style_obj.hatch_styles[i % len(style_obj.hatch_styles)]
                         for i, lb in enumerate(all_sp_order)}

            # organic above
            for label, ydata in org_curves.items():
                lc = color_map[label] if color else 'white'
                ht = hatch_map[label]
                patch = ax.fill_between(xs, bot_pos, bot_pos + ydata,
                                        facecolor=lc, edgecolor='black',
                                        linewidth=style_obj.fill_linewidth, hatch=ht,
                                        label=label, alpha=style_obj.fill_alpha)
                handles.append(patch)
                bot_pos = bot_pos + ydata

            # aqueous below (fill downward, display as positive magnitude)
            for label, ydata in aq_curves.items():
                lc = color_map[label] if color else 'white'
                ht = hatch_map[label]
                patch = ax.fill_between(xs, bot_neg - ydata, bot_neg,
                                        facecolor=lc, edgecolor='black',
                                        linewidth=style_obj.fill_linewidth, hatch=ht,
                                        label=label, alpha=style_obj.fill_alpha)
                handles.append(patch)
                bot_neg = bot_neg - ydata

            ax.axhline(0, color='black', linewidth=1.2)

            # y-axis: show absolute values on both sides
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):.3g}'))
            _ax_labels(ax, 'Concentration (M)  [org ↑ | aq ↓]')

        else:
            # extent: positive stack up, negative stack down
            for idx, (label, ydata) in enumerate(curves.items()):
                lc  = style_obj.colors[idx % len(style_obj.colors)] if color else 'white'
                ht  = style_obj.hatch_styles[idx % len(style_obj.hatch_styles)]
                pos = np.maximum(ydata, 0)
                neg = np.minimum(ydata, 0)
                patch = ax.fill_between(xs, bot_pos, bot_pos + pos,
                                        facecolor=lc, edgecolor='black',
                                        linewidth=style_obj.fill_linewidth, hatch=ht,
                                        label=label, alpha=style_obj.fill_alpha)
                if np.any(neg != 0):
                    ax.fill_between(xs, bot_neg + neg, bot_neg,
                                    facecolor=lc, edgecolor='black',
                                    linewidth=style_obj.fill_linewidth, hatch=ht,
                                    alpha=style_obj.fill_alpha)
                bot_pos = bot_pos + pos
                bot_neg = bot_neg + neg
                handles.append(patch)
            ax.axhline(0, color='black', linewidth=1.2)
            _ax_labels(ax, ylabel)

        ax.legend(handles[::-1], [h.get_label() for h in handles[::-1]],
                  loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  frameon=False,
                  fontsize=style_obj._fs('legend') if style_obj.legend_fontsize is not None
                  else (style_obj.legend_fontsize_many if plot_count > 10
                  else style_obj.legend_fontsize_few))
        plt.tight_layout()
        return fig, ax



# ---------------------------------------------------------------------------
# Speciation fraction plots
# ---------------------------------------------------------------------------

def _plot_fractions(reactions_obj, component, c0, maintain=None, logx=False,
                    n_points=20, select=None, type='plot', color=False,
                    init_scale=0.1, recheck=False, recheck_tol=1e-4,
                    recheck_density=5, presolver_timeout=2.0):
    """Implement reactions.fractions() – separated here to keep reactions.py tidy."""
    VALID = ('plot', 'data', 'layer', 'bilayer', 'norm-bilayer')
    if type not in VALID:
        raise InputError(f"type must be one of {VALID}, got {type!r}")

    frac_dict = reactions_obj._compute_fractions()
    style_obj = globals()['style']   # singleton; avoid shadowing by 'type' param

    def _to_neutral(name):
        s = species(name)
        s = re.sub(r'\s*\[[^\]]*[+\-][^\]]*\]\s*$', '', s)
        s = re.sub(r'\s*\([slgorg]+\)\s*$', '', s, flags=re.IGNORECASE)
        return s.strip()

    # resolve component:
    # 1. direct cluster key (e.g. 'C', 'H', 'Fe')
    # 2. neutral-stripped name as cluster key
    # 3. species name → find which cluster contains it
    comp_neutral = _to_neutral(component)
    if component in frac_dict:
        component_key = component
    elif comp_neutral in frac_dict:
        component_key = comp_neutral
    else:
        # try to find a cluster that contains this species
        sp_norm = species(component) if component else component
        matched = [k for k, members in frac_dict.items() if sp_norm in members]
        if not matched:
            # also try neutral form
            matched = [k for k, members in frac_dict.items()
                       if comp_neutral in members]
        if len(matched) == 1:
            component_key = matched[0]
        elif len(matched) > 1:
            # species spans multiple clusters — pick the smallest (most specific)
            component_key = min(matched, key=lambda k: len(frac_dict[k]))
        else:
            raise InputError(
                f"Component {component!r} not found in this reaction system.\n"
                f"Available cluster labels: {list(frac_dict.keys())}")

    component_species = frac_dict[component_key]

    if select is None: select = []
    if select: select = [species(str(s)) for s in select]

    data = reactions_obj._sweep_data(
        c0, y='concentration', logx=logx, logy=False, init_scale=init_scale,
        n_points=n_points, recheck=recheck, recheck_tol=recheck_tol,
        recheck_density=recheck_density, presolver_timeout=presolver_timeout,
        maintain=maintain)

    sweep_name = data['variable']
    xs         = np.array(data['x'])
    sp_concs   = data['concentrations']
    fixed_c0   = data['fixed_c0']

    visible = [s for s in component_species
               if not is_nonaqueous(s) and not is_electron(s)]
    if select:
        unknown = [s for s in select if s not in sp_concs]
        if unknown:
            import warnings
            warnings.warn(f"fractions select: not found: {unknown}")
        visible = [s for s in visible if s in select]

    total = np.zeros(len(xs))
    for s in component_species:
        if s in sp_concs: total += np.array(sp_concs[s], dtype=float)
    total_safe = np.where(total > 0, total, np.nan)

    frac_curves: Dict[str, np.ndarray] = {}
    for s in visible:
        if s in sp_concs:
            frac_curves[s] = np.array(sp_concs[s], dtype=float) / total_safe

    if type == 'data':
        return {**data, 'fractions': {s: list(v) for s, v in frac_curves.items()},
                'component': component_key}

    sweep_fmt = _format_species(sweep_name)
    comp_fmt  = _format_species(component_key)
    n_org     = sum(1 for s in component_species if is_organic(s))
    org_note  = f',  v_oa={fixed_c0.get("O/A","?")}' if n_org > 0 else ''
    rc_note   = f',  recheck={recheck_tol:.0e}' if recheck else ''

    if type == 'plot':
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        _setup_ax(ax, logx, xs)
        for idx, (s, ydata) in enumerate(frac_curves.items()):
            ls    = style_obj.line_styles[idx % len(style_obj.line_styles)]
            lc    = style_obj.colors[idx % len(style_obj.colors)] if color else 'black'
            fs    = _format_species(s)
            label = f'[{fs}](org)' if is_organic(s) else f'[{fs}]'
            ax.plot(xs, ydata, color=lc, linewidth=style_obj.linewidth, linestyle=ls, label=label)
        ax.set_ylim(-0.02, 1.05)
        subtitle = (f'({len(reactions_obj._reactions)} reaction(s),  '
                    f'{len(frac_curves)} species shown' + org_note + rc_note + ')')
        ax.set_xlabel(f'[{sweep_fmt}]₀  (M)', fontsize=style_obj._fs('x_title'))
        ax.set_ylabel(f'Fraction of total [{comp_fmt}]', fontsize=style_obj._fs('y_title'))
        ax.set_title(f'Speciation fractions of {comp_fmt}  vs  [{sweep_fmt}]₀\n{subtitle}', fontsize=style_obj._fs('x_title') + 1)
        n_shown = len(frac_curves)
        ax.tick_params(axis='x', labelsize=style_obj._fs('x_tick'))
        ax.tick_params(axis='y', labelsize=style_obj._fs('y_tick'))
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  frameon=False,
                  fontsize=style_obj._fs('legend') if style_obj.legend_fontsize is not None
                           else (style_obj.legend_fontsize_many if n_shown > 10
                           else style_obj.legend_fontsize_few))
        plt.tight_layout()
        return fig, ax

    if type == 'layer':
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        _setup_ax(ax, logx, xs)
        names   = list(frac_curves.keys())
        bottom  = np.zeros(len(xs))
        handles = []
        for idx, s in enumerate(names):
            ydata = frac_curves[s]
            lc    = style_obj.colors[idx % len(style_obj.colors)] if color else 'white'
            ht    = style_obj.hatch_styles[idx % len(style_obj.hatch_styles)]
            fs    = _format_species(s)
            label = f'[{fs}](org)' if is_organic(s) else f'[{fs}]'
            patch = ax.fill_between(xs, bottom, bottom + ydata,
                                    facecolor=lc, edgecolor='black',
                                    linewidth=style_obj.fill_linewidth, hatch=ht,
                                    label=label, alpha=style_obj.fill_alpha)
            handles.append(patch)
            bottom = bottom + ydata
        ax.set_ylim(0, 1)
        subtitle = (f'({len(reactions_obj._reactions)} reaction(s),  '
                    f'{len(frac_curves)} species shown' + org_note + rc_note + ')')
        ax.set_xlabel(f'[{sweep_fmt}]₀  (M)', fontsize=style_obj._fs('x_title'))
        ax.set_ylabel(f'Fraction of total [{comp_fmt}]', fontsize=style_obj._fs('y_title'))
        ax.set_title(f'Speciation fractions of {comp_fmt}  vs  [{sweep_fmt}]₀\n{subtitle}', fontsize=style_obj._fs('x_title') + 1)
        ax.tick_params(axis='x', labelsize=style_obj._fs('x_tick'))
        ax.tick_params(axis='y', labelsize=style_obj._fs('y_tick'))
        ax.legend(handles[::-1], [h.get_label() for h in handles[::-1]],
                  loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  frameon=False,
                  fontsize=style_obj._fs('legend') if style_obj.legend_fontsize is not None
                  else (8 if len(frac_curves) > 10 else 9))
        plt.tight_layout()
        return fig, ax

    # bilayer and norm-bilayer
    aq_species  = [s for s in frac_curves if not is_organic(s)]
    org_species = [s for s in frac_curves if is_organic(s)]
    if not org_species:
        raise InputError(
            f"type='{type}' requires organic species, but none were found.\n"
            "Add a reaction with organic '(org)' species to use bilayer plots.")

    total_aq  = np.zeros(len(xs))
    total_org = np.zeros(len(xs))
    for s in component_species:
        if s in sp_concs:
            arr = np.array(sp_concs[s], dtype=float)
            if is_organic(s): total_org += arr
            else:             total_aq  += arr

    if type == 'bilayer':
        total_all   = np.where(total_aq + total_org > 0, total_aq + total_org, np.nan)
        aq_frac_g   = {s: np.array(sp_concs[s], dtype=float) / total_all
                       for s in aq_species  if s in sp_concs}
        org_frac_g  = {s: np.array(sp_concs[s], dtype=float) / total_all
                       for s in org_species if s in sp_concs}
        aq_norm     = aq_frac_g
        org_norm    = org_frac_g
        y_label     = f'Fraction of total [{comp_fmt}]'
        title_pre   = 'Bilayer speciation'
    else:   # norm-bilayer (type == 'norm-bilayer')
        aq_norm  = {s: np.array(sp_concs[s], dtype=float) /
                       np.where(total_aq > 0, total_aq, np.nan)
                    for s in aq_species  if s in sp_concs}
        org_norm = {s: np.array(sp_concs[s], dtype=float) /
                       np.where(total_org > 0, total_org, np.nan)
                    for s in org_species if s in sp_concs}
        y_label   = 'Fraction within phase'
        title_pre = 'Normalised bilayer speciation'

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    _setup_ax(ax, logx, xs)

    all_sp_ord = org_species + aq_species
    color_map  = {s: style_obj.colors[i % len(style_obj.colors)]  for i, s in enumerate(all_sp_ord)}
    hatch_map  = {s: style_obj.hatch_styles[i % len(style_obj.hatch_styles)] for i, s in enumerate(all_sp_ord)}

    handles = []
    bottom  = np.zeros(len(xs))
    for s in org_species:
        if s not in org_norm: continue
        ydata = org_norm[s]
        lc    = color_map[s] if color else 'white'
        ht    = hatch_map[s]
        fs    = _format_species(s)
        patch = ax.fill_between(xs, bottom, bottom + np.nan_to_num(ydata),
                                facecolor=lc, edgecolor='black',
                                linewidth=style_obj.fill_linewidth, hatch=ht,
                                alpha=style_obj.fill_alpha, label=f'[{fs}](org)')
        handles.append(patch)
        bottom = bottom + np.nan_to_num(ydata)

    bottom = np.zeros(len(xs))
    for s in aq_species:
        if s not in aq_norm: continue
        ydata = aq_norm[s]
        lc    = color_map[s] if color else 'white'
        ht    = hatch_map[s]
        fs    = _format_species(s)
        patch = ax.fill_between(xs, bottom, bottom - np.nan_to_num(ydata),
                                facecolor=lc, edgecolor='black',
                                linewidth=style_obj.fill_linewidth, hatch=ht,
                                alpha=style_obj.fill_alpha, label=f'[{fs}]')
        handles.append(patch)
        bottom = bottom - np.nan_to_num(ydata)

    ax.axhline(0, color='black', linewidth=1.2)
    ax.set_ylim(-1, 1)

    from matplotlib.ticker import FixedLocator
    yticks = [t for t in ax.get_yticks() if -1 <= t <= 1]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.set_yticklabels([f'{abs(t):.2g}' for t in yticks])

    subtitle = (f'({len(reactions_obj._reactions)} reaction(s),  '
                f'{len(org_species)} org + {len(aq_species)} aq species'
                + org_note + rc_note + ')')
    ax.set_xlabel(f'[{sweep_fmt}]₀  (M)', fontsize=style_obj._fs('x_title'))
    ax.set_ylabel(y_label, fontsize=style_obj._fs('y_title'))
    ax.set_title(f'{title_pre} of {comp_fmt}  vs  [{sweep_fmt}]₀\n{subtitle}', fontsize=style_obj._fs('x_title') + 1)
    n_handles = len(handles)
    ax.tick_params(axis='x', labelsize=style_obj._fs('x_tick'))
    ax.tick_params(axis='y', labelsize=style_obj._fs('y_tick'))
    ax.legend(handles, [h.get_label() for h in handles],
              loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
              frameon=False,
              fontsize=style_obj._fs('legend') if style_obj.legend_fontsize is not None
                       else (style_obj.legend_fontsize_many if n_handles > 10
                       else style_obj.legend_fontsize_few))
    plt.tight_layout()
    return fig, ax