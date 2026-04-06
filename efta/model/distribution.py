"""
efta.model.distribution
=======================
:func:`distribution` — plot a parameter distribution as a styled histogram
with optional best-fit probability distribution overlay.
"""

from __future__ import annotations

from typing import List, Optional, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ..plotting import style as _style

__all__ = ['distribution']

# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------
# Each entry: (display_name, scipy_dist_or_None, needs_positive, needs_unit_bounded)
# needs_positive=True  → data is shifted to (0, ∞) before fitting
# needs_unit_bounded=True → data is scaled to (0, 1) before fitting
# None for scipy_dist means it is handled specially (Dirac delta, Uniform).

_ALL_CANDIDATES = {
    # ── original ──────────────────────────────────────────────────────────────
    'Normal':           (stats.norm,          False, False),
    'Log-normal':       (stats.lognorm,       True,  False),
    'Exponential':      (stats.expon,         True,  False),
    'Gamma':            (stats.gamma,         True,  False),
    'Weibull':          (stats.weibull_min,   True,  False),
    'Beta':             (stats.beta,          False, True),
    'Cauchy':           (stats.cauchy,        False, False),
    'Laplace':          (stats.laplace,       False, False),
    # ── new continuous ────────────────────────────────────────────────────────
    'Uniform':          (stats.uniform,       False, False),
    'Student-t':        (stats.t,             False, False),
    'Inverse-Gaussian': (stats.invgauss,      True,  False),
    'Log-logistic':     (stats.fisk,          True,  False),
    'Burr':             (stats.burr,          True,  False),
    'Wald':             (stats.wald,          True,  False),
    'Pareto':           (stats.pareto,        True,  False),
    # ── chemistry-relevant additions ──────────────────────────────────────
    'Gamma':            (stats.gamma,         True,  False),
    'Chi-squared':      (stats.chi2,          True,  False),
    'Inverse-Gamma':    (stats.invgamma,      True,  False),
    # ── special ───────────────────────────────────────────────────────────────
    'Dirac':            (None,                False, False),  # spike, handled separately
}

#: Default set — only the most common for chemical parameter distributions
_DEFAULT_CANDIDATES = ['Normal', 'Log-normal', 'Exponential',
                        'Gamma', 'Chi-squared', 'Inverse-Gamma']

#: Threshold for Dirac detection: relative range < this → treat as delta
_DIRAC_THRESHOLD = 1e-6


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _is_dirac(data: np.ndarray) -> bool:
    rng = np.ptp(data)
    return rng == 0 or (len(data) > 1 and
                        rng / (abs(np.mean(data)) + 1e-300) < _DIRAC_THRESHOLD)


def _best_fit(data: np.ndarray, candidates: List[str]):
    """
    Fit candidate distributions and return the one with the highest
    log-likelihood.  Returns ``(name, dist, params)`` or
    ``('Dirac', None, (mean,))`` for a spike.
    """
    if _is_dirac(data):
        return 'Dirac', None, (float(np.mean(data)),)

    shifted = data - data.min() + 1e-10 if data.min() <= 0 else data
    lo, hi  = data.min(), data.max()
    span    = hi - lo if hi > lo else 1.0
    scaled  = (data - lo) / span   # maps to [0, 1] for Beta

    best_name, best_dist, best_params, best_ll = None, None, None, -np.inf

    for name in candidates:
        if name == 'Dirac':
            continue
        if name not in _ALL_CANDIDATES:
            raise ValueError(
                f"Unknown distribution {name!r}. "
                f"Available: {sorted(_ALL_CANDIDATES)}")
        dist, needs_pos, needs_unit = _ALL_CANDIDATES[name]
        if dist is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if needs_unit:
                    fit_data = np.clip(scaled, 1e-9, 1 - 1e-9)
                    params   = dist.fit(fit_data, floc=0, fscale=1)
                    ll       = np.sum(dist.logpdf(fit_data, *params))
                elif needs_pos:
                    params = dist.fit(shifted, floc=0)
                    ll     = np.sum(dist.logpdf(shifted, *params))
                else:
                    params = dist.fit(data)
                    ll     = np.sum(dist.logpdf(data, *params))
            if np.isfinite(ll) and ll > best_ll:
                best_ll     = ll
                best_name   = name
                best_dist   = dist
                best_params = params
        except Exception:
            continue

    return best_name, best_dist, best_params


def _eval_pdf(name, dist, params, xs, data):
    """Evaluate PDF of a fitted distribution on xs, accounting for transforms."""
    if dist is None:
        return None
    _, needs_pos, needs_unit = _ALL_CANDIDATES[name]
    lo, hi = data.min(), data.max()
    span   = hi - lo if hi > lo else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if needs_unit:
            xs_t = (xs - lo) / span
            ys   = dist.pdf(xs_t, *params) / span
        elif needs_pos:
            shift = -data.min() + 1e-10 if data.min() <= 0 else 0.0
            ys    = dist.pdf(xs + shift, *params)
        else:
            ys = dist.pdf(xs, *params)
    return np.where(np.isfinite(ys), ys, 0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distribution(data:         Union[List[float], np.ndarray],
                 fit:          bool = True,
                 distributions: Optional[List[str]] = None,
                 title:        str  = None,
                 xlabel:       str  = 'Value',
                 bins:         int  = None,
                 ax                 = None,
                 color:        str  = None,
                 log_scale:    bool = False) -> tuple:
    """
    Plot a histogram of *data* with an optional best-fit distribution overlay.

    Automatically detects and labels a **Dirac delta** when all values are
    near-identical (relative range < 1e-6) — common for discrete parameters
    from bootstrap.

    Parameters
    ----------
    data : list or array
        The values to plot.
    fit : bool, optional
        If True, fit and overlay the best-matching probability distribution.
    distributions : list of str, optional
        Candidate distribution names to consider.  If ``None``, all
        distributions are tried.  Available names::

            'Normal', 'Log-normal', 'Exponential', 'Gamma', 'Weibull',
            'Beta', 'Cauchy', 'Laplace',                      # original
            'Uniform', 'Student-t', 'Inverse-Gaussian',       # new
            'Log-logistic', 'Burr', 'Wald', 'Pareto',         # new
            'Dirac'                                            # auto-detected

        ``'Dirac'`` in the list is ignored (it is always auto-detected).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label (default ``'Value'``).
    bins : int, optional
        Number of histogram bins.  Defaults to Sturges' rule.
    ax : matplotlib Axes, optional
        Axes to draw on.  If None, a new figure is created.
    color : str, optional
        Bar colour.  Defaults to the first colour in the style palette.
    log_scale : bool, optional
        If True, use log₁₀ scale on the x-axis (data must be positive).

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> fig, ax = distribution(results[0].distributions[0], xlabel='Ka',
    ...                        log_scale=True)
    >>> # restrict candidates
    >>> fig, ax = distribution(data, distributions=['Normal', 'Gamma'])
    >>> # Dirac delta auto-detected for constant discrete params
    >>> fig, ax = distribution([3.0]*50, xlabel='a (stoich)')
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("No finite values in data.")

    candidates = [c for c in (distributions or _DEFAULT_CANDIDATES)
                  if c != 'Dirac']

    is_spike = _is_dirac(data)

    # ── figure setup ─────────────────────────────────────────────────────────
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('white')
    else:
        fig = ax.figure

    ax.set_facecolor('#f8f8f8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.8)
    ax.grid(axis='y', color='white', linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    bar_color = color or _style.colors[0]
    fit_color = _style.colors[3] if len(_style.colors) > 3 else '#E63946'
    n_bins    = bins or max(8, int(1 + 3.322 * np.log10(max(len(data), 2))))

    # log transform if requested
    if log_scale and np.all(data > 0):
        hist_data          = np.log10(data)
        x_label            = f'log₁₀({xlabel})'
        # In log-space the data is linearised — only Normal makes sense.
        # Override candidates to avoid nonsensical log-normal-of-log fits.
        candidates = ['Normal']
    else:
        hist_data = data
        x_label   = xlabel
        log_scale = False

    fit_name = ''
    ax2      = None

    if is_spike:
        # ── Dirac delta ───────────────────────────────────────────────────────
        val  = float(np.mean(hist_data))
        raw  = float(np.mean(data))
        span = max(abs(val) * 0.15, 0.5)
        ax.bar([val], [len(data)], width=span * 0.25,
               color=bar_color, alpha=0.75, edgecolor='white',
               linewidth=0.6, zorder=2)
        ax.set_xlim(val - span, val + span)

        # secondary axis: Dirac delta label
        ax2 = ax.twinx()
        ax2.axvline(val, color=fit_color, linewidth=_style.linewidth,
                    zorder=3, label=f'Dirac  δ(x − {raw:.4g})')
        ax2.set_ylabel('Density (PDF)', fontsize=_style._fs('y_title'),
                       color=fit_color)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelsize=_style._fs('y_tick'),
                        colors=fit_color)
        ax2.spines['right'].set_edgecolor(fit_color)
        ax2.spines['right'].set_linewidth(0.8)
        for sp in ('top', 'left', 'bottom'):
            ax2.spines[sp].set_visible(False)
        ax2.legend(frameon=False, fontsize=_style._fs('legend'),
                   loc='upper left')
        fit_name = f'Dirac  δ(x − {raw:.4g})'

    else:
        # ── binning algorithm ─────────────────────────────────────────────────
        # Sort data, compute delta between each adjacent unique pair.
        # The maximum delta is used as the bin width — guaranteeing that
        # every bin spans at least one gap, so there are never empty bars
        # between populated ones.
        sorted_unique = np.unique(hist_data)
        lo, hi        = sorted_unique[0], sorted_unique[-1]

        if len(sorted_unique) == 1:
            # all identical — single bar
            bar_width = max(abs(lo) * 0.15, 0.5)
            ax.bar([lo], [len(hist_data)], width=bar_width * 0.6,
                   color=bar_color, alpha=0.75, edgecolor='white',
                   linewidth=0.6, zorder=2, align='center')
            ax.set_xlim(lo - bar_width, lo + bar_width)
            counts    = np.array([len(hist_data)])
            bin_edges = np.array([lo - bar_width * 0.3, lo + bar_width * 0.3])
        else:
            max_delta = float(np.max(np.diff(sorted_unique)))
            bin_edges = np.arange(lo, hi + max_delta * 1.001, max_delta)
            counts, bin_edges, _ = ax.hist(
                hist_data, bins=bin_edges, density=False,
                color=bar_color, alpha=0.75,
                edgecolor='white', linewidth=0.6, zorder=2,
            )

        if fit and candidates:
            fname, fdist, fparams = _best_fit(hist_data, candidates)
            if fdist is not None:
                xs = np.linspace(bin_edges[0], bin_edges[-1], 400)
                ys = _eval_pdf(fname, fdist, fparams, xs, hist_data)
                if ys is not None and np.any(ys > 0):
                    # ── peak alignment ────────────────────────────────────────
                    # Only align when there is a clear single mode.
                    # Multi-modal or flat histograms should not be shifted.
                    bin_centres   = (bin_edges[:-1] + bin_edges[1:]) / 2
                    max_count     = np.max(counts)
                    mode_bins     = np.where(counts == max_count)[0]
                    is_clear_mode = (len(mode_bins) == 1 and
                                     max_count > np.mean(counts) * 1.5)

                    if is_clear_mode:
                        hist_peak_x = float(bin_centres[mode_bins[0]])
                        xs_fine     = np.linspace(bin_edges[0], bin_edges[-1], 2000)
                        ys_fine     = _eval_pdf(fname, fdist, fparams, xs_fine, hist_data)
                        if ys_fine is not None and np.any(ys_fine > 0):
                            pdf_peak_x  = float(xs_fine[np.argmax(ys_fine)])
                            shift       = hist_peak_x - pdf_peak_x
                        else:
                            shift = 0.0
                    else:
                        shift = 0.0

                    xs_shifted = xs + shift
                    ys_shifted = _eval_pdf(fname, fdist, fparams, xs_shifted, hist_data)
                    if ys_shifted is None or not np.any(ys_shifted > 0):
                        xs_shifted = xs
                        ys_shifted = ys

                    ax2 = ax.twinx()
                    ax2.plot(xs_shifted, ys_shifted, color=fit_color,
                             linewidth=_style.linewidth, zorder=3,
                             label=f'Best fit: {fname}')
                    ax2.set_ylabel('Density (PDF)',
                                   fontsize=_style._fs('y_title'),
                                   color=fit_color)
                    ax2.tick_params(axis='y', labelsize=_style._fs('y_tick'),
                                    colors=fit_color)
                    ax2.spines['right'].set_edgecolor(fit_color)
                    ax2.spines['right'].set_linewidth(0.8)
                    for sp in ('top', 'left', 'bottom'):
                        ax2.spines[sp].set_visible(False)
                    ax2.set_ylim(bottom=0)
                    ax2.legend(frameon=False, fontsize=_style._fs('legend'),
                               loc='upper left')
                    fit_name = fname

    # ── statistics box ────────────────────────────────────────────────────────
    mu, sd = np.mean(data), np.std(data)
    med    = np.median(data)
    stats_text = (f'n = {len(data)}\n'
                  f'mean = {mu:.4g}\n'
                  f'std  = {sd:.4g}\n'
                  f'med  = {med:.4g}')
    if fit_name:
        stats_text += f'\nfit: {fit_name}'

    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=_style._fs('legend'),
            family='monospace',
            color='#444444',
            bbox=dict(facecolor='white', edgecolor='#dddddd',
                      boxstyle='round,pad=0.4', alpha=0.85),
            zorder=4)

    # ── axis labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(x_label, fontsize=_style._fs('x_title'))
    ax.set_ylabel('Count',  fontsize=_style._fs('y_title'))
    ax.tick_params(axis='x', labelsize=_style._fs('x_tick'))
    ax.tick_params(axis='y', labelsize=_style._fs('y_tick'))

    if title:
        ax.set_title(title, fontsize=_style._fs('x_title') + 1, pad=10)

    if created:
        plt.tight_layout()

    return fig, ax
