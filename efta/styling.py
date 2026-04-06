"""
efta.styling
============
Helpers for modifying the :data:`efta.style` singleton at runtime.

Functions
---------
randomize_linestyle(range=[])
    Shuffle a slice of ``style.line_styles`` in place.

randomize_color(range=[])
    Shuffle a slice of ``style.colors`` in place.

randomize_pattern(range=[])
    Shuffle a slice of ``style.hatch_styles`` in place.

coloring(colors='default')
    Apply a named preset palette or a custom list of colour strings.
    Darker and lighter variants are derived automatically so the palette
    always contains exactly 16 entries.

    Built-in presets
    ~~~~~~~~~~~~~~~~
    'default'      – original efta palette (black anchor + warm/cool accents)
    'colorblind'   – Okabe-Ito 8-colour set, safe under all colorblindness types
    'pastel'       – soft, desaturated tones; suits filled / area plots
    'dark'         – deep, saturated colours; suits presentations / dark backgrounds
    'earth'        – ochres, clay reds, muted greens; suits geochemistry / env. science
    'ocean'        – blues, teals, aquamarines; suits aqueous chemistry plots
    'monochrome'   – spread of greys from near-black to near-white
    'vivid'        – fully saturated primaries and secondaries; maximum contrast
"""

from __future__ import annotations

import colorsys
import random
from typing import List, Sequence, Union

from .errors import InputError
from .plotting import style   # the PlotStyle singleton

__all__ = [
    'randomize_linestyle', 'randomize_color', 'randomize_pattern',
    'coloring', 'PRESETS',
    'fontsize', 'legend_fontsize',
    'x_fontsize', 'x_tick_fontsize', 'x_title_fontsize',
    'y_fontsize', 'y_tick_fontsize', 'y_title_fontsize',
]


# ---------------------------------------------------------------------------
# Built-in preset palettes  (8 base colours each; expanded to 16 at call time)
# ---------------------------------------------------------------------------

PRESETS: dict = {
    # Original efta palette
    'default': [
        '#000000', '#F8CD5A', '#364C86', '#B3514E',
        '#61A363', '#D7D0BD', '#C49A1A', '#1E2F5C',
    ],

    # Okabe-Ito (2008) — tested for deuteranopia, protanopia, tritanopia
    # Order: orange, sky-blue, green, yellow, blue, vermillion, pink, black
    'colorblind': [
        '#E69F00', '#56B4E9', '#009E73', '#F0E442',
        '#0072B2', '#D55E00', '#CC79A7', '#000000',
    ],

    # Soft, desaturated — good for filled / stacked area plots
    'pastel': [
        '#F4A9A8', '#A8D8EA', '#A8E6CF', '#FFEAA7',
        '#B8B5FF', '#FFB347', '#C3B1E1', '#D5E8D4',
    ],

    # Deep, saturated — good for presentations and dark backgrounds
    'dark': [
        '#1B1B2F', '#162447', '#1F4068', '#1B262C',
        '#0F3460', '#533483', '#E94560', '#C84B31',
    ],

    # Ochres, clay reds, muted greens — geochemistry / environmental science
    'earth': [
        '#6B4226', '#A0522D', '#C68642', '#E8B84B',
        '#556B2F', '#8B7355', '#CD853F', '#D2691E',
    ],

    # Blues, teals, aquamarines — aqueous chemistry
    'ocean': [
        '#03045E', '#0077B6', '#00B4D8', '#90E0EF',
        '#006D77', '#83C5BE', '#1A759F', '#168AAD',
    ],

    # Greys pre-spread across the full lightness range
    'monochrome': [
        '#0D0D0D', '#2B2B2B', '#4A4A4A', '#6B6B6B',
        '#8C8C8C', '#ADADAD', '#CECECE', '#EFEFEF',
    ],

    # Fully saturated — maximum perceptual contrast
    'vivid': [
        '#E63946', '#F4A261', '#E9C46A', '#2A9D8F',
        '#264653', '#457B9D', '#A8DADC', '#6A0572',
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_range(range_arg: list, length: int):
    """
    Convert a ``range=[i, j]`` argument into a (start, stop) slice pair.

    - ``[]``        -> whole list  (0, length)
    - ``[i]``       -> from i to end
    - ``[i, j]``    -> from i to j  (inclusive -> exclusive stop = j+1)
    """
    if not range_arg:
        return 0, length
    if len(range_arg) == 1:
        return int(range_arg[0]), length
    i, j = int(range_arg[0]), int(range_arg[1])
    i = max(0, i)
    j = min(length - 1, j)
    return i, j + 1


def _hex_to_rgb(hex_color: str):
    """Return (r, g, b) floats in [0, 1] from a hex string like '#RRGGBB'."""
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r / 255.0, g / 255.0, b / 255.0


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """Return '#RRGGBB' from (r, g, b) floats in [0, 1]."""
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return '#{:02X}{:02X}{:02X}'.format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def _darken(hex_color: str, factor: float = 0.55) -> str:
    """Return a darker version of *hex_color* by scaling the L channel in HLS."""
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return _rgb_to_hex(*colorsys.hls_to_rgb(h, max(0.0, l * factor), s))


def _lighten(hex_color: str, factor: float = 1.55) -> str:
    """Return a lighter version of *hex_color* by scaling the L channel in HLS."""
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return _rgb_to_hex(*colorsys.hls_to_rgb(h, min(1.0, l * factor), s))


def _lightness(hex_color: str) -> float:
    """Return the HLS lightness (0-1) of a hex colour."""
    r, g, b = _hex_to_rgb(hex_color)
    _, l, _ = colorsys.rgb_to_hls(r, g, b)
    return l


def _derive(base: str, dark_f: float, light_f: float) -> str:
    """
    Derive a variant of *base* that is guaranteed to differ from it.
    For very dark colours (L < 0.15) always lighten; for very light
    colours (L > 0.85) always darken; otherwise darken first.
    """
    l = _lightness(base)
    if l < 0.15:
        return _lighten(base, light_f)
    if l > 0.85:
        return _darken(base, dark_f)
    return _darken(base, dark_f)


def _expand_to_16(base_colors: List[str]) -> List[str]:
    """
    Expand an arbitrary-length list of base colours to exactly 16 slots.

    If >= 16 entries are supplied, the first 16 are taken as-is.
    Otherwise, progressively derived variants are generated by cycling
    through the base colours until 16 unique slots are filled.
    Very dark base colours are lightened rather than darkened to avoid
    producing duplicate near-black entries.
    """
    if len(base_colors) >= 16:
        return list(base_colors[:16])

    palette: List[str] = list(base_colors)
    palette_set: set = set(palette)

    shade_steps = [
        (0.60, 1.45),
        (0.38, 1.65),
        (0.75, 1.28),
        (0.22, 1.80),
        (0.85, 1.15),
        (0.12, 1.90),
        (0.50, 1.55),
        (0.30, 1.70),
    ]

    for dark_f, light_f in shade_steps:
        for b in base_colors:
            if len(palette) >= 16:
                break
            candidate = _derive(b, dark_f, light_f)
            if candidate not in palette_set:
                palette.append(candidate)
                palette_set.add(candidate)
        for b in base_colors:
            if len(palette) >= 16:
                break
            l = _lightness(b)
            # for the light pass, invert direction for very light bases
            candidate = _lighten(b, light_f) if l < 0.85 else _darken(b, dark_f)
            if candidate not in palette_set:
                palette.append(candidate)
                palette_set.add(candidate)
        if len(palette) >= 16:
            break

    # last resort: if still short, force-add slightly offset shades
    offset = 0.05
    while len(palette) < 16:
        for b in base_colors:
            if len(palette) >= 16:
                break
            r, g, b2 = _hex_to_rgb(b)
            hh, ll, ss = colorsys.rgb_to_hls(r, g, b2)
            ll_new = min(1.0, max(0.0, ll + offset))
            candidate = _rgb_to_hex(*colorsys.hls_to_rgb(hh, ll_new, ss))
            if candidate not in palette_set:
                palette.append(candidate)
                palette_set.add(candidate)
        offset += 0.05

    return palette[:16]


def _normalise_color(c: str) -> str:
    """Convert any matplotlib colour string to '#RRGGBB' hex."""
    c = c.strip()
    if not c.startswith('#'):
        try:
            import matplotlib.colors as _mc
            c = _mc.to_hex(c)
        except Exception:
            pass
    return c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def randomize_linestyle(range: list = []) -> None:
    """
    Shuffle ``style.line_styles`` entries in the given index range in place.

    Parameters
    ----------
    range : list, optional
        ``[]``     -> shuffle all entries  (default)
        ``[i]``    -> shuffle from index *i* to the end
        ``[i, j]`` -> shuffle entries *i* through *j* (inclusive)

    Examples
    --------
    >>> randomize_linestyle()        # shuffle all
    >>> randomize_linestyle([0, 3])  # shuffle first 4 entries only
    """
    start, stop = _resolve_range(range, len(style.line_styles))
    segment = style.line_styles[start:stop]
    random.shuffle(segment)
    style.line_styles[start:stop] = segment


def randomize_color(range: list = []) -> None:
    """
    Shuffle ``style.colors`` entries in the given index range in place.

    Parameters
    ----------
    range : list, optional
        ``[]``     -> shuffle all entries  (default)
        ``[i]``    -> shuffle from index *i* to the end
        ``[i, j]`` -> shuffle entries *i* through *j* (inclusive)

    Examples
    --------
    >>> randomize_color()          # shuffle all 16 colour slots
    >>> randomize_color([8, 15])   # shuffle only the second half
    """
    start, stop = _resolve_range(range, len(style.colors))
    segment = style.colors[start:stop]
    random.shuffle(segment)
    style.colors[start:stop] = segment


def randomize_pattern(range: list = []) -> None:
    """
    Shuffle ``style.hatch_styles`` entries in the given index range in place.

    Parameters
    ----------
    range : list, optional
        ``[]``     -> shuffle all entries  (default)
        ``[i]``    -> shuffle from index *i* to the end
        ``[i, j]`` -> shuffle entries *i* through *j* (inclusive)

    Examples
    --------
    >>> randomize_pattern()        # shuffle all hatch patterns
    >>> randomize_pattern([0, 5])  # shuffle first 6 patterns
    """
    start, stop = _resolve_range(range, len(style.hatch_styles))
    segment = style.hatch_styles[start:stop]
    random.shuffle(segment)
    style.hatch_styles[start:stop] = segment


def coloring(colors: Union[str, Sequence[str]] = 'default') -> None:
    """
    Apply a colour palette to :data:`efta.style`.

    The palette is always expanded to exactly 16 slots; darker and lighter
    variants are derived automatically from the base colours.

    Parameters
    ----------
    colors : str or sequence of str, default ``'default'``
        **Named preset** (string):

        +------------------+----------------------------------------------+
        | ``'default'``    | Original efta palette                        |
        +------------------+----------------------------------------------+
        | ``'colorblind'`` | Okabe-Ito 8-colour set, safe under all       |
        |                  | types of colour vision deficiency            |
        +------------------+----------------------------------------------+
        | ``'pastel'``     | Soft, desaturated tones; suits area plots    |
        +------------------+----------------------------------------------+
        | ``'dark'``       | Deep, saturated; suits dark backgrounds      |
        +------------------+----------------------------------------------+
        | ``'earth'``      | Ochres, clay reds, muted greens              |
        +------------------+----------------------------------------------+
        | ``'ocean'``      | Blues, teals, aquamarines                    |
        +------------------+----------------------------------------------+
        | ``'monochrome'`` | Greys from near-black to near-white          |
        +------------------+----------------------------------------------+
        | ``'vivid'``      | Fully saturated; maximum perceptual contrast |
        +------------------+----------------------------------------------+

        **Custom list** — any number of hex or named matplotlib colour
        strings.  Supply as few as 1 (monochrome derivation) or 16+
        (taken as-is).

    Examples
    --------
    >>> coloring()                          # restore default
    >>> coloring('colorblind')              # Okabe-Ito preset
    >>> coloring('ocean')                   # ocean blues
    >>> coloring(['#E63946', '#457B9D'])    # 2 custom colours -> 16 derived
    >>> coloring('#457B9D')                 # 1 colour -> 16 monochrome shades
    """
    # --- named preset or bare colour string ---
    if isinstance(colors, str):
        name = colors.strip().lower()
        if name in PRESETS:
            style.colors = _expand_to_16(list(PRESETS[name]))
            return
        # treat as a single custom colour string (e.g. '#457B9D' or 'red')
        c = _normalise_color(colors)
        style.colors = _expand_to_16([c])
        return

    # --- list / sequence of colour strings ---
    base = [_normalise_color(c) for c in colors]

    if not base:
        raise InputError(
            "coloring() requires at least one colour string.\n"
            "Example: coloring(['#E63946', '#457B9D'])")

    style.colors = _expand_to_16(base)

# ---------------------------------------------------------------------------
# Font size setters
# ---------------------------------------------------------------------------

def _set_fs(attr: str, value: float) -> None:
    """Validate and set a font size attribute on the style singleton."""
    value = float(value)
    if value <= 0:
        raise InputError(f"Font size must be positive, got {value}.")
    setattr(style, attr, value)


def fontsize(size: float) -> None:
    """
    Set the global fallback font size used for all text elements that do
    not have a more specific override set.

    Parameters
    ----------
    size : float
        Font size in points.

    Examples
    --------
    >>> fontsize(12)
    """
    _set_fs('fontsize', size)


def legend_fontsize(size: float) -> None:
    """
    Set the legend font size.

    Overrides the ``legend_fontsize_many`` / ``legend_fontsize_few``
    logic when set.  Pass ``None`` to restore the automatic behaviour.

    Parameters
    ----------
    size : float or None
    """
    if size is None:
        style.legend_fontsize = None
        return
    _set_fs('legend_fontsize', size)


def x_fontsize(size: float) -> None:
    """
    Set both x-axis tick label and title font sizes together.

    Parameters
    ----------
    size : float
    """
    _set_fs('x_tick_fontsize', size)
    _set_fs('x_title_fontsize', size)


def x_tick_fontsize(size: float) -> None:
    """
    Set the x-axis tick label font size only.

    Parameters
    ----------
    size : float
    """
    _set_fs('x_tick_fontsize', size)


def x_title_fontsize(size: float) -> None:
    """
    Set the x-axis title (label) font size only.

    Parameters
    ----------
    size : float
    """
    _set_fs('x_title_fontsize', size)


def y_fontsize(size: float) -> None:
    """
    Set both y-axis tick label and title font sizes together.

    Parameters
    ----------
    size : float
    """
    _set_fs('y_tick_fontsize', size)
    _set_fs('y_title_fontsize', size)


def y_tick_fontsize(size: float) -> None:
    """
    Set the y-axis tick label font size only.

    Parameters
    ----------
    size : float
    """
    _set_fs('y_tick_fontsize', size)


def y_title_fontsize(size: float) -> None:
    """
    Set the y-axis title (label) font size only.

    Parameters
    ----------
    size : float
    """
    _set_fs('y_title_fontsize', size)
