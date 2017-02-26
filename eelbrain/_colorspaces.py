"""Data-specific colormaps

Meas values
-----------
B
    Magnetic field strength (MEG).
V
    Voltage (EEG).
p
    Probability (statistics).
r, t, f
    Statistic (correlation, t- and f- values).
"""
from __future__ import division

from itertools import izip

from colormath.color_objects import LCHabColor, sRGBColor
from colormath.color_conversions import convert_color
from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def lch_to_rgb(lightness, chroma, hue):
    "Convert Lightness/Chroma/Hue color representation to RGB"
    psych = LCHabColor(lightness, chroma, hue * 360)
    rgb = convert_color(psych, sRGBColor)
    return rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b


def make_seq_cmap(seq, val, name):
    """Colormap from sequence of RGB values

    Parameters
    ----------
    seq : iterator
        Each entry is either an RGB tuple (if pre- and post color are
        identical) or a tuple with two RGB tuples (separate pre- and post-
        colors).
    val : iterator
        For each entry in ``seq``, the coordinate on the colormap.
    name : str
        Colormap name.
    """
    red = []
    green = []
    blue = []
    for v, col in zip(val, seq):
        if len(col) == 3:
            r0, g0, b0 = col
            r1, g1, b1 = col
        elif len(col) == 2:
            (r0, g0, b0), (r1, g1, b1) = col
        else:
            raise ValueError('col in seq: %s' % str(col))
        red.append((v, r0, r1))
        green.append((v, g0, g1))
        blue.append((v, b0, b1))
    cdict = {'red': red, 'green': green, 'blue': blue}
    return LinearSegmentedColormap(name, cdict)


def twoway_cmap(n1, hue_start=0.1, hue_shift=0.5, name=None, hues=None):
    """Create colormap for two-way interaction

    Parameters
    ----------
    n1 : int
        Number of levels on the first factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    name : str
        Name of the colormap.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    """
    # within each hue, create values for [-1, -0.5, 0.5, 1]
    # return list of [i, (r, g, b), (r, g, b)]
    if hues is None:
        hues = np.linspace(hue_start, hue_start + 1, n1, False) % 1.
    hue_shift *= (0.5 / n1)

    seqs = []
    for h in hues:
        h_pre = (h - hue_shift) % 1
        h_post = (h + hue_shift) % 1
        seqs.append((lch_to_rgb(0, 100, h_pre),
                     lch_to_rgb(50, 100, h),
                     lch_to_rgb(100, 100, h_post)))

    seq = []
    for i in xrange(n1):
        seq.append((seqs[i - 1][-1], seqs[i][0]))
        seq.append(seqs[i][1])
        if i == n1 - 1:
            seq.append((seqs[i][2], seqs[0][0]))

    loc = np.linspace(0, 1, n1 * 2 + 1)

    if name is None:
        name = "%i_by_n" % n1

    return make_seq_cmap(seq, loc, name)


def oneway_colors(n, hue_start=0.2, light_range=0.5):
    """Create colors for categories

    Parameters
    ----------
    n : int
        Number of levels.
    hue_start : 0 <= scalar < 1 | sequence of scalar
        First hue value (default 0.2) or list of hue values.
    light_range : scalar | tuple
        Amount of lightness variation. If a positive scalar, the first color is
        lightest; if a negative scalar, the first color is darkest. Tuple with
        two scalar to define a specific range.
    """
    if isinstance(hue_start, float):
        hue = np.linspace(hue_start, hue_start + 1, n, False) % 1.
    elif len(hue_start) >= n:
        hue = hue_start
    else:
        raise ValueError("If list of hues is provided it needs ot contain at "
                         "least as many hues as there are cells")

    if isinstance(light_range, (list, tuple)):
        start, stop = light_range
        lightness = np.linspace(100 * start, 100 * stop, n)
    else:
        l_edge = 50 * light_range
        lightness = np.linspace(50 + l_edge, 50 - l_edge, n)
    return [lch_to_rgb(l, 100, h) for l, h in izip(lightness, hue)]


def twoway_colors(n1, n2, hue_start=0.2, hue_shift=0., hues=None):
    """Create colors for two-way interaction

    Parameters
    ----------
    n1, n2 : int
        Number of levels on the first and second factors.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).
    """
    if hues is None:
        hues = np.linspace(hue_start, hue_start + 1, n1, False) % 1.
    else:
        hues = np.asarray(hues)
        if np.any(hues > 1) or np.any(hues < 0):
            raise ValueError("hue values out of range; need to be in [0, 1]")
        elif len(hues) < n1:
            raise ValueError("Need at least as many hues as levels in the "
                             "first factor (got %i, need %i)" % (len(hues), n1))
    hue_shift *= (1. / 3. / n1)
    lstart = 60. / n2
    ls = np.linspace(lstart, 100 - lstart, n2)

    colors = []
    for hue in hues:
        hs = np.linspace(hue - hue_shift, hue + hue_shift, n2) % 1
        colors.extend(lch_to_rgb(l, 100, h) for l, h in izip(ls, hs))

    return colors


def make_cmaps():
    """Create some custom colormaps and register them with matplotlib"""
    # polar:  blue-white-red
    cmap = LinearSegmentedColormap.from_list(
        "polar", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (1.0, (1.0, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # xpolar ("extra-polar"): fade ends into black
    cmap = LinearSegmentedColormap.from_list(
        "xpolar", (
            (0.0, (0.0, 0.0, 0.0)),
            (0.3, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (0.7, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # extra-polar alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "xpolar-a", (
            (0.0, (0.0, 0.0, 0.0, 1.0)),
            (0.3, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.7, (1.0, 0.0, 0.0, 1.0)),
            (1.0, (0.0, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # phase
    cmap = LinearSegmentedColormap.from_list(
        "phase", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # sig:  significance map for specific vmax=0.05
    cmap = LinearSegmentedColormap.from_list(
        "sig", (
            (0.0,  (1.0, 1.0, 1.0)),
            (0.02, (1.0, 1.0, 0.0)),
            (0.2,  (1.0, 0.5, 0.0)),
            (1.0,  (1.0, 0.0, 0.0)),
        ))
    cmap.set_over('k', alpha=0.)
    cmap.set_bad('b', alpha=0.)
    register_cmap(cmap=cmap)


make_cmaps()

symmetric_cmaps = ('polar', 'xpolar', 'xpolar-a',
                   'BrBG', 'BrBG_r', 'PRGn', 'PRGn_r', 'PuOr', 'PuOr_',
                   'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'seismic', 'seismic_r')
zerobased_cmaps = ('sig',)
# corresponding cmaps with transparency (alpha channel)
ALPHA_CMAPS = {'xpolar': 'xpolar-a',
               'RdBu_r': 'xpolar-a'}


def set_info_cs(info, cs={'cmap': 'jet'}, copy=True):
    """Update the plotting arguments in info to reflect a new colorspace

    Parameters
    ----------
    info : dict
        The previous info dictionary.
    cs : dict
        The new colorspace info dictionary.
    copy : bool
        Make a copy of the dictionary before modifying it.

    Returns
    -------
    info : dict
        The updated dictionary.
    """
    if copy:
        info = info.copy()
    for key in ('meas', 'unit', 'cmap', 'vmin', 'vmax', 'contours'):
        if key in info and key not in cs:
            info.pop(key)
    info.update(cs)
    return info


def default_info(meas, **kwargs):
    "Default colorspace info"
    kwargs['meas'] = meas
    return kwargs


def cluster_pmap_info():
    contours = {0.05: (0., 0., 0.)}
    info = {'meas': 'p', 'contours': contours, 'cmap': 'sig', 'vmax': 0.05}
    return info


def sig_info(p=.05, contours={.01: '.5', .001: '0'}):
    "Info dict for significance map"
    info = {'meas': 'p', 'cmap': 'sig', 'vmax': p, 'contours': contours}
    return info


def stat_info(meas, c0=None, c1=None, c2=None, tail=0, **kwargs):
    if 'contours' not in kwargs:
        contours = kwargs['contours'] = {}
        if c0 is not None:
            if tail >= 0:
                contours[c0] = (1.0, 0.5, 0.1)
            if tail <= 0:
                contours[-c0] = (0.5, 0.1, 1.0)
        if c1 is not None:
            if tail >= 0:
                contours[c1] = (1.0, 0.9, 0.2)
            if tail <= 0:
                contours[-c1] = (0.9, 0.2, 1.0)
        if c2 is not None:
            if tail >= 0:
                contours[c2] = (1.0, 1.0, 0.8)
            if tail <= 0:
                contours[-c2] = (1.0, 0.8, 1.0)

    if meas == 'r':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 't':
        info = {'meas': meas, 'cmap': 'RdBu_r'}
    elif meas == 'f':
        info = {'meas': meas, 'cmap': 'BuPu_r', 'vmin': 0}
    else:
        info = default_info(meas)
    info.update(kwargs)
    return info


_unit_fmt = {1: "%s",
             1e-3: "m%s",
             1e-6: r"$\mu$%s",
             1e-9: "n%s",
             1e-12: "p%s",
             1e-15: "f%s"}


def eeg_info(vmax=None, mult=1, unit='V', meas="V"):
    unit = _unit_fmt[1 / mult] % unit
    out = dict(cmap='xpolar', meas=meas, unit=unit)
    if vmax is not None:
        out['vmax'] = vmax
    return out


def meg_info(vmax=None, mult=1, unit='T', meas="B"):
    unit = _unit_fmt[1 / mult] % unit
    out = dict(cmap='xpolar', meas=meas, unit=unit)
    if vmax is not None:
        out['vmax'] = vmax
    return out
