# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import division

from functools import partial
from itertools import izip, product
import os
from tempfile import mkdtemp
from warnings import warn

from nibabel.freesurfer import read_annot
import numpy as np
import mne

from .._data_obj import asndvar, NDVar, SourceSpace
from .._utils import deprecated
from ..fmtxt import Image, im_table, ms
from ._base import EelFigure, ImLayout, ColorBarMixin
from ._colors import ColorList


# defaults
FOREGROUND = (0, 0, 0)
BACKGROUND = (1, 1, 1)


def assert_can_save_movies():
    from ._brain_fix import assert_can_save_movies
    assert_can_save_movies()


def annot(annot, subject='fsaverage', surf='smoothwm', borders=False, alpha=0.7,
          hemi=None, views=('lat', 'med'), w=None, h=None, axw=None, axh=None,
          foreground=None, background=None, parallel=True, cortex='classic',
          subjects_dir=None):
    """Plot the parcellation in an annotation file

    Parameters
    ----------
    annot : str
        Name of the annotation (e.g., "PALS_B12_LOBES").
    subject : str
        Name of the subject (default 'fsaverage').
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    borders : bool | int
        Show only label borders (PySurfer Brain.add_annotation() argument).
    alpha : scalar
        Alpha of the annotation (1=opaque, 0=transparent, default 0.7).
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default includes hemisphere with more than one
        label in the annot file).
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    subjects_dir : None | str
        Override the default subjects_dir.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance.

    Notes
    -----
    The ``Brain`` object that is returned has a
    :meth:`~plot._brain_fixes.plot_legend` method to plot the color legend.

    See Also
    --------
    eelbrain.plot.brain.annot_legend : plot a corresponding legend without the brain
    """
    if hemi is None:
        annot_lh = mne.read_labels_from_annot(subject, annot, 'lh',
                                              subjects_dir=subjects_dir)
        use_lh = len(annot_lh) > 1
        annot_rh = mne.read_labels_from_annot(subject, annot, 'rh',
                                              subjects_dir=subjects_dir)
        use_rh = len(annot_rh) > 1
        if use_lh and use_rh:
            hemi = 'split'
        elif use_lh:
            hemi = 'lh'
        elif use_rh:
            hemi = 'rh'
        else:
            raise ValueError("Neither hemisphere contains more than one label")

    brain = _surfer_brain(subject, surf, hemi, views, w, h, axw, axh,
                          foreground, background, cortex, subjects_dir)
    brain._set_annot(annot, borders, alpha)
    if parallel:
        brain.set_parallel_view(scale=True)
    return brain


def annot_legend(lh, rh, *args, **kwargs):
    """Plot a legend for a freesurfer parcellation

    Parameters
    ----------
    lh : str
        Path to the lh annot-file.
    rh : str
        Path to the rh annot-file.
    labels : dict (optional)
        Alternative (text) label for (brain) labels.
    h : 'auto' | scalar
        Height of the figure in inches. If 'auto' (default), the height is
        automatically increased to fit all labels.

    Returns
    -------
    legend : :class:`~eelbrain.plot.ColorList`
        Figure with legend for the parcellation.

    Notes
    -----
    Instead of :func:`~eelbrain.plot.brain.annot_legend` it is usually
    easier to use::

    >>> brain = plot.brain.annoot(annot, ...)
    >>> legend = brain.plot_legend()

    See Also
    --------
    eelbrain.plot.brain.annot : plot the parcellation on a brain model
    """
    _, lh_colors, lh_names = read_annot(lh)
    _, rh_colors, rh_names = read_annot(rh)
    lh_colors = dict(izip(lh_names, lh_colors[:, :4] / 255.))
    rh_colors = dict(izip(rh_names, rh_colors[:, :4] / 255.))
    names = set(lh_names)
    names.update(rh_names)
    colors = {}
    seq = []  # sequential order in legend
    seq_lh = []
    seq_rh = []
    for name in names:
        if name in lh_colors and name in rh_colors:
            if np.array_equal(lh_colors[name], rh_colors[name]):
                colors[name] = lh_colors[name]
                seq.append(name)
            else:
                colors[name + '-lh'] = lh_colors[name]
                colors[name + '-rh'] = rh_colors[name]
                seq_lh.append(name + '-lh')
                seq_rh.append(name + '-rh')
        elif name in lh_colors:
            colors[name + '-lh'] = lh_colors[name]
            seq_lh.append(name + '-lh')
        else:
            colors[name + '-rh'] = rh_colors[name]
            seq_rh.append(name + '-rh')
    return ColorList(colors, seq + seq_lh + seq_rh, *args, **kwargs)


def _plot(data, *args, **kwargs):
    "Plot depending on source space kind"
    if data.source.kind == 'vol':
        return _voxel_brain(data, *args, **kwargs)
    else:
        return brain(data, *args, **kwargs)


def dspm(src, fmin=13, fmax=22, fmid=None, *args, **kwargs):
    """
    Plot a source estimate with coloring for dSPM values (bipolar).

    Parameters
    ----------
    src : NDVar, dims = ([case,] source, [time])
        NDVar with SourceSpace dimension. If stc contains a case dimension,
        the average across cases is taken.
    fmin, fmax : scalar >= 0
        Start- and end-point for the color gradient for positive values. The
        gradient for negative values goes from -fmin to -fmax. Values between
        -fmin and fmin are transparent.
    fmid : None | scalar
        Midpoint for the color gradient. If fmid is None (default) it is set
        half way between fmin and fmax.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Add a colorbar to the figure (use ``.plot_colorbar()`` to plot a
        colorbar separately).
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    if fmid is None:
        fmid = (fmax + fmin) / 2
    lut = _dspm_lut(fmin, fmid, fmax)
    return _plot(src, lut, -fmax, fmax, *args, **kwargs)


def p_map(p_map, param_map=None, p0=0.05, p1=0.01, p0alpha=0.5, *args,
          **kwargs):
    """Plot a map of p-values in source space.

    Parameters
    ----------
    p_map : NDVar
        Statistic to plot (normally a map of p values).
    param_map : NDVar
        Statistical parameter covering the same data points as p_map. Only the
        sign is used, for incorporating the directionality of the effect into
        the plot.
    p0, p1 : scalar
        Threshold p-values for the color map.
    p0alpha : 1 >= float >= 0
        Alpha for greatest p-value that is still displayed (default: 0.5 -
        clearly indicate border).
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Add a colorbar to the figure (use ``.plot_colorbar()`` to plot a
        colorbar separately).
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    if 'solid' in kwargs:
        warn("The solid parameter for plot.brain.p_map() is deprecated and "
             "will stop working after Eelbrain 0.25, use p0alpha instead.",
             DeprecationWarning)
        p0alpha = float(kwargs.pop('solid'))
    pmap, lut, vmax = _p_lut(p_map, param_map, p0, p1, p0alpha)
    return _plot(pmap, lut, -vmax, vmax, *args, **kwargs)


def cluster(cluster, vmax=None, *args, **kwargs):
    """Plot a spatio-temporal cluster

    Plots a cluster with the assumption that all non-zero data should be
    visible, while areas that not part of the cluster are 0.

    Parameters
    ----------
    cluster : NDVar
        The cluster.
    vmax : scalar != 0
        Maximum value in the colormap. Default is the maximum value in the
        cluster.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Add a colorbar to the figure (use ``.plot_colorbar()`` to plot a
        colorbar separately).
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    if vmax is None:
        vmax = max(cluster.x.max(), -cluster.x.min())
        if vmax == 0:
            raise ValueError("The cluster's data is all zeros")
    elif vmax == 0:
        raise ValueError("vmax can't be 0")
    elif vmax < 0:
        vmax = -vmax

    lut = _dspm_lut(0, vmax / 10, vmax)
    return _plot(cluster, lut, -vmax, vmax, *args, **kwargs)


def _surfer_brain(subject='fsaverage', surf='smoothwm', hemi='split',
                  views=('lat', 'med'), w=None, h=None, axw=None, axh=None,
                  foreground=None, background=None, cortex='classic',
                  subjects_dir=None):
    """Create surfer.Brain instance

    Parameters
    ----------
    subject : str
        Name of the subject (default 'fsaverage').
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot.
    views : str | iterator of str
        View or views to show in the figure.
    colorbar : bool
        Whether to add a colorbar to the figure.
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance.
    """
    from ._brain_fix import Brain

    if isinstance(views, basestring):
        views = [views]
    elif not isinstance(views, list):
        views = list(views)

    if hemi == 'split':
        n_views_x = 2
    elif hemi in ('lh', 'rh', 'both'):
        n_views_x = 1
    else:
        raise ValueError("Unknown value for hemi parameter: %s" % repr(hemi))

    title = None
    if w is not None:
        width = w
    elif axw is not None:
        width = axw * n_views_x
    else:
        width = 500 * n_views_x

    if h is not None:
        height = h
    elif axh is not None:
        height = axh * len(views)
    else:
        height = 400 * len(views)

    if foreground is None:
        foreground = FOREGROUND

    if background is None:
        background = BACKGROUND

    return Brain(subject, hemi, surf, title=title, cortex=cortex, alpha=1.,
                 size=(width, height), views=views, background=background,
                 foreground=foreground, subjects_dir=subjects_dir)


def brain(src, cmap=None, vmin=None, vmax=None, surf='smoothwm',
          views=('lat', 'med'), hemi=None, colorbar=False, time_label='ms',
          w=None, h=None, axw=None, axh=None, foreground=None, background=None,
          parallel=True, cortex='classic', smoothing_steps=None, mask=True,
          subjects_dir=None, colormap=None):
    """Create a PySurfer Brain object with a data layer

    Parameters
    ----------
    src : NDVar ([case,] source, [time]) | SourceSpace
        NDVar with SourceSpace dimension. If stc contains a case dimension,
        the average across cases is taken. If a SourceSpace, the Brain is
        returned without adding any data and corresponding arguments are
        ignored.
    cmap : str | array
        Colormap (name of a matplotlib colormap) or LUT array.
    vmin, vmax : scalar
        Endpoints for the colormap. Need to be set explicitly if ``cmap`` is
        a LUT array.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : str | iterator of str
        View or views to show in the figure. Options are: 'rostral', 'parietal',
        'frontal', 'ventral', 'lateral', 'caudal', 'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both' | 'split'
        Which hemispheres to plot (default based on data).
    colorbar : bool
        Add a colorbar to the figure (use ``.plot_colorbar()`` to plot a
        colorbar separately).
    time_label : str
        Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
        milliseconds or in seconds, or supply a custom format string to format
        time values (in seconds; default is ``'ms'``).
    w, h, axw, axh : scalar
        Layout parameters (figure width/height, subplot width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    mask : bool
        Shade areas that are not in ``src``.
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.

    Returns
    -------
    brain : surfer.Brain
        PySurfer Brain instance containing the plot.
    """
    if colormap is not None:
        warn("The colormap parameter is deprecated, use cmap instead",
             DeprecationWarning)
        cmap = colormap

    if isinstance(src, SourceSpace):
        if cmap is not None or vmin is not None or vmax is not None:
            raise TypeError("When plotting SourceSpace, cmap, vmin and vmax "
                            "can not be specified (got %s)" %
                            ', '.join((cmap, vmin, vmax)))
        ndvar = None
        source = src
    else:
        ndvar = asndvar(src)
        if ndvar.has_case:
            ndvar = ndvar.summary()
        source = ndvar.source

    if hemi is None:
        if source.lh_n and source.rh_n:
            hemi = 'split'
        elif source.lh_n:
            hemi = 'lh'
        elif not source.rh_n:
            raise ValueError('No data')
        else:
            hemi = 'rh'
    elif (hemi == 'lh' and source.rh_n) or (hemi == 'rh' and source.lh_n):
        if ndvar is None:
            source = source[source.dimindex(hemi)]
        else:
            ndvar = ndvar.sub(source=hemi)
            source = ndvar.source

    if subjects_dir is None:
        subjects_dir = source.subjects_dir

    brain = _surfer_brain(source.subject, surf, hemi, views, w, h, axw, axh,
                          foreground, background, cortex, subjects_dir)

    if ndvar is not None:
        brain.add_ndvar(ndvar, cmap, vmin, vmax, smoothing_steps, colorbar,
                        time_label)

    if mask:
        alpha = 0.5 if mask is True else mask
        brain.add_mask(source, alpha, (0, 0, 0), smoothing_steps, subjects_dir)

    if parallel:
        brain.set_parallel_view(scale=True)
    return brain


@deprecated("0.25", brain)
def surfer_brain(*args, **kwargs):
    pass


def _dspm_lut(fmin, fmid, fmax, n=256):
    """Create a color look up table (lut) for a dSPM plot

    Parameters
    ----------
    fmin, fmid, fmax : scalar
        Start-, mid- and endpoint for the color gradient.
    n : int
        Number of distinct color values in the table.

    Notes
    -----
    Transitions:

    0-fmin
        Transparent.
    fmin-fmid
        Transparent - opaque.
    fmid-fmax
        Hue shift.
    """
    if not (fmin < fmid < fmax):
        raise ValueError("Invalid colormap, we need fmin < fmid < fmax")
    elif fmin < 0:
        raise ValueError("The dSPM color gradient is symmetric around 0, fmin "
                         "needs to be > 0 (got %s)." % fmin)

    lut = np.zeros((n, 4), dtype=np.uint8)
    i0 = int(round(n / 2))  # v=0 (middle of the LUT)
    imin = int(round((fmin / fmax) * i0))  # i0 is the range of one side of the LUT
    min_n = i0 - imin
    min_p = i0 + imin
    imid = int(round((fmid / fmax) * i0))
    mid_n = i0 - imid
    mid_p = i0 + imid

    # red end
    lut[i0:, 0] = 255
    lut[mid_p:, 1] = np.linspace(0, 255, n - mid_p)

    # blue end
    lut[:i0, 2] = 255
    lut[:mid_n, 0] = np.linspace(127, 0, mid_n)
    lut[:mid_n, 1] = np.linspace(127, 0, mid_n)

    # alpha
    lut[:mid_n, 3] = 255
    lut[mid_n:min_n, 3] = np.linspace(255, 0, min_n - mid_n)
    lut[min_n:min_p, 3] = 0
    lut[min_p:mid_p, 3] = np.linspace(0, 255, mid_p - min_p)
    lut[mid_p:, 3] = 255

    return lut


def _p_lut(pmap, tmap, p0, p1, p0alpha, n=256):
    """Creat a color look up table (lut) for p-values

    Parameters
    ----------
    pmap : NDVar
        Map of p-values.
    tmap : NDVar
        Map of signed statistic (only used to code the sign of each p-value).
    p0 : scalar
        Highest p-vale that should be visible.
    p1 : scalar
        P-value where the colormap changes from ramping alpha to ramping color.
    p0alpha : bool
        Alpha at p0.
    n : int
        Number of color categories in the lut.
    """
    if p1 >= p0:
        raise ValueError("p1 needs to be smaller than p0.")

    pstep = 2 * p0 / (n - 3)  # there are n - 1 steps, 2 leave the visible range

    # max p-value that needs to be represented (1 step out of visible)
    vmax = p0 + pstep

    # bring interesting p-values to the range [pstep vmax]
    pmap = vmax - pmap

    # set uninteresting values to zero
    pmap.x.clip(0, vmax, pmap.x)

    # add sign to p-values
    if tmap is not None:
        pmap.x *= np.sign(tmap.x)

    # http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
    lut = np.zeros((n, 4), dtype=np.uint8)

    middle = n // 2
    p0p = middle + 1
    p1n = int(round(p1 / pstep))
    p1p = n - p1n

    # negative colors
    lut[:middle, 2] = 255
    lut[:p1n, 0] = np.linspace(255, 0, p1n)

    # positive colors
    lut[middle:, 0] = 255
    lut[p1p:, 1] = np.linspace(0, 255, n - p1p)

    # alpha
    if p0alpha == 1:
        lut[:middle, 3] = 255
        lut[p0p:, 3] = 255
    elif not 0 <= p0alpha <= 1:
        raise ValueError("p0alpha=%r" % (p0alpha,))
    else:
        p0_alpha = int(round(p0alpha * 255))
        lut[:p1n, 3] = 255
        lut[p1n:middle, 3] = np.linspace(255, p0_alpha, middle - p1n)
        lut[p0p:p1p, 3] = np.linspace(p0_alpha, 255, p1p - p0p)
        lut[p1p:, 3] = 255

    pmap.info['cmap ticks'] = {
        -vmax: '<' + str(p1 / 10)[1:],
        -vmax + p1: str(p1)[1:],
        0: str(p0)[1:],
        vmax - p1: str(p1)[1:],
        vmax: '<' + str(p1 / 10)[1:],
    }

    return pmap, lut, vmax


def _activation_lut(threshold=3, vmax=8):
    """Color map for plotting stc activation.

    Parameters
    ----------
    threshold : scalar
        threshold is point at which values gain 50% visibility.
        from threshold and beyond, alpha = 100% visibility.
    vmax : scalar
        the upper range of activation values. values are clipped above this range. When None,
        vmax = two standard deviations above and below the mean.

    Notes
    -----
    Colors:
    - negative is blue.
    - super negative is purple.
    - positive is red.
    - super positive is yellow.
    """
    values = np.linspace(-vmax, vmax, 256)
    trans_uidx = np.argmin(np.abs(values - threshold))
    trans_lidx = np.argmin(np.abs(values + threshold))

    # Transparent ramping
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[127:trans_uidx, 3] = np.linspace(0, 128, trans_uidx - 127)
    lut[trans_lidx:127, 3] = np.linspace(128, 0, 127 - trans_lidx)
    lut[trans_uidx:, 3] = 255
    lut[:trans_lidx, 3] = 255

    # negative -> Blue
    lut[:127, 2] = 255
    # super negative -> Purple
    lut[:trans_lidx, 0] = np.linspace(0, 255, trans_lidx)
    # positive -> Red
    lut[127:, 0] = 255
    # super positive -> Yellow
    lut[trans_uidx:, 1] = np.linspace(0, 255, 256 - trans_uidx)

    return lut


def _voxel_brain(data, lut, vmin, vmax):
    """Plot spheres for volume source space

    Parameters
    ----------
    data : NDVar
        Data to plot.
    lut : array
        Color LUT.
    vmin, vmax : scalar
        Data range.
    """
    if data.dimnames != ('source',):
        raise ValueError("Can only plot 1 dimensional source space NDVars")

    from mayavi import mlab

    x, y, z = data.source.coordinates.T

    figure = mlab.figure()
    mlab.points3d(x, y, z, scale_factor=0.002, opacity=0.5)
    pts = mlab.points3d(x, y, z, data.x, vmin=vmin, vmax=vmax)
    pts.module_manager.scalar_lut_manager.lut.table = lut
    return figure


################################################################################
# Bin-Tables
############
# Top-level functions for fmtxt image tables and classes for Eelfigures.
# - _x_bin_table_ims() wrap 'x' brain plot function
# - _bin_table_ims() creates ims given a brain plot function

class ImageTable(EelFigure, ColorBarMixin):
    # Initialize in two steps
    #
    #  1) Initialize class to generate layout
    #  2) Use ._res_h and ._res_w to generate images
    #  3) Finalize bu calling ._add_ims()
    #

    def __init__(self, n_rows, n_columns, title=None, *args, **kwargs):
        layout = ImLayout(n_rows * n_columns, 0, 0.5, 4/3, 2, title, *args,
                          nrow=n_rows, ncol=n_columns, **kwargs)
        EelFigure.__init__(self, "ImageTable", layout)

        self._n_rows = n_rows
        self._n_columns = n_columns
        self._res_w = int(round(layout.axw * layout.dpi))
        self._res_h = int(round(layout.axh * layout.dpi))

    def _add_ims(self, ims, header, cmap_params, cmap_data):
        for row, column in product(xrange(self._n_rows), xrange(self._n_columns)):
            ax = self._axes[row * self._n_columns + column]
            ax.imshow(ims[row][column])

        # time labels
        y = 0.25 / self._layout.h
        for i, label in enumerate(header):
            x = (0.5 + i) / self._layout.ncol
            self.figure.text(x, y, label, va='center', ha='center')

        ColorBarMixin.__init__(self, lambda: cmap_params, cmap_data)
        self._show()

    def _fill_toolbar(self, tb):
        ColorBarMixin._fill_toolbar(self, tb)


class _BinTable(EelFigure, ColorBarMixin):
    """Super-class"""
    def __init__(self, ndvar, tstart, tstop, tstep, im_func, surf, views, hemi,
                 summary, title, foreground=None, background=None,
                 parallel=True, smoothing_steps=None, mask=True,
                 *args, **kwargs):
        if isinstance(views, str):
            views = (views,)
        data = ndvar.bin(tstep, tstart, tstop, summary)
        n_columns = len(data.time)
        n_hemis = (data.source.lh_n > 0) + (data.source.rh_n > 0)
        n_rows = len(views) * n_hemis

        layout = ImLayout(n_rows * n_columns, 0, 0.5, 4/3, 2, title, *args,
                          nrow=n_rows, ncol=n_columns, **kwargs)
        EelFigure.__init__(self, "BinTable", layout)

        res_w = int(layout.axw * layout.dpi)
        res_h = int(layout.axh * layout.dpi)
        ims, header, cmap_params = im_func(data, surf, views, hemi, axw=res_w,
                                           axh=res_h, foreground=foreground,
                                           background=background,
                                           parallel=parallel,
                                           smoothing_steps=smoothing_steps,
                                           mask=mask)
        for row in xrange(n_rows):
            for column in xrange(n_columns):
                ax = self._axes[row * n_columns + column]
                ax.imshow(ims[row][column])

        # time labels
        y = 0.25 / layout.h
        for i, label in enumerate(header):
            x = (0.5 + i) / layout.ncol
            self.figure.text(x, y, label, va='center', ha='center')

        ColorBarMixin.__init__(self, lambda: cmap_params, data)
        self._show()

    def _fill_toolbar(self, tb):
        ColorBarMixin._fill_toolbar(self, tb)


class BinTable(_BinTable):
    """DSPM plot bin-table"""
    def __init__(self, ndvar, tstart=None, tstop=None, tstep=0.1,
                 fmin=13, fmax=22, fmid=None,
                 surf='smoothwm', views=('lat', 'med'), hemi=None,
                 summary='sum', title=None, *args, **kwargs):
        im_func = partial(_dspm_bin_table_ims, fmin, fmax, fmid)
        _BinTable.__init__(self, ndvar, tstart, tstop, tstep, im_func, surf,
                           views, hemi, summary, title, *args, **kwargs)


class ClusterBinTable(_BinTable):
    """Data plotted on brain for different time bins and views

    Parameters
    ----------
    ndvar : NDVar (time x source)
        Data to be plotted.
    tstart : None | scalar
        Time point of the start of the first bin (inclusive; None to use the
        first time point in ndvar).
    tstop : None | scalar
        End of the last bin (exclusive; None to end with the last time point
        in ndvar).
    tstep : scalar
        Size of each bin (in seconds).
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : list of str
        Views to display (for each hemisphere, lh first). Options are:
        'rostral', 'parietal', 'frontal', 'ventral', 'lateral', 'caudal',
        'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both'
        Which hemispheres to plot (default based on data).
    summary : str
        How to summarize data in each time bin. Can be the name of a numpy
        function that takes an axis parameter (e.g., 'sum', 'mean', 'max') or
        'extrema' which selects the value with the maximum absolute value.
        Default is sum.
    vmax : scalar != 0
        Maximum value in the colormap. Default is the maximum value in the
        cluster.
    title : str
        Figure title.
    """
    def __init__(self, ndvar, tstart=None, tstop=None, tstep=0.1,
                 surf='smoothwm', views=('lat', 'med'), hemi=None,
                 summary='sum', vmax=None, title=None, *args, **kwargs):
        im_func = partial(_cluster_bin_table_ims, vmax)
        _BinTable.__init__(self, ndvar, tstart, tstop, tstep, im_func, surf,
                           views, hemi, summary, title, *args, **kwargs)


def dspm_bin_table(ndvar, fmin=2, fmax=8, fmid=None,
                   tstart=None, tstop=None, tstep=0.1, surf='smoothwm',
                   views=('lat', 'med'), hemi=None, summary='extrema',
                   axw=300, axh=250, *args, **kwargs):
    """Create a table with images for time bins

    Parameters
    ----------
    ndvar : NDVar (time x source)
        Data to be plotted.
    fmin, fmax : scalar >= 0
        Start- and end-point for the color gradient for positive values. The
        gradient for negative values goes from -fmin to -fmax. Values between
        -fmin and fmin are transparent.
    fmid : None | scalar
        Midpoint for the color gradient. If fmid is None (default) it is set
        half way between fmin and fmax.
    tstart : None | scalar
        Time point of the start of the first bin (inclusive; None to use the
        first time point in ndvar).
    tstop : None | scalar
        End of the last bin (exclusive; None to end with the last time point
        in ndvar).
    tstep : scalar
        Size of each bin (in seconds).
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : list of str
        Views to display (for each hemisphere, lh first). Options are:
        'rostral', 'parietal', 'frontal', 'ventral', 'lateral', 'caudal',
        'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both'
        Which hemispheres to plot (default based on data).
    summary : str
        How to summarize data in each time bin. Can be the name of a numpy
        function that takes an axis parameter (e.g., 'sum', 'mean', 'max') or
        'extrema' which selects the value with the maximum absolute value.
        Default is extrema.
    axw, axh : scalar
        Subplot width/height (default axw=300, axh=250).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.


    Returns
    -------
    images : Image
        FMTXT Image object that can be saved as SVG or integrated into an
        FMTXT document.


    Notes
    -----
    Plotting is based on :func:`plot.brain.dspm`.

    The resulting image can be saved with either of the following (currently
    the only supported formats)::

    >>> image.save_html("name.html")
    >>> image.save_image("name.svg")


    See Also
    --------
    plot.brain.bin_table: plotting clusters as bin-table
    """
    data = ndvar.bin(tstep, tstart, tstop, summary)
    ims, header, _ = _dspm_bin_table_ims(fmin, fmax, fmid, data, surf, views,
                                         hemi, axw, axh, *args, **kwargs)
    return im_table(ims, header)


def bin_table(ndvar, tstart=None, tstop=None, tstep=0.1, surf='smoothwm',
              views=('lat', 'med'), hemi=None, summary='sum', vmax=None,
              axw=300, axh=250, *args, **kwargs):
    """Create a table with images for time bins

    Parameters
    ----------
    ndvar : NDVar (time x source)
        Data to be plotted.
    tstart : None | scalar
        Time point of the start of the first bin (inclusive; None to use the
        first time point in ndvar).
    tstop : None | scalar
        End of the last bin (exclusive; None to end with the last time point
        in ndvar).
    tstep : scalar
        Size of each bin (in seconds).
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    views : list of str
        Views to display (for each hemisphere, lh first). Options are:
        'rostral', 'parietal', 'frontal', 'ventral', 'lateral', 'caudal',
        'medial', 'dorsal'.
    hemi : 'lh' | 'rh' | 'both'
        Which hemispheres to plot (default based on data).
    summary : str
        How to summarize data in each time bin. Can be the name of a numpy
        function that takes an axis parameter (e.g., 'sum', 'mean', 'max') or
        'extrema' which selects the value with the maximum absolute value.
        Default is sum.
    vmax : scalar != 0
        Maximum value in the colormap. Default is the maximum value in the
        cluster.
    out : 'image' | 'figure'
        Format in which to return the plot. ``'image'`` (default) returns an
        Image object that
    axw, axh : scalar
        Subplot width/height (default axw=300, axh=250).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.


    Returns
    -------
    images : Image
        FMTXT Image object that can be saved as SVG or integrated into an
        FMTXT document.


    Notes
    -----
    Plotting is based on :func:`plot.brain.cluster`.

    The resulting image can be saved with either of the following (currently
    the only supported formats)::

    >>> image.save_html("name.html")
    >>> image.save_image("name.svg")


    See Also
    --------
    plot.brain.dspm_bin_table: plotting SPMs as bin-table
    """
    data = ndvar.bin(tstep, tstart, tstop, summary)
    ims, header, _ = _cluster_bin_table_ims(vmax, data, surf, views, hemi, axw,
                                            axh, *args, **kwargs)
    return im_table(ims, header)


def _dspm_bin_table_ims(fmin, fmax, fmid, data, surf, views, hemi, axw, axh,
                        *args, **kwargs):
    if fmax is None:
        fmax = max(data.max(), -data.min())

    def brain_(hemi_):
        return dspm(data, fmin, fmax, fmid, surf, views[0], hemi_, False, None,
                    axw, axh, *args, **kwargs)

    return _bin_table_ims(data, hemi, views, brain_)


def _cluster_bin_table_ims(vmax, data, surf, views, hemi, axw, axh, *args,
                           **kwargs):
    if vmax is None:
        vmax = max(-data.min(), data.max())
        if vmax == 0:
            raise NotImplementedError("The data of ndvar is all zeros")
    elif vmax == 0:
        raise ValueError("vmax can't be 0")
    elif vmax < 0:
        vmax = -vmax

    def brain_(hemi_):
        return cluster(data, vmax, surf, views[0], hemi_, False, None, axw, axh,
                        None, None, *args, **kwargs)

    return _bin_table_ims(data, hemi, views, brain_)


def _bin_table_ims(data, hemi, views, brain_func):
    ims = []
    if hemi is None:
        hemis = []
        if data.source.lh_n:
            hemis.append('lh')
        if data.source.rh_n:
            hemis.append('rh')
    elif hemi == 'both':
        hemis = ['lh', 'rh']
    elif hemi == 'lh' or hemi == 'rh':
        hemis = [hemi]
    else:
        raise ValueError("hemi=%s" % repr(hemi))

    cmap_params = None
    for hemi in hemis:
        brain = brain_func(hemi)

        hemi_lines = [[] for _ in views]
        for i in xrange(len(data.time)):
            brain.set_data_time_index(i)
            for line, view in izip(hemi_lines, views):
                brain.show_view(view)
                im = brain.screenshot_single('rgba', True)
                line.append(im)
        ims += hemi_lines
        if cmap_params is None:
            cmap_params = brain._get_cmap_params()
        brain.close()

    header = ['%i - %i ms' % (ms(t0), ms(t1)) for t0, t1 in data.info['bins']]
    return ims, header, cmap_params


class SequencePlotter(object):
    """Plot multiple images of the same data

    Parameters
    ----------
    source : SourceSpace
        Source space which to plot.
    surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
        Freesurfer surface to use as brain geometry.
    w, h : scalar
        Layout parameters (figure width/height).
    foreground : mayavi color
        Figure foreground color (i.e., the text color).
    background : mayavi color
        Figure background color.
    parallel : bool
        Set views to parallel projection (default ``True``).
    cortex : str | tuple | dict
        See :class:`surfer.Brain`.
    smoothing_steps : None | int
        Number of smoothing steps if data is spatially undersampled (pysurfer
        ``Brain.add_data()`` argument).
    mask : bool
        Shade areas that are not in ``src``.
    subjects_dir : None | str
        Override the subjects_dir associated with the source space dimension.
    """
    def __init__(self, source):
        self.source = source
        self._data = []
        self._time = None
        self._bins = None
        self._brain_args = {}

    def set_brain_args(self, surf='smoothwm', foreground=None, background=None,
                       parallel=True, cortex='classic', mask=True):
        self._brain_args = {
            'surf': surf, 'foreground': foreground, 'background': background,
            'parallel': parallel, 'cortex': cortex, 'mask': mask}

    def add_ndvar(self, ndvar, *args, **kwargs):
        self._data.append(('data', ndvar, args, kwargs))
        if ndvar.has_dim('time'):
            if self._time is None:
                self._time = ndvar.time
            elif not ndvar.time == self._time:
                raise ValueError("Incompatible time axes")

            if self._bins is None and 'bins' in ndvar.info:
                self._bins = ndvar.info['bins']

    def plot_table(self, hemi=('lh', 'rh'), view=('lateral', 'medial'),
                   *args, **kwargs):
        """Add ims to a figure

        Parameters
        ----------
        figure : _ImTable
            Figure to which to add images.
        """
        if self._time is None:
            raise RuntimeError("No data with time axis")

        if isinstance(hemi, basestring):
            hemis = (hemi,)
        else:
            hemis = hemi

        if isinstance(view, basestring):
            views = (view,)
        else:
            views = view

        figure = ImageTable(len(hemis) * len(views), len(self._time), *args,
                            **kwargs)

        im_rows = []
        cmap_params = None
        cmap_data = None
        for hemi in hemis:
            hemi_rows = [[] for _ in views]

            # plot brain
            b = brain(self.source, hemi=hemi, views=views[0], w=figure._res_w,
                      h=figure._res_h, time_label='', **self._brain_args)
            # add data layers
            for layer in self._data:
                if layer[0] == 'data':
                    ndvar, args, kwargs = layer[1:]
                    b.add_ndvar(ndvar, *args, time_label='', **kwargs)
                else:
                    raise RuntimeError("Data of kind %r" % (layer[0],))
            b.set_parallel_view(scale=True)

            # capture images
            for i in xrange(len(self._time)):
                b.set_data_time_index(i)
                for row, view in izip(hemi_rows, views):
                    b.show_view(view)
                    row.append(b.screenshot_single('rgba', True))
            im_rows += hemi_rows

            if cmap_params is None:
                cmap_params = b._get_cmap_params()
                cmap_data = self._data[-1][1]
            b.close()

        # table header
        if self._bins is None:
            header = ['%i ms' % ms(t) for t in self._time]
        else:
            header = ['%i - %i ms' % (ms(t0), ms(t1)) for t0, t1 in self._bins]

        figure._add_ims(im_rows, header, cmap_params, cmap_data)
        return figure


def connectivity(source):
    """Plot source space connectivity

    Parameters
    ----------
    source : SourceSpace | NDVar
        Source space or NDVar containing the source space.

    Returns
    -------
    figure : mayavi Figure
        The figure.
    """
    from mayavi import mlab

    if isinstance(source, NDVar):
        source = source.get_dim('source')

    connections = source.connectivity()
    coords = source.coordinates
    x, y, z = coords.T

    figure = mlab.figure()
    src = mlab.pipeline.scalar_scatter(x, y, z, figure=figure)
    src.mlab_source.dataset.lines = connections
    lines = mlab.pipeline.stripper(src)
    mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=1.,
                          figure=figure)
    return figure


def copy(brain):
    "Copy the figure to the clip board"
    import wx

    tempdir = mkdtemp()
    tempfile = os.path.join(tempdir, "brain.png")
    brain.save_image(tempfile, 'rgba', True)

    bitmap = wx.Bitmap(tempfile, wx.BITMAP_TYPE_PNG)
    bitmap_obj = wx.BitmapDataObject(bitmap)

    if not wx.TheClipboard.IsOpened():
        open_success = wx.TheClipboard.Open()
        if open_success:
            wx.TheClipboard.SetData(bitmap_obj)
            wx.TheClipboard.Close()
            wx.TheClipboard.Flush()
