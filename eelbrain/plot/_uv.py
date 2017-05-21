# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot univariate data (:class:`~eelbrain.Var` objects)."""




import logging

import numpy as np
import scipy.stats  # without this sp.stats is not available
import matplotlib as mpl

from .._stats import test, stats
from .._data_obj import asvar, ascategorial, assub, cellname, Celltable
from ._base import EelFigure, Layout, LegendMixin, frame_title
from ._colors import find_cell_colors


defaults = dict(title_kwargs={'size': 14,
                                'family': 'serif'},
              mono=False,  # use monochrome instead of colors
              # defaults for color
              hatch=['', '', '//', '', '*', 'O', '.', '', '/', '//'],
              linestyle=['-', '-', '--', ':'],
              c={'pw': ['#00FF00', '#FFCC00', '#FF6600', '#FF3300'],
#                   'colors': ['.3', '.7', '1.', '1.'],
                   'colors': [(0.99609375, 0.12890625, 0.0),
                              (0.99609375, 0.5859375, 0.0),
                              (0.98046875, 0.99609375, 0.0),
                              (0.19921875, 0.99609375, 0.0)],
                   'markers': False,
                   },
              # defaults for monochrome
              cm={'pw': ['.6', '.4', '.2', '.0'],
                    'colors': ['.3', '.7', '1.', '1.'],
                    'markers': ['o', 'o', '^', 'o'],
                    },
                )  # set by __init__


def _mark_plot_pairwise(ax, ct, parametric, bottom, y_unit, corr, trend, markers,
                        levels=True, pwcolors=None, x0=0, top=None):
    """Mark pairwise significance

    Parameters
    ----------
    ax : Axes
        Axes to plot to.
    ct : Celltable
        Data for the tests.
    parametric : bool
        Whether to perform parametric tests.
    bottom : scalar
        Bottom of the space to use for the connectors (in data coordinates, i.e.
        highest point reached by the data plot).
    y_unit : scalar
        Suggested scale for half the vertical distance between connectors (only used
        if top is None).
    corr : None | 'hochberg' | 'bonferroni' | 'holm'
        Method for multiple comparison correction.
    trend : None | str
        Symbol to mark trends.
    markers : bool
        Plot markers indicating significance level (stars).
    ...
    top : scalar
        Impose a fixed top end of the y-axis.

    Returns
    -------
    top : scalar
        The top most value on the y axis.
    """
    if levels is not True:  # to avoid test.star() conflict
        trend = False

    # visual parameters
    if not pwcolors:
        if defaults['mono']:
            pwcolors = defaults['cm']['pw'][1 - bool(trend):]
        else:
            pwcolors = defaults['c']['pw'][1 - bool(trend):]
    font_size = mpl.rcParams['font.size'] * 1.5

    tests = test._pairwise(ct.get_data(), ct.all_within, parametric, corr, levels,
                           trend)

    # plan grid layout
    k = len(ct.cells)
    reservation = np.zeros((sum(range(1, k)), k - 1))
    connections = []
    for distance in range(1, k):
        for i in range(0, k - distance):
            # i, j are data indexes for the categories being compared
            j = i + distance
            index = tests['pw_indexes'][(i, j)]
            stars = tests['stars'][index]
            if not stars:
                continue

            free_levels = np.flatnonzero(reservation[:, i:j].sum(1) == 0)
            level = free_levels.min()
            reservation[level, i:j] = 1
            connections.append((level, i, j, index, stars))

    # plan spatial distances
    used_levels = np.flatnonzero(reservation[:, i:j].sum(1) > 0)
    if len(used_levels) == 0:
        if top is None:
            return bottom + y_unit
        else:
            return top
    n_levels = used_levels.max() + 1
    n_steps = n_levels * 2 + 1 + bool(markers)
    if top is None:
        top = bottom + y_unit * n_steps
    else:
        y_unit = (top - bottom) / n_steps

    # draw connections
    for level, i, j, index, stars in connections:
        c = pwcolors[stars - 1]
        y1 = bottom + y_unit * (level * 2 + 1)
        y2 = y1 + y_unit
        x1 = (x0 + i) + .025
        x2 = (x0 + j) - .025
        ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], color=c)
        if markers:
            symbol = tests['symbols'][index]
            ax.text((x1 + x2) / 2, y2, symbol, color=c, size=font_size,
                    ha='center', va='center', clip_on=False)

    return top


def _mark_plot_1sample(ax, ct, par, y_min, y_unit, popmean=0, corr='Hochberg',
                       trend="'", levels=True, pwcolors=None, x0=0):
    """Mark significance for one-sample test

    Returns
    -------
    y_max : float
        Top of space used on y axis.
    """
    if levels is not True:  # to avoid test.star() conflict
        trend = False
    # tests
    if not pwcolors:
        if defaults['mono']:
            pwcolors = defaults['cm']['pw'][1 - bool(trend):]
        else:
            pwcolors = defaults['c']['pw'][1 - bool(trend):]
    # mod
    ps = []
    if par:
        for d in ct.get_data():
            t, p = scipy.stats.ttest_1samp(d, popmean)
            ps.append(p)
    else:
        raise NotImplementedError("nonparametric 1-sample test")
    ps_adjusted = test.mcp_adjust(ps, corr)
    stars = test.star(ps_adjusted, int, levels, trend)
    stars_str = test.star(ps_adjusted, str, levels, trend)
    font_size = mpl.rcParams['font.size'] * 1.5
    if any(stars):
        y_stars = y_min + 1.75 * y_unit
        for i, n_stars in enumerate(stars):
            if n_stars > 0:
                c = pwcolors[n_stars - 1]
                ax.text(x0 + i, y_stars, stars_str[i], color=c, size=font_size,
                        ha='center', va='center', clip_on=False)
        return y_min + 4. * y_unit
    else:
        return y_min


class PairwiseLegend(EelFigure):
    """Legend for colors used in pairwise comparisons

    Parameters
    ----------
    size : scalar
        Side length in inches of a virtual square containing each bar.
    trend : bool
        Also include a bar for trends (p<0.1). Default is True.
    """
    def __init__(self, size=.3, trend=True, *args, **kwargs):
        if trend:
            levels = [.1, .05, .01, .001]
            colors = defaults['c']['pw']
        else:
            levels = [.05, .01, .001]
            colors = defaults['c']['pw'][1:]

        # layout
        n_levels = len(levels)
        ax_h = n_levels * size
        y_unit = size / 5
        ax_aspect = 4 / n_levels
        layout = Layout(None, ax_aspect, ax_h, False, *args, **kwargs)
        EelFigure.__init__(self, "ColorGrid", layout)
        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()

        x1 = .1 * size
        x2 = .9 * size
        x = (x1, x1, x2, x2)
        x_text = 1.2 * size
        for i, level, color in zip(range(n_levels), levels, colors):
            y1 = y_unit * (i * 5 + 2)
            y2 = y1 + y_unit
            ax.plot(x, (y1, y2, y2, y1), color=color)
            label = "p<%s" % (str(level)[1:])
            ax.text(x_text, y1 + y_unit / 2, label, ha='left', va='center')

        ax.set_ylim(0, self._layout.h)
        ax.set_xlim(0, self._layout.w)
        self._show()


class _SimpleFigure(EelFigure):
    def __init__(self, wintitle, *args, **kwargs):
        layout = Layout(1, 1, 5, *args, **kwargs)
        EelFigure.__init__(self, wintitle, layout)
        self._ax = ax = self._axes[0]

        # remove x-axis ticks
        xax = ax.get_xaxis()
        xax.set_ticks_position('none')

        # collector for handles for figure legend
        self._handles = []
        self._legend = None

    def _set_xlabel_categorial(self, xlabel, model):
        if xlabel is True:
            if model and model.name:
                xlabel = model.name.replace('_', ' ')
            else:
                xlabel = False

        if xlabel:
            self._ax.set_xlabel(xlabel)

    def _show(self):
        if not self._layout.w_fixed:
            # make sure x axis labels don't overlap
            self.draw()
            labels = self._ax.get_xticklabels()
            if len(labels) > 1:
                bbs = [l.get_window_extent() for l in labels]
                overlap = max(bbs[i].x1 - bbs[i + 1].x0 for i in range(len(bbs)-1))
                extend = len(bbs) * (overlap + 10)
                w, h = self._frame.GetSize()
                w += int(extend)
                self._frame.SetSize((w, h))
        EelFigure._show(self)


class Boxplot(_SimpleFigure):
    r"""Boxplot for a continuous variable

    Parameters
    ----------
    Y : Var
        Dependent variable.
    X : categorial
        Category definition (draw one box for every cell in X).
    match : None | categorial
        Match cases for a repeated measures design.
    sub : None | index-array
        Use a subset of the data.
    cells : None | sequence of cells of X
        Cells to plot (optional). All entries have to be cells of X). Can be
        used to change the order of the bars or plot only certain cells.
    datalabels : scalar
        Threshold for labeling outliers (in standard-deviation).
    bottom : scalar
        Lowest possible value on the y axis (default is 0 or slightly
        below the lowest value).
    top : scalar
        Set the upper x axis limit (default is to fit all the data).
    ylabel : str | None
        Y axis label (default is inferred from the data).
    xlabel : str | None
        X axis label (default is ``X.name``).
    xtick_delim : str
        Delimiter for x axis category descriptors (default is ``'\n'``,
        i.e. the level on each Factor of ``X`` on a separate line).
    titlekwargs : dict
        Keyword arguments for the figure title.
    test : bool | scalar
        True (default): perform pairwise tests;  False/None: no tests;
        scalar: 1-sample tests against this value.
    par : bool
        Use parametric test for pairwise comparisons (use non-parametric
        tests if False).
    trend : None | str
        Marker for a trend in pairwise comparisons.
    test_markers : bool
        For pairwise tests, plot markers indicating significance level
        (stars).
    corr : None | 'hochberg' | 'bonferroni' | 'holm'
        Method for multiple comparison correction (default 'hochberg').
    hatch : bool | str
        Matplotlib Hatch pattern to fill boxes (True to use the module
        default; default is False).
    colors : bool | sequence | dict of matplitlib colors
        Matplotlib colors to use for boxes (True to use the module default;
        default is False, i.e. no colors).
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    frame : bool
        Draw a frame containing the figure from the top and the right
        (default ``True``).
    tight : bool
        Use matplotlib's tight_layout to resize all axes to fill the figure
        (default True).
    title : str
        Figure title.
    """
    def __init__(self, Y, X=None, match=None, sub=None, cells=None, datalabels=None,
                 bottom=None, top=None, ylabel=True, xlabel=True,
                 xtick_delim='\n', test=True, par=True, trend="'", test_markers=True,
                 corr='Hochberg', hatch=False, colors=False, ds=None, *args,
                 **kwargs):
        # get data
        ct = Celltable(Y, X, match, sub, cells, ds, asvar)
        if ct.X is None and test is True:
            test = 0.

        # kwargs
        if hatch is True:
            hatch = defaults['hatch']

        if hatch and len(hatch) < ct.n_cells:
            msg = ("hatch needs at least as many values as there are cells "
                   "(%i) got %s" % (ct.n_cells, repr(hatch)))
            raise ValueError(msg)

        if colors is True:
            if defaults['mono']:
                colors = defaults['cm']['colors']
            else:
                colors = defaults['c']['colors']
        elif isinstance(colors, dict):
            colors = [colors[cell] for cell in ct.cells]

        if colors and len(colors) < ct.n_cells:
            msg = ("colors needs at least as many values as there are cells "
                   "(%i) got %s" % (ct.n_cells, repr(colors)))
            raise ValueError(msg)

        # get axes
        frame_title_ = frame_title("Boxplot", ct.Y, ct.X)
        _SimpleFigure.__init__(self, frame_title_, *args, **kwargs)
        self._set_xlabel_categorial(xlabel, ct.X)
        self._configure_yaxis(ct.Y, ylabel)
        ax = self._axes[0]

        # determine ax lim
        if bottom is None:
            if np.min(ct.Y.x) >= 0:
                bottom = 0
            else:
                d_min = np.min(ct.Y.x)
                d_max = np.max(ct.Y.x)
                d_range = d_max - d_min
                bottom = d_min - .05 * d_range

        # boxplot
        k = len(ct.cells)
        all_data = ct.get_data()
        bp = ax.boxplot(all_data, labels=ct.cellnames(xtick_delim))

        # Now fill the boxes with desired colors
        if hatch or colors:
            for i in range(ct.n_cells):
                box = bp['boxes'][i]
                box_x = box.get_xdata()[:5]  # []
                box_y = box.get_ydata()[:5]  # []
                box_coords = list(zip(box_x, box_y))
                if colors:
                    c = colors[i]
                else:
                    c = '.5'

                if hatch:
                    h = hatch[i]
                else:
                    h = ''

                poly = mpl.patches.Polygon(box_coords, facecolor=c, hatch=h,
                                           zorder=-999)
                ax.add_patch(poly)
        if defaults['mono']:
            for itemname in bp:
                bp[itemname].set_color('black')

        # labelling
        y_min = np.max(np.hstack(all_data))
        y_unit = (y_min - bottom) / 15

        # tests
        if test is True:
            y_top = _mark_plot_pairwise(ax, ct, par, y_min, y_unit, corr, trend,
                                        test_markers, x0=1)
        elif test is not False and test is not None:
            ax.axhline(test, color='black')
            y_top = _mark_plot_1sample(ax, ct, par, y_min, y_unit, test, corr, trend,
                                       x0=1)
        else:
            y_top = None

        if top is None:
            top = y_top

        # data labels
        if datalabels:
            for i, cell in enumerate(ct.cells):
                d = ct.data[cell]
                indexes = np.where(np.abs(d) / d.std() >= datalabels)[0]
                for index in indexes:
                    label = ct.matchlabels[cell][index]
                    ax.annotate(label, (i + 1, d[index]))

        # set ax limits
        ax.set_ylim(bottom, top)
        ax.set_xlim(.5, k + .5)

        self._show()


class Barplot(_SimpleFigure):
    r"""Barplot for a continuous variable

    Parameters
    ----------
    Y : Var
        Dependent variable.
    X : categorial
        Model (Factor or Interaction).
    match : None | categorial
        Match cases for a repeated measures design.
    sub : None | index-array
        Use a subset of the data.
    cells : None | sequence of cells of X
        Cells to plot (optional). All entries have to be cells of X). Can be
        used to change the order of the bars or plot only certain cells.
    test : bool | scalar
        True (default): perform pairwise tests;  False: no tests;
        scalar: 1-sample tests against this value
    par : bool
        Use parametric test for pairwise comparisons (use non-parametric
        tests if False).
    corr : None | 'hochberg' | 'bonferroni' | 'holm'
        Method for multiple comparison correction (default 'hochberg').
    trend : None | str
        Marker for a trend in pairwise comparisons.
    test_markers : bool
        For pairwise tests, plot markers indicating significance level
        (stars).
    ylabel : str | None
        Y axis label (default is inferred from the data).
    error : str
        Measure of variability to display in the error bars (default: 1
        SEM). Examples:
        'ci': 95% confidence interval;
        '99%ci': 99% confidence interval (default);
        '2sem': 2 standard error of the mean.
    pool_error : bool
        Pool the errors for the estimate of variability (default is True
        for related measures designs, False for others). See Loftus & Masson
        (1994).
    ec : matplotlib color
        Error bar color.
    xlabel : str | None
        X axis label (default is ``X.name``).
    xticks : None | sequence of str
        X-axis tick labels describing the categories. None to plot no labels
        (Default uses cell names from ``X``).
    xtick_delim : str
        Delimiter for x axis category descriptors (default is ``'\n'``,
        i.e. the level on each Factor of ``X`` on a separate line).
    hatch : bool | str
        Matplotlib Hatch pattern to fill boxes (True to use the module
        default; default is False).
    colors : bool | dict | sequence of matplitlib colors
        Matplotlib colors to use for boxes (True to use the module default;
        default is False, i.e. no colors).
    bottom : scalar
        Lower end of the y axis (default is 0).
    top : scalar
        Upper end of the y axis (default is determined from the data).
    origin : scalar
        Origin of the bars on the y-axis (the default is ``max(bottom, 0)``)
    c : matplotlib color
        Bar color (ignored if colors is specified).
    edgec : matplotlib color
        Barplot edge color.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    frame : bool
        Draw a frame containing the figure from the top and the right
        (default ``True``).
    tight : bool
        Use matplotlib's tight_layout to resize all axes to fill the figure
        (default True).
    title : str
        Figure title.
    """
    def __init__(self, Y, X=None, match=None, sub=None, cells=None, test=True, par=True,
                 corr='Hochberg', trend="'", test_markers=True, ylabel=True,
                 error='sem', pool_error=None, ec='k', xlabel=True, xticks=True,
                 xtick_delim='\n', hatch=False, colors=False, bottom=0, top=None,
                 origin=None, c='#0099FF', edgec=None, ds=None, *args, **kwargs):
        ct = Celltable(Y, X, match, sub, cells, ds, asvar)

        if pool_error is None:
            pool_error = ct.all_within

        frame_title_ = frame_title("Barplot", ct.Y, ct.X)
        _SimpleFigure.__init__(self, frame_title_, *args, **kwargs)
        self._set_xlabel_categorial(xlabel, ct.X)
        self._configure_yaxis(ct.Y, ylabel)

        x0, x1, y0, y1 = _plt_barplot(self._ax, ct, error, pool_error, hatch,
                                      colors, bottom, top, origin, c=c,
                                      edgec=edgec, ec=ec, test=test, par=par,
                                      trend=trend, corr=corr,
                                      test_markers=test_markers)

        self._ax.set_xlim(x0, x1)
        self._ax.set_ylim(y0, y1)
        if y1 > 0 > y0:
            self._ax.axhline(0, color='k')

        # figure decoration
        if xticks:
            if xticks is True:
                xticks = ct.cellnames(xtick_delim)
            self._ax.set_xticklabels(xticks)
            self._ax.set_xticks(list(range(len(ct.cells))))
        else:
            self._ax.set_xticks(())

        self._show()


def _plt_barplot(ax, ct, error, pool_error, hatch, colors, bottom, top=None,
                 origin=None, left=None, width=.5, c='#0099FF', edgec=None,
                 ec='k', test=True, par=True, trend="'", corr='Hochberg',
                 test_markers=True):
    """Draw a barplot to axes ax for Celltable ct.

    Parameters
    ----------
    ax : mpl Axes
        Axes to which to plot
    ct : Celltable
        Data to plot.
    error : str
        Variability description (e.g., "95%ci").
    pool_error : bool
        Pool the errors for the estimate of variability.
    ...
    """
    # kwargs
    if hatch is True:
        hatch = defaults['hatch']

    if colors is True:
        if defaults['mono']:
            colors = defaults['cm']['colors']
        else:
            colors = defaults['c']['colors']
    elif isinstance(colors, dict):
        colors = [colors[cell] for cell in ct.cells]

    # data means
    k = len(ct.cells)
    if left is None:
        left = np.arange(k) - width / 2
    height = np.array(ct.get_statistic(np.mean))

    # origin
    if origin is None:
        origin = max(0, bottom)

    # error bars
    if ct.X is None:
        error_match = None
    else:
        error_match = ct.match
    y_error = stats.variability(ct.Y.x, ct.X, error_match, error, pool_error, ct.cells)

    # fig spacing
    plot_max = np.max(height + y_error)
    plot_min = np.min(height - y_error)
    plot_span = plot_max - plot_min
    y_bottom = min(bottom, plot_min - plot_span * .05)

    # main BARPLOT
    bars = ax.bar(left, height - origin, width, bottom=origin,
                  color=c, edgecolor=edgec, ecolor=ec, yerr=y_error)

    # hatch
    if hatch:
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
    if colors:
        for bar, c in zip(bars, colors):
            bar.set_facecolor(c)

    # pairwise tests
    if ct.X is None and test is True:
        test = 0.
    y_unit = (plot_max - y_bottom) / 15
    if test is True:
        y_top = _mark_plot_pairwise(ax, ct, par, plot_max, y_unit, corr, trend,
                                    test_markers, top=top)
    elif (test is False) or (test is None):
        y_top = plot_max + y_unit
    else:
        ax.axhline(test, color='black')
        y_top = _mark_plot_1sample(ax, ct, par, plot_max, y_unit, test, corr, trend)

    if top is None:
        top = y_top

    #      x0,                     x1,                      y0,       y1
    lim = (min(left) - .5 * width, max(left) + 1.5 * width, y_bottom, top)
    return lim


class Timeplot(EelFigure, LegendMixin):
    """Plot a variable over time

    Parameters
    ----------
    Y : Var
        Dependent variable.
    categories : categorial
        Model (Factor or Interaction)
    time : Var
        Variable describing the time.
    match : None | categorial
        Match cases for a repeated measures design.
    sub : None | index-array
        Use a subset of the data.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    main : numpy function
        draw lines to connect values across time (default: np.mean).
        Can be 'bar' for barplots or False.
    spread : str
        How to indicate data spread.
        None: no indication;
        'box': boxplots;
        '{x}sem': x standard error of the means (e.g. '2sem');
        '{x}std': x standard deviations;
    x_jitter : bool
        When plotting error bars, jitter their location on the x-axis to
        increase readability.
    bottom : scalar
        Lower end of the y axis (default is 0).
    top : scalar
        Upper end of the y axis (default is determined from the data).
    ylabel : str | None
        Y axis label (default is inferred from the data).
    xlabel : str | None
        X axis label (default is inferred from the data).
    timelabels : sequence | dict
        Labels for the x (time) axis. Can be provided in the form of a list
        of labels from left to right, or a {time_value: label} dictionary.
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    colors : str | list | dict
        Colors for the categories.
        **str**: A colormap name; cells are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as the cells.
        **dict**: A dictionary mapping each cell to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    frame : bool
        Draw a frame containing the figure from the top and the right
        (default ``True``).
    tight : bool
        Use matplotlib's tight_layout to resize all axes to fill the figure
        (default True).
    title : str
        Figure title.
    """
    def __init__(self, Y, categories, time, match=None, sub=None, ds=None,
                 # data plotting
                 main=np.mean, spread='sem', x_jitter=False,
                 bottom=None, top=None,
                 # labelling
                 ylabel=True, xlabel=True, timelabels=None, legend='upper right',
                 colors=None, hatch=False, markers=True, *args, **kwargs):
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        categories = ascategorial(categories, sub, ds)
        time = asvar(time, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        # transform to 3 kwargs:
        # - local_plot ('bar' or 'box')
        # - line_plot (function for values to connect)
        # - spread
        if main == 'bar':
            assert spread != 'box'
            local_plot = 'bar'
            line_plot = None
        else:
            line_plot = main
            if spread == 'box':
                local_plot = 'box'
                spread = None
            else:
                local_plot = None
        del main

        # hatch/marker
        if hatch is True:
            hatch = defaults['hatch']
        if markers is True:
            if defaults['mono']:
                markers = defaults['cm']['markers']
            else:
                markers = defaults['c']['markers']

        # colors: {category index -> color, ...}
        colors = find_cell_colors(categories, colors)
        color_list = [colors[i] for i in categories.cells]

        # get axes
        layout = Layout(1, 1, 5, *args, **kwargs)
        EelFigure.__init__(self, "Timeplot", layout)
        self._configure_yaxis(Y, ylabel)
        self._configure_xaxis(time, xlabel)
        ax = self._axes[0]

        # categories
        n_cat = len(categories.cells)

        # find time points
        time_points = np.unique(time.x)
        n_time_points = len(time_points)
        time_step = min(np.diff(time_points))
        if local_plot in ['box', 'bar']:
            within_spacing = time_step / (2 * n_cat)
            padding = (2 + n_cat / 2) * within_spacing
        else:
            within_spacing = time_step / (8 * n_cat)
            padding = time_step / 4  # (2 + n_cat/2) * within_spacing

        rel_pos = np.arange(0, n_cat * within_spacing, within_spacing)
        rel_pos -= np.mean(rel_pos)

        t_min = min(time_points) - padding
        t_max = max(time_points) + padding

        # prepare container for time series
        if line_plot:
            line_values = np.empty((n_cat, n_time_points))
        if spread and local_plot != 'bar':
            yerr = np.empty((n_cat, n_time_points))

        # loop through time points
        for i_t, t in enumerate(time_points):
            ct = Celltable(Y, categories, match=match, sub=(time == t))
            if line_plot:
                line_values[:, i_t] = ct.get_statistic(line_plot)

            pos = rel_pos + t
            if local_plot == 'box':
                # boxplot
                bp = ax.boxplot(ct.get_data(out=list), positions=pos, widths=within_spacing)

                # Now fill the boxes with desired colors
#               if hatch or colors:
#                   numBoxes = len(bp['boxes'])
                for i, cell in enumerate(ct.cells):
                    box = bp['boxes'][i]
                    boxX = box.get_xdata()[:5]
                    boxY = box.get_ydata()[:5]
                    boxCoords = list(zip(boxX, boxY))

                    c = colors[cell]
                    try:
                        h = hatch[i]
                    except:
                        h = ''
                    patch = mpl.patches.Polygon(boxCoords, facecolor=c, hatch=h,
                                                zorder=-999)
                    ax.add_patch(patch)

                if True:  # defaults['mono']:
                    for itemname in bp:
                        plt.setp(bp[itemname], color='black')
            elif local_plot == 'bar':
                _plt_barplot(ax, ct, spread, False, hatch, color_list, 0, left=pos,
                             width=within_spacing, test=False)
            elif spread:
                yerr[:, i_t] = ct.get_statistic(spread)

        legend_handles = {}
        if line_plot:
            # plot means
            x = time_points
            for i, cell in enumerate(categories.cells):
                y = line_values[i]
                name = cellname(cell)
                color = colors[cell]

                if hatch:
                    ls = defaults['linestyle'][i]
                    if color == '1.':
                        color = '0.'
                else:
                    ls = '-'

                if ls == '-':
                    mfc = color
                else:
                    mfc = '1.'

                if not markers or len(markers) <= i:
                    marker = None
                else:
                    marker = markers[i]

                handles = ax.plot(x, y, color=color, linestyle=ls, label=name,
                                  zorder=6, marker=marker, mfc=mfc)
                legend_handles[cell] = handles[0]

                if spread:
                    if x_jitter:
                        x_errbars = x + rel_pos[i]
                    else:
                        x_errbars = x
                    ax.errorbar(x_errbars, y, yerr=yerr[i], fmt=None, zorder=5,
                                ecolor=color, linestyle=ls, label=name)
        else:
            legend = False

        # finalize
        ax.set_xlim(t_min, t_max)
        ax.set_xticks(time_points)
        ax.set_ylim(bottom, top)
        if timelabels is not None:
            if isinstance(timelabels, dict):
                xticklabels = [timelabels[t] for t in time_points]
            else:
                xticklabels = [str(l) for l in timelabels]
            ax.set_xticklabels(xticklabels)

        LegendMixin.__init__(self, legend, legend_handles)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


def _reg_line(Y, reg):
    coeff = np.hstack([np.ones((len(reg), 1)), reg[:, None]])
    (a, b), residues, rank, s = np.linalg.lstsq(coeff, Y)
    regline_x = np.array([min(reg), max(reg)])
    regline_y = a + b * regline_x
    return regline_x, regline_y


class Correlation(EelFigure, LegendMixin):
    """Plot the correlation between two variables

    Parameters
    ----------
    Y : Var
        Variable for the y-axis.
    X : Var
        Variable for the x-axis.
    cat : categorial
        Plot the correlation separately for different categories.
    sub : None | index-array
        Use a subset of the data.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    c : list
        List of colors for cells.
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    xlabel, ylabel : str
        Label for the x- and y-axis (default based on data).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : str
        Figure title.
    """
    def __init__(self, y, x, cat=None, sub=None, ds=None,
                 c=['b', 'r', 'k', 'c', 'm', 'y', 'g'], legend='upper right',
                 xlabel=True, ylabel=True, *args, **kwargs):
        sub = assub(sub, ds)
        y = asvar(y, sub, ds)
        x = asvar(x, sub, ds)
        if cat is not None:
            cat = ascategorial(cat, sub, ds)

        # figure
        frame_title_ = frame_title("Correlation", y, x, cat)
        layout = Layout(1, 1, 5, *args, **kwargs)
        EelFigure.__init__(self, frame_title_, layout)
        self._configure_yaxis(y, ylabel)
        self._configure_xaxis(x, xlabel)

        ax = self._axes[0]
        legend_handles = {}
        if cat is None:
            legend = False
            ax.scatter(x.x, y.x, alpha=.5)
        else:
            for color, cell in zip(c, cat.cells):
                idx = (cat == cell)
                label = cellname(cell)
                h = ax.scatter(x[idx].x, y[idx].x, c=color, label=label, alpha=.5)
                legend_handles[label] = h

        # limits
        for func, data in ((ax.set_xlim, x), (ax.set_ylim, y)):
            min_ = data.min()
            max_ = data.max()
            range_ = max_ - min_
            delta = range_ / 20.
            func(min_ - delta, max_ + delta)

        LegendMixin.__init__(self, legend, legend_handles)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


class Regression(EelFigure, LegendMixin):
    """Plot the regression of Y on X

    parameters
    ----------
    Y : Var
        Variable for the y-axis.
    X : Var
        Variable for the x-axis.
    cat : categorial
        Plot the regression separately for different categories.
    match : None | categorial
        Match cases for a repeated measures design.
    sub : None | index-array
        Use a subset of the data.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    ylabel : str
        Y-axis label (default is ``Y.name``).
    alpha : scalar
        alpha for individual data points (to control visualization of
        overlap)
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    ...
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : str
        Figure title.
    """
    def __init__(self, Y, X, cat=None, match=None, sub=None, ds=None,
                 xlabel=True, ylabel=True, alpha=.2, legend='upper right',
                 c=['#009CFF', '#FF7D26', '#54AF3A', '#FE58C6', '#20F2C3'],
                 *args, **kwargs):
        sub = assub(sub, ds)
        Y = asvar(Y, sub, ds)
        X = asvar(X, sub, ds)
        if cat is not None:
            cat = ascategorial(cat, sub, ds)
        if match is not None:
            raise NotImplementedError("match kwarg")

        if ylabel is True:
            ylabel = Y.name

        # figure
        frame_title_ = frame_title("Regression", Y, X, cat)
        layout = Layout(1, 1, 5, *args, **kwargs)
        EelFigure.__init__(self, frame_title_, layout)
        self._configure_xaxis(X, xlabel)
        self._configure_yaxis(Y, ylabel)

        ax = self._axes[0]
        legend_handles = {}
        if cat is None:
            legend = False
            if type(c) in [list, tuple]:
                color = c[0]
            else:
                color = c
            y = Y.x
            reg = X.x
            ax.scatter(reg, y, edgecolor=color, facecolor=color, s=100,
                       alpha=alpha, marker='o', label='_nolegend_')
            x, y = _reg_line(y, reg)
            ax.plot(x, y, c=color)
        else:
            for i, cell in enumerate(cat.cells):
                idx = (cat == cell)
                # scatter
                y = Y.x[idx]
                reg = X.x[idx]
                color = c[i % len(c)]
                h = ax.scatter(reg, y, edgecolor=color, facecolor=color, s=100,
                               alpha=alpha, marker='o', label='_nolegend_')
                legend_handles[cell] = h
                # regression line
                x, y = _reg_line(y, reg)
                ax.plot(x, y, c=color, label=cellname(cell))

        LegendMixin.__init__(self, legend, legend_handles)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


# MARK: Requirements for Statistical Tests

def _difference(data, names):
    "Data condition x subject"
    data_differences = []; diffnames = []; diffnames_2lines = []
    for i, (name1, data1) in enumerate(list(zip(names, data))):
        for name2, data2 in zip(names[i + 1:], data[i + 1:]):
            data_differences.append(data1 - data2)
            diffnames.append('-'.join([name1, name2]))
            diffnames_2lines.append('-\n'.join([name1, name2]))
    return data_differences, diffnames, diffnames_2lines


def _ax_histogram(ax, data, normed, **kwargs):
    """Create normality test figure"""
    n, bins, patches = ax.hist(data, normed=normed, **kwargs)
    data = np.ravel(data)

    # normal line
    mu = np.mean(data)
    sigma = np.std(data)
    y = mpl.mlab.normpdf(bins, mu, sigma)
    if not normed:
        y *= len(data) * np.diff(bins)[0]
    ax.plot(bins, y, 'r--', linewidth=1)

    # TESTS
    # test Anderson
    A2, thresh, sig = scipy.stats.morestats.anderson(data)
    index = sum(A2 >= thresh)
    if index > 0:
        ax.text(.95, .95, '$^{*}%s$' % str(sig[index - 1] / 100), color='r', size=11,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,)
    logging.debug(" Anderson: %s, %s, %s" % (A2, thresh, sig))
    # test Lilliefors
    n_test = test.lilliefors(data)
    ax.set_xlabel(r"$D=%.3f$, $p_{est}=%.2f$" % n_test)  # \chi ^{2}
    # make sure ticks display int values


class Histogram(EelFigure):
    """
    Histogram plots with tests of normality

    Parameters
    ----------
    Y : Var
        Dependent variable.
    X : categorial
        Categories for separate histograms.
    match : None | categorial
        Match cases for a repeated measures design.
    sub : None | index-array
        Use a subset of the data.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables
    pooled : bool
        Add one plot with all values/differences pooled.
    normed : bool
        Norm counts to approximate a probability density (default False).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | str
        Figure title.
    """
    def __init__(self, Y, X=None, match=None, sub=None, ds=None, pooled=True,
                 normed=False, tight=True, title=True, *args, **kwargs):
        ct = Celltable(Y, X, match=match, sub=sub, ds=ds, coercion=asvar)

        # layout
        if X is None:
            nax = 1
            if title is True:
                title = "Test for Normality"
        elif ct.all_within:
            n_comp = len(ct.cells) - 1
            nax = n_comp ** 2
            kwargs['nrow'] = n_comp
            kwargs['ncol'] = n_comp
            self._make_axes = False
            self._default_ylabel_ax = -1
            if title is True:
                title = "Tests for Normality of the Differences"
        else:
            nax = len(ct.cells)
            if title is True:
                title = "Tests for Normality"

        frame_title_ = frame_title("Histogram", ct.Y, ct.X)
        layout = Layout(nax, 1, 3, tight, title, *args, **kwargs)
        EelFigure.__init__(self, frame_title_, layout)

        if X is None:
            _ax_histogram(self._axes[0], ct.Y.x, normed)
        elif ct.all_within:  # distribution of differences
            data = ct.get_data()
            names = ct.cellnames()

            pooled_data = []
            # i: row
            # j: -column
            for i in range(n_comp + 1):
                for j in range(i + 1, n_comp + 1):
                    difference = data[i] - data[j]
                    pooled_data.append(scipy.stats.zscore(difference))  # z transform?? (scipy.stats.zs())
                    ax_i = n_comp * i + (n_comp + 1 - j)
                    ax = self.figure.add_subplot(n_comp, n_comp, ax_i)
                    self._axes.append(ax)
                    _ax_histogram(ax, difference, normed)
                    if i == 0:
                        ax.set_title(names[j], size=12)
                    if j == n_comp:
                        ax.set_ylabel(names[i], size=12)
            # pooled diffs
            if pooled and len(names) > 2:
                ax = self.figure.add_subplot(n_comp, n_comp, n_comp ** 2)
                self._axes.append(ax)
                _ax_histogram(ax, pooled_data, normed, facecolor='g')
                ax.set_title("Pooled Differences (n=%s)" % len(pooled_data),
                             weight='bold')
                self.figure.text(.99, .01, "$^{*}$ Anderson and Darling test "
                                 "thresholded at $[ .15,   .10,    .05,    "
                                 ".025,   .01 ]$.", color='r',
                                 verticalalignment='bottom',
                                 horizontalalignment='right')
        else:  # independent measures
            for cell, ax in zip(ct.cells, self._axes):
                ax.set_title(cellname(cell))
                _ax_histogram(ax, ct.data[cell], normed)

        if normed:
            self._configure_yaxis('p', 'probability density')
        else:
            self._configure_yaxis('n', 'count')
        self._configure_xaxis(ct.Y, False)
        self._show()


def boxcox_explore(Y, params=[-1, -.5, 0, .5, 1], crange=False, ax=None, box=True):
    """
    Plot the distribution resulting from a Box-Cox transformation

    Parameters
    ----------
    Y : array-like
        array containing data whose distribution is to be examined
    crange :
        correct range of transformed data
    ax :
        ax to plot to (``None`` creates a new figure)
    box :
        use Box-Cox family

    """
    if hasattr(Y, 'x'):
        Y = Y.x
    else:
        Y = np.ravel(Y)

    if np.any(Y == 0):
        raise ValueError("data contains 0")

    y = []
    for p in params:
        if p == 0:
            if box:
                xi = np.log(Y)
            else:
                xi = np.log10(Y)
                # xi = np.log1p(x)
        else:
            if box:
                xi = (Y ** p - 1) / p
            else:
                xi = Y ** p
        if crange:
            xi -= min(xi)
            xi /= max(xi)
        y.append(xi)

    if not ax:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111)

    ax.boxplot(y)
    ax.set_xticks(list(range(1, 1 + len(params))))
    ax.set_xticklabels(params)
    ax.set_xlabel("p")
    if crange:
        ax.set_ylabel("Value (Range Corrected)")
