# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot uniform time-series of one variable."""



import operator

import matplotlib as mpl
import numpy as np

from .._data_obj import (
    ascategorial, asndvar, assub, cellname, Celltable, longname)
from .._stats import stats
from . import _base
from ._base import EelFigure, Layout, LegendMixin, YLimMixin, XAxisMixin
from ._colors import colors_for_oneway, find_cell_colors
from .._colorspaces import oneway_colors
from functools import reduce


class UTSStat(LegendMixin, YLimMixin, EelFigure):
    """
    Plot statistics for a one-dimensional NDVar

    Parameters
    ----------
    Y : 1d-NDVar
        Dependent variable (one-dimensional NDVar).
    X : categorial or None
        Model: specification of conditions which should be plotted separately.
    Xax : None | categorial
        Make separate axes for each category in this categorial model.
    match : Factor
        Identifier for repeated measures data.
    sub : None | index array
        Only use a subset of the data provided.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    main : func | None
        Measure for the central tendency (function that takes an ``axis``
        argument). The default is numpy.mean.
    error : None | str
        Measure of variability to plot (default: 1 SEM). Examples:
        'ci': 95% confidence interval;
        '99%ci': 99% confidence interval (default);
        '2sem': 2 standard error of the mean;
        'all': plot all traces.
    pool_error : bool
        Pool the errors for the estimate of variability (default is True
        for related measures designs, False otherwise). See Loftus & Masson
        (1994).
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlabel : str | None
        X-axis labels. By default the label is inferred from the data.
    ylabel : str | None
        Y-axis labels. By default the label is inferred from the data.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    invy : bool
        Invert the y axis (if ``bottom`` and/or ``top`` are specified explicitly
        they take precedence; an inverted y-axis can also be produced by
        specifying ``bottom > top``).
    bottom, top | None | scalar
        Set an absolute range for the plot's y axis.
    hline : None | scalar | (value, kwarg-dict) tuple
        Add a horizontal line to each plot. If provided as a tuple, the second
        element can include any keyword arguments that should be submitted to
        the call to matplotlib axhline call.
    xdim : str
        dimension for the x-axis (default is 'time')
    xlim : None | (scalar, scalar)
        Tuple of xmin and xmax to set the initial x-axis limits.
    color : matplotlib color
        Color if just a single category of data is plotted.
    colors : str | list | dict
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; Cells of X are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as X.cells.
        **dict**: A dictionary mapping each cell in X to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    clusters : None | Dataset
        Clusters to add to the plots. The clusters should be provided as
        Dataset, as stored in test results' :py:attr:`.clusters`.
    pmax : scalar
        Maximum p-value of clusters to plot as solid.
    ptrend : scalar
        Maximum p-value of clusters to plot as trend.
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : str | None
        Figure title.
    frame : bool | 't'
        How to frame the plots.
        ``True`` (default): normal matplotlib frame;
        ``False``: omit top and right lines;
        ``'t'``: draw spines at x=0 and y=0, common for ERPs.

    Notes
    -----
    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    def __init__(self, Y='Y', X=None, Xax=None, match=None, sub=None, ds=None,
                 main=np.mean, error='sem', pool_error=None, legend='upper right',
                 axtitle=True, xlabel=True, ylabel=True, xticklabels=True,
                 invy=False, bottom=None, top=None, hline=None, xdim='time',
                 xlim=None, color='b', colors=None, clusters=None, pmax=0.05,
                 ptrend=0.1, *args, **kwargs):
        # coerce input variables
        sub = assub(sub, ds)
        Y = asndvar(Y, sub, ds)
        if X is not None:
            X = ascategorial(X, sub, ds)
        if Xax is not None:
            Xax = ascategorial(Xax, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        if error and error != 'all' and \
                (pool_error or (pool_error is None and match is not None)):
            all_x = [i for i in (Xax, X) if i is not None]
            if len(all_x) > 0:
                full_x = reduce(operator.mod, all_x)
                ct = Celltable(Y, full_x, match)
                dev_data = stats.variability(ct.Y.x, ct.X, ct.match, error, True)
                error = 'data'
            else:
                dev_data = None
        else:
            dev_data = None

        if Xax is None:
            nax = 1
            ct = Celltable(Y, X, match)
            if X is None:
                color_x = None
            else:
                color_x = ct.X
        else:
            ct = Celltable(Y, Xax)
            if X is None:
                color_x = None
                X_ = None
            else:
                Xct = Celltable(X, Xax)
                color_x = Xct.Y
            if match is not None:
                matchct = Celltable(match, Xax)
            nax = len(ct.cells)

        # assemble colors
        if color_x is None:
            colors = {None: color}
        else:
            colors = find_cell_colors(color_x, colors)

        frame_title = _base.frame_title("UTSStat", Y, X, Xax)
        layout = Layout(nax, 2, 4, *args, share_axes=True, **kwargs)
        EelFigure.__init__(self, frame_title, layout)
        clip = layout.frame

        # create plots
        self._plots = []
        legend_handles = {}
        if Xax is None:
            p = _ax_uts_stat(self._axes[0], ct, colors, main, error, dev_data,
                             xdim, xlim, invy, bottom, top, hline, clusters,
                             pmax, ptrend, clip)
            self._plots.append(p)
            legend_handles.update(p.legend_handles)
            if len(ct) < 2:
                legend = False
        else:
            for i, ax, cell in zip(range(nax), self._axes, ct.cells):
                if X is not None:
                    X_ = Xct.data[cell]

                if match is not None:
                    match = matchct.data[cell]

                ct_ = Celltable(ct.data[cell], X_, match=match, coercion=asndvar)
                p = _ax_uts_stat(ax, ct_, colors, main, error, dev_data,
                                 xdim, xlim, invy, bottom, top, hline, clusters,
                                 pmax, ptrend, clip)
                self._plots.append(p)
                legend_handles.update(p.legend_handles)
            self._set_axtitle(axtitle, names=list(map(cellname, ct.cells)))

        self._configure_yaxis(ct.Y, ylabel)
        self._configure_xaxis_dim(ct.Y.get_dim(xdim), xlabel, xticklabels)
        YLimMixin.__init__(self, self._plots)
        LegendMixin.__init__(self, legend, legend_handles)
        self._update_ui_cluster_button()
        self._show()

    def _fill_toolbar(self, tb):
        import wx

        btn = self._cluster_btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        btn.Enable(False)
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

        LegendMixin._fill_toolbar(self, tb)

    def _OnShowClusterInfo(self, event):
        from .._wxutils import show_text_dialog

        if len(self._plots) == 1:
            clusters = self._plots[0].cluster_plt.clusters
            all_plots_same = True
        else:
            all_clusters = [p.cluster_plt.clusters is None for p in self._plots]
            clusters = all_clusters[0]
            if all(c is clusters for c in all_clusters[1:]):
                all_plots_same = True
            else:
                all_plots_same = False

        if all_plots_same:
            info = str(clusters)
        else:
            info = []
            for i, clusters in enumerate(all_clusters):
                if clusters is None:
                    continue
                title = "Axes %i" % i
                info.append(title)
                info.append('\n')
                info.append('-' * len(title))
                info.append(str(clusters))
            info = '\n'.join(info)

        show_text_dialog(self._frame, info, "Clusters")

    def _update_ui_cluster_button(self):
        if hasattr(self, '_cluster_btn'):
            enable = not all(p.cluster_plt.clusters is None for p in self._plots)
            self._cluster_btn.Enable(enable)

    def _get_axes(self, axes):
        "Iterate axes corresponding to ``axes`` parameter"
        if axes is None:
            return self._axes
        elif isinstance(axes, int):
            return self._axes[axes],
        else:
            return (self._axes[i] for i in axes)

    def add_hline(self, y, axes=None, *args, **kwargs):
        """Draw a vertical bar on all axes

        Parameters
        ----------
        y : scalar
            Level at which to plot the line.
        axes : int | list of int
            Which axes to mark (default is all axes).
        ...
            :meth:`matplotlib.axes.Axes.axhline` parameters.


        Notes
        -----
        See Matplotlib's :meth:`matplotlib.axes.Axes.axhline` for more
        arguments.
        """
        for ax in self._get_axes(axes):
            ax.axhline(y, *args, **kwargs)

    def add_vspan(self, xmin, xmax, axes=None, *args, **kwargs):
        """Draw a vertical bar on all axes

        Parameters
        ----------
        xmin : scalar
            Start value on the x-axis.
        xmax : scalar
            Last value on the x-axis.
        axes : int | list of int
            Which axes to mark (default is all axes).
        ...
            :meth:`matplotlib.axes.Axes.axvspan` parameters.


        Notes
        -----
        See Matplotlib's :meth:`matplotlib.axes.Axes.axvspan` for more
        arguments.
        """
        for ax in self._get_axes(axes):
            ax.axvspan(xmin, xmax, *args, **kwargs)

    def set_clusters(self, clusters, pmax=0.05, ptrend=0.1, color='.7', ax=None):
        """Add clusters from a cluster test to the plot (as shaded area).

        Parameters
        ----------
        clusters : None | Dataset
            The clusters, as stored in test results' :py:attr:`.clusters`.
            Use ``None`` to remove the clusters plotted on a given axis.
        pmax : scalar
            Maximum p-value of clusters to plot as solid.
        ptrend : scalar
            Maximum p-value of clusters to plot as trend.
        color : matplotlib color
            Color for the clusters.
        ax : None | int
            Index of the axes to which the clusters are to be added. If None,
            add the clusters to all axes.
        """
        nax = len(self._axes)
        if ax is None:
            axes = range(nax)
        else:
            axes = [ax]

        # update plots
        for ax in axes:
            p = self._plots[ax].cluster_plt
            p.set_clusters(clusters, False)
            p.set_color(color, False)
            p.set_pmax(pmax, ptrend)
        self.draw()

        self._update_ui_cluster_button()

    def set_xlim(self, xmin, xmax):
        "Adjust the x-axis limits on all axes"
        for ax in self._axes:
            ax.set_xlim(xmin, xmax)
        self.draw()

    def set_ylim(self, bottom=None, top=None):
        "Adjust the y-axis limits on all axes"
        self._axes[0].set_ylim(bottom, top)
        self.draw()


class UTS(LegendMixin, YLimMixin, XAxisMixin, EelFigure):
    """Value by time plot for UTS data

    Parameters
    ----------
    epochs : epochs
        Uts data epochs to plot.
    xax : None | categorial
        Make separate axes for each category in this categorial model.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    ds : None | Dataset
        If a Dataset is specified, all data-objects can be specified as
        names of Dataset variables.
    sub : str | array
        Specify a subset of the data.
    xlabel, ylabel : str | None
        X- and y axis labels. By default the labels will be inferred from
        the data.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    bottom, top : scalar
        Y-axis limits.
    legend : str | int | 'fig' | None
        Matplotlib figure legend location argument or 'fig' to plot the
        legend in a separate figure.
    xlim : (scalar, scalar)
        Initial x-axis view limits (default is the full x-axis in the data).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | str
        Figure title.

    Notes
    -----
    Navigation:
     - ``↑``: scroll up
     - ``↓``: scroll down
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: x-axis zoom in (reduce x axis range)
     - ``d``: x-axis zoom out (increase x axis range)
     - ``r``: y-axis zoom in (reduce y-axis range)
     - ``c``: y-axis zoom out (increase y-axis range)
    """
    def __init__(self, epochs, xax=None, axtitle=True, ds=None, sub=None,
                 xlabel=True, ylabel=True, xticklabels=True, bottom=None,
                 top=None, legend='upper right', xlim=None, *args, **kwargs):
        epochs, (xdim,), frame_title = _base.unpack_epochs_arg(
            epochs, (None,), xax, ds, "UTS", sub
        )
        layout = Layout(len(epochs), 2, 4, *args, **kwargs)
        EelFigure.__init__(self, frame_title, layout)
        self._set_axtitle(axtitle, epochs)

        e0 = epochs[0][0]
        self._configure_xaxis_dim(e0.get_dim(xdim), xlabel, xticklabels)
        self._configure_yaxis(e0, ylabel)

        self.plots = []
        legend_handles = {}
        vlims = _base.find_fig_vlims(epochs, top, bottom)
        colors = oneway_colors(max(list(map(len, epochs))))
        for ax, layers in zip(self._axes, epochs):
            h = _ax_uts(ax, layers, xdim, vlims, colors)
            self.plots.append(h)
            legend_handles.update(h.legend_handles)

        self.epochs = epochs
        XAxisMixin.__init__(self, epochs, xdim, xlim)
        YLimMixin.__init__(self, self.plots)
        LegendMixin.__init__(self, legend, legend_handles)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)


class _ax_uts_stat(object):

    def __init__(self, ax, ct, colors, main, error, dev_data, xdim, xlim,
                 invy, bottom, top, hline, clusters, pmax, ptrend, clip):
        # stat plots
        self.ax = ax
        self.stat_plots = []
        self.legend_handles = {}

        x = ct.Y.get_dim(xdim)
        for cell in ct.cells:
            ndvar = ct.data[cell]
            y = ndvar.get_data(('case', xdim))
            plt = _plt_uts_stat(ax, x, y, main, error, dev_data, colors[cell],
                                cellname(cell), clip)
            self.stat_plots.append(plt)
            if plt.main is not None:
                self.legend_handles[cell] = plt.main[0]

        # hline
        if hline is not None:
            if isinstance(hline, tuple):
                if len(hline) != 2:
                    raise ValueError("hline must be None, scalar or length 2 tuple")
                hline, hline_kw = hline
                hline_kw = dict(hline_kw)
            else:
                hline_kw = {'color': 'k'}

            hline = float(hline)
            ax.axhline(hline, **hline_kw)

        # cluster plot
        self.cluster_plt = _plt_uts_clusters(ax, clusters, pmax, ptrend)

        # format x axis
        if xlim is None:
            ax.set_xlim(x[0], x[-1])
        else:
            xmin, xmax = xlim
            ax.set_xlim(xmin, xmax)

        # format y axis
        if invy:
            y0, y1 = ax.get_ylim()
            if bottom is None:
                bottom = y1

            if top is None:
                top = y0
        self.set_ylim(bottom, top)

    @property
    def title(self):
        return self.ax.get_title()

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class UTSClusters(EelFigure):
    """Plot permutation cluster test results

    Parameters
    ----------
    res : testnd.anova
        ANOVA with permutation cluster test result object.
    pmax : scalar
        Maximum p-value of clusters to plot as solid.
    ptrend : scalar
        Maximum p-value of clusters to plot as trend.
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    cm : str
        Colormap to use for coloring different effects.
    overlay : bool
        Plot epochs (time course for different effects) on top of each
        other (as opposed to on separate axes).
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : str
        Figure title.
    """
    def __init__(self, res, pmax=0.05, ptrend=0.1, axtitle=True, cm=None,
                 overlay=False, xticklabels=True, *args, **kwargs):
        clusters_ = res.clusters

        epochs, (xdim,), frame_title = _base.unpack_epochs_arg(
            res, (None,), plot_name="UTSClusters"
        )
        # create figure
        n = len(epochs)
        nax = 1 if overlay else n
        layout = Layout(nax, 2, 4, *args, **kwargs)
        EelFigure.__init__(self, frame_title, layout)
        self._set_axtitle(axtitle, epochs)

        colors = colors_for_oneway(list(range(n)), cmap=cm)
        self._caxes = []
        if overlay:
            ax = self._axes[0]

        for i, layers in enumerate(epochs):
            stat = layers[0]
            if not overlay:
                ax = self._axes[i]

            # ax clusters
            if clusters_:
                if 'effect' in clusters_:
                    cs = clusters_.sub('effect == %r' % stat.name)
                else:
                    cs = clusters_
            else:
                cs = None

            cax = _ax_uts_clusters(ax, stat, cs, colors[i], pmax, ptrend, xdim)
            self._caxes.append(cax)

        e0 = epochs[0][0]
        self._configure_yaxis(e0, True)
        self._configure_xaxis_dim(e0.get_dim(xdim), True, xticklabels)
        self.clusters = clusters_
        self._show()

    def _fill_toolbar(self, tb):
        import wx

        btn = wx.Button(tb, wx.ID_ABOUT, "Clusters")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnShowClusterInfo)

    def _OnShowClusterInfo(self, event):
        from .._wxutils import show_text_dialog
        info = str(self.clusters)
        show_text_dialog(self._frame, info, "Clusters")

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        "Set the threshold p-value for clusters to be displayed"
        for cax in self._caxes:
            cax.set_pmax(pmax, ptrend)
        self.draw()


class _ax_uts(object):

    def __init__(self, ax, layers, xdim, vlims, colors):
        l0 = layers[0]
        vmin, vmax = _base.find_uts_ax_vlim(layers, vlims)

        self.legend_handles = {}
        for l, color in zip(layers, colors):
            color = l.info.get('color', color)
            p = _plt_uts(ax, l, xdim, color)
            self.legend_handles[longname(l)] = p.plot_handle
            contours = l.info.get('contours', None)
            if contours:
                for v, color in contours.items():
                    if v in contours:
                        continue
                    contours[v] = ax.axhline(v, color=color)

        x = l0.get_dim(xdim)
        ax.set_xlim(x[0], x[-1])

        self.ax = ax
        self.set_ylim(vmin, vmax)

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class _plt_uts(object):

    def __init__(self, ax, ndvar, xdim, color=None, kwargs={}):
        y = ndvar.get_data((xdim,))
        x = ndvar.get_dim(xdim).x
        if color is not None:
            kwargs['color'] = color
        self.plot_handle = ax.plot(x, y, label=longname(ndvar), **kwargs)[0]

        for y, kwa in _base.find_uts_hlines(ndvar):
            if color is not None:
                kwa['color'] = color
            ax.axhline(y, **kwa)


class _ax_uts_clusters:
    def __init__(self, ax, Y, clusters, color=None, pmax=0.05, ptrend=0.1,
                 xdim='time'):
        uts_args = _base.find_uts_args(Y, False, color)
        self._bottom, self._top = _base.find_vlim_args(Y)

        _plt_uts(ax, Y, xdim=xdim, **uts_args)

        if np.any(Y.x < 0) and np.any(Y.x > 0):
            ax.axhline(0, color='k')

        # pmap
        self.cluster_plt = _plt_uts_clusters(ax, clusters, pmax, ptrend, color)

        # save ax attr
        self.ax = ax
        x = Y.get_dim(xdim).x
        self.xlim = (x[0], x[-1])

        ax.set_xlim(*self.xlim)
        ax.set_ylim(bottom=self._bottom, top=self._top)

    def set_clusters(self, clusters):
        self.cluster_plt.set_clusters(clusters)

    def set_pmax(self, pmax=0.05, ptrend=0.1):
        self.cluster_plt.set_pmax(pmax, ptrend)


class _plt_uts_clusters:
    """UTS cluster plot

    Parameters
    ----------
    ax : Axes
        Axes.
    clusters : Dataset
        Dataset with entries for 'tstart', 'tstop' and 'p'.
    """
    def __init__(self, ax, clusters, pmax, ptrend, color=None, hatch='/'):
        self.pmax = pmax
        self.ptrend = ptrend
        self.h = []
        self.ax = ax
        self.clusters = clusters
        self.color = color
        self.hatch = hatch
        self.update()

    def set_clusters(self, clusters, update=True):
        self.clusters = clusters
        if update:
            self.update()

    def set_color(self, color, update=True):
        self.color = color
        if update:
            self.update()

    def set_pmax(self, pmax, ptrend, update=True):
        self.pmax = pmax
        self.ptrend = ptrend
        if update:
            self.update()

    def update(self):
        h = self.h
        while len(h):
            h.pop().remove()

        clusters = self.clusters
        if clusters is None:
            return

        p_include = self.ptrend or self.pmax
        for cluster in clusters.itercases():
            if 'p' in cluster:
                p = cluster['p']
                if p > p_include:
                    continue
                alpha = 0.5 if p < self.pmax else 0.2
            else:
                alpha = 0.5

            x0 = cluster['tstart']
            x1 = cluster['tstop']
            h = self.ax.axvspan(x0, x1, color=self.color,  # , hatch=self.hatch,
                                fill=True, alpha=alpha, zorder=-1)
            self.h.append(h)


class _plt_uts_stat(object):

    def __init__(self, ax, x, y, main, error, dev_data, color, label, clip):
        # plot main
        if hasattr(main, '__call__'):
            y_main = main(y, axis=0)
            lw = mpl.rcParams['lines.linewidth']
            if error == 'all':
                lw *= 2
            self.main = ax.plot(x, y_main, color=color, label=label, lw=lw,
                                zorder=5, clip_on=clip)
        elif error == 'all':
            self.main = None
        else:
            raise ValueError("Invalid argument: main=%r" % main)

        # plot error
        if error == 'all':
            self.error = ax.plot(x, y.T, color=color, alpha=0.3, clip_on=clip)
        elif error:
            if error == 'data':
                pass
            elif hasattr(error, '__call__'):
                dev_data = error(y, axis=0)
            else:
                dev_data = stats.variability(y, None, None, error, False)
            lower = y_main - dev_data
            upper = y_main + dev_data
            self.error = ax.fill_between(x, lower, upper, color=color, alpha=0.3,
                                         linewidth=0, zorder=0, clip_on=clip)
        else:
            self.error = None
