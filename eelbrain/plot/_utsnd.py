# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Plot multidimensional uniform time series."""





import numpy as np

from .._names import INTERPOLATE_CHANNELS
from . import _base
from ._base import EelFigure, Layout, ColorMapMixin, LegendMixin, YLimMixin, \
    XAxisMixin, TopoMapKey


class _plt_im(object):

    _aspect = 'auto'

    def __init__(self, ax, ndvar, overlay, cmaps, vlims, contours, extent,
                 interpolation, mask=None):
        self.ax = ax
        im_kwa = _base.find_im_args(ndvar, overlay, vlims, cmaps)
        self._meas = meas = ndvar.info.get('meas')
        self._contours = contours.get(meas, None)
        self._data = self._data_from_ndvar(ndvar)
        self._extent = extent
        self._mask = mask

        if im_kwa is not None:
            self.im = ax.imshow(self._data, origin='lower', aspect=self._aspect,
                                extent=extent, interpolation=interpolation,
                                **im_kwa)
            self._cmap = im_kwa['cmap']
            if mask is not None:
                self.im.set_clip_path(mask)
            self.vmin, self.vmax = self.im.get_clim()
        else:
            self.im = None
            self.vmin = self.vmax = None

        # draw flexible parts
        self._contour_h = None
        self._draw_contours()

    def _data_from_ndvar(self, ndvar):
        raise NotImplementedError

    def _draw_contours(self):
        if self._contour_h:
            for c in self._contour_h.collections:
                c.remove()
            self._contour_h = None

        if self._contours:
            h = self.ax.contour(self._data, origin='lower', aspect=self._aspect,
                                extent=self._extent, **self._contours)
            if self._mask is not None:
                for c in h.collections:
                    c.set_clip_path(self._mask)
            self._contour_h = h

    def add_contour(self, meas, level, color):
        if self._meas == meas:
            levels = tuple(self._contours['levels']) + (level,)
            colors = tuple(self._contours['colors']) + (color,)
            self._contours = {'levels': levels, 'colors': colors}
            self._draw_contours()

    def get_kwargs(self):
        "Get the arguments required for plotting the im"
        if self.im:
            vmin, vmax = self.im.get_clim()
            args = dict(cmap=self._cmap, vmin=vmin, vmax=vmax)
        else:
            args = {}
        return args

    def set_cmap(self, cmap, meas=None):
        if (self.im is not None) and (meas is None or meas == self._meas):
            self.im.set_cmap(cmap)
            self._cmap = cmap

    def set_data(self, ndvar, vlim=False):
        data = self._data_from_ndvar(ndvar)
        if self.im is not None:
            self.im.set_data(data)
            if vlim:
                vmin, vmax = _base.find_vlim_args(ndvar)
                self.set_vlim(vmax, None, vmin)

        self._data = data
        self._draw_contours()

    def set_vlim(self, vmax, meas, vmin):
        if self.im is None:
            return
        elif (meas is not None) and (self._meas != meas):
            return

        if vmax is None:
            _, vmax = self.im.get_clim()

        vmin, vmax = _base.fix_vlim_for_cmap(vmin, vmax, self._cmap)
        self.im.set_clim(vmin, vmax)
        self.vmin, self.vmax = self.im.get_clim()


class _plt_im_array(_plt_im):

    def __init__(self, ax, ndvar, overlay, dimnames, interpolation, vlims,
                 cmaps, contours):
        self._dimnames = dimnames[::-1]
        xdim, ydim = ndvar.get_dims(dimnames)
        extent = xdim._axis_im_extent() + ydim._axis_im_extent()
        _plt_im.__init__(self, ax, ndvar, overlay, cmaps, vlims, contours,
                         extent, interpolation)

    def _data_from_ndvar(self, ndvar):
        return ndvar.get_data(self._dimnames)


class _ax_im_array(object):

    def __init__(self, ax, layers, x='time', interpolation=None, vlims={},
                 cmaps={}, contours={}):
        self.ax = ax
        self.data = layers
        self.layers = []
        dimnames = layers[0].get_dimnames((x, None))

        # plot
        overlay = False
        for l in layers:
            p = _plt_im_array(ax, l, overlay, dimnames, interpolation, vlims,
                              cmaps, contours)
            self.layers.append(p)
            overlay = True

    @property
    def title(self):
        return self.ax.get_title()

    def add_contour(self, meas, level, color):
        for l in self.layers:
            l.add_contour(meas, level, color)

    def set_cmap(self, cmap, meas=None):
        """Change the colormap in the array plot

        Parameters
        ----------
        cmap : str | colormap
            New colormap.
        meas : None | str
            Measurement to which to apply the colormap. With None, it is
            applied to all.
        """
        for l in self.layers:
            l.set_cmap(cmap, meas)

    def set_data(self, layers, vlim=False):
        """Update the plotted data

        Parameters
        ----------
        layers : list of NDVar
            Data to plot
        vlim : bool
            Update vlims for the new data.
        """
        self.data = layers
        for l, p in zip(layers, self.layers):
            p.set_data(l, vlim)

    def set_vlim(self, vmax=None, meas=None, vmin=None):
        for l in self.layers:
            l.set_vlim(vmax, meas, vmin)


class Array(ColorMapMixin, XAxisMixin, EelFigure):
    """Plot UTS data to a rectangular grid.

    Parameters
    ----------
    epochs : NDVar
        If data has only 1 dimension, the x-axis defines epochs.
    Xax : None | categorial
        Create a separate plot for each cell in this model.
    xlabel, ylabel : bool | str
        Labels for x- and y-axis; the default is determined from the data.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    sub : str | array
        Specify a subset of the data.
    x : str
        Dimension to plot on the x axis (default 'time').
    vmax : scalar
        Upper limits for the colormap.
    vmin : scalar
        Lower limit for the colormap.
    cmap : str
        Colormap (default depends on the data).
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    interpolation : str
        Array image interpolation (see Matplotlib's
        :meth:`~matplotlib.axes.Axes.imshow`). Matplotlib 1.5.3's SVG output
        can't handle uneven aspect with ``interpolation='none'``, use
        ``interpolation='nearest'`` intead.
    xlim : (scalar, scalar)
        Initial x-axis view limits (default is the full x-axis in the data).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | string
        Figure title.

    Notes
    -----
    Navigation:
     - ``←``: scroll left
     - ``→``: scroll right
     - ``home``: scroll to beginning
     - ``end``: scroll to end
     - ``f``: zoom in (reduce x axis range)
     - ``d``: zoom out (increase x axis range)
    """
    def __init__(self, epochs, Xax=None, xlabel=True, ylabel=True,
                 xticklabels=True, ds=None, sub=None, x='time', vmax=None,
                 vmin=None, cmap=None, axtitle=True, interpolation=None,
                 xlim=None, *args, **kwargs):
        epochs, (xdim, ydim), frame_title = _base.unpack_epochs_arg(
            epochs, (x, None), Xax, ds, "Array", sub
        )
        ColorMapMixin.__init__(self, epochs, cmap, vmax, vmin)

        nax = len(epochs)
        layout = Layout(nax, 2, 4, *args, **kwargs)
        EelFigure.__init__(self, frame_title, layout)
        self._set_axtitle(axtitle, epochs)

        self.plots = []
        for i, ax, layers in zip(range(nax), self._axes, epochs):
            p = _ax_im_array(ax, layers, x, interpolation, self._vlims,
                             self._cmaps, self._contours)
            self.plots.append(p)

        e0 = epochs[0][0]
        self._configure_xaxis_dim(e0.get_dim(xdim), xlabel, xticklabels)
        self._configure_yaxis_dim(e0.get_dim(ydim), ylabel, scalar=False)
        XAxisMixin.__init__(self, epochs, xdim, xlim, im=True)
        self._show()

    def _fill_toolbar(self, tb):
        ColorMapMixin._fill_toolbar(self, tb)


class _plt_utsnd(object):
    """
    UTS-plot for a single epoch

    Parameters
    ----------
    ax : matplotlib axes
        Target axes.
    epoch : NDVar (sensor by time)
        Epoch to plot.
    sensors : None | True | numpy index
        The sensors to plot (None or True -> all sensors).
    others :
        Matplotlib plot() arguments.
    """
    def __init__(self, ax, epoch, xdim, linedim, sensors=None, *args, **kwargs):
        if sensors is not None and sensors is not True:
            epoch = epoch.sub(sensor=sensors)

        self._dims = (linedim, xdim)
        kwargs['label'] = epoch.name
        self.lines = ax.plot(epoch.get_dim(xdim),
                             epoch.get_data((xdim, linedim)), *args, **kwargs)

        for y, kwa in _base.find_uts_hlines(epoch):
            ax.axhline(y, **kwa)

        self.epoch = epoch
        self._sensors = sensors
        # FIXME:  implement actual Case dimension that supports iteration etc.
        labels = list(range(len(epoch))) if linedim == 'case' else epoch.get_dim(linedim)
        self.legend_handles = {name: line for name, line in zip(labels, self.lines)}

    def remove(self):
        while self.lines:
            self.lines.pop().remove()

    def set_visible(self, visible=True):
        for line in self.lines:
            line.set_visible(visible)

    def set_ydata(self, epoch):
        if self._sensors:
            epoch = epoch.sub(sensor=self._sensors)
        for line, y in zip(self.lines, epoch.get_data(self._dims)):
            line.set_ydata(y)


class _ax_butterfly(object):
    """Axis with butterfly plot

    Parameters
    ----------
    vmin, vmax: None | scalar
        Y axis limits.
    """
    def __init__(self, ax, layers, xdim, linedim, sensors=None, color=None,
                 vlims={}):
        self.ax = ax
        self.data = layers
        self.layers = []
        self.legend_handles = {}
        self._xvalues = []  # values on the x axis
        self._meas = None

        vmin, vmax = _base.find_uts_ax_vlim(layers, vlims)

        name = ''
        overlay = False
        for l in layers:
            uts_args = _base.find_uts_args(l, overlay, color)
            if uts_args is None:
                continue
            overlay = True

            # plot
            h = _plt_utsnd(ax, l, xdim, linedim, sensors, **uts_args)
            self.layers.append(h)
            if not name and l.name:
                name = l.name

            self._xvalues = np.union1d(self._xvalues, l.get_dim(xdim))
            self.legend_handles.update(h.legend_handles)

        # axes decoration
        ax.set_xlim(self._xvalues[0], self._xvalues[-1])

    #    ax.yaxis.set_offset_position('right')
        ax.yaxis.offsetText.set_va('top')

        self.set_ylim(vmin, vmax)

    @property
    def title(self):
        return self.ax.get_title()

    def set_ylim(self, vmin, vmax):
        self.ax.set_ylim(vmin, vmax)
        self.vmin, self.vmax = self.ax.get_ylim()


class Butterfly(LegendMixin, TopoMapKey, YLimMixin, XAxisMixin, EelFigure):
    """Butterfly plot for NDVars

    Parameters
    ----------
    epochs : (list of) NDVar
        Data to plot.
    xax : None | categorial
        Create a separate plot for each cell in this model.
    sensors: None or list of sensor IDs
        sensors to plot (``None`` = all)
    axtitle : bool | sequence of str
        Title for the individual axes. The default is to show the names of the
        epochs, but only if multiple axes are plotted.
    xlabel : str | None
        X-axis labels. By default the label is inferred from the data.
    ylabel : str | None
        Y-axis labels. By default the label is inferred from the data.
    xticklabels : bool
        Add tick-labels to the x-axis (default True).
    color : matplotlib color
        default (``None``): use segment color if available, otherwise
        black; ``True``: alternate colors (mpl default)
    ds : None | Dataset
        If a Dataset is provided, ``epochs`` and ``Xax`` can be specified
        as strings.
    sub : str | array
        Specify a subset of the data.
    x : str
        Dimension to plot on the x-axis (default 'time').
    vmax : scalar
        Top of the y axis (default depends on data).
    vmin : scalar
        Bottom of the y axis (default depends on data).
    xlim : (scalar, scalar)
        Initial x-axis view limits (default is the full x-axis in the data).
    tight : bool
        Use matplotlib's tight_layout to expand all axes to fill the figure
        (default True)
    title : None | string
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

    Keys available for sensor data:
     - ``t``: open a ``Topomap`` plot for the time point under the mouse pointer.
     - ``T``: open a larger ``Topomap`` plot with visible sensor names for the
       time point under the mouse pointer.
    """
    _cmaps = None  # for TopoMapKey mixin
    _contours = None

    def __init__(self, epochs, xax=None, sensors=None, axtitle=True,
                 xlabel=True, ylabel=True, xticklabels=True, color=None,
                 ds=None, sub=None, x='time', vmax=None, vmin=None, xlim=None,
                 *args, **kwargs):
        epochs, (xdim, linedim), frame_title = _base.unpack_epochs_arg(
            epochs, (x, None), xax, ds, "Butterfly", sub
        )
        layout = Layout(len(epochs), 2, 4, *args, **kwargs)
        EelFigure.__init__(self, frame_title, layout)
        self._set_axtitle(axtitle, epochs)
        e0 = epochs[0][0]
        self._configure_xaxis_dim(e0.get_dim(xdim), xlabel, xticklabels)
        self._configure_yaxis(e0, ylabel)

        self.plots = []
        self._vlims = _base.find_fig_vlims(epochs, vmax, vmin)
        legend_handles = {}
        for ax, layers in zip(self._axes, epochs):
            h = _ax_butterfly(ax, layers, xdim, linedim, sensors, color,
                              self._vlims)
            self.plots.append(h)
            legend_handles.update(h.legend_handles)

        XAxisMixin.__init__(self, epochs, xdim, xlim)
        YLimMixin.__init__(self, self.plots)
        if linedim == 'sensor':
            TopoMapKey.__init__(self, self._topo_data)
        LegendMixin.__init__(self, 'invisible', legend_handles)
        self._show()

    def _fill_toolbar(self, tb):
        LegendMixin._fill_toolbar(self, tb)

    def _topo_data(self, event):
        if not event.inaxes:
            return
        p = self.plots[self._axes.index(event.inaxes)]
        t = event.xdata
        data = [l.sub(time=t) for l in p.data]
        return data, p.title + ' %i ms' % round(t), 'default'



class _ax_bfly_epoch:
    def __init__(self, ax, epoch, mark=None, state=True, label=None, color='k',
                 lw=0.2, mcolor='r', mlw=0.8, antialiased=True, vlims={}):
        """Specific plot for showing a single sensor by time epoch

        Parameters
        ----------
        ax : mpl Axes
            Plot target axes.
        epoch : NDVar
            Sensor by time epoch.
        mark : dict {int: mpl color}
            Channel: color dict of channels with custom color.
        color : mpl color
            Color for unmarked traces.
        lw : scalar
            Sensor trace plot Line width (default 0.5).
        mlw : scalar
            Marked sensor plot line width (default 1).
        """
        self.lines = _plt_utsnd(ax, epoch, 'time', 'sensor', color=color, lw=lw,
                                antialiased=antialiased)

        self.ax = ax
        self.epoch = epoch
        self._state_h = []
        self._visible = True
        self._ylim = _base.find_uts_ax_vlim([epoch], vlims)
        self._update_ax_lim()
        self._styles = {None: {'color': color, 'lw': lw, 'ls': '-',
                               'zorder': 2},
                        'mark': {'color': mcolor, 'lw': mlw, 'ls': '-',
                                 'zorder': 10},
                        INTERPOLATE_CHANNELS: {'color': 'b', 'lw': 1.2,
                                               'ls': ':', 'zorder': 6}}
        self._marked = {'mark': set(), INTERPOLATE_CHANNELS: set()}
        if mark:
            self.set_marked('mark', mark)

        if label is None:
            label = ''
        self._label = ax.text(0, 1.01, label, va='bottom', ha='left', transform=ax.transAxes)

        # create initial plots
        self.set_state(state)

    def set_data(self, epoch, label=None):
        self.epoch = epoch
        self.lines.set_ydata(epoch)
        if label is not None:
            self._label.set_text(label)

    def _update_ax_lim(self):
        self.ax.set_xlim(self.epoch.time[0], self.epoch.time[-1])
        ylim = self._ylim
        if ylim:
            if np.isscalar(ylim):
                self.ax.set_ylim(-ylim, ylim)
            else:
                y_min, y_max = ylim
                self.ax.set_ylim(y_min, y_max)

    def set_marked(self, kind, sensors):
        """Set the channels which should be marked for a specific style

        Parameters
        ----------
        kind : str
            The style.
        sensors : collection of int
            Channel index for the channels to mark as ``kind``.
        """
        old = self._marked[kind]
        new = self._marked[kind] = set(sensors)
        if not old and not new:
            return
        # mark new channels
        for i in new.difference(old):
            self.lines.lines[i].update(self._styles[kind])
        # find channels to unmark
        old.difference_update(new)
        if not old:
            return
        # possible alternate style
        if kind == 'mark':
            other_kind = INTERPOLATE_CHANNELS
        else:
            other_kind = 'mark'
        # reset old channels
        for i in old:
            if i in self._marked[other_kind]:
                self.lines.lines[i].update(self._styles[other_kind])
            else:
                self.lines.lines[i].update(self._styles[None])

    def set_state(self, state):
        "Set the state (True=accept / False=reject)"
        if state:
            while self._state_h:
                h = self._state_h.pop()
                h.remove()
        else:
            if not self._state_h:
                h1 = self.ax.plot([0, 1], [0, 1], color='r', linewidth=1,
                                  transform=self.ax.transAxes)
                h2 = self.ax.plot([0, 1], [1, 0], color='r', linewidth=1,
                                  transform=self.ax.transAxes)
                self._state_h.extend(h1 + h2)

    def set_visible(self, visible=True):
        if self._visible != visible:
            self.lines.set_visible(visible)
            self._label.set_visible(visible)
            for line in self._state_h:
                line.set_visible(visible)
            self._visible = visible

    def set_ylim(self, ylim):
        self._ylim = ylim
        self._update_ax_lim()
