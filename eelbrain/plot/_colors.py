# -*- coding: utf-8 -*-
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Color tools for plotting."""


from itertools import product
import operator
from warnings import warn

import numpy as np
import matplotlib as mpl

from .. import _colorspaces as cs
from .._data_obj import Factor, Interaction, cellname
from ._base import EelFigure, Layout
from functools import reduce


POINT_SIZE = 0.0138889  # 1 point in inches
LEGEND_SIZE = 1.2  # times font.size


def find_cell_colors(x, colors):
    """Process the colors arg from plotting functions

    Parameters
    ----------
    x : categorial
        Model for which colors are needed.
    colors : str | list | dict
        Colors for the plots if multiple categories of data are plotted.
        **str**: A colormap name; cells are mapped onto the colormap in
        regular intervals.
        **list**: A list of colors in the same sequence as cells.
        **dict**: A dictionary mapping each cell to a color.
        Colors are specified as `matplotlib compatible color arguments
        <http://matplotlib.org/api/colors_api.html>`_.
    """
    if isinstance(colors, (list, tuple)):
        cells = x.cells
        if len(colors) < len(cells):
            err = ("The `colors` argument %s does not supply enough "
                   "colors (%i) for %i "
                   "cells." % (str(colors), len(colors), len(cells)))
            raise ValueError(err)
        return dict(list(zip(cells, colors)))
    elif isinstance(colors, dict):
        for cell in x.cells:
            if cell not in colors:
                raise KeyError("%s not in colors" % repr(cell))
        return colors
    elif colors is None:
        return colors_for_categorial(x, cmap=colors)
    else:
        raise TypeError("Invalid type: colors=%s" % repr(colors))


def colors_for_categorial(x, hue_start=0.2, cmap=None):
    """Automatically select colors for a categorial model

    Parameters
    ----------
    x : categorial
        Model defining the cells for which to define colors.
    hue_start : 0 <= scalar < 1
        First hue value (only for two-way or higher level models).
    cmap : str (optional)
        Name of a matplotlib colormap to use instead of default hue-based
        colors (only used for one-way models).

    Returns
    -------
    colors : dict {cell -> color}
        Dictionary providing colors for the cells in x.
    """
    if isinstance(x, Factor):
        return colors_for_oneway(x.cells, hue_start, cmap=cmap)
    elif isinstance(x, Interaction):
        return colors_for_nway([f.cells for f in x.base], hue_start)
    else:
        msg = ("x needs to be Factor or Interaction, got %s" % repr(x))
        raise TypeError(msg)


def colors_for_oneway(cells, hue_start=0.2, light_range=0.5, cmap=None):
    """Define colors for a single factor design

    Parameters
    ----------
    cells : sequence of str
        Cells for which to assign colors.
    hue_start : 0 <= scalar < 1 | sequence of scalar
        First hue value (default 0.2) or list of hue values.
    light_range : scalar | tuple of 2 scalar
        Scalar that specifies the amount of lightness variation (default 0.5).
        If positive, the first color is lightest; if negative, the first color
        is darkest. A tuple can be used to specify exact end-points (e.g.,
        ``(1.0, 0.4)``). ``0.2`` is equivalent to ``(0.4, 0.6)``.
    cmap : str (optional)
        Use a matplotlib colormap instead of the default color generation
        algorithm. Name of a matplotlib colormap to use (e.g., 'jet'). If
        specified, ``hue_start`` and ``light_range`` are ignored.

    Returns
    -------
    dict : {str: tuple}
        Mapping from cells to colors.
    """
    if isinstance(hue_start, str):
        warn("The function signature of colors_for_oneway has changed. "
             "Specifying a colormap as second parameter will stop working after "
             "Eelbrain 0.25. Use cmap=%r as keyword argument instead." %
             hue_start, DeprecationWarning)
        cmap = hue_start
    n = len(cells)
    if cmap is None:
        return dict(zip(cells, cs.oneway_colors(n, hue_start, light_range)))
    else:
        cm = mpl.cm.get_cmap(cmap)
        return {cell: cm(i / n) for i, cell in enumerate(cells)}


def colors_for_twoway(x1_cells, x2_cells, hue_start=0.2, hue_shift=0.,
                      hues=None):
    """Define cell colors for a two-way design

    Parameters
    ----------
    x1_cells : tuple of str
        Cells of the major factor.
    x2_cells : tuple of str
        Cells of the minor factor.
    hue_start : 0 <= scalar < 1
        First hue value.
    hue_shift : 0 <= scalar < 1
        Use that part of the hue continuum between categories to shift hue
        within categories.
    hues : list of scalar
        List of hue values corresponding to the levels of the first factor
        (overrides regular hue distribution).

    Returns
    -------
    dict : {tuple: tuple}
        Mapping from cells to colors.
    """
    n1 = len(x1_cells)
    n2 = len(x2_cells)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 cells on each factor")

    clist = cs.twoway_colors(n1, n2, hue_start, hue_shift, hues)
    return dict(zip(product(x1_cells, x2_cells), clist))


def colors_for_nway(cell_lists, hue_start=0.2):
    """Define cell colors for a two-way design

    Parameters
    ----------
    cell_lists : sequence of of tuple of str
        List of the cells for each factor. E.g. for ``A % B``:
        ``[('a1', 'a2'), ('b1', 'b2', 'b3')]``.
    hue_start : 0 <= scalar < 1
        First hue value.

    Returns
    -------
    dict : {tuple: tuple}
        Mapping from cells to colors.
    """
    if len(cell_lists) == 1:
        return colors_for_oneway(cell_lists[0])
    elif len(cell_lists) == 2:
        return colors_for_twoway(cell_lists[0], cell_lists[1], hue_start)
    elif len(cell_lists) > 2:
        ns = list(map(len, cell_lists))
        n_outer = reduce(operator.mul, ns[:-1])
        n_inner = ns[-1]

        # outer circle
        hues = np.linspace(hue_start, 1 + hue_start, ns[0], False)
        # subdivide for each  level
        distance = 1. / ns[0]
        for n_current in ns[1:-1]:
            new = []
            d = distance / 3
            for hue in hues:
                new.extend(np.linspace(hue - d, hue + d, n_current))
            hues = new
            distance = 2 * d / (n_current - 1)
        hues = np.asarray(hues)
        hues %= 1
        colors = cs.twoway_colors(n_outer, n_inner, hues=hues)
        return dict(zip(product(*cell_lists), colors))
    else:
        return {}


class ColorGrid(EelFigure):
    """Plot colors for a two-way design in a grid

    Parameters
    ----------
    row_cells : tuple of str
        Cells contained in the rows.
    column_cells : tuple of str
        Cells contained in the columns.
    colors : dict
        Colors for cells.
    size : scalar
        Size (width and height) of the color squares (the default is to
        scale them to fit the figure).
    column_label_position : 'top' | 'bottom'
        Where to place the column labels (default is 'top').
    row_first : bool
        Whether the row cell precedes the column cell in color keys. By
        default this is inferred from the existing keys.
    labels : dict (optional)
        Condition labels that are used instead of the keys in ``row_cells`` and
        ``column_cells``.
    """
    def __init__(self, row_cells, column_cells, colors, size=None,
                 column_label_position='top', row_first=None, labels=None,
                 *args, **kwargs):
        if row_first is None:
            row_cell_0 = row_cells[0]
            col_cell_0 = column_cells[0]
            if (row_cell_0, col_cell_0) in colors:
                row_first = True
            elif (col_cell_0, row_cell_0) in colors:
                row_first = False
            else:
                msg = ("Neither %s nor %s exist as a key in colors" %
                       ((row_cell_0, col_cell_0), (col_cell_0, row_cell_0)))
                raise KeyError(msg)

        if size is None:
            size = mpl.rcParams['font.size'] * LEGEND_SIZE * POINT_SIZE
        layout = Layout(None, 1, 3, False, *args, **kwargs)
        EelFigure.__init__(self, "ColorGrid", layout)
        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()
        self._ax = ax

        # reverse rows so we can plot upwards
        row_cells = tuple(reversed(row_cells))
        n_rows = len(row_cells)
        n_cols = len(column_cells)

        # color patches
        for col in range(n_cols):
            for row in range(n_rows):
                if row_first:
                    cell = (row_cells[row], column_cells[col])
                else:
                    cell = (column_cells[col], row_cells[row])
                patch = mpl.patches.Rectangle((col, row), 1, 1, fc=colors[cell],
                                              ec='none')
                ax.add_patch(patch)

        # prepare labels
        if labels:
            column_labels = [labels.get(c, c) for c in column_cells]
            row_labels = [labels.get(c, c) for c in row_cells]
        else:
            column_labels = column_cells
            row_labels = row_cells

        # column labels
        tilt_labels = any(len(label) > 1 for label in column_labels)
        self._labels = []
        if column_label_position == 'top':
            y = n_rows + 0.1
            va = 'bottom'
            rotation = 40 if tilt_labels else 0
            ymin = 0
            ymax = self._layout.h / size
        elif column_label_position == 'bottom':
            y = -0.1
            va = 'top'
            rotation = -40 if tilt_labels else 0
            ymax = n_rows
            ymin = n_rows - self._layout.h / size
        else:
            msg = "column_label_position=%s" % repr(column_label_position)
            raise ValueError(msg)

        for col, label in enumerate(column_labels):
            h = ax.text(col + 0.5, y, label, va=va,
                        ha='left' if tilt_labels else 'center',
                        rotation=rotation)
            self._labels.append(h)

        # row labels
        x = n_cols + 0.1
        for row, label in enumerate(row_labels):
            h = ax.text(x, row + 0.5, label, va='center', ha='left')
            self._labels.append(h)

        if size is not None:
            self._ax.set_xlim(0, self._layout.w / size)
            self._ax.set_ylim(ymin, ymax)

        self._show()

    def _tight(self):
        # arbitrary default with equal aspect
        self._ax.set_ylim(0, 1)
        self._ax.set_xlim(0, 1 * self._layout.w / self._layout.h)

        # draw to compute text coordinates
        self.draw()

        # find label bounding box
        xmax = 0
        ymax = 0
        for h in self._labels:
            bbox = h.get_window_extent()
            if bbox.xmax > xmax:
                xmax = bbox.xmax
                xpos = h.get_position()[0]
            if bbox.ymax > ymax:
                ymax = bbox.ymax
                ypos = h.get_position()[1]
        xmax += 2
        ymax += 2

        # transform from display coordinates -> data coordinates
        trans = self._ax.transData.inverted()
        xmax, ymax = trans.transform((xmax, ymax))

        # calculate required movement
        _, ax_xmax = self._ax.get_xlim()
        _, ax_ymax = self._ax.get_ylim()
        xtrans = ax_xmax - xmax
        ytrans = ax_ymax - ymax

        # calculate the scale factor:
        # new_coord = x * coord
        # new_coord = coord + trans
        # x = (coord + trans) / coord
        scale = (xpos + xtrans) / xpos
        scale_y = (ypos + ytrans) / ypos
        if scale_y <= scale:
            scale = scale_y

        self._ax.set_xlim(0, ax_xmax / scale)
        self._ax.set_ylim(0, ax_ymax / scale)


class ColorList(EelFigure):
    """Plot colors with labels

    Parameters
    ----------
    colors : dict
        Colors for cells.
    cells : tuple
        Cells for which to plot colors (default is ``colors.keys()``).
    labels : dict (optional)
        Condition labels that are used instead of the keys in ``colors``. This
        is useful if ``colors`` uses abbreviated labels, but the color legend
        should contain more intelligible labels.
    h : 'auto' | scalar
        Height of the figure in inches. If 'auto' (default), the height is
        automatically increased to fit all labels.
    """
    def __init__(self, colors, cells=None, labels=None, h='auto', *args,
                 **kwargs):
        if cells is None:
            cells = list(colors.keys())

        if h == 'auto':
            h = len(cells) * mpl.rcParams['font.size'] * POINT_SIZE * LEGEND_SIZE

        if labels is None:
            labels = {cell: cellname(cell) for cell in cells}
        elif not isinstance(labels, dict):
            raise TypeError("labels=%s" % repr(labels))

        layout = Layout(None, 1.5, 2, False, False, h, *args, **kwargs)
        EelFigure.__init__(self, "Colors", layout)

        ax = self.figure.add_axes((0, 0, 1, 1), frameon=False)
        ax.set_axis_off()

        n = len(cells)
        self._labels = []
        for i, cell in enumerate(cells):
            bottom = n - i - 1
            y = bottom + 0.5
            patch = mpl.patches.Rectangle((0, bottom), 1, 1, fc=colors[cell],
                                          ec='none', zorder=1)
            ax.add_patch(patch)
            h = ax.text(1.1, y, labels[cell], va='center', ha='left', zorder=2)
            self._labels.append(h)

        ax.set_ylim(0, n)
        ax.set_xlim(0, n * self._layout.w / self._layout.h)

        # resize figure to match legend
        if not self._layout.w_fixed:
            self.draw()
            # all calculation in pixels
            fig_bb = self.figure.get_window_extent()
            x_max = max(h.get_window_extent().x1 for h in self._labels)
            w0, h0 = self._frame.GetSize()
            new_w = w0 + (x_max - fig_bb.x1) + 5
            self._frame.SetSize((new_w, h0))
            # adjust x-limits
            # self.draw()
            ax_bb = ax.get_window_extent()
            ax.set_xlim(0, n * ax_bb.width / ax_bb.height)

        self._ax = ax
        self._show()


class ColorBar(EelFigure):
    """A color-bar for a matplotlib color-map

    Parameters
    ----------
    cmap : str | Colormap | array
        Name of the color-map, or a matplotlib Colormap, or LUT.
    vmin : scalar
        Lower end of the scale mapped onto cmap.
    vmax : scalar
        Upper end of the scale mapped onto cmap.
    label : None | str
        Label for the x-axis (default is the unit, or if no unit is provided
        the name of the colormap).
    label_position : 'left' | 'right' | 'top' | 'bottom'
        Position of the axis label. Valid values depend on orientation.
    label_rotation : scalar
        Angle of the label in degrees (For horizontal colorbars, the default is
        0; for vertical colorbars, the default is 0 for labels of 3 characters
        and shorter, and 90 for longer labels).
    clipmin : scalar
        Clip the color-bar below this value.
    clipmax : scalar
        Clip the color-bar above this value.
    orientation : 'horizontal' | 'vertical'
        Orientation of the bar (default is horizontal).
    unit : str
        Unit for the axis to determine tick labels (for example, ``u'µV'`` to
        label 0.000001 as '1').
    contours : iterator of scalar (optional)
        Plot contour lines at these values.
    width : scalar
        Width of the color-bar in inches.
    ticks : {float: str} dict | sequence of float
        Customize tick-labels on the colormap; either a dictionary with
        tick-locations and labels, or a sequence of tick locations.
    """
    def __init__(self, cmap, vmin, vmax, label=True, label_position=None,
                 label_rotation=None,
                 clipmin=None, clipmax=None, orientation='horizontal',
                 unit=None, contours=(), width=None, ticks=None, h=None, w=None,
                 *args, **kwargs):
        if isinstance(cmap, np.ndarray):
            if cmap.max() > 1:
                cmap = cmap / 255.
            cm = mpl.colors.ListedColormap(cmap, 'LUT')
            lut = cmap
        else:
            cm = mpl.cm.get_cmap(cmap)
            lut = cm(np.arange(cm.N))

        # prepare im and sizing
        if orientation == 'horizontal':
            if h is None and w is None:
                h = 1
            ax_aspect = 4
            im = lut.reshape((1, cm.N, 4))
        elif orientation == 'vertical':
            if h is None and w is None:
                h = 4
            ax_aspect = 0.3
            im = lut.reshape((cm.N, 1, 4))
        else:
            raise ValueError("orientation=%s" % repr(orientation))

        if label is True:
            if unit:
                label = unit
            else:
                label = cm.name

        layout = Layout(1, ax_aspect, 2, True, False, h, w, *args, **kwargs)
        EelFigure.__init__(self, "ColorBar: %s" % cm.name, layout)
        ax = self._axes[0]

        if orientation == 'horizontal':
            ax.imshow(im, extent=(vmin, vmax, 0, 1), aspect='auto')
            ax.set_xlim(clipmin, clipmax)
            ax.yaxis.set_ticks(())
            self._contours = [ax.axvline(c, c='k') for c in contours]
            if unit:
                self._configure_xaxis(unit, label)
            elif label:
                ax.set_xlabel(label)

            if label_position is not None:
                ax.xaxis.set_label_position(label_position)

            if label_rotation is not None:
                ax.xaxis.label.set_rotation(label_rotation)

            axis = ax.xaxis
        elif orientation == 'vertical':
            ax.imshow(im, extent=(0, 1, vmin, vmax), aspect='auto', origin='lower')
            ax.set_ylim(clipmin, clipmax)
            ax.xaxis.set_ticks(())
            self._contours = [ax.axhline(c, c='k') for c in contours]
            if unit:
                self._configure_yaxis(unit, label)
            elif label:
                ax.set_ylabel(label)

            if label_position is not None:
                ax.yaxis.set_label_position(label_position)

            if label_rotation is not None:
                ax.yaxis.label.set_rotation(label_rotation)
                if (label_rotation + 10) % 360 < 20:
                    ax.yaxis.label.set_va('center')
            elif label and len(label) <= 3:
                ax.yaxis.label.set_rotation(0)
                ax.yaxis.label.set_va('center')

            axis = ax.yaxis
        else:
            raise ValueError("orientation=%s" % repr(orientation))

        # value ticks
        if isinstance(ticks, dict):
            tick_locs = sorted(ticks)
            axis.set_ticks(tick_locs)
            axis.set_ticklabels([ticks[t] for t in tick_locs])
        elif ticks is not None:
            axis.set_ticks(ticks)

        self._orientation = orientation
        self._width = width
        self._show()

    def _tight(self):
        # make sure ticklabels have space
        ax = self._axes[0]
        if self._orientation == 'vertical' and not self._layout.w_fixed:
            self.draw()
            labels = ax.get_yticklabels()
            # wmax = max(l.get_window_extent().width for l in labels)
            x0 = min(l.get_window_extent().x0 for l in labels)
            if x0 < 0:
                w, h = self.figure.get_size_inches()
                w -= (x0 / self._layout.dpi)
                self.figure.set_size_inches(w, h, forward=True)
        super(ColorBar, self)._tight()
        # Override to keep bar thickness
        if self._width:
            x = (self._width, self._width)
            x = self.figure.dpi_scale_trans.transform(x)
            x = self.figure.transFigure.inverted().transform(x)
            pos = ax.get_position()
            if self._orientation == 'vertical':
                ax.set_position((pos.xmin, pos.ymin, x[0], pos.height))
            else:
                ax.set_position((pos.xmin, pos.ymin, pos.width, x[1]))
            self.draw()
