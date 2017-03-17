# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from __future__ import division

from distutils.version import LooseVersion
import os
import sys
from tempfile import mkdtemp
from time import time, sleep

from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, ListedColormap, colorConverter
import numpy as np
import wx

from .._data_obj import NDVar, SourceSpace
from .._wxgui import run as run_gui
from ..fmtxt import Image
from ..mne_fixes import reset_logger
from ._base import (backend, do_autorun, find_axis_params_data, find_fig_cmaps,
                    find_fig_vlims)
from ._colors import ColorBar
from ._wx_brain import BrainFrame

# Traits-GUI related imports after BrainFrame
from mayavi import mlab
# surfer imports, lower screen logging level
first_import = 'surfer' not in sys.modules
import surfer
if first_import:
    reset_logger(surfer.utils.logger)


def assert_can_save_movies():
    if LooseVersion(surfer.__version__) < LooseVersion('0.6'):
        raise ImportError("Saving movies requires PySurfer 0.6")


class Brain(surfer.Brain):
    # Subclass that adds Eelbrain functionality to surfer.Brain
    def __init__(self, subject, hemi, views, surf, title, width, height,
                 show, run, **kwargs):
        self.__data = []
        self.__annot = None

        if title is None:
            title = "Brain: %s" % subject
        elif not isinstance(title, basestring):
            raise TypeError("title needs to be a string, got %r" % (title,))

        assert isinstance(views, list)
        n_rows = len(views)
        n_columns = 2 if hemi == 'split' else 1
        self._frame = frame = BrainFrame(None, title, width, height, n_rows, n_columns)

        surfer.Brain.__init__(self, subject, hemi, surf, views=views, figure=frame.figure, **kwargs)

        if backend['show'] and show:
            self._frame.Show()
            if backend['eelbrain'] and do_autorun(run):
                run_gui()

    def add_mask(self, source, color=(1, 1, 1), smoothing_steps=None,
                 alpha=None, subjects_dir=None):
        """Add a mask shading areas that are not included in an NDVar

        Parameters
        ----------
        source : SourceSpace
            SourceSpace.
        color : matplotlib color
            Mask color, can include alpha (defauls is black with alpha=0.5:
            ``(0, 0, 0, 0.5)``).
        smoothing_steps : scalar (optional)
            Smooth transition at the mask's border.
        alpha : scalar
            Alpha for the mask (supercedes alpha in ``color``).
        subjects_dir : str
            Use this directory as the subjects directory.
        """
        if isinstance(source, NDVar):
            source = source.get_dim('source')
        if not isinstance(source, SourceSpace):
            raise TypeError("source needs to be a SourceSpace or NDVar, got "
                            "%s" % (source,))

        color = colorConverter.to_rgba(color, alpha)
        if smoothing_steps is not None:
            # generate LUT
            lut = np.repeat(np.reshape(color, (1, 4)), 256, 0)
            lut[:, 3] = np.linspace(color[-1], 0, 256)
            np.clip(lut, 0, 1, lut)
            lut *= 255
            lut = np.round(lut).astype(np.uint8)
            # generate mask Label
            mask_ndvar = source._mask_ndvar(subjects_dir)
            self.add_ndvar(mask_ndvar, lut, 0., 1., smoothing_steps, False,
                           None, False)
        else:
            lh, rh = source._mask_label(subjects_dir)
            if self._hemi == 'lh':
                rh = None
            elif self._hemi == 'rh':
                lh = None

            if source.lh_n and lh:
                self.add_label(lh, color[:3], color[3])
                self.labels_dict['mask-lh'][0].actor.property.lighting = False
            if source.rh_n and rh:
                self.add_label(rh, color[:3], color[3])
                self.labels_dict['mask-rh'][0].actor.property.lighting = False

    def add_ndvar(self, ndvar, cmap=None, vmin=None, vmax=None,
                  smoothing_steps=None, colorbar=False, time_label='ms',
                  lighting=True, contours=None):
        """Add data layer form an NDVar

        Parameters
        ----------
        ndvar : NDVar  (source[, time])
            NDVar with SourceSpace dimension and optional time dimension.
        cmap : str | array
            Colormap (name of a matplotlib colormap) or LUT array.
        vmin, vmax : scalar
            Endpoints for the colormap. Need to be set explicitly if ``cmap`` is
            a LUT array.
        smoothing_steps : None | int
            Number of smoothing steps if data is spatially undersampled
            (PySurfer ``Brain.add_data()`` argument).
        colorbar : bool
            Add a colorbar to the figure (use ``.plot_colorbar()`` to plot a
            colorbar separately).
        time_label : str
            Label to show time point. Use ``'ms'`` or ``'s'`` to display time in
            milliseconds or in seconds, or supply a custom format string to format
            time values (in seconds; default is ``'ms'``).
        lighting : bool
            The data overlay is affected by light sources (set to False to make
            the data overlay luminescent).
        contours : bool | sequence of scalar
            Draw contour lines instead of a solid overlay. Set to a list of
            contour levels or ``True`` for automatic contours.
        """
        # find standard args
        meas = ndvar.info.get('meas')
        if cmap is None or isinstance(cmap, basestring):
            epochs = ((ndvar,),)
            cmaps = find_fig_cmaps(epochs, cmap, alpha=True)
            vlims = find_fig_vlims(epochs, vmax, vmin, cmaps)
            vmin, vmax = vlims[meas]
        # colormap
        if contours is not None:
            if cmap is None:
                cmap = ('w', 'w')
            elif isinstance(cmap, basestring) and len(cmap) > 1:
                cmap = cmaps[meas]
            else:
                contour_color = colorConverter.to_rgb(cmap)
                cmap = (contour_color, contour_color)
        elif cmap is None or isinstance(cmap, basestring):
            cmap = cmaps[meas]

        # convert ColorMap to LUT (PySurfer can't handle ColorMap instances)
        if isinstance(cmap, basestring):
            cmap = get_cmap(cmap)
        if isinstance(cmap, Colormap):
            cmap = np.round(cmap(np.arange(cmap.N)) * 255).astype(np.uint8)

        # general PySurfer data args
        alpha = 1
        if smoothing_steps is None and ndvar.source.kind == 'ico':
            smoothing_steps = ndvar.source.grade + 1

        if ndvar.has_dim('time'):
            times = ndvar.time.times
            data_dims = ('source', 'time')
            if time_label == 'ms':
                import surfer
                if LooseVersion(surfer.__version__) > LooseVersion('0.5'):
                    time_label = lambda x: '%s ms' % int(round(x * 1000))
                else:
                    times = times * 1000
                    time_label = '%i ms'
            elif time_label == 's':
                time_label = '%.3f s'
        else:
            times = None
            data_dims = ('source',)

        # determine which hemi we're adding data to
        if self._hemi in ('lh', 'rh'):
            data_hemi = self._hemi
        elif not ndvar.source.lh_n:
            data_hemi = 'rh'
        elif not ndvar.source.rh_n:
            data_hemi = 'lh'
        else:
            data_hemi = 'both'
        # remember where to find data_dict
        dict_hemi = 'rh' if data_hemi == 'rh' else 'lh'
        data_index = len(self._data_dicts[dict_hemi])

        # add data
        new_surfaces = []
        if data_hemi != 'rh':
            if self._hemi == 'lh':
                colorbar_ = colorbar
                colorbar = False
                time_label_ = time_label
                time_label = None
            else:
                colorbar_ = False
                time_label_ = None

            src_hemi = ndvar.sub(source='lh')
            data = src_hemi.get_data(data_dims)
            vertices = ndvar.source.lh_vertno
            self.add_data(data, vmin, vmax, None, cmap, alpha, vertices,
                          smoothing_steps, times, time_label_, colorbar_, 'lh')
            new_surfaces.extend(self.data_dict['lh']['surfaces'])

        if data_hemi != 'lh':
            src_hemi = ndvar.sub(source='rh')
            data = src_hemi.get_data(data_dims)
            vertices = ndvar.source.rh_vertno
            self.add_data(data, vmin, vmax, None, cmap, alpha, vertices,
                          smoothing_steps, times, time_label, colorbar, 'rh')
            new_surfaces.extend(self.data_dict['rh']['surfaces'])

        # update surfaces
        for surface in new_surfaces:
            if contours is not None:
                surface.enable_contours = True
                # http://code.enthought.com/projects/files/ets_api/enthought.mayavi.components.contour.Contour.html
                surface.contour.auto_update_range = False
                # surface.contour.maximum_contour = ndvar.max()
                # surface.contour.minimum_contour = ndvar.min()
                if contours is not True:
                    surface.contour.contours = contours
                    surface.contour.auto_contours = False

            if not lighting:
                surface.actor.property.lighting = False

        self.__data.append({
            'hemi': data_hemi,
            'data': ndvar,
            'dict_hemi': dict_hemi,
            'dict_index': data_index,
        })

    def close(self):
        "Close the figure window"
        self._frame.Close()
        surfer.Brain.close(self)

    def copy_screenshot(self):
        "Copy the currently shown image to the clipboard"
        tempdir = mkdtemp()
        tempfile = os.path.join(tempdir, "brain.png")
        self.save_image(tempfile, 'rgba', True)

        bitmap = wx.Bitmap(tempfile, wx.BITMAP_TYPE_PNG)
        bitmap_obj = wx.BitmapDataObject(bitmap)

        if not wx.TheClipboard.IsOpened():
            open_success = wx.TheClipboard.Open()
            if open_success:
                wx.TheClipboard.SetData(bitmap_obj)
                wx.TheClipboard.Close()
                wx.TheClipboard.Flush()

    def _get_cmap_params(self, layer=0, label=True):
        """Return parameters required to plot a colorbar"""
        data = self.__data[layer]
        data_dict = self._data_dicts[data['dict_hemi']][data['dict_index']]
        colormap = ListedColormap(data_dict['orig_ctable'] / 255., label)
        return colormap, data_dict['fmin'], data_dict['fmax']

    def image(self, name=None, format='png', alt=None):
        """Create an FMText Image from a screenshot

        Parameters
        ----------
        name : str
            Name for the file (without extension; default is ``data.name`` or
            'brain').
        format : str
            File format (default 'png').
        alt : None | str
            Alternate text, placeholder in case the image can not be found
            (HTML `alt` tag).
        """
        if name is None:
            for data in self.__data:
                name = data['data'].name
                if name:
                    break
            else:
                name = 'brain'
        im = self.screenshot('rgba', True)
        return Image.from_array(im, name, format, alt)

    def plot_colorbar(self, label=True, label_position=None, label_rotation=None,
                      clipmin=None, clipmax=None, orientation='horizontal',
                      width=None, ticks=None, layer=None, *args, **kwargs):
        """Plot a colorbar corresponding to the displayed data

        Parameters
        ----------
        label : str | bool
            Label for the x-axis (default is based on the data).
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
        width : scalar
            Width of the color-bar in inches.
        ticks : {float: str} dict | sequence of float
            Customize tick-labels on the colormap; either a dictionary with
            tick-locations and labels, or a sequence of tick locations.
        layer : int
            If the brain contains multiple data layers, plot a colorbar for
            only one (int in the order ndvars were added; default is to plot
            colorbars for all layers).

        Returns
        -------
        colorbar : :class:`~eelbrain.plot.ColorBar` | list
            ColorBar plot object (list of colorbars if more than one data layer
            are present).
        """
        if not self.__data:
            raise RuntimeError("Brain has no data to plot colorbar for")

        if layer is None:
            layers = xrange(len(self.__data))
        else:
            layers = (layer,)

        out = []
        for layer in layers:
            data = self.__data[layer]
            ndvar = data['data']
            unit = ndvar.info.get('unit', None)
            if ticks is None:
                ticks = ndvar.info.get('cmap ticks')
            _, label = find_axis_params_data(ndvar, label)
            colormap, vmin, vmax = self._get_cmap_params(layer, label)
            out.append(ColorBar(
                colormap, vmin, vmax, label, label_position, label_rotation,
                clipmin, clipmax, orientation, unit, (), width, ticks, *args,
                **kwargs))

            # reset parames
            label = True

        if len(out) == 1:
            return out[0]
        else:
            return out

    def _set_annot(self, annot, borders, alpha):
        "Store annot name to enable plot_legend()"
        self.add_annotation(annot, borders, alpha)
        self.__annot = annot

    def play_movie(self, tstart=None, tstop=None, fps=10):
        """Play an animation by setting time"""
        max_wait = 1 / fps
        start = 0 if tstart is None else self.index_for_time(tstart)
        stop = self.n_times if tstop is None else self.index_for_time(tstop)
        for i in xrange(start, stop):
            t0 = time()
            self.set_data_time_index(i)
            dt = time() - t0
            if dt < max_wait:
                sleep(max_wait - dt)

    def plot_legend(self, *args, **kwargs):
        """Plot legend for parcellation

        Parameters
        ----------
        labels : dict (optional)
            Alternative (text) label for (brain) labels.
        h : 'auto' | scalar
            Height of the figure in inches. If 'auto' (default), the height is
            automatically increased to fit all labels.

        Returns
        -------
        legend : :class:`~eelbrain.plot.ColorList`
            Figure with legend for the parcellation.

        See Also
        --------
        plot.brain.annot_legend : plot a legend without plotting the brain
        """
        from ._brain import annot_legend
        if self.__annot is None:
            raise RuntimeError("Can only plot legend for brain displaying "
                               "parcellation")

        lh = os.path.join(self.subjects_dir, self.subject_id, 'label',
                          'lh.%s.annot' % self.__annot)
        rh = os.path.join(self.subjects_dir, self.subject_id, 'label',
                          'rh.%s.annot' % self.__annot)

        return annot_legend(lh, rh, *args, **kwargs)

    def remove_data(self):
        """Remove data shown with ``Brain.add_ndvar``"""
        surfer.Brain.remove_data(self)
        del self.__data[:]

    def set_parallel_view(self, forward=None, up=None, scale=None):
        """Set view to parallel projection

        Parameters
        ----------
        forward : scalar
            Move the view forward (mm).
        up : scalar
            Move the view upward (mm).
        scale : scalar
            Mayavi parallel_scale parameter. Default is 95 for the inflated
            surface, 75 otherwise.
        """
        if scale is True:
            surf = self.geo.values()[0].surf
            if surf == 'inflated':
                scale = 95
            else:
                scale = 75  # was 65 for WX backend

        for figs in self._figures:
            for fig in figs:
                if forward is not None or up is not None:
                    mlab.view(focalpoint=(0, forward or 0, up or 0),
                              figure=fig)
                if scale is not None:
                    fig.scene.camera.parallel_scale = scale
                fig.scene.camera.parallel_projection = True
                fig.render()

        # without this sometimes the brain position is off
        self.screenshot()
