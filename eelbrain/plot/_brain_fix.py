# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from distutils.version import LooseVersion
import os
import sys

from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import numpy as np

from .._data_obj import NDVar, SourceSpace
from ..fmtxt import Image
from ._base import (backend, find_axis_params_data, find_fig_cmaps,
                    find_fig_vlims)
from ._colors import ColorBar

# pyface imports: set GUI backend (ETS don't support wxPython 3.0)
if backend['ets_toolkit']:
    os.environ['ETS_TOOLKIT'] = backend['ets_toolkit']

# surfer imports, lower screen logging level
first_import = 'surfer' not in sys.modules
import surfer
if first_import:
    from ..mne_fixes import reset_logger
    reset_logger(surfer.utils.logger)


def assert_can_save_movies():
    if LooseVersion(surfer.__version__) < LooseVersion('0.6'):
        raise ImportError("Saving movies requires PySurfer 0.6")


class Brain(surfer.Brain):
    # Subclass that adds Eelbrain functionality to the PySurfer surfer.Brain class
    def __init__(self, *args, **kwargs):
        self.__data = None
        self.__annot = None
        surfer.Brain.__init__(self, *args, **kwargs)

        from traits.trait_base import ETSConfig
        self._prevent_close = ETSConfig.toolkit == 'wx'

    def add_mask(self, source, alpha=0.5, smoothing_steps=None,
                 subjects_dir=None):
        """Add a mask shading areas that are not included in an NDVar

        Parameters
        ----------
        source : SourceSpace
            SourceSpace.
        alpha : scalar
            Opacity of the mask layer.
        smoothing_steps : scalar (optional)
            Smooth transition at the mask's border.
        subjects_dir : str
            Use this directory as the subjects directory.
        """
        if isinstance(source, NDVar):
            source = source.get_dim('source')
        if not isinstance(source, SourceSpace):
            raise TypeError("source needs to be a SourceSpace or NDVar, got "
                            "%s" % (source,))
        if not 0. <= alpha <= 1.:
            raise ValueError("alpha needs to be between 0 and 1, got %s" % alpha)

        if smoothing_steps is not None:
            mask_ndvar = source._mask_ndvar(subjects_dir)
            lut = np.zeros((256, 4), np.uint8)
            lut[::-1, 3] = np.arange(256, dtype=np.uint8) * alpha
            self.add_ndvar(mask_ndvar, lut, 0., 1., smoothing_steps, False, None)
        else:
            lh, rh = source._mask_label(subjects_dir)
            if self._hemi == 'lh':
                rh = None
            elif self._hemi == 'rh':
                lh = None

            if source.lh_n and lh:
                self.add_label(lh, alpha=alpha)
            if source.rh_n and rh:
                self.add_label(rh, alpha=alpha)

    def add_ndvar(self, ndvar, cmap=None, vmin=None, vmax=None,
                  smoothing_steps=None, colorbar=False, time_label='ms'):
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
        """
        # colormap
        if cmap is None or isinstance(cmap, basestring):
            epochs = ((ndvar,),)
            cmaps = find_fig_cmaps(epochs, cmap, alpha=True)
            vlims = find_fig_vlims(epochs, vmax, vmin, cmaps)
            meas = ndvar.info.get('meas')
            vmin, vmax = vlims[meas]
            # convert to LUT
            cmap = get_cmap(cmaps[meas])
            cmap = np.round(cmap(np.arange(256)) * 255).astype(np.uint8)

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

        # add data
        surfaces = []
        if ndvar.source.lh_n and self._hemi != 'rh':
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
            surfaces.append(self.data_dict['lh']['surfaces'][0])

        if ndvar.source.rh_n and self._hemi != 'lh':
            src_hemi = ndvar.sub(source='rh')
            data = src_hemi.get_data(data_dims)
            vertices = ndvar.source.rh_vertno
            self.add_data(data, vmin, vmax, None, cmap, alpha, vertices,
                          smoothing_steps, times, time_label, colorbar, 'rh')
            surfaces.append(self.data_dict['rh']['surfaces'][0])

        for s in surfaces:
            s.actor.property.lighting = False

        self.__data = ndvar

    def close(self):
        "Prevent close() call that causes segmentation fault"
        if not self._prevent_close:
            surfer.Brain.close(self)

    def _get_cmap_params(self, label=True):
        """Return parameters required to plot a colorbar"""
        if self._hemi in ('both', 'split', 'lh'):
            data = self.data_dict['lh']
        elif self._hemi == 'rh':
            data = self.data_dict[self._hemi]
        else:
            raise RuntimeError("Brain._hemi=%s" % repr(self._hemi))
        cmap = ListedColormap(data['orig_ctable'] / 255., label)
        return cmap, data['fmin'], data['fmax']

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
            name = getattr(self.__data, 'name', None) or 'brain'
        im = self.screenshot('rgba', True)
        return Image.from_array(im, name, format, alt)

    def plot_colorbar(self, label=True, label_position=None, label_rotation=None,
                      clipmin=None, clipmax=None, orientation='horizontal',
                      width=None, ticks=None, *args, **kwargs):
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

        Returns
        -------
        colorbar : :class:`~eelbrain.plot.ColorBar`
            ColorBar plot object.
        """
        if self.__data is None:
            raise RuntimeError("Brain has no data to plot colorbar for")
        unit = self.__data.info.get('unit', None)
        if ticks is None:
            ticks = self.__data.info.get('cmap ticks')
        _, label = find_axis_params_data(self.__data, label)
        cmap, vmin, vmax = self._get_cmap_params(label)
        return ColorBar(cmap, vmin, vmax, label, label_position, label_rotation,
                        clipmin, clipmax, orientation, unit, (), width, ticks,
                        *args, **kwargs)

    def _set_annot(self, annot, borders, alpha):
        "Store annot name to enable plot_legend()"
        self.add_annotation(annot, borders, alpha)
        self.__annot = annot

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

    def set_parallel_scale(self, scale=None):
        """Set view to parallel projection

        Parameters
        ----------
        scale : scalar
            Mayavi parallel_scale parameter. Default is 95 for the inflated
            surface, 75 otherwise.
        """
        if scale is None:
            surf = self.geo.values()[0].surf
            if surf == 'inflated':
                scale = 95
            else:
                scale = 75  # was 65 for WX backend

        for figs in self._figures:
            for fig in figs:
                fig.scene.camera.parallel_scale = scale
                fig.scene.camera.parallel_projection = True
                fig.render()

        # without this sometimes the brain position is off
        # self.screenshot()
