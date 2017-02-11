# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Fix up surfer.Brain"""
from distutils.version import LooseVersion
import os
import sys

from matplotlib.colors import ListedColormap

from ..fmtxt import Image
from ._base import backend, find_axis_params_data
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
    def __init__(self, data, *args, **kwargs):
        self.__data = data
        self.__annot = None
        surfer.Brain.__init__(self, *args, **kwargs)

        from traits.trait_base import ETSConfig
        self._prevent_close = ETSConfig.toolkit == 'wx'

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
            name = self.__data.name or 'brain'
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
