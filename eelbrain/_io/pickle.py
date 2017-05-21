# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>


from pickle import dump, HIGHEST_PROTOCOL, Unpickler
from importlib import import_module
from itertools import chain
import os

from .._data_obj import NDVar
from .._utils import ui


def pickle(obj, dest=None, protocol=HIGHEST_PROTOCOL):
    """Pickle a Python object.

    Parameters
    ----------
    dest : None | str
        Path to destination where to save the  file. If no destination is
        provided, a file dialog is shown. If a destination without extension is
        provided, '.pickled' is appended.
    protocol : int
        Pickle protocol (default is HIGHEST_PROTOCOL).
    """
    if dest is None:
        filetypes = [("Pickled Python Objects (*.pickled)", '*.pickled')]
        dest = ui.ask_saveas("Pickle Destination", "", filetypes)
        if dest is False:
            raise RuntimeError("User canceled")
        else:
            print('dest=%r' % dest)
    else:
        dest = os.path.expanduser(dest)
        if not os.path.splitext(dest)[1]:
            dest += '.pickled'

    try:
        with open(dest, 'wb') as fid:
            dump(obj, fid, protocol)
    except SystemError as exception:
        if exception.args[0] == 'error return without exception set':
            if os.path.exists(dest):
                os.remove(dest)
            raise IOError("An error occurred while pickling. This could be "
                          "due to an attempt to pickle an array (or NDVar) "
                          "that is too big. Try saving several smaller arrays.")
        else:
            raise


def map_paths(module_name, class_name):
    "Subclass to handle changes in module paths"
    if module_name == 'eelbrain.vessels.data':
        module_name = 'eelbrain._data_obj'
        class_names = {'var': 'Var', 'factor': 'Factor', 'ndvar': 'NDVar',
                       'datalist': 'Datalist', 'dataset': 'Dataset'}
        class_name = class_names[class_name]
    elif module_name.startswith('eelbrain.data.'):
        if module_name.startswith('eelbrain.data.load'):
            rev = module_name.replace('.data.load', '.load')
        elif module_name.startswith('eelbrain.data.stats'):
            rev = module_name.replace('.data.stats', '._stats')
        elif module_name.startswith('eelbrain.data.data_obj'):
            rev = module_name.replace('.data.data_obj', '._data_obj')
        else:
            raise NotImplementedError("%r / %r" % (module_name, class_name))
        module_name = rev

    module = import_module(module_name)
    return getattr(module, class_name)


def unpickle(file_path=None):
    """Load pickled Python objects from a file.

    Almost like ``cPickle.load(open(file_path))``, but also loads object saved
    with older versions of Eelbrain, and allows using a system file dialog to
    select a file.

    Parameters
    ----------
    file_path : None | str
        Path to a pickled file. If None (default), a system file dialog will be
        shown. If the user cancels the file dialog, a RuntimeError is raised.
    """
    if file_path is None:
        filetypes = [("Pickles (*.pickled)", '*.pickled'), ("All files", '*')]
        file_path = ui.ask_file("Select File to Unpickle", "Select a pickled "
                                "file to unpickle", filetypes)
        if file_path is False:
            raise RuntimeError("User canceled")
        else:
            print(repr(file_path))
    else:
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            new_path = os.extsep.join((file_path, 'pickled'))
            if os.path.exists(new_path):
                file_path = new_path

    with open(file_path, 'rb') as fid:
        unpickler = Unpickler(fid)
        unpickler.find_global = map_paths
        obj = unpickler.load()

    return obj


def update_subjects_dir(obj, subjects_dir, depth=0):
    """Update NDVar SourceSpace.subjects_dir attributes

    Examine elements of ``obj`` recursively and replace ``subjects_dir`` on all
    NDVars with SourceSpace dimension that are found.

    Parameters
    ----------
    obj : object
        Object to examine.
    subjects_dir : str
        New values for subjects_dir.
    depth : int
        Recursion depth for examining attributes (default 2). Negative number
        for exhaustive search.

    Notes
    -----
    The following elements are searched:

      - Attributes of objects that have a ``__dict__``.
      - ``dict`` values.
      - list/tuple items.
    """
    if isinstance(obj, NDVar):
        if hasattr(obj, 'source'):
            obj.source.subjects_dir = subjects_dir
    elif depth:
        if hasattr(obj, '__dict__'):
            values = iter(obj.__dict__.values())
        else:
            values = ()

        if isinstance(obj, dict):
            values = chain(values, iter(obj.values()))
        elif isinstance(obj, (tuple, list)):
            values = chain(values, obj)

        for v in values:
            update_subjects_dir(v, subjects_dir, depth - 1)
