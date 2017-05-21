'''
Tools for loading data from text files.

.. autosummary::
   :toctree: generated

   tsv
   var
'''
import os
import re

import numpy as np

from .._utils import ui
from .._utils.parse import FLOAT_NAN_PATTERN
from .. import _data_obj as _data

__all__ = ('tsv', 'var')


# could use csv module (http://docs.python.org/2/library/csv.html) but it
# currently does not support unicode
def tsv(path=None, names=True, types='auto', delimiter='\t', skiprows=0,
        start_tag=None, ignore_missing=False, empty=None):
    r"""
    Load a :class:`Dataset` from a tab-separated values file.

    Parameters
    ----------
    path : None | str
        Path to the tsv file. If None, a system file dialog will open.
    names : list of str | bool
        * ``['name1', ...]`` use these names
        * ``True``: look for names on the first line of the file
        * ``False``: use "v1", "v2", ...
    types : 'auto' | list of int
        * ``'auto'`` -> import as Var if all values can be converted float,
          otherwise as Factor
        * list of 0=auto, 1=Factor, 2=Var. e.g. ``[0,1,1,0]``
    delimiter : None | str
        Value delimiting cells in the input file (default: ``'\t'`` (tab);
        None = any whitespace).
    skiprows : int
        Skip so many rows at the beginning of the file (for tsv files with
        headers). Column names (if names==True) are expected to come after
        the skipped rows. Skiprows is applied after start_tag.
    start_tag : None | str
        Alternative way to skip header rows. The table is assumed to start
        on the line following the last line in the file that starts with
        ``start_tag``.
    ignore_missing : bool
        Ignore rows with missing values (default False). Append ``NaN`` for
        numerical and ``""`` for categorial variables. Missing values occur when
        not all lines in a file contain the same number of occurrences of the
        delimiter. For reading empty values (i.e., "") see the empty_to_nan
        argument.
    empty : str
        For numerical variables, substitute this value for empty entries (i.e., for
        ""). For example, if a column in a file contains ``"5", "3", ""``, this is
        read by default as ``Factor(['5', '3', ''])``. With ``empty='nan'``, it is
        read as ``Var([5, 3, nan])``.
    """
    if path is None:
        path = ui.ask_file("Load TSV", "Select tsv file to import as Dataset")
        if not path:
            return

    with open(path) as fid:
        lines = fid.readlines()
        if len(lines) == 1:
            # tsv file exported by excel had carriage return only
            text = lines[0]
            if text.count('\r') > 1:
                lines = text.split('\r')

    # find start position
    if start_tag:
        start = 0
        for i, line in enumerate(lines, 1):
            if line.startswith(start_tag):
                start = i
        if start:
            lines = lines[start:]
    if skiprows:
        lines = lines[skiprows:]

    # read / create names
    if names is True:
        head_line = lines.pop(0)
        names = head_line.split(delimiter)
        names = [n.strip().strip('"') for n in names]

    # separate lines into values
    rows = [[v.strip() for v in line.split(delimiter)] for line in lines]

    row_lens = set(len(row) for row in rows)
    if len(row_lens) > 1 and not ignore_missing:
        msg = ("Not all rows have same number of entries. Set ignore_missing "
               "to True in order to ignore this error.")
        raise ValueError(msg)
    n_cols = max(row_lens)
    n_rows = len(rows)

    if names:
        if len(names) != n_cols:
            msg = ("The number of names in the header (%i) does not "
                   "correspond to the number of columns in the table (%i)"
                   % (len(names), n_cols))
            raise ValueError(msg)
    else:
        names = ['v%i' % i for i in range(n_cols)]

    if types in ('auto', None, False, True):
        types = [0] * n_cols
    else:
        assert len(types) == n_cols

    # find quotes (imply type 1)
    quotes = "'\""
    data = np.empty((n_rows, n_cols), object)
    for r, line in enumerate(rows):
        for c, v in enumerate(line):
            for str_del in quotes:
                if len(v) > 0 and v[0] == str_del:
                    v = v.strip(str_del)
                    types[c] = 1
            data[r, c] = v

    # convert values to data-objects
    float_pattern = re.compile(FLOAT_NAN_PATTERN)
    ds = _data.Dataset(name=os.path.basename(path))
    np_vars = vars(np)
    bool_dict = {'True': True, 'False': False, None: False}
    for name, values, type_ in zip(names, data.T, types):
        # infer type
        if type_ > 0:
            pass
        elif all(v in bool_dict for v in values):
            type_ = 3
        elif empty is not None:
            if all(v in (None, '') or float_pattern.match(v) for v in values):
                type_ = 2
            else:
                type_ = 1
        elif all(v is None or float_pattern.match(v) for v in values):
            type_ = 2
        else:
            type_ = 1

        # substitute values
        if type_ == 2:
            if empty is not None:
                values = [empty if v == '' else v for v in values]
            values = [np.nan if v is None else eval(v, np_vars) for v in values]
        elif type_ == 3:
            values = [bool_dict[v] for v in values]

        # create data-object
        if type_ == 1:
            dob = _data.Factor(values, labels={None: ''}, name=name)
        else:
            dob = _data.Var(values, name=name)
        ds.add(dob)

    return ds


def var(path=None, name=None):
    """
    Load a :class:`Var` object from a text file by splitting at white-spaces.

    Parameters
    ----------
    path : str(path) | None
        Source file. If None, a system file dialog is opened.
    name : str | None
        Name for the Var.
    """
    if path is None:
        path = ui.ask_file("Load Var File", "Select Var File")
        if not path:
            return

    FILE = open(path)
    lines = FILE.read().split()
    FILE.close()
    is_bool = all(line in ['True', 'False'] for line in lines)

    if is_bool:
        x = np.genfromtxt(path, dtype=bool)
    else:
        x = np.loadtxt(path)

    return _data.Var(x, name=name)


def eeg_montage(path=None, kind='Polhemus digitized montage'):
    """Read Montage saved with the Montage App.

    Reads the montage and transforms it to MNE space, ready for use with
    :func:`mne.io.read_raw_brainvision`.

    Parameters
    ----------
    path : str | None
        Source file. If None, a system file dialog is opened.
    kind : str
        Name for the Montage.

    Notes
    -----
    Requires MNE-Python 0.9 or later.
    """
    import mne
    try:
        from mne.channels.layout import Montage
    except:
        raise ImportError("Requires MNE-Python 0.9 or later.")

    if path is None:
        path = ui.ask_file("Load Montage File", "Select Montage File",
                           [("Montage Text Files", "*.txt")])
        if path is None:
            return

    ds = tsv(path, ['name', 'x', 'y', 'z'])
    ds = ds.sub("name != ''")

    names_lower = [name.lower() for name in ds['name']]
    locs = np.hstack((ds['x'].x[:, None], ds['y'].x[:, None], ds['z'].x[:, None]))
    locs /= 1000

    # check that all needed points are present
    missing = [name for name in ('nasion', 'lpa', 'rpa')
               if name not in names_lower]
    if missing:
        raise ValueError("The points %s are missing, but are needed "
                         "to transform the points to the MNE coordinate "
                         "system. Either add the points, or read the "
                         "montage with transform=False." % missing)

    # transform to MNE (Neuromag) space
    nasion = locs[names_lower.index('nasion')]
    lpa = locs[names_lower.index('lpa')]
    rpa = locs[names_lower.index('rpa')]
    trans = mne.transforms.get_ras_to_neuromag_trans(nasion, lpa, rpa)
    locs = mne.transforms.apply_trans(trans, locs)

    return Montage(locs, ds['name'].as_labels(), kind, np.arange(len(locs)))

