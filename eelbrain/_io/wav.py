# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""I/O for wave files"""
import os

import numpy as np

from .._data_obj import NDVar, UTS, Ordered
from .._utils import ui


FILETYPES = [("WAV files", "*.wav")]


def load_wav(filename=None, name=None):
    """Load a wav file as NDVar

    Parameters
    ----------
    filename : str
        Filename of the wav file. If not filename is specified, a file dialog is
        shown to select one.
    name : str
        NDVar name (default is the file name).

    Returns
    -------
    wav : NDVar
        NDVar with the wav file's data. If the file contains a single channel,
        the NDVar dimensions are ``(time,)``; if it contains several channels,
        they are ``(channel, time)``. ``wav.info`` contains entries for
        ``filename`` and ``samplingrate``.

    Notes
    -----
    Uses :mod:`scipy.io.wavfile`.
    """
    from scipy.io import wavfile

    if filename is None:
        filename = ui.ask_file("Load WAV File", "Select WAV file to load as "
                               "NDVar", FILETYPES)
        if not filename:
            return
    elif not isinstance(filename, str):
        raise TypeError("filename must be string, got %s" % repr(filename))
    elif not os.path.exists(filename):
        _, ext = os.path.splitext(filename)
        if not ext:
            filename += '.wav'

    srate, data = wavfile.read(filename)
    time = UTS(0, 1. / srate, data.shape[-1])
    if name is None:
        name = os.path.basename(filename)
    info = {'filename': filename, 'samplingrate': srate}
    if data.ndim == 1:
        return NDVar(data, (time,), info, name)
    elif data.ndim == 2:
        chan = Ordered('channel', np.arange(len(data)))
        return NDVar(data, (chan, time), info, name)
    else:
        raise NotImplementedError("Data with %i dimensions" % data.ndim)


def save_wav(ndvar, filename=None, toint=False):
    """Save an NDVar as wav file

    Parameters
    ----------
    ndvar : NDVar (time,)
        Sound data. Values should either be floating point numbers between -1
        and +1, or 16 bit integers.
    filename : str
        Where to save. If unspecified a file dialog will ask for the location.
    toint : bool
        Convert floating point data to 16 bit integer (default False).

    Notes
    -----
    Uses :mod:`scipy.io.wavfile`.
    """
    from scipy.io import wavfile

    data = ndvar.get_data('time')
    if toint and data.dtype != np.int16:
        above = data >= 2**15
        below = data < -2**15
        if np.any(above) or np.any(below):
            n = np.sum(above) + np.sum(below)
            print(("WARNING: clipping %i samples" % n))
            data[above] = 2**15 - 1
            data[below] = -2**15

        data = data.astype(np.int16)
    elif data.dtype.kind != 'i' and (data.max() > 1. or data.min() < -1.):
        raise ValueError("Floating point data should be in range [-1, 1]. Set "
                         "toint=True to save as 16 bit integer data.")

    if filename is None:
        msg = "Save %s..." % ndvar.name
        filename = ui.ask_saveas(msg, msg, FILETYPES)
    srate = int(round(1. / ndvar.time.tstep))
    wavfile.write(filename, srate, data)
