import os

import numpy as np

from .._data_obj import NDVar, UTS, Ordered
from .._utils import ui


def wav(filename=None):
    from scipy.io import wavfile

    if filename is None:
        filename = ui.ask_file("Load WAV File", "Select WAV file to load as "
                               "NDVar", [("WAV files", "*.wav")])
        if not filename:
            return
    elif not isinstance(filename, str):
        raise TypeError("filename must be string, got %s" % repr(filename))

    srate, data = wavfile.read(filename)
    time = UTS(0, 1. / srate, data.shape[-1])
    name = os.path.basename(filename)
    info = {'filename': filename, 'samplingrate': srate}
    if data.ndim == 1:
        return NDVar(data, (time,), info, name)
    elif data.ndim == 2:
        chan = Ordered('channel', np.arange(len(data)))
        return NDVar(data, (chan, time), info, name)
    else:
        raise NotImplementedError("Data with %i dimensions" % data.ndim)
