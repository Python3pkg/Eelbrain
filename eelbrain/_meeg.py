# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import defaultdict


import numpy as np

from ._data_obj import Datalist, Dataset, Var
from ._ndvar import neighbor_correlation
from ._info import BAD_CHANNELS
from ._names import INTERPOLATE_CHANNELS


def _out(out, epochs):
    if out is None:
        return Datalist([[] for _ in range(len(epochs))])
    elif len(out) != len(epochs):
        raise ValueError("out needs same length as epochs, got %i/%i" %
                         (len(out), len(epochs)))
    return out


def new_rejection_ds(ds):
    """Create a rejection Dataset from a Dataset with epochs"""
    out = Dataset(info={BAD_CHANNELS: []})
    out['trigger'] = ds['trigger']
    out[:, 'accept'] = True
    out[:, 'tag'] = ''
    return out


def find_flat_epochs(epochs, flat=1e-13, out=None):
    out = _out(out, epochs)
    d = epochs.max('time') - epochs.min('time')
    for i, chi in zip(*np.nonzero(d.get_data(('case', 'sensor')) < flat)):
        ch = epochs.sensor.names[chi]
        if ch not in out[i]:
            out[i].append(ch)

    return out


def find_flat_evoked(epochs, flat=1e-14):
    average = epochs.mean('case')
    d = average.max('time') - average.min('time')
    return epochs.sensor.names[d < flat]


def find_noisy_channels(epochs, mincorr=0.35):
    names = epochs.sensor.names
    out_e = Datalist([list(names[neighbor_correlation(ep) < mincorr]) for ep in epochs])
    return out_e


def find_bad_channels(epochs, flat, flat_average, mincorr):
    "Find flat and noisy channels"
    interpolate = find_noisy_channels(epochs, mincorr)
    interpolate = find_flat_epochs(epochs, flat, interpolate)
    bad_channels = find_flat_evoked(epochs, flat_average)
    return bad_channels, interpolate


def make_rej(ds):
    epochs = ds['meg']

    # find rejections
    bad_channels, interpolate = find_bad_channels(epochs)

    # construct dataset
    out = ds[('trigger',)]
    out['accept'] = Var(np.array(list(map(len, interpolate))) <= 5)
    out[:, 'tag'] = ''
    out[INTERPOLATE_CHANNELS] = interpolate
    out.info[BAD_CHANNELS] = bad_channels
    return out


def channel_listlist_to_dict(listlist):
    out = defaultdict(list)
    for i, chs in enumerate(listlist):
        for ch in chs:
            out[ch].append(i)
    return dict(out)
