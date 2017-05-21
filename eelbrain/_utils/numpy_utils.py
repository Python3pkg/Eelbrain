# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from collections import Sequence
from distutils.version import LooseVersion


import numpy as np


full_slice = slice(None)


def digitize_index(index, values, tol=None):
    """Locate a scalar ``index`` on ``values``

    Parameters
    ----------
    index : scalar
        Index to locate on ``values``.
    values : array
        1-dimensional array on which to locate ``index``.
    tol : float
        Tolerance for suppressing an IndexError when index only falls near a
        value.
    """
    i = int(digitize(index, values, True))
    if index == values[i]:
        return i
    elif not tol:
        raise IndexError("Index %r does not match any value" % (index,))
    elif i > 0 and values[i] - index > index - values[i - 1]:
        i -= 1
    diff = abs(index - values[i])
    if i == 0:
        if len(values) == 1:
            if diff <= abs(values[0]) * tol:
                return i
            else:
                raise IndexError("Index %r outside of tolerance from only "
                                 "value %r" % (index, values[0]))
        elif diff <= abs(values[1] - values[0]) * tol:
            return i
        else:
            raise IndexError("Index %r outside of tolerance" % (index,))
    elif diff <= abs(values[i] - values[i - 1]) * tol:
        return i
    else:
        raise IndexError("Index %r outside of tolerance" % (index,))


def digitize_slice_endpoint(index, values):
    """Locate a scalar slice endpoint on ``values``

    (Whenever index is between two values, move to the larger)

    Parameters
    ----------
    index : scalar
        Index to locate on ``values``.
    values : array
        1-dimensional array on which to locate ``index``.
    """
    return int(digitize(index, values, True))


def index_to_int_array(index, n):
    if isinstance(index, np.ndarray):
        if index.dtype.kind == 'i':
            return index
        elif index.dtype.kind == 'b':
            return np.flatnonzero(index)
    return np.arange(n)[index]


def apply_numpy_index(data, index):
    "Apply numpy index to non-numpy sequence"
    if isinstance(index, (int, slice)):
        return data[index]
    elif isinstance(index, list):
        return (data[i] for i in index)
    array = np.asarray(index)
    assert array.ndim == 1, "Index must be 1 dimensional, got %r" % (index,)
    if array.dtype.kind == 'i':
        return (data[i] for i in array)
    elif array.dtype.kind == 'b':
        assert len(array) == len(data), "Index must have same length as data"
        return (d for d, i in zip(data, array) if i)
    raise TypeError("Invalid numpy-like index: %r" % (index,))


def slice_to_arange(s, length):
    """Convert a slice into a numerical index

    Parameters
    ----------
    s : slice
        Slice.
    length : int
        Length of the target for the index (only needed if ``slice.stop`` is
        None).

    Returns
    -------
    arange : array of int
        Numerical index equivalent to slice.
    """
    if s.start is None:
        start = 0
    else:
        start = s.start

    if s.stop is None:
        stop = length
    else:
        stop = s.stop

    return np.arange(start, stop, s.step)


# pre numpy 0.10, digitize requires 1d-array
if LooseVersion(np.__version__) < LooseVersion('1.10'):
    def digitize(x, bins, right=False):
        if np.isscalar(x):
            return np.digitize(np.atleast_1d(x), bins, right)[0]
        elif x.ndim != 1:
            raise NotImplementedError("digitize for pre 1.10 numpy with ndim > "
                                      "1 array")
        return np.digitize(x, bins, right)
else:
    digitize = np.digitize
