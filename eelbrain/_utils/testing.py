# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Utilities for testing"
from distutils.version import LooseVersion
from functools import wraps
from importlib import import_module
import os
from operator import mul
import shutil
import tempfile

from nose.plugins.skip import SkipTest
from nose.tools import assert_equal, assert_true, eq_
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .._data_obj import Dataset, NDVar, Var, isdatalist, isdatacontainer, isuv
from functools import reduce


class TempDir(str):
    "After MNE-Python mne.utils"
    def __new__(cls):
        return str.__new__(cls, tempfile.mkdtemp())

    def __del__(self):
        shutil.rmtree(self, ignore_errors=True)


def assert_dataset_equal(ds1, ds2, msg="Datasets unequal", decimal=None):
    """
    Raise an assertion if two Datasets are not equal up to desired precision.

    Parameters
    ----------
    ds1, ds2 : Dataset
        Datasets to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    decimal : None | int
        Desired precision (default is exact match).
    """
    assert_equal(list(ds1.keys()), list(ds2.keys()), "%s: different keys (%s vs %s)" %
                 (msg, list(ds1.keys()), list(ds2.keys())))
    for k in list(ds1.keys()):
        assert_dataobj_equal(ds1[k], ds2[k], msg=msg, decimal=decimal)
    assert_equal(list(ds1.info.keys()), list(ds2.info.keys()), "%s: keys in info" % msg)


def assert_dataobj_equal(d1, d2, msg="Data-objects unequal", decimal=None):
    """Assert that two data-objects are equal up to desired precision.

    Parameters
    ----------
    d1, d2 : data-objects
        Data-objects to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    decimal : None | int
        Desired precision (default is exact match).
    """
    if not isdatacontainer(d1):
        raise TypeError("d1 is not a data-object but %s" % repr(d1))
    elif not isdatacontainer(d2):
        raise TypeError("d2 is not a data-object but %s" % repr(d2))
    else:
        eq_(type(d1), type(d2))
    msg += ":"
    assert_equal(d1.name, d2.name, "%s unequal names (%r vs %r"
                 ")" % (msg, d1.name, d2.name))
    msg += ' Two %ss named %r have' % (d1.__class__.__name__, d1.name)
    len1 = len(d1)
    len2 = len(d2)
    assert_equal(len1, len2, "%s unequal length: %i/%i" % (msg, len1, len2))
    if isinstance(d1, Var) and decimal:
        assert_array_almost_equal(d1.x, d2.x, decimal)
    elif isuv(d1):
        assert_true(np.all(d1 == d2), "%s unequal values: %r vs "
                    "%r" % (msg, d1, d2))
    elif isinstance(d1, NDVar):
        if decimal:
            is_different = np.max(np.abs(d1.x - d2.x)) >= 10**-decimal
        else:
            is_different = np.any(d1.x != d2.x)

        if is_different:
            n = reduce(mul, d1.x.shape)
            n_different = (d1.x != d2.x).sum()
            mean_diff = np.abs(d1.x - d2.x).sum() / n_different
            raise AssertionError("%s unequal values. Difference in %i of %i "
                                 "values, average difference=%s." %
                                 (msg, n_different, n, mean_diff))
    elif isdatalist(d1):
        for i in range(len(d1)):
            assert_equal(d1[i], d2[i], "%s unequal values" % msg)


def assert_source_space_equal(src1, src2, msg="SourceSpace Dimension objects "
                              "unequal"):
    """Assert that two SourceSpace objects are identical

    Parameters
    ----------
    src1, src2 : SourceSpace objects
        SourceSpace objects to compare.
    msg : str
        Prefix of the error message to be printed in case of failure.
    """
    msg = "%s:" % msg
    assert_array_equal(src1.vertno[0], src2.vertno[0], "%s unequal lh vertno "
                       "(%r vs %r)" % (msg, src1.vertno[0], src2.vertno[0]))
    assert_array_equal(src1.vertno[1], src2.vertno[1], "%s unequal rh vertno "
                       "(%r vs %r)" % (msg, src1.vertno[1], src2.vertno[1]))
    assert_equal(src1.subject, src2.subject, "%s unequal subject (%r vs %r"
                 ")" % (msg, src1.subject, src2.subject))
    assert_equal(src1.src, src2.src, "%s unequal names (%r vs %r"
                 ")" % (msg, src1.src, src2.src))
    assert_equal(src1.subjects_dir, src2.subjects_dir, "%s unequal names (%r "
                 "vs %r)" % (msg, src1.subjects_dir, src2.subjects_dir))


def requires_mayavi(function):
    """Decorator to skip test if mayavi is not available"""
    try:
        import mayavi
    except ImportError:
        @wraps(function)
        def decorator(*args, **kwargs):
            raise SkipTest('Skipped %s, requires mayavi' % function.__name__)
    else:
        @wraps(function)
        def decorator(*args, **kwargs):
            return function(*args, **kwargs)
    return decorator


def requires_mne_sample_data(function):
    import mne
    if mne.datasets.sample.data_path(download=False):
        @wraps(function)
        def decorator(*args, **kwargs):
            return function(*args, **kwargs)
    else:
        @wraps(function)
        def decorator(*args, **kwargs):
            raise SkipTest('Skipped %s, requires mne sample data' % function.__name__)
    return decorator


def requires_module(name, version):
    mod = import_module(name)

    def wrapper(function):
        if LooseVersion(mod.__version__) >= LooseVersion(version):
            @wraps(function)
            def decorator(*args, **kwargs):
                return function(*args, **kwargs)
        else:
            @wraps(function)
            def decorator(*args, **kwargs):
                raise SkipTest('Skipped %s, requires %s %s, found mne %s' %
                               (function.__name__, name, version,
                                mod.__version__))
        return decorator
    return wrapper


def skip_on_windows(function):
    @wraps(function)
    def decorator(*args, **kwargs):
        if os.name == 'nt':
            raise SkipTest('Skipped %s on Windows' % function.__name__)
        else:
            return function(*args, **kwargs)
    return decorator


def file_path(name):
    "Path to test data file in the test_data directory"
    path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..',
                                         'test_data', name))
    if os.path.exists(path):
        return path
    else:
        raise IOError("Testing file does not exist. Test can only be executed "
                      "from source repository.")
