# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from math import floor
import os

from nose.tools import eq_, assert_almost_equal, assert_is_none, assert_raises
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pickle as pickle
import scipy.io
from eelbrain import boosting, convolve, datasets
from eelbrain._trf import _boosting
from eelbrain._trf._boosting import boost_1seg, evaluate_kernel
from eelbrain._utils.testing import assert_dataobj_equal


def assert_res_equal(res1, res):
    assert_array_equal(res1.h, res.h)
    eq_(res1.r, res.r)
    eq_(res1.spearmanr, res.spearmanr)


def run_boosting(ds, mp):
    "Run boosting tests"
    if not mp:
        n_workers = _boosting.N_WORKERS
        _boosting.N_WORKERS = 0

    y = ds['y']
    x1 = ds['x1']
    x2 = ds['x2']
    y_mean = y.mean()
    x2_mean = x2.mean()

    # test values from running function, not verified independently
    res = boosting(y, x1 * 2000, 0, 1, scale_data=False, mindelta=0.0025)
    eq_(repr(res),
        '<boosting y ~ x1, 0 - 1, scale_data=False, mindelta=0.0025>')
    eq_(round(res.r, 2), 0.75)
    assert_is_none(res.y_mean)

    res = boosting(y, x1, 0, 1)
    eq_(repr(res), '<boosting y ~ x1, 0 - 1>')
    eq_(round(res.r, 2), 0.83)
    eq_(res.y_mean, y_mean)
    eq_(res.y_scale, y.std())
    eq_(res.x_mean, x1.mean())
    eq_(res.x_scale, x1.std())
    # inplace
    res_ip = boosting(y.copy(), x1.copy(), 0, 1, 'inplace')
    assert_res_equal(res_ip, res)
    # persistence
    res_p = pickle.loads(pickle.dumps(res, pickle.HIGHEST_PROTOCOL))
    assert_res_equal(res_p, res)

    res = boosting(y, x2, 0, 1)
    eq_(round(res.r, 2), 0.60)

    res = boosting(y, x2, 0, 1, error='l1')
    eq_(round(res.r, 2), 0.55)
    eq_(res.y_mean, y.mean())
    eq_(res.y_scale, (y - y_mean).abs().mean())
    eq_(res.x_mean, x2_mean)
    eq_(res.x_scale, (x2 - x2_mean).abs().mean())

    res = boosting(y, [x1, x2], 0, 1)
    eq_(round(res.r, 2), 0.98)

    if not mp:
        _boosting.N_WORKERS = n_workers


def test_boosting():
    "Test boosting NDVars"
    ds = datasets._get_continuous()

    # test boosting results
    yield run_boosting, ds, True
    yield run_boosting, ds, False


def test_result():
    "Test boosting results"
    ds = datasets._get_continuous()

    # convolve function
    y = convolve([ds['h1'], ds['h2']], [ds['x1'], ds['x2']])
    y.name = 'y'
    assert_dataobj_equal(y, ds['y'])

    # test prediction with res.h and res.h_scaled
    res = boosting(ds['y'], ds['x1'], 0, 1)
    y1 = convolve(res.h_scaled, ds['x1'])
    x_scaled = ds['x1'] / res.x_scale
    y2 = convolve(res.h, x_scaled)
    y2 *= res.y_scale
    y2 += y1.mean() - y2.mean()  # mean can't be reconstructed
    assert_dataobj_equal(y1, y2, decimal=12)

    # test NaN checks  (modifies data)
    ds['x2'].x[1, 50] = np.nan
    assert_raises(ValueError, boosting, ds['y'], ds['x2'], 0, .5)
    assert_raises(ValueError, boosting, ds['y'], ds['x2'], 0, .5, False)
    ds['x2'].x[1, :] = 1
    assert_raises(ValueError, boosting, ds['y'], ds['x2'], 0, .5)
    ds['y'].x[50] = np.nan
    assert_raises(ValueError, boosting, ds['y'], ds['x1'], 0, .5)
    assert_raises(ValueError, boosting, ds['y'], ds['x1'], 0, .5, False)


def test_boosting_func():
    "Test boosting() against svdboostV4pred.m"
    # 1d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting.mat')
    mat = scipy.io.loadmat(path)
    x = mat['stim']
    y = mat['signal'][0]

    h, test_sse_history = boost_1seg(x, y, 10, 0.005, 40, 0, 0.01, 'l2', True)
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr, _ = evaluate_kernel(y[:test_seg_len], x[:, :test_seg_len], h, 'l2')

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0], 10)
    assert_allclose(test_sse_history, mat['Str_testE'][0])

    # 2d-TRF
    path = os.path.join(os.path.dirname(__file__), 'test_boosting_2d.mat')
    mat = scipy.io.loadmat(path)
    x = mat['stim']
    y = mat['signal'][0]

    h, test_sse_history = boost_1seg(x, y, 10, 0.005, 40, 0, 0.01, 'l2', True)
    test_seg_len = int(floor(x.shape[1] / 40))
    r, rr, _ = evaluate_kernel(y[:test_seg_len], x[:, :test_seg_len], h, 'l2')

    assert_array_equal(h, mat['h'])
    assert_almost_equal(r, mat['crlt'][0, 0], 10)
    assert_almost_equal(rr, mat['crlt'][1, 0])
    # svdboostV4pred multiplies error by number of predictors
    assert_allclose(test_sse_history, mat['Str_testE'][0] / 3)
