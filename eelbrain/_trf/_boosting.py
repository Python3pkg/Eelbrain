from __future__ import division
from inspect import getargspec
from itertools import chain, izip, product
import logging
from math import floor
import time

import numpy as np
from numpy import newaxis
from scipy.stats import spearmanr
from tqdm import tqdm

from .. import _colorspaces as cs
from .._data_obj import NDVar, UTS


# BoostingResult version
VERSION = 6


class BoostingResult(object):
    """Result from boosting a temporal response function

    Attributes
    ----------
    h : NDVar | tuple of NDVar
        The temporal response function. Whether ``h`` is an NDVar or a tuple of
        NDVars depends on whether the ``x`` parameter to :func:`boosting` was
        an NDVar or a sequence of NDVars.
    r : float | NDVar
        Correlation between the measured response and the response predicted
        with ``h``. Type depends on the ``y`` parameter to :func:`boosting`.
    spearmanr : float | NDVar
        As ``r``, the Spearman rank correlation.
    t_run : float
        Time it took to run the boosting algorithm (in seconds).
    error : str
        The error evaluation method used.
    fit_error : float | NDVar
        The fit error, i.e. the result of the ``error`` error function on the
        final fit.
    delta : scalar
        Kernel modification step used.
    mindelta : None | scalar
        Mindelta parameter used.
    scale_data : bool
        Scale_data parameter used.
    y_mean : NDVar | scalar
        Mean that was subtracted from ``y``.
    y_scale : NDVar | scalar
        Scale by which ``y`` was divided.
    x_mean : NDVar | scalar | tuple
        Mean that was subtracted from ``x``.
    x_scale : NDVar | scalar | tuple
        Scale by which ``x`` was divided.
    """
    def __init__(self, h, r, isnan, t_run, version, delta, mindelta, error,
                 spearmanr, fit_error, scale_data, y_mean, y_scale, x_mean,
                 x_scale):
        self.h = h
        self.r = r
        self.isnan = isnan
        self.t_run = t_run
        self.version = version
        self.delta = delta
        self.mindelta = mindelta
        self.error = error
        self.spearmanr = spearmanr
        self.fit_error = fit_error
        self.scale_data = scale_data
        self.y_mean = y_mean
        self.y_scale = y_scale
        self.x_mean = x_mean
        self.x_scale = x_scale

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in
                getargspec(self.__init__).args[1:]}

    def __setstate__(self, state):
        self.__init__(**state)


def boosting(y, x, tstart, tstop, scale_data=True, delta=0.005, mindelta=None,
             error='l2'):
    """Estimate a temporal response function through boosting

    Parameters
    ----------
    y : NDVar
        Signal to predict.
    x : NDVar | sequence of NDVar
        Signal to use to predict ``y``. Can be sequence of NDVars to include
        multiple predictors. Time dimension must correspond to ``y``.
    tstart : float
        Start of the TRF in seconds.
    tstop : float
        Stop of the TRF in seconds.
    scale_data : bool | 'inplace'
        Scale ``y`` and ``x`` before boosting: subtract the mean and divide by
        the standard deviation (when ``error='l2'``) or the mean absolute
        value (when ``error='l1'``). With ``scale_data=True`` (default) the
        original ``y`` and ``x`` are left untouched; use ``'inplace'`` to save
        memory by scaling the original ``y`` and ``x``.
    delta : scalar
        Step for changes in the kernel.
    mindelta : scalar
        If set, ``delta`` is divided in half after each unsuccessful iteration
        until ``delta < mindelta``. The default is ``mindelta = delta``.
    error : 'l2' | 'l1' | 'l2centered' | 'l1centered'
        Error function to use (default is ``l2``).

    Returns
    -------
    result : BoostingResult
        Object containing results from the boosting estimation (see
        :class:`BoostingResult`).
    """
    # check y and x
    if isinstance(x, NDVar):
        x = (x,)
        multiple_x = False
    else:
        x = tuple(x)
        assert all(isinstance(x_, NDVar) for x_ in x)
        multiple_x = True
    time_dim = y.get_dim('time')
    if any(x_.get_dim('time') != time_dim for x_ in x):
        raise ValueError("Not all NDVars have the same time dimension")

    # scale y and x appropriately for error function
    if scale_data:
        data = (y,) + x
        data_mean = tuple(d.mean('time') for d in data)
        if isinstance(scale_data, int):
            data = tuple(d - d_mean for d, d_mean in izip(data, data_mean))
        elif isinstance(scale_data, str):
            if scale_data == 'inplace':
                for d, d_mean in izip(data, data_mean):
                    d -= d_mean
            else:
                raise ValueError("scale_data=%r" % scale_data)
        else:
            raise TypeError("scale_data=%r" % (scale_data,))

        if error == 'l1':
            data_scale = tuple(d.abs().mean('time') for d in data)
        elif error == 'l2':
            data_scale = tuple(d.std('time') for d in data)
        else:
            raise ValueError("error=%r; needs to be 'l1' or 'l2' if "
                             "scale_data=True." % (error,))
        for d, d_scale in izip(data, data_scale):
            d /= d_scale
        y = data[0]
        x = data[1:]
    else:
        data_mean = data_scale = (None,) * (len(x) + 1)

    # x_data:  predictor x time array
    x_data = []
    x_meta = []
    i = 0
    for x_ in x:
        if x_.ndim == 1:
            xdim = None
            data = x_.x[newaxis, :]
            index = i
        elif x_.ndim == 2:
            xdim = x_.dims[not x_.get_axis('time')]
            data = x_.get_data((xdim.name, 'time'))
            index = slice(i, i + len(data))
        else:
            raise NotImplementedError("x with more than 2 dimensions")
        x_data.append(data)
        x_meta.append((x_.name, xdim, index))
        i += len(data)

    if len(x_data) == 1:
        x_data = x_data[0]
    else:
        x_data = np.vstack(x_data)

    # y_data:  ydim x time array
    if y.ndim == 1:
        ydim = None
        y_data = y.x[None, :]
    elif y.ndim == 2:
        ydim = y.dims[not y.get_axis('time')]
        y_data = y.get_data((ydim.name, 'time'))
    else:
        raise NotImplementedError("y with more than 2 dimensions")

    # TRF extra dimension
    trf_dim = ydim

    # prepare trf (by cropping data)
    i_start = int(round(tstart / time_dim.tstep))
    i_stop = int(round(tstop / time_dim.tstep))
    trf_length = i_stop - i_start
    if i_start < 0:
        x_data = x_data[:, -i_start:]
        y_data = y_data[:, :i_start]
    elif i_start > 0:
        x_data = x_data[:, :-i_start]
        y_data = y_data[:, i_start:]

    # do boosting
    n_responses = len(y_data)
    desc = "Boosting %i response" % n_responses + 's' * (n_responses > 1)
    total = n_responses * 10
    pbar = tqdm(desc=desc, total=total)
    res = [boosting_continuous(x_data, y_, trf_length, delta, error, mindelta,
                               pbar=pbar)
           for y_ in y_data]
    hs, rs, rrs, errs = zip(*res)
    h_x = np.array(hs)
    pbar.close()
    dt = time.time() - pbar.start_t

    # correlation
    if ydim is None:
        r = rs[0]
        rr = rrs[0]
        err = errs[0]
        isnan = np.isnan(r)
    else:
        rs = np.asarray(rs)
        isnan = np.isnan(rs)
        rs[isnan] = 0
        r = NDVar(rs, (ydim,), cs.stat_info('r'), 'correlation')
        rr = NDVar(np.asarray(rrs), (ydim,), cs.stat_info('r'), 'rank correlation')
        err = NDVar(np.asarray(errs), (ydim,), y.info.copy(), 'fit error')

    # TRF
    h_time = UTS(tstart, time_dim.tstep, trf_length)
    hs = []
    for name, dim, index in x_meta:
        h_x_ = h_x[:, index, :]
        if dim is None:
            dims = (h_time,)
        else:
            dims = (dim, h_time)
        if trf_dim is None:
            h_x_ = h_x_[0]
        else:
            dims = (trf_dim,) + dims
        hs.append(NDVar(h_x_, dims, y.info.copy(), name))

    if multiple_x:
        hs = tuple(hs)
        idx = slice(1, None)
    else:
        hs = hs[0]
        idx = 1

    return BoostingResult(hs, r, isnan, dt, VERSION, delta, mindelta, error, rr,
                          err, scale_data, data_mean[0], data_scale[0],
                          data_mean[idx], data_scale[idx])


def boosting_continuous(x, y, trf_length, delta, error, mindelta=None, nsegs=10,
                        pbar=None):
    """Boosting for a continuous data segment, cycle through even splits for
    test segment

    Parameters
    ----------
    ...

    Returns
    -------
    h : array
        Average of the Estimated kernel (all zeros if all partitions failed).
    r : scalar | array
        Pearson correlation (array when ``forward`` is set).
    rank_r : scalar | array
        Rank correlation (array when ``forward`` is set).
    error : scalar | array
        Error of the final fit (array when ``forward`` is set).
    """
    logger = logging.getLogger('eelbrain.boosting')
    if mindelta is None:
        mindelta = delta
    hs = []

    for i in xrange(nsegs):
        h, test_sse_history, msg = boost_1seg(x, y, trf_length, delta, nsegs, i,
                                              mindelta, error)
        logger.debug(msg)
        if np.any(h):
            hs.append(h)
        if pbar is not None:
            pbar.update()

    if hs:
        h = np.mean(hs, 0)
        r, rr, err = evaluate_kernel(y, x, h, error)
    else:
        h = np.zeros(h.shape)
        r = rr = err = 0.
    return h, r, rr, err


def boost_1seg(x, y, trf_length, delta, nsegs, segno, mindelta, error):
    """boosting with one test segment determined by regular division

    Based on port of svdboostV4pred

    Parameters
    ----------
    x : array (n_stims, n_times)
        Stimulus.
    y : array (n_times,)
        Dependent signal, time series to predict.
    trf_length : int
        Length of the TRF (in time samples).
    delta : scalar
        Step of the adjustment.
    nsegs : int
        Number of segments
    segno : int [0, nsegs-1]
        which segment to use for testing
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error : 'l2' | 'Sabs'
        Error function to use.

    Returns
    -------
    history[best_iter] : array like h
        Winning kernel.
    test_corr[best_iter] : scalar
        Test data correlation for winning kernel.
    test_rcorr[best_iter] : scalar
        Test data rank correlation for winning kernel.
    test_sse_history : list of len n_iterations
        SSE for test data at each iteration
    train_corr : list of len n_iterations
        Correlation for training data at each iteration.
    """
    assert x.ndim == 2
    assert y.shape == (x.shape[1],)

    # separate training and testing signal
    test_seg_len = int(floor(x.shape[1] / nsegs))
    test_index = slice(test_seg_len * segno, test_seg_len * (segno + 1))
    if segno == 0:
        train_index = (slice(test_seg_len, None),)
    elif segno == nsegs-1:
        train_index = (slice(0, -test_seg_len),)
    elif segno < 0 or segno >= nsegs:
        raise ValueError("segno=%r" % segno)
    else:
        train_index = (slice(0, test_seg_len * segno),
                       slice(test_seg_len * (segno + 1), None))

    y_train = tuple(y[..., i] for i in train_index)
    y_test = (y[..., test_index],)
    x_train = tuple(x[:, i] for i in train_index)
    x_test = (x[:, test_index],)

    return boost_segs(y_train, y_test, x_train, x_test, trf_length, delta,
                      mindelta, error)


def boost_segs(y_train, y_test, x_train, x_test, trf_length, delta, mindelta,
               error):
    """Boosting supporting multiple array segments

    Parameters
    ----------
    y_train, y_test : tuple of array (n_times,)
        Dependent signal, time series to predict.
    x_train, x_test : array (n_stims, n_times)
        Stimulus.
    trf_length : int
        Length of the TRF (in time samples).
    delta : scalar
        Step of the adjustment.
    mindelta : scalar
        Smallest delta to use. If no improvement can be found in an iteration,
        the first step is to divide delta in half, but stop if delta becomes
        smaller than ``mindelta``.
    error : str
        Error function to use.
    """
    error = ERROR_FUNC[error]
    n_stims = len(x_train[0])
    if any(len(x) != n_stims for x in chain(x_train, x_test)):
        raise ValueError("Not all x have same number of stimuli")
    n_times = [len(y) for y in chain(y_train, y_test)]
    if any(x.shape[1] != n for x, n in izip(chain(x_train, x_test), n_times)):
        raise ValueError("y and x have inconsistent number of time points")

    h = np.zeros((n_stims, trf_length))

    # buffers
    y_train_error = tuple(y.copy() for y in y_train)
    y_train_buf = tuple(np.empty(y.shape) for y in y_train)
    y_test_error = tuple(y.copy() for y in y_test)
    y_test_buf = tuple(np.empty(y.shape) for y in y_test)

    ys_error = y_train_error + y_test_error
    ys_delta = tuple(np.empty(y.shape) for y in ys_error)
    xs = x_train + x_test

    new_error = np.empty(h.shape)
    new_sign = np.empty(h.shape, np.int8)

    # history lists
    history = []
    test_error_history = []
    i_boost = 0
    while True:
        history.append(h.copy())

        # evaluate current h
        e_test = sum(error(y, buf) for y, buf in izip(y_test_error, y_test_buf))
        e_train = sum(error(y, buf) for y, buf in izip(y_train_error, y_train_buf))

        test_error_history.append(e_test)

        # stop the iteration if all the following requirements are met
        # 1. more than 10 iterations are done
        # 2. The testing error in the latest iteration is higher than that in
        #    the previous two iterations
        if i_boost > 10:
            if (test_error_history[-1] > test_error_history[-2] and
                    test_error_history[-1] > test_error_history[-3]):
                reason = "error(test) not improving in 2 steps"
                break
        else:
            i_boost += 1

        # generate possible movements -> training error
        for i_stim, i_time in product(xrange(h.shape[0]), xrange(h.shape[1])):
            # y_delta = change in y from delta change in h
            for yd, x in izip(ys_delta, x_train):
                yd[:i_time] = 0.
                yd[i_time:] = x[i_stim, :-i_time or None]
                yd *= delta

            # +/- delta
            e_add = 0
            e_sub = 0
            for y_err, dy, buf in izip(y_train_error, ys_delta, y_train_buf):
                # + delta
                np.subtract(y_err, dy, buf)
                e_add += error(buf, buf)
                # - delta
                np.add(y_err, dy, buf)
                e_sub += error(buf, buf)

            if e_add > e_sub:
                new_error[i_stim, i_time] = e_sub
                new_sign[i_stim, i_time] = -1
            else:
                new_error[i_stim, i_time] = e_add
                new_sign[i_stim, i_time] = 1

        i_stim, i_time = np.unravel_index(np.argmin(new_error), h.shape)
        new_train_error = new_error[i_stim, i_time]
        delta_signed = new_sign[i_stim, i_time] * delta

        # If no improvements can be found reduce delta
        if new_train_error > e_train:
            if delta < mindelta:
                reason = ("No improvement possible for training data, "
                          "stopping...")
                break
            else:
                delta *= 0.5
                # print("No improvement, new delta=%s..." % delta)
                continue

        # update h with best movement
        h[i_stim, i_time] += delta_signed

        # abort if we're moving in circles
        if len(history) >= 2 and np.array_equal(h, history[-2]):
            reason = "Same h after 2 iterations"
            break
        elif len(history) >= 3 and np.array_equal(h, history[-3]):
            reason = "Same h after 3 iterations"
            break

        # update error
        for err, yd, x in izip(ys_error, ys_delta, xs):
            yd[:i_time] = 0.
            yd[i_time:] = x[i_stim, :-i_time or None]
            yd *= delta_signed
            err -= yd

    else:
        reason = "maxiter exceeded"

    best_iter = np.argmin(test_error_history)

    # Keep predictive power as the correlation for the best iteration
    return (history[best_iter], test_error_history,
            reason + ' (%i iterations)' % len(test_error_history))


def apply_kernel(x, h, out=None):
    """Predict ``y`` by applying kernel ``h`` to ``x``

    x.shape is (n_stims, n_samples)
    h.shape is (n_stims, n_trf_samples)
    """
    if out is None:
        out = np.zeros(x.shape[1])
    else:
        out.fill(0)

    for ind in xrange(len(h)):
        out += np.convolve(h[ind], x[ind])[:len(out)]

    return out


def evaluate_kernel(y, x, h, error):
    """Fit quality statistics

    Returns
    -------
    r : float | array
        Pearson correlation.
    rank_r : float | array
        Spearman rank correlation.
    error : float | array
        Error corresponding to error_func.
    """
    y_pred = apply_kernel(x, h)

    # discard onset (length of kernel)
    i0 = h.shape[-1] - 1
    y = y[..., i0:]
    y_pred = y_pred[..., i0:]

    error_func = ERROR_FUNC[error]
    return (np.corrcoef(y, y_pred)[0, 1],
            spearmanr(y, y_pred)[0],
            error_func(y - y_pred))


# Error functions
def ss(error, buf=None):
    "Sum squared error"
    return np.dot(error, error[:, None])[0]


def ss_centered(error, buf=None):
    "Sum squared of the centered error"
    error = np.subtract(error, error.mean(), buf)
    return np.dot(error, error[:, None])[0]


def sum_abs(error, buf=None):
    "Sum of absolute error"
    return np.abs(error, buf).sum()


def sum_abs_centered(error, buf=None):
    "Sum of absolute centered error"
    error = np.subtract(error, error.mean(), buf)
    return np.abs(error, buf).sum()


ERROR_FUNC = {'l2': ss, 'l2centered': ss_centered,
              'l1': sum_abs, 'l1centered': sum_abs_centered}