# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import logging

from nose.tools import eq_, ok_, assert_raises
import numpy as np

from eelbrain import Factor, Var
from eelbrain._stats.permutation import resample, permute_sign_flip


def test_permutation():
    """Test permutation"""
    v = Var(np.arange(6))
    res = np.empty((5, 6))
    for i, y in enumerate(resample(v, samples=5)):
        res[i] = y.x
    logging.info('Standard Permutation:\n%s' % res)

    # with unit
    s = Factor('abc', tile=2)
    for i, y in enumerate(resample(v, samples=5, unit=s)):
        res[i] = y.x
    logging.info('Permutation with Unit:\n%s' % res)

    # check we have only appropriate cells
    cols = [np.unique(res[:, i]) for i in range(res.shape[1])]
    for i in range(3):
        eq_(len(np.setdiff1d(cols[i], [i, i + 3])), 0)
    for i in range(3, 6):
        eq_(len(np.setdiff1d(cols[i], [i, i - 3])), 0)

    # check we have some variability
    eq_(max(list(map(len, cols))), 2)


def test_permutation_sign_flip():
    "Test permute_sign_flip()"
    res = np.empty((2 ** 6 - 1, 6), dtype=np.int8)
    for i, sign in enumerate(permute_sign_flip(6, samples=-1)):
        res[i] = sign
    eq_(i, 2 ** 6 - 2)
    logging.info('Permutation with sign_flip:\n%s' % res)
    ok_(np.all(res.min(1) < 0), "Not all permutations have a sign flip")
    for i, row in enumerate(res):
        eq_(np.any(np.all(row == res[:i], 1)), False)

    assert_raises(NotImplementedError, permute_sign_flip(63).__next__)
