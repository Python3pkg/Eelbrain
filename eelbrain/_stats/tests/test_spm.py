# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import pickle as pickle
from nose.tools import eq_
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import datasets
from eelbrain._stats.spm import LM, RandomLM


def test_lm():
    ds = datasets.get_uts()
    model = ds.eval("A*B*Y")
    coeffs = ds['uts'].ols(model)

    lm = LM('uts', 'A*B*Y', ds, 'effect')
    for i, effect in enumerate(model.effects):
        assert_array_equal(lm._coefficient(effect.name), coeffs.x[i: i+1])


def test_random_lm():
    # dummy coding
    ds = datasets.get_uts()
    lms = []
    for i in range(5):
        ds['uts'].x += np.random.normal(0, 2, ds['uts'].shape)
        lms.append(LM('uts', 'A*B*Y', ds))
    rlm = RandomLM(lms)
    res = rlm.column_ttest('A x B', samples=100, pmin=0.05, mintime=0.025)
    eq_(res.clusters.n_cases, 1)

    # effect coding
    ds = datasets.get_uts()
    lms = []
    for i in range(5):
        ds['uts'].x += np.random.normal(0, 2, ds['uts'].shape)
        lms.append(LM('uts', 'A*B*Y', ds, 'effect'))
    rlm = RandomLM(lms)
    res = rlm.column_ttest('A x B', samples=100, pmin=0.05, mintime=0.025)
    eq_(res.clusters.n_cases, 6)

    # persistence
    rlm_p = pickle.loads(pickle.dumps(rlm, pickle.HIGHEST_PROTOCOL))
    eq_(rlm_p.dims, rlm.dims)
