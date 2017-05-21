# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from itertools import repeat
import warnings

from nose.tools import (eq_, assert_almost_equal, assert_is_instance,
    assert_raises, nottest)
import numpy as np
from numpy import newaxis
from numpy.testing import assert_allclose

from eelbrain import datasets, test, testnd, Dataset, NDVar
from eelbrain._data_obj import UTS
from eelbrain._stats import glm
from eelbrain._stats.permutation import permute_order


def r_require(package):
    from rpy2.rinterface import RRuntimeWarning
    from rpy2.robjects import r

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RRuntimeWarning)
        success = r('require(%s)' % package)[0]
    
    if not success:
        print(r("install.packages('%s', repos='http://cran.us.r-project.org')"
                % package))
        success = r('require(%s)' % package)[0]
        if not success:
            raise RuntimeError("Could not install R package %r" % package)


@nottest
def assert_f_test_equal(f_test, r_res, r_row, f_lmf, f_nd, r_kind='aov'):
    if r_kind in ('aov', 'rmaov'):
        r_res = {'df': r_res[0][r_row], 'SS': r_res[1][r_row],
                 'MS': r_res[2][r_row], 'F': r_res[3][r_row],
                 'p': r_res[4][r_row]}
    elif r_kind == 'ez':
        pass
    elif r_kind == 'Anova':
        r_res = {'df': r_res[1][r_row], 'SS': r_res[0][r_row],
                 'F': r_res[2][r_row], 'p': r_res[3][r_row]}
    else:
        raise ValueError("invalid r_kind=%r" % r_kind)

    eq_(f_test.df, r_res['df'])
    if 'SS' is r_res:
        assert_almost_equal(f_test.SS, r_res['SS'])
    if 'MS' in r_res:
        assert_almost_equal(f_test.MS, r_res['MS'])
    assert_almost_equal(f_test.F, r_res['F'])
    assert_almost_equal(f_test.p, r_res['p'])
    assert_almost_equal(f_lmf, r_res['F'])  # lm-fitter comparison"
    assert_almost_equal(f_nd, r_res['F'])  # nd-anova comparison"


@nottest
def assert_f_tests_equal(f_tests, r_res, fs, fnds, r_kind='aov'):
    if r_kind == 'ez':
        r_results = []
        for f_test in f_tests:
            f_test_name = set(f_test.name.split(' x '))
            for i, r_name in enumerate(r_res[0][0]):
                if set(r_name.split(':')) == f_test_name:
                    r_results.append({'df': r_res[0][1][i], 'F': r_res[0][3][i],
                                      'p': r_res[0][4][i]})
                    break
            else:
                raise RuntimeError("Effect %s not in ezANOVA" % f_test_name)
    elif r_kind == 'rmaov':
        r_results = [r_res[i][0] for i in range(len(f_tests))]
    else:
        r_results = repeat(r_res, len(f_tests))

    r_row = 0
    for i, r_res in enumerate(r_results):
        if r_kind != 'rmaov':
            r_row = i
        assert_f_test_equal(f_tests[i], r_res, r_row, fs[i], fnds[i], r_kind)


def run_on_lm_fitter(y, x, ds):
    y = ds.eval(y)
    y = y.x[:, newaxis]
    y = np.hstack((y, y))
    x = ds.eval(x)
    fitter = glm._nd_anova(x)
    fmaps = fitter.map(y)
    fs = fmaps[:, 0]
    return fs


def run_as_ndanova(y, x, ds):
    yt = ds.eval(y).x[:, None]
    y2 = np.concatenate((yt, yt * 2), 1)
    ndvar = NDVar(y2, ('case', UTS(0, 0.1, 2)))
    res = testnd.anova(ndvar, x, ds=ds)
    f1 = [fmap.x[0] for fmap in res.f]
    f2 = [fmap.x[1] for fmap in res.f]
    for f1_, f2_ in zip(f1, f2):
        eq_(f1_, f2_)
    return f1


def test_anova():
    "Test ANOVA"
    from rpy2.robjects import r
    r_require('car')
    r_require('ez')

    ds = datasets.get_uv(nrm=True)
    ds.to_r('ds')

    # fixed effects
    aov = test.ANOVA('fltvar', 'A*B', ds=ds)
    print(aov)
    fs = run_on_lm_fitter('fltvar', 'A*B', ds)
    fnds = run_as_ndanova('fltvar', 'A*B', ds)
    r_res = r("Anova(lm(fltvar ~ A * B, ds, type=2))")
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'Anova')

    # random effect
    aov = test.ANOVA('fltvar', 'A*B*rm', ds=ds)
    print(aov)
    fs = run_on_lm_fitter('fltvar', 'A*B*rm', ds)
    fnds = run_as_ndanova('fltvar', 'A*B*rm', ds)
    r('test.aov <- aov(fltvar ~ A * B + Error(rm / (A * B)), ds)')
    print(r('test.summary <- summary(test.aov)'))
    r_res = r['test.summary'][1:]
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'rmaov')

    # nested random effect
    aov = test.ANOVA('fltvar', 'B + A%B + A * nrm(B)', ds=ds)
    print(aov)
    fs = run_on_lm_fitter('fltvar', 'B + A%B + A * nrm(B)', ds)
    fnds = run_as_ndanova('fltvar', 'B + A%B + A * nrm(B)', ds)
    r_res = r('ezANOVA(ds, fltvar, nrm, A, between=B)')
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'ez')

    # not fully specified model with random effects
    assert_raises(NotImplementedError, test.anova, 'fltvar', 'A*rm', ds=ds)

    # empty cells
    dss = ds.sub("A%B != ('a2', 'b2')")
    assert_raises(NotImplementedError, test.anova, 'fltvar', 'A*B', ds=dss)
    assert_raises(NotImplementedError, run_on_lm_fitter, 'fltvar', 'A*B', ds=dss)
    dss = ds.sub("A%B != ('a1', 'b1')")
    assert_raises(NotImplementedError, test.anova, 'fltvar', 'A*B', ds=dss)
    assert_raises(NotImplementedError, run_on_lm_fitter, 'fltvar', 'A*B', ds=dss)


def test_ndanova():
    ds = datasets.get_uts(nrm=True)
    ds['An'] = ds['A'].as_var({'a0': 0, 'a1': 1})

    assert_raises(NotImplementedError, testnd.anova, 'uts', 'An*B*rm', ds=ds)

    # nested random effect
    res = testnd.anova('uts', 'A + A%B + B * nrm(A)', ds=ds, match='nrm',
                       samples=100, pmin=0.05)
    eq_(len(res.find_clusters(0.05)), 8)


def test_anova_perm():
    "Test permutation argument for ANOVA"
    ds = datasets.get_uts()
    y = ds['uts'].x
    y_perm = np.empty_like(y)
    n_cases, n_tests = y.shape

    # balanced anova
    aov = glm._BalancedFixedNDANOVA(ds.eval('A*B'))
    r1 = aov.preallocate(y.shape)
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_allclose(r2, r1, 1e-6, 1e-6)

    # full repeated measures anova
    aov = glm._FullNDANOVA(ds.eval('A*B*rm'))
    r1 = aov.preallocate(y.shape)
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_allclose(r2, r1, 1e-6, 1e-6)

    # incremental anova
    ds = ds[1:]
    y = ds['uts'].x
    y_perm = np.empty_like(y)
    n_cases, n_tests = y.shape
    aov = glm._IncrementalNDANOVA(ds.eval('A*B'))
    r1 = aov.preallocate(y.shape)
    for perm in permute_order(n_cases, 2):
        aov.map(y, perm)
        r2 = r1.copy()
        y_perm[perm] = y
        aov.map(y_perm)
        assert_allclose(r2, r1, 1e-6, 1e-6)


def test_anova_r_adler():
    """Test ANOVA accuracy by comparing with R (Adler dataset of car package)

    An unbalanced 3 by 2 independent measures design.
    """
    from rpy2.robjects import r

    # "Adler" dataset
    r_require('car')
    ds = Dataset.from_r('Adler')

    # with balanced data
    dsb = ds.equalize_counts('expectation % instruction')
    dsb.to_r('AdlerB')
    aov = test.ANOVA('rating', 'instruction * expectation', ds=dsb)
    fs = run_on_lm_fitter('rating', 'instruction * expectation', dsb)
    fnds = run_as_ndanova('rating', 'instruction * expectation', dsb)
    print(r('a.aov <- aov(rating ~ instruction * expectation, AdlerB)'))
    print(r('a.summary <- summary(a.aov)'))
    r_res = r['a.summary'][0]
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds)

    # with unbalanced data; for Type II SS use car package
    aov = test.ANOVA('rating', 'instruction * expectation', ds=ds)
    fs = run_on_lm_fitter('rating', 'instruction * expectation', ds)
    fnds = run_as_ndanova('rating', 'instruction * expectation', ds)
    r_res = r("Anova(lm(rating ~ instruction * expectation, Adler, type=2))")
    assert_f_tests_equal(aov.f_tests, r_res, fs, fnds, 'Anova')

    # single predictor
    aov = test.ANOVA('rating', 'instruction', ds=ds)
    fs = run_on_lm_fitter('rating', 'instruction', ds)
    fnds = run_as_ndanova('rating', 'instruction', ds)
    r_res = r("Anova(lm(rating ~ instruction, Adler, type=2))")
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0], 'Anova')


def test_anova_r_sleep():
    "Test ANOVA accuracy by comparing with R (sleep dataset)"
    from rpy2.robjects import r

    # "sleep" dataset
    print(r('data(sleep)'))
    ds = Dataset.from_r('sleep')
    ds['ID'].random = True

    # independent measures
    aov = test.ANOVA('extra', 'group', ds=ds)
    fs = run_on_lm_fitter('extra', 'group', ds)
    fnds = run_as_ndanova('extra', 'group', ds)
    print(r('sleep.aov <- aov(extra ~ group, sleep)'))
    print(r('sleep.summary <- summary(sleep.aov)'))
    r_res = r['sleep.summary'][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0])

    # repeated measures
    aov = test.ANOVA('extra', 'group * ID', ds=ds)
    fs = run_on_lm_fitter('extra', 'group * ID', ds)
    fnds = run_as_ndanova('extra', 'group * ID', ds)
    print(r('sleep.aov <- aov(extra ~ group + Error(ID / group), sleep)'))
    print(r('sleep.summary <- summary(sleep.aov)'))
    r_res = r['sleep.summary'][1][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0])

    # unbalanced (independent measures)
    ds2 = ds[1:]
    print(r('sleep2 <- subset(sleep, (group == 2) | (ID != 1))'))
    aov = test.ANOVA('extra', 'group', ds=ds2)
    fs = run_on_lm_fitter('extra', 'group', ds2)
    fnds = run_as_ndanova('extra', 'group', ds2)
    print(r('sleep2.aov <- aov(extra ~ group, sleep2)'))
    print(r('sleep2.summary <- summary(sleep2.aov)'))
    r_res = r['sleep2.summary'][0]
    assert_f_test_equal(aov.f_tests[0], r_res, 0, fs[0], fnds[0])


def test_lmfitter():
    "Test the _nd_anova class"
    ds = datasets.get_uts()

    # independent, residuals vs. Hopkins
    y = ds['uts'].x

    x = ds.eval("A * B")
    lm = glm._nd_anova(x)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    x_full = ds.eval("A * B + ind(A%B)")
    lm_full = glm._nd_anova(x_full)
    assert_is_instance(lm_full, glm._FullNDANOVA)
    f_maps_full = lm_full.map(y)
    p_maps_full = lm_full.p_maps(f_maps)

    for f, f_full in zip(f_maps, f_maps_full):
        assert_allclose(f, f_full)
    for p, p_full in zip(p_maps, p_maps_full):
        assert_allclose(p, p_full)

    # repeated measures
    x = ds.eval("A * B * rm")
    lm = glm._nd_anova(x)
    f_maps = lm.map(y)
    p_maps = lm.p_maps(f_maps)

    aov = test.ANOVA(y[:, 0], x)
    for f_test, f_map, p_map in zip(aov.f_tests, f_maps, p_maps):
        assert_almost_equal(f_map[0], f_test.F)
        assert_almost_equal(p_map[0], f_test.p)
