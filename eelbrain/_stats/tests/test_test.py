# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>


from nose.tools import eq_, assert_almost_equal
from numpy.testing import assert_array_equal
import scipy.stats

from eelbrain import datasets, test
from eelbrain.fmtxt import asfmtext
from eelbrain._stats import test as _test


def test_correlations():
    "Test test.correlations()"
    ds = datasets.get_uv()

    res = test.correlations('fltvar', 'fltvar2', ds=ds)
    print(res)
    eq_(str(res[2][0]).strip(), '.398')
    res = test.correlations('fltvar', 'fltvar2', ds=ds, asds=True)
    assert_almost_equal(res[0, 'r'], .398, 3)

    res = test.correlations('fltvar', 'fltvar2', 'A', ds=ds)
    print(res)
    eq_(str(res[2][0]).strip(), 'a1')
    eq_(str(res[2][1]).strip(), '-0.149')
    eq_(str(res[3][1]).strip(), '.740')
    res = test.correlations('fltvar', 'fltvar2', 'A', ds=ds, asds=True)
    eq_(res[0, 'A'], 'a1')
    assert_almost_equal(res[0, 'r'], -0.149, 3)
    assert_almost_equal(res[1, 'r'], .740, 3)

    res = test.correlations('fltvar', 'fltvar2', 'A%B', ds=ds)
    print(res)
    eq_(str(res[2][2]).strip(), '-0.276')
    res = test.correlations('fltvar', 'fltvar2', 'A%B', ds=ds, asds=True)
    assert_almost_equal(res[0, 'r'], -0.276, 3)

    res = test.correlations('fltvar', ('fltvar2', 'intvar'), 'A%B', ds=ds)
    print(res)
    eq_(str(res[2][1]).strip(), 'a1')
    eq_(str(res[2][2]).strip(), 'b1')
    eq_(str(res[2][3]).strip(), '-0.276')
    res = test.correlations('fltvar', ('fltvar2', 'intvar'), 'A%B', ds=ds, asds=True)
    assert_almost_equal(res[0, 'r'], -0.276, 3)


def test_star():
    "Test the star function"
    assert_array_equal(_test.star([0.1, 0.04, 0.01], int), [0, 1, 2])
    assert_array_equal(_test.star([0.001, 0.04, 0.1], int), [3, 1, 0])


def test_ttest():
    """Test test.ttest()"""
    ds = datasets.get_uv()

    print(test.ttest('fltvar', ds=ds))
    print(test.ttest('fltvar', 'A', ds=ds))
    print(test.ttest('fltvar', 'A%B', ds=ds))
    print(test.ttest('fltvar', 'A', match='rm', ds=ds))
    print(test.ttest('fltvar', 'A', 'a1', match='rm', ds=ds))
    print(test.ttest('fltvar', 'A%B', ('a1', 'b1'), match='rm', ds=ds))


def test_ttest_rel():
    """Test test.TTestRel()"""
    ds = datasets.get_uv()

    res = test.TTestRel('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds)
    i1 = ds.eval("logical_and(A=='a1', B=='b1')")
    i2 = ds.eval("logical_and(A=='a2', B=='b1')")
    a1 = ds[i1, 'fltvar'].x
    a2 = ds[i2, 'fltvar'].x
    diff = a1 - a2
    t, p = scipy.stats.ttest_rel(a1, a2)
    assert_array_equal(res.diff.x, diff)
    eq_(res.df, len(a1) - 1)
    eq_(res.tail, 0)
    assert_almost_equal(res.t, t, 10)
    assert_almost_equal(res.p, p, 10)
    print(res)
    print(asfmtext(res))

    res = test.TTestRel('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds, 1)
    assert_array_equal(res.diff.x, diff)
    eq_(res.df, len(a1) - 1)
    eq_(res.tail, 1)
    assert_almost_equal(res.t, t, 10)
    assert_almost_equal(res.p, p / 2 if t > 0 else 1 - p / 2, 10)
    print(res)
    print(asfmtext(res))

    res = test.TTestRel('fltvar', 'A', 'a1', 'a2', 'rm', "B=='b1'", ds, -1)
    assert_array_equal(res.diff.x, diff)
    eq_(res.df, len(a1) - 1)
    eq_(res.tail, -1)
    assert_almost_equal(res.t, t, 10)
    assert_almost_equal(res.p, p / 2 if t < 0 else 1 - p / 2, 10)
    print(res)
    print(asfmtext(res))
