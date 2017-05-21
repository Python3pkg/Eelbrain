# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from nose.tools import eq_, ok_
import numpy as np

from eelbrain import Factor, Var, datasets, plot


def test_barplot():
    "Test plot.Barplot"
    ds = datasets.get_uv()

    # one category
    plot.Barplot('fltvar', ds=ds, test=False, show=False)
    plot.Barplot('fltvar', ds=ds, show=False)
    plot.Barplot('fltvar', match='rm', ds=ds, show=False)

    # multiple categories
    plot.Barplot('fltvar', 'A%B', match='rm', ds=ds, show=False)
    plot.Barplot('fltvar', 'A%B', match='rm', ds=ds, pool_error=False,
                 show=False)
    plot.Barplot('fltvar', 'A%B', match='rm', test=0, ds=ds, show=False)

    # cells
    plot.Barplot('fltvar', 'A%B', cells=(('a2', 'b2'), ('a1', 'b1')), ds=ds, show=False)
    plot.Barplot('fltvar', 'A%B', match='rm', cells=(('a2', 'b2'), ('a1', 'b1')),
                 ds=ds, show=False)

    # Fixed top
    p = plot.Barplot('fltvar', 'A%B', ds=ds, top=2, test_markers=False,
                     show=False)
    ax = p._axes[0]
    eq_(ax.get_ylim()[1], 2)


def test_boxplot():
    "Test plot.Boxplot"
    ds = datasets.get_uv()
    plot.Boxplot('fltvar', 'A%B', match='rm', ds=ds, show=False)

    # one category
    plot.Boxplot('fltvar', ds=ds, test=False, show=False)
    plot.Boxplot('fltvar', ds=ds, show=False)
    plot.Boxplot('fltvar', match='rm', ds=ds, show=False)

    # cells
    plot.Boxplot('fltvar', 'A%B', cells=(('a2', 'b2'), ('a1', 'b1')), ds=ds, show=False)
    plot.Boxplot('fltvar', 'A%B', match='rm', cells=(('a2', 'b2'), ('a1', 'b1')),
                 ds=ds, show=False)

    # many pairwise significances
    ds['fltvar'][ds.eval("A%B==('a1','b1')")] += 1
    ds['fltvar'][ds.eval("A%B==('a2','b2')")] -= 1
    ds['C'] = Factor('qw', repeat=10, tile=4)
    plot.Boxplot('fltvar', 'A%B%C', ds=ds, show=False)

    # long labels
    ds['A'].update_labels({'a1': 'a very long label', 'a2': 'another very long label'})
    p = plot.Barplot('fltvar', 'A%B', ds=ds, show=False)
    labels = p._ax.get_xticklabels()
    bbs = [l.get_window_extent() for l in labels]
    for i in range(len(bbs) - 1):
        ok_(bbs[i].x1 < bbs[i + 1].x0)


def test_correlation():
    "Test plot.Correlation()"
    ds = datasets.get_uv()

    plot.Correlation('fltvar', 'fltvar2', ds=ds, show=False)
    plot.Correlation('fltvar', 'fltvar2', 'A', ds=ds, show=False)
    plot.Correlation('fltvar', 'fltvar2', 'A%B', ds=ds, show=False)


def test_histogram():
    "Test plot.Histogram"
    ds = datasets.get_uts()
    plot.Histogram('Y', 'A%B', ds=ds, show=False)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds, show=False)
    plot.Histogram('Y', 'A%B', match='rm', ds=ds, normed=True, show=False)


def test_regression():
    "Test plot.Regression"
    ds = datasets.get_uv()

    plot.Regression('fltvar', 'fltvar2', ds=ds, show=False)
    plot.Regression('fltvar', 'fltvar2', 'A', ds=ds, show=False)
    plot.Regression('fltvar', 'fltvar2', 'A%B', ds=ds, show=False)


def test_timeplot():
    "Test plot.Timeplot"
    ds = datasets.get_uts()
    ds['seq'] = Var(np.arange(2).repeat(30))
    plot.Timeplot('Y', 'B', 'seq', match='rm', ds=ds, show=False)
