# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from copy import deepcopy
from itertools import product
from operator import (
    add, iadd, sub, isub, mul, imul, div, idiv, floordiv, ifloordiv, mod, imod)
import os
import pickle as pickle
import shutil
from string import ascii_lowercase
import tempfile
import warnings

import mne
from nose.tools import (
    eq_, ok_, assert_almost_equal, assert_false, assert_is_instance,
    assert_raises, assert_not_equal, nottest)
import numpy as np
from numpy.testing import (
    assert_equal, assert_array_equal, assert_allclose,
    assert_array_almost_equal)

from eelbrain import (
    datasets, load, Var, Factor, NDVar, Datalist, Dataset, Celltable, align,
    align1, choose, combine, cwt_morlet, shuffled_index)
from eelbrain._data_obj import (
    all_equal, asvar, assub, full_slice, longname, Categorial, Sensor,
    SourceSpace, UTS, DimensionMismatchError, assert_has_no_empty_cells)
from eelbrain._stats.stats import rms
from eelbrain._utils.testing import (
    assert_dataobj_equal, assert_dataset_equal, assert_source_space_equal,
    requires_mne_sample_data, skip_on_windows)


OPERATORS = ((add, iadd, '+'),
             (sub, isub, '-'),
             (mul, imul, '*'),
             (div, idiv, '/'),
             (floordiv, ifloordiv, '//'),
             (mod, imod, '%'))


def test_aggregate():
    "Test aggregation methods"
    ds = datasets.get_uts()

    # don't handle inconsistencies silently
    assert_raises(ValueError, ds.aggregate, 'A%B')

    dsa = ds.aggregate('A%B', drop_bad=True)
    assert_array_equal(dsa['n'], [15, 15, 15, 15])
    idx1 = ds.eval("logical_and(A=='a0', B=='b0')")
    eq_(dsa['Y', 0], ds['Y', idx1].mean())

    # unequal cell counts
    ds = ds[:-3]
    dsa = ds.aggregate('A%B', drop_bad=True)
    assert_array_equal(dsa['n'], [15, 15, 15, 12])
    idx1 = ds.eval("logical_and(A=='a0', B=='b0')")
    eq_(dsa['Y', 0], ds['Y', idx1].mean())

    # equalize count
    dsa = ds.aggregate('A%B', drop_bad=True, equal_count=True)
    assert_array_equal(dsa['n'], [12, 12, 12, 12])
    idx1_12 = np.logical_and(idx1, idx1.cumsum() <= 12)
    eq_(dsa['Y', 0], ds['Y', idx1_12].mean())

    # equalize count with empty cell
    sds = ds.sub("logical_or(A == 'a1', B == 'b1')")
    dsa = sds.aggregate('A%B', drop_bad=True, equal_count=True)
    assert_array_equal(dsa['n'], [12, 12, 12])


def test_align():
    "Testing align() and align1() functions"
    ds = datasets.get_uv()
    # index the dataset
    ds.index()
    ds['aindex'] = ds.eval("A.enumerate_cells()")
    # subset
    idx4 = np.arange(0, ds.n_cases, 4)
    idx4i = idx4[::-1]
    ds2 = ds.sub(np.arange(0, ds.n_cases, 2))
    # shuffle the whole dataset
    shuffle_index = np.arange(ds.n_cases)
    np.random.shuffle(shuffle_index)
    ds_shuffled = ds[shuffle_index]

    # align1: align Dataset to index
    dsa = align1(ds2, idx4)
    assert_array_equal(dsa['index'], idx4, "align1() failure")
    dsa = align1(ds2, idx4i)
    assert_array_equal(dsa['index'], idx4i, "align1() failure")
    # d_idx as Var
    dsa = align1(ds2[::2], idx4, idx4i)
    assert_array_equal(dsa['index'], idx4i, "align1() failure")
    assert_raises(ValueError, align1, ds2, idx4, idx4i)

    # Factor index
    assert_raises(ValueError, align1, ds, ds['rm', ::-1], 'rm')
    fds = ds[:20]
    dsa = align1(fds, fds['rm', ::-1], 'rm')
    assert_array_equal(dsa['index'], np.arange(19, -1, -1), "align1 Factor")

    # align two datasets
    dsa1, dsa2 = align(ds, ds2)
    assert_array_equal(dsa1['index'], dsa2['index'], "align() failure")
    dsa1, dsa2 = align(ds, ds2[::-1])
    assert_array_equal(dsa1['index'], dsa2['index'], "align() failure")
    dsa1, dsa2 = align(ds, ds_shuffled)
    assert_dataset_equal(dsa1, dsa2)

    # align using categorial
    dsa1, dsa2 = align(ds, ds_shuffled, 'A % aindex')
    assert_dataset_equal(dsa1, dsa2)
    dsa1, dsa2 = align(ds, ds_shuffled, 'aindex % A')
    assert_dataset_equal(dsa1, dsa2)


def test_celltable():
    "Test the Celltable class."
    ds = datasets.get_uts()
    ds['cat'] = Factor('abcd', repeat=15)

    ct = Celltable('Y', 'A', ds=ds)
    eq_(ct.n_cases, 60)
    eq_(ct.n_cells, 2)
    eq_(repr(ct), "Celltable(Y, A)")
    eq_(repr(Celltable(ds['Y'].x, 'A', ds=ds)), "Celltable(<ndarray>, A)")
    eq_(repr(Celltable(ds['Y'].x, ds['A'].x, ds=ds)),
        "Celltable(<ndarray>, <Factor>)")

    ct = Celltable('Y', 'A', match='rm', ds=ds)
    eq_(ct.n_cases, 30)
    eq_(ct.n_cells, 2)

    # cat argument
    ct = Celltable('Y', 'cat', cat=('c', 'b'), ds=ds)
    eq_(ct.n_cases, 30)
    eq_(ct.X[0], 'c')
    eq_(ct.X[-1], 'b')
    assert_raises(ValueError, Celltable, 'Y', 'cat', cat=('c', 'e'), ds=ds)

    ct = Celltable('Y', 'A', match='rm', ds=ds)
    eq_(ct.n_cases, 30)
    assert np.all(ct.groups['a0'] == ct.groups['a1'])

    ct = Celltable('Y', 'cat', match='rm', cat=('c', 'b'), ds=ds)
    eq_(ct.n_cases, 30)
    eq_(ct.X[0], 'c')
    eq_(ct.X[-1], 'b')

    # catch unequal length
    assert_raises(ValueError, Celltable, ds['Y', :-1], 'cat', ds=ds)
    assert_raises(ValueError, Celltable, ds['Y', :-1], 'cat', match='rm', ds=ds)

    # coercion of numerical X
    X = ds.eval("A == 'a0'")
    ct = Celltable('Y', X, cat=(None, None), ds=ds)
    eq_(('False', 'True'), ct.cat)
    assert_array_equal(ct.data['True'], ds['Y', X])

    ct = Celltable('Y', X, cat=('True', 'False'), ds=ds)
    eq_(('True', 'False'), ct.cat)
    assert_array_equal(ct.data['True'], ds['Y', X])

    # test coercion of Y
    ct = Celltable(ds['Y'].x, 'A', ds=ds)
    assert_is_instance(ct.Y, np.ndarray)
    ct = Celltable(ds['Y'].x, 'A', ds=ds, coercion=asvar)
    assert_is_instance(ct.Y, Var)

    # test sub
    ds_sub = ds.sub("A == 'a0'")
    ct_sub = Celltable('Y', 'B', ds=ds_sub)
    ct = Celltable('Y', 'B', sub="A == 'a0'", ds=ds)
    assert_dataobj_equal(ct_sub.Y, ct.Y)

    # test sub with rm
    ct_sub = Celltable('Y', 'B', match='rm', ds=ds_sub)
    ct = Celltable('Y', 'B', match='rm', sub="A == 'a0'", ds=ds)
    assert_dataobj_equal(ct_sub.Y, ct.Y)

    # Interaction match
    ct = Celltable('Y', 'A', match='B % rm', ds=ds)
    ok_(ct.all_within)
    assert_dataobj_equal(combine((ct.data['a0'], ct.data['a1'])), ds['Y'])

    # test rm sorting
    ds = Dataset()
    ds['rm'] = Factor('abc', repeat=4)
    ds['Y'] = Var(np.arange(3.).repeat(4))
    ds['X'] = Factor('ab', repeat=2, tile=3)
    idx = np.arange(12)
    np.random.shuffle(idx)
    ds = ds[idx]
    ct = Celltable('Y', 'X', 'rm', ds=ds)
    assert_array_equal(ct.match, Factor('abc', tile=2))
    assert_array_equal(ct.Y, np.tile(np.arange(3.), 2))
    assert_array_equal(ct.X, Factor('ab', repeat=3))


def test_coercion():
    "Test data class coercion"
    ds = datasets.get_uts()
    ds['avar'] = Var.from_dict(ds['A'], {'a0': 0, 'a1': 1})

    assert_array_equal(assub("A == 'a0'", ds), ds['A'] == 'a0')
    assert_array_equal(assub("avar == 0", ds), ds['avar'] == 0)
    with warnings.catch_warnings():  # element-wise comparison
        warnings.simplefilter("ignore")
        assert_raises(TypeError, assub, "avar == '0'", ds)


def test_choose():
    "Test choose()"
    ds = datasets.get_uts(True)[::4]
    utsnd = ds['utsnd']
    utsnd2 = utsnd + 1.
    idx = ds['B'] == 'b0'
    idxi = np.invert(idx)

    y = choose(idx, (utsnd, utsnd2))
    assert_array_equal(y.x[idx], utsnd2.x[idx])
    assert_array_equal(y.x[idxi], utsnd.x[idxi])

    assert_raises(DimensionMismatchError, choose, idx, (utsnd, utsnd.sub(sensor='1')))


def test_combine():
    "Test combine()"
    ds1 = datasets.get_uts()
    ds2 = datasets.get_uts()
    n = ds1.n_cases
    ds = combine((ds1, ds2))
    assert_array_equal(ds2['Y'].x, ds['Y'].x[n:])

    # combine Datasets with unequal keys
    del ds1['Y']
    # raise
    assert_raises(KeyError, combine, (ds1, ds2))
    assert_raises(KeyError, combine, (ds2, ds1))
    # drop
    del ds2['YCat']
    ds = combine((ds1, ds2), incomplete='drop')
    ok_('Y' not in ds)
    ok_('YCat' not in ds)
    # fill in
    ds = combine((ds1, ds2), incomplete='fill in')
    assert_array_equal(ds['Y'].x[n:], ds2['Y'].x)
    assert_array_equal(np.isnan(ds['Y'].x[:n]), True)
    assert_array_equal(ds['YCat'][:n], ds1['YCat'])
    assert_array_equal(ds['YCat'][n:], '')

    # invalid input
    assert_raises(ValueError, combine, ())
    assert_raises(TypeError, combine, (ds2['A'], ds2['Y']))

    # combine NDVar with unequel dimensions
    ds = datasets.get_uts(utsnd=True)
    y = ds['utsnd']
    y1 = y.sub(sensor=['0', '1', '2', '3'])
    y2 = y.sub(sensor=['1', '2', '3', '4'])
    ds1 = Dataset((y1,), info={'a': np.arange(2), 'b': [np.arange(2)]})
    ds2 = Dataset((y2,), info={'a': np.arange(2), 'b': [np.arange(2)]})
    dsc = combine((ds1, ds2))
    y = dsc['utsnd']
    eq_(list(y.sensor.names), ['1', '2', '3'], "Sensor dimension intersection")
    dims = ('case', 'sensor', 'time')
    ref = np.concatenate((y1.get_data(dims)[:, 1:], y2.get_data(dims)[:, :3]))
    assert_array_equal(y.get_data(dims), ref, "combine utsnd")
    # info
    assert_array_equal(dsc.info['a'], np.arange(2))
    eq_(len(dsc.info['b']), 1)
    assert_array_equal(dsc.info['b'][0], np.arange(2))


def test_datalist():
    "Test Datalist class"
    dl = Datalist(list(range(10)))

    # indexing
    eq_(dl[3], 3)
    x = dl[:3]
    assert_is_instance(x, Datalist)
    assert_array_equal(x, list(range(3)))
    assert_array_equal(dl[8:], list(range(8, 10)))
    x = dl[np.arange(10) < 3]
    assert_is_instance(x, Datalist)
    assert_array_equal(x, list(range(3)))
    assert_array_equal(dl[np.arange(3)], list(range(3)))

    # __add__
    x = dl + list(range(10, 12))
    assert_is_instance(x, Datalist)
    assert_array_equal(x, list(range(12)))

    # aggregate
    x = dl.aggregate(Factor('ab', repeat=5))
    assert_is_instance(x, Datalist)
    assert_array_equal(x, [2.0, 7.0])

    # repr
    dl = Datalist([['a', 'b'], [], ['a']])
    eq_(str(dl), "[['a', 'b'], [], ['a']]")
    dl = Datalist([['a', 'b'], [], ['a']], fmt='strlist')
    eq_(str(dl), '[[a, b], [], [a]]')
    eq_(str(dl[:2]), '[[a, b], []]')

    # eq
    a = Datalist([[], [1], [], [1]])
    b = Datalist([[], [], [2], [1]])
    assert_array_equal(a == b, [True, False, False, True])
    assert_array_equal(a != b, [False, True, True, False])

    # deepcopy
    ac = deepcopy(a)
    ok_(ac is not a)
    assert_array_equal(ac, a)
    ac[0].append(1)
    assert_array_equal(ac == a, [False, True, True, True])

    # __setitem__
    ac[:2] = (1, 2)
    assert_array_equal(ac == [1, 2, [], [1]], True)
    ac[np.arange(2, 4)] = [3, 4]
    assert_array_equal(ac == list(range(1, 5)), True)
    assert_raises(ValueError, ac.__setitem__, np.arange(2), np.arange(3))

    # update
    a._update_listlist(b)
    assert_array_equal(a, [[], [1], [2], [1]])


def test_dataset():
    "Basic dataset operations"
    ds = Dataset()
    # naming
    ds['f'] = Factor('abab')
    eq_(ds['f'].name, 'f')

    # ds.add()
    assert_raises(ValueError, ds.add, Factor('aabb'))
    ds.add(Factor('aabb', name='g'))
    eq_(ds['g'].name, 'g')

    # ds.update()
    ds = Dataset()
    ds.update({'f': Factor('abab')})
    eq_(ds['f'].name, 'f')


def test_dataset_combining():
    "Test Dataset combination methods"
    ds = datasets.get_uv()
    del ds['fltvar'], ds['intvar'], ds['A']

    ds2 = datasets.get_uv()
    del ds2['fltvar'], ds2['intvar']
    ds.update(ds2)
    assert_array_equal(ds['A'], ds2['A'])

    ds2 = datasets.get_uv()
    del ds2['fltvar'], ds2['intvar']
    ds2['B'][5] = 'something_else'
    del ds['A']
    assert_raises(ValueError, ds.update, ds2)


def test_dataset_indexing():
    """Test Dataset indexing"""
    ds = datasets.get_uv()

    # indexing values
    eq_(ds['A', 1], ds['A'][1])
    eq_(ds[1, 'A'], ds['A'][1])

    # indexing variables
    assert_dataobj_equal(ds[:, 'A'], ds['A'])
    assert_dataobj_equal(ds['A', :], ds['A'])
    assert_dataobj_equal(ds[:10, 'A'], ds['A'][:10])
    assert_dataobj_equal(ds['A', :10], ds['A'][:10])

    # new Dataset through indexing
    ds2 = Dataset()
    ds2['A'] = ds['A']
    assert_dataset_equal(ds[('A',)], ds2)
    ds2['B'] = ds['B']
    assert_dataset_equal(ds['A', 'B'], ds2)
    assert_dataset_equal(ds[('A', 'B'), :10], ds2[:10])
    assert_dataset_equal(ds[:10, ('A', 'B')], ds2[:10])

    # assigning value
    ds[2, 'A'] = 'hello'
    eq_(ds[2, 'A'], 'hello')
    ds['A', 2] = 'not_hello'
    eq_(ds[2, 'A'], 'not_hello')

    # assigning new factor
    ds['C', :] = 'c'
    ok_(np.all(ds.eval("C == 'c'")))

    # assigning new Var
    ds['D1', :] = 5.
    ds[:, 'D2'] = 5.
    assert_array_equal(ds['D1'], 5)
    assert_array_equal(ds['D2'], 5)

    # test illegal names
    f = Factor('aaabbb')
    assert_raises(ValueError, ds.__setitem__, '%dsa', f)
    assert_raises(ValueError, ds.__setitem__, '432', f)
    assert_raises(ValueError, ds.__setitem__, ('%dsa', slice(None)), 'value')
    assert_raises(ValueError, ds.__setitem__, (slice(None), '%dsa'), 'value')
    assert_raises(ValueError, ds.__setitem__, ('432', slice(None)), 4.)
    assert_raises(ValueError, ds.__setitem__, (slice(None), '432'), 4.)

    # deleting items
    del ds['A']
    ok_('A' not in ds)
    assert_raises(KeyError, ds.__getitem__, 'A')
    del ds['B', 'rm']
    ok_('B' not in ds and 'rm' not in ds)


def test_dataset_repr():
    "Test Dataset string representation methods"
    ds = datasets.get_uts()

    print(ds)
    print(repr(ds))

    eq_(str(ds.head()), str(ds[:10]))
    eq_(str(ds.tail()), str(ds[-10:]))

    print(ds['A'])
    print(repr(ds['A']))

    print(ds['Y'])
    print(repr(ds['Y']))

    print(ds['uts'])
    print(repr(ds['uts']))


def test_dataset_sorting():
    "Test Dataset sorting methods"
    test_array = np.arange(10)
    ds = Dataset()
    ds['v'] = Var(test_array)
    ds['f'] = Factor(test_array)

    # shuffle the Dataset
    rand_idx = test_array.copy()
    np.random.shuffle(rand_idx)
    ds_shuffled = ds[rand_idx]

    # ascending, Var, copy
    dsa = ds_shuffled.sorted('v')
    assert_dataset_equal(dsa, ds, "Copy sorted by Var, ascending")

    # descending, Factor, in-place
    ds_shuffled.sort('f', descending=True)
    assert_dataset_equal(ds_shuffled, ds[::-1], "In-place sorted by Factor, "
                         "descending")


def test_dim_categorial():
    "Test Categorial Dimension"
    values = ['a', 'b', 'c', 'abc']
    name = 'cat'
    dim = Categorial(name, values)

    # basic properties
    print(dim)
    eq_(len(dim), len(values))

    # persistence
    s = pickle.dumps(dim, pickle.HIGHEST_PROTOCOL)
    dim_ = pickle.loads(s)
    eq_(dim_, dim)

    # indexing
    sub_values = values[:2]
    idx = dim.dimindex(sub_values)
    eq_(dim[idx], Categorial(name, sub_values))
    eq_(dim.dimindex('a'), values.index('a'))
    eq_(dim.dimindex('abc'), values.index('abc'))
    assert_raises(TypeError, dim.dimindex, ('a', 'b', 'c'))

    # intersection
    dim2 = Categorial(name, ['c', 'b', 'e'])
    dim_i = dim.intersect(dim2)
    eq_(dim_i, Categorial(name, ['b', 'c']))

    # unicode
    dimu = Categorial(name, ['c', 'b', 'e'])
    eq_(dimu, dim2)


def test_dim_uts():
    "Test UTS Dimension"
    uts = UTS(-0.1, 0.005, 301)

    # basic indexing
    assert_raises(ValueError, uts.dimindex, 1.5)
    assert_raises(ValueError, uts.dimindex, -.15)

    # make sure indexing rounds correctly for floats
    for i, s in enumerate(np.arange(0, 1.4, 0.05)):
        idx = uts.dimindex((-0.1 + s, s))
        eq_(idx.start, 10 * i)
        eq_(idx.stop, 20 + 10 * i)

    # intersection
    uts1 = UTS(-0.1, 0.01, 50)
    uts2 = UTS(0, 0.01, 20)
    intersection = uts1.intersect(uts2)
    eq_(intersection, uts2)
    idx = uts1.dimindex((0, 0.2))
    eq_(uts1[idx], uts2)


def test_effect():
    "Test _Effect class"
    # .enumerate_cells()
    f1 = Factor('aabbccaabbcc')
    f2 = Factor('abababababab')
    i = f1 % f2

    n1 = np.concatenate((np.tile([0, 1], 3), np.tile([2, 3], 3)))
    assert_array_equal(f1.enumerate_cells(), n1)
    assert_array_equal(f2.enumerate_cells(), np.arange(6).repeat(2))
    assert_array_equal(i.enumerate_cells(), np.arange(2).repeat(6))


def test_equality():
    u = Var(np.arange(5.))
    v = Var(np.arange(5.))
    ok_(all_equal(u, v))
    u[-1] = np.nan
    assert_false(all_equal(u, v))
    v[-1] = np.nan
    assert_false(all_equal(u, v))
    ok_(all_equal(u, v, True))


def test_factor():
    "Test basic Factor functionality"
    # initializing
    assert_array_equal(Factor('ab'), ['a', 'b'])
    assert_array_equal(Factor('ab', repeat=2), ['a', 'a', 'b', 'b'])
    assert_array_equal(Factor('ab', repeat=np.array([2, 1])), ['a', 'a', 'b'])
    empty_factor = Factor([])
    eq_(len(empty_factor), 0)
    assert_dataobj_equal(Factor(np.empty(0)), empty_factor)
    # from Factor
    f = Factor('aabbcc')
    assert_array_equal(Factor(f), f)
    assert_array_equal(Factor(f, labels={'a': 'b'}), Factor('bbbbcc'))

    # removing a cell
    f = Factor('aabbcc')
    eq_(f.cells, ('a', 'b', 'c'))
    eq_(f.n_cells, 3)
    f[f == 'c'] = 'a'
    eq_(f.cells, ('a', 'b'))
    eq_(f.n_cells, 2)

    # cell order
    a = np.tile(np.arange(3), 3)
    # alphabetical
    f = Factor(a, labels={0: 'c', 1: 'b', 2: 'a'})
    eq_(f.cells, ('a', 'b', 'c'))
    # ordered
    f = Factor(a, labels=((0, 'c'), (1, 'b'), (2, 'a')))
    eq_(f.cells, ('c', 'b', 'a'))
    eq_(f[:2].cells, ('c', 'b'))
    f[f == 'b'] = 'c'
    eq_(f.cells, ('c', 'a'))

    # label length
    lens = [2, 5, 32, 2, 32, 524]
    f = Factor(['a' * l for l in lens], 'f')
    fl = f.label_length()
    assert_array_equal(fl, lens)
    eq_(fl.info['longname'], 'f.label_length()')
    lens2 = [3, 5, 32, 2, 32, 523]
    f2 = Factor(['b' * l for l in lens2], 'f2')
    assert_array_equal(fl - f2.label_length(), [a - b for a, b in zip(lens, lens2)])

    # equality
    f = Factor('aabbcc')
    assert_equal(f == Factor('aabbcc'), True)
    assert_equal(f == Factor('bbccaa'), False)
    assert_equal(f == Factor('aabxxx'), (True, True, True, False, False, False))
    assert_equal(f == Var(np.ones(6)), False)

    # Factor.as_var()
    assert_array_equal(f.as_var(dict(list(zip('abc', list(range(3)))))), [0, 0, 1, 1, 2, 2])
    assert_array_equal(f.as_var({'a': 1}, 2), [1, 1, 2, 2, 2, 2])
    assert_raises(KeyError, f.as_var, {'a': 1})

    # Factor.floodfill()
    f = Factor([' ', ' ', '1', '2', ' ', ' ', '3', ' ', ' ', '2', ' ', ' ', '1'])
    regions =  [ 1,   1,   1,   2,   2,   2,   3,   3,   3,   2,   2,   1,   1]
    regions2 = [ 1,   1,   1,   2,   2,   3,   3,   2,   2,   2,   2,   1,   1]
    regions3 = [ 1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2]
    target3 =  ['1', '1', '1', '2', '2', '2', '3', '3', '2', '2', '2', '2', '1']
    target_p = [' ', ' ', '1', '2', '2', '2', '3', '3', '3', '2', '2', '2', '1']
    assert_array_equal(f.floodfill(regions, ' '), Var(regions).as_factor())
    assert_array_equal(f.floodfill(regions2, ' '), Var(regions2).as_factor())
    assert_array_equal(f.floodfill(regions3, ' '), target3)
    assert_array_equal(f.floodfill('previous', ' '), target_p)
    f = Factor(['', '', 'a', '', 'e', 'r', ''])
    assert_array_equal(f.floodfill([1, 1, 1, 11, 11, 11, 11]), Factor('aaaeerr'))


def test_factor_relabel():
    "Test Factor.relabel() method"
    f = Factor('aaabbbccc')
    f.update_labels({'a': 'd'})
    assert_array_equal(f, Factor('dddbbbccc'))
    f.update_labels({'d': 'c', 'c': 'd'})
    assert_array_equal(f, Factor('cccbbbddd'))
    f.update_labels({'d': 'c'})
    assert_array_equal(f, Factor('cccbbbccc'))
    assert_raises(KeyError, f.update_labels, {'a':'c'})


def test_interaction():
    "Test Interaction"
    ds = datasets.get_uv()
    A = ds['A']
    B = ds['B']
    i = A % B
    # eq for sequence
    assert_array_equal(i == A % B, True)
    assert_array_equal(i == B % A, False)
    assert_array_equal(i == A, False)
    assert_array_equal(i == ds['fltvar'], False)
    assert_array_equal(ds.eval("A%B") == Factor(ds['A']) % B, True)
    # eq for element
    for a, b in product(A.cells, B.cells):
        assert_array_equal(i == (a, b), np.logical_and(A == a, B == b))

    # Interaction.as_factor()
    a = Factor('aabb')
    i = a % Factor('cdcd')
    assert_dataobj_equal(i.as_factor(), Factor(['a c', 'a d', 'b c', 'b d']))
    i = a % Factor(['c', '', 'c', ''])
    assert_dataobj_equal(i.as_factor(), Factor(['a c', 'a', 'b c', 'b']))

    # pickling
    ip = pickle.loads(pickle.dumps(i))
    assert_dataobj_equal(ip, i)


def test_isin():
    "Test .isin() methods"
    values = np.array([  6, -6, 6, -2, -1, 0, -10, -5, -10, -6])
    v = values[0]
    v2 = values[:2]
    labels = {i: c for i, c in enumerate(ascii_lowercase, -10)}
    vl = labels[v]
    v2l = [labels[v_] for v_ in v2]

    target = np.logical_or(values == v2[0], values == v2[1])
    inv_target = np.invert(target)
    index_target = np.flatnonzero(values == v)
    empty = np.array([])

    var = Var(values)
    assert_array_equal(var.index(v), index_target)
    assert_array_equal(var.isin(v2), target)
    assert_array_equal(var.isany(*v2), target)
    assert_array_equal(var.isnot(*v2), inv_target)
    assert_array_equal(var.isnotin(v2), inv_target)

    var0 = Var([])
    assert_array_equal(var0.isin(v2), empty)
    assert_array_equal(var0.isany(*v2), empty)
    assert_array_equal(var0.isnot(*v2), empty)
    assert_array_equal(var0.isnotin(v2), empty)

    f = Factor(values, labels=labels)
    assert_array_equal(f.index(vl), index_target)
    assert_array_equal(f.isin(v2l), target)
    assert_array_equal(f.isany(*v2l), target)
    assert_array_equal(f.isnot(*v2l), inv_target)
    assert_array_equal(f.isnotin(v2l), inv_target)

    f0 = Factor([])
    assert_array_equal(f0.isin(v2l), empty)
    assert_array_equal(f0.isany(*v2l), empty)
    assert_array_equal(f0.isnot(*v2l), empty)
    assert_array_equal(f0.isnotin(v2l), empty)


def test_longname():
    "Test info['longname'] entry"
    ds = Dataset()
    u = Var([2], 'u')
    v = Var([1], 'v')

    # simple operations, also tested in test_var()
    eq_(longname(v.abs()), 'abs(v)')
    eq_(longname(u * v), "u * v")
    eq_(longname(u * v.abs()), "u * abs(v)")

    # Dataset assigning
    ds['abs_v'] = v.abs()
    eq_(longname(ds['abs_v']), 'abs_v')


def test_model():
    "Test Model class"
    a = Factor('ab', repeat=3, name='a')
    b = Factor('ab', tile=3, name='b')
    u = Var([1, 1, 1, -1, -1, -1], 'u')
    v = Var([1., 2., 3., 4., 5., 6.], 'v')
    w = Var([1., 0., 0., 1., 1., 0.], 'w')

    # model repr
    m = a * b + v
    eq_(repr(m), "a + b + a % b + v")
    lines = ("intercept   a   b   a x b   v",
             "-----------------------------",
             "1           1   1   1       1",
             "1           1   0   0       2",
             "1           1   1   1       3",
             "1           0   0   0       4",
             "1           0   1   0       5",
             "1           0   0   0       6")
    eq_(str(m), '\n'.join(lines))
    eq_(str(m.head(2)), '\n'.join(lines[:4]))
    eq_(str(m.tail(2)), '\n'.join(lines[:2] + lines[-2:]))

    # model without explicit names
    x1 = Factor('ab', repeat=2)
    x2 = Factor('ab', tile=2)
    m = x1 * x2
    eq_(repr(m), "<?> + <?> + <?> % <?>")

    # catch explicit intercept
    intercept = Factor('i', repeat=4, name='intercept')
    assert_raises(ValueError, a.__mul__, intercept)

    # different var/factor combinations
    eq_(a * b, a + b + a % b)
    eq_(a * v, a + v + a % v)
    eq_(a * (v + w), a + v + w + a % v + a % w)

    # parametrization
    m = v + w + v * w
    p = m._parametrize('dummy')
    eq_(p.column_names, ['intercept', 'v', 'w', 'v * w'])
    assert_array_equal(p.x[:, p.terms['intercept']], 1)
    assert_array_equal(p.x[:, p.terms['v']], v.x[:, None])
    assert_array_equal(p.x[:, p.terms['w']], w.x[:, None])
    assert_array_equal(p.x[:, p.terms['v * w']], (v * w).x[:, None])

    # persistence
    mp = pickle.loads(pickle.dumps(m, pickle.HIGHEST_PROTOCOL))
    assert_array_equal(m.full, mp.full)

    # nested Vars
    m = (v + w) * u
    assert_dataobj_equal(m.effects[2], u)
    assert_dataobj_equal(m.effects[3], v * u)
    assert_dataobj_equal(m.effects[4], w * u)
    m = u * (v + w)
    assert_dataobj_equal(m.effects[0], u)
    assert_dataobj_equal(m.effects[3], u * v)
    assert_dataobj_equal(m.effects[4], u * w)
    m = (v + w) % u
    assert_dataobj_equal(m.effects[0], v * u)
    assert_dataobj_equal(m.effects[1], w * u)
    m = u % (v + w)
    assert_dataobj_equal(m.effects[0], u * v)
    assert_dataobj_equal(m.effects[1], u * w)


def test_ndvar():
    "Test the NDVar class"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    # meaningful slicing
    assert_raises(KeyError, x.sub, sensor='5')
    assert_equal(x.sub(sensor='4'), x.x[:, 4])
    assert_equal(x.sub(sensor=['4', '3', '2']), x.x[:, [4, 3, 2]])
    assert_equal(x.sub(sensor=['4']), x.x[:, [4]])
    assert_equal(x.sub(case=1, sensor='4'), x.x[1, 4])

    # setup indices
    s_case = slice(10, 13)
    s_sensor = slice('2', '4')
    s_time = slice(0.1, 0.2)
    b_case = np.bincount([10, 11, 12], minlength=len(x)).astype(bool)
    b_sensor = np.array([False, False, True, True, False])
    b_time = np.bincount(range(30, 40), minlength=len(x.time)).astype(bool)
    a_case = np.arange(10, 13)
    a_sensor = ['2', '3']
    a_time = np.arange(0.1, 0.2, 0.01)

    # slicing with different index kinds
    tgt = x.x[s_case, 2:4, 30:40]
    eq_(tgt.shape, (3, 2, 10))
    # single
    assert_equal(x.sub(case=s_case, sensor=s_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=a_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=b_sensor, time=b_time), tgt)
    # bool & slice
    assert_equal(x.sub(case=b_case, sensor=s_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=b_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=s_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=b_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=b_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=s_sensor, time=b_time), tgt)
    # bool & array
    assert_equal(x.sub(case=b_case, sensor=a_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=b_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=a_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=b_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=b_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=a_sensor, time=b_time), tgt)
    # slice & array
    assert_equal(x.sub(case=s_case, sensor=a_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=s_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=a_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=s_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=s_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=a_sensor, time=s_time), tgt)
    # all three
    assert_equal(x.sub(case=a_case, sensor=b_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=a_case, sensor=s_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=a_sensor, time=s_time), tgt)
    assert_equal(x.sub(case=b_case, sensor=s_sensor, time=a_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=a_sensor, time=b_time), tgt)
    assert_equal(x.sub(case=s_case, sensor=b_sensor, time=a_time), tgt)

    # norm
    y = x / x.norm('sensor')
    assert_allclose(y.norm('sensor'), 1.)
    y = ds['uts'].mean('case').norm('time')
    assert_is_instance(y, float)

    # Var
    v_case = Var(b_case)
    assert_equal(x.sub(case=v_case, sensor=b_sensor, time=a_time), tgt)

    # univariate result
    assert_dataobj_equal(x.sub(sensor='2', time=0.1),
                         Var(x.x[:, 2, 30], x.name))
    eq_(x.sub(case=0, sensor='2', time=0.1), x.x[0, 2, 30])

    # baseline correction
    x_bl = x - x.summary(time=(None, 0))
    # assert that the baseline is 0
    bl = x_bl.summary('case', 'sensor', time=(None, 0))
    ok_(abs(bl) < 1e-10, "Baseline correction")

    # NDVar as index
    sens_mean = x.mean(('case', 'time'))
    idx = sens_mean > 0
    pos = sens_mean[idx]
    assert_array_equal(pos.x > 0, True)

    # NDVar as index along one dimension
    x_tc = x.sub(sensor='1')
    x_time = NDVar(x_tc.time.times >= 0.3, dims=(x_tc.time,))
    assert_dataobj_equal(x_tc[x_time], x_tc.sub(time=(0.3, None)))

    # out of range index
    assert_raises(ValueError, x.sub, time=(0.1, 0.81))
    assert_raises(IndexError, x.sub, time=(-0.25, 0.1))

    # iteration
    for i, xi in enumerate(x):
        assert_dataobj_equal(xi, x[i])
        if i > 4:
            break


def test_ndvar_binning():
    "Test NDVar.bin()"
    x = np.arange(10)
    time = UTS(-0.1, 0.1, 10)
    x_dst = x.reshape((5, 2)).mean(1)
    time_dst = np.arange(0., 0.9, 0.2)

    # 1-d
    ndvar = NDVar(x, (time,))
    b = ndvar.bin(0.2)
    assert_array_equal(b.x, x_dst, "Binned data")
    assert_array_equal(b.time.x, time_dst, "Bin times")
    b = ndvar.sub(time=(0, 0.8)).bin(0.4)
    eq_(b.shape, (2,))

    # 2-d
    ndvar = NDVar(np.vstack((x, x, x)), ('case', time))
    b = ndvar.bin(0.2)
    assert_array_equal(b.x, np.vstack((x_dst, x_dst, x_dst)), "Binned data")
    assert_array_equal(b.time.x, time_dst, "Bin times")

    # time:
    x = np.ones((5, 70))
    ndvar = NDVar(x, ('case', UTS(0.45000000000000007, 0.005, 70)))
    binned_ndvar = ndvar.bin(0.05)
    assert_array_equal(binned_ndvar.x, 1.)
    eq_(binned_ndvar.shape, (5, 7))


def test_ndvar_graph_dim():
    "Test NDVar dimensions with conectvity graph"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    # non-monotonic index
    sub_mono = x.sub(sensor=['2', '3', '4'])
    sub_nonmono = x.sub(sensor=['4', '3', '2'])
    argsort = np.array([2,1,0])
    conn = argsort[sub_mono.sensor.connectivity().ravel()].reshape((-1, 2))
    assert_equal(sub_nonmono.sensor.connectivity(), conn)

    # labeling
    x1 = ds.eval("utsnd[logical_and(A=='a0', B=='b0')].mean('case')")
    x2 = ds.eval("utsnd[A=='a1'].mean('case')")
    x = x1 + x2
    l = x.label_clusters(3)
    eq_(len(l.info['cids']), 5)
    eq_(len(np.unique(l.x)), 6)


@nottest
def test_ndvar_index(x, dimname, index, a_index, index_repr=True):
    "Helper function for test_ndvar_indexing"
    ax = x.get_axis(dimname)
    index_prefix = (full_slice,) * ax
    if dimname != 'case':
        dim = x.get_dim(dimname)
        assert_equal(dim.dimindex(index), a_index)
        if index_repr is not False:
            if index_repr is True:
                index_repr = index
            eq_(dim._index_repr(a_index), index_repr)
    x_array = x.x[index_prefix + (a_index,)]
    x1 = x.sub(**{dimname: index})
    x2 = x[index_prefix + (index,)]
    assert_array_equal(x1.x, x_array)
    assert_dataobj_equal(x2, x1)


def test_ndvar_indexing():
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    # case
    test_ndvar_index(x, 'case', 1, 1)
    test_ndvar_index(x, 'case', [0, 3], [0, 3])
    test_ndvar_index(x, 'case', slice(0, 10, 2), slice(0, 10, 2))

    # sensor
    test_ndvar_index(x, 'sensor', '0', 0)
    test_ndvar_index(x, 'sensor', ['0', '2'], [0, 2])
    test_ndvar_index(x, 'sensor', slice('0', '2'), slice(0, 2))
    test_ndvar_index(x, 'sensor', 0, 0, False)
    test_ndvar_index(x, 'sensor', [0, 2], [0, 2], False)
    test_ndvar_index(x, 'sensor', slice(0, 2), slice(0, 2), False)

    # time
    test_ndvar_index(x, 'time', 0, 20)
    test_ndvar_index(x, 'time', 0.1, 30)
    test_ndvar_index(x, 'time', 0.102, 30, False)
    test_ndvar_index(x, 'time', [0, 0.1, 0.2], [20, 30, 40])
    test_ndvar_index(x, 'time', slice(0.1, None), slice(30, None))
    test_ndvar_index(x, 'time', slice(0.2), slice(40))
    test_ndvar_index(x, 'time', slice(0.202), slice(41), False)
    test_ndvar_index(x, 'time', slice(0.1, 0.2), slice(30, 40))
    test_ndvar_index(x, 'time', slice(0.102, 0.2), slice(31, 40), False)
    test_ndvar_index(x, 'time', slice(0.1, None, 0.1), slice(30, None, 10))
    test_ndvar_index(x, 'time', slice(0.1, None, 1), slice(30, None, 100))

    # Ordered
    x = cwt_morlet(ds['uts'], [8, 10, 13, 17])
    assert_raises(IndexError, x.__getitem__, (full_slice, 9))
    assert_raises(IndexError, x.__getitem__, (full_slice, 6))
    test_ndvar_index(x, 'frequency', 10, 1)
    test_ndvar_index(x, 'frequency', 10.1, 1, False)
    test_ndvar_index(x, 'frequency', 9.9, 1, False)
    test_ndvar_index(x, 'frequency', [8.1, 10.1], [0, 1], False)
    test_ndvar_index(x, 'frequency', slice(8, 13), slice(0, 2))
    test_ndvar_index(x, 'frequency', slice(8, 13.1), slice(0, 3), False)
    test_ndvar_index(x, 'frequency', slice(8, 13.1, 2), slice(0, 3, 2), False)

    # Categorial
    x = NDVar(x.x, ('case', Categorial('cat', ['8', '10', '13', '17']), x.time))
    assert_raises(TypeError, x.__getitem__, (full_slice, 9))
    assert_raises(IndexError, x.__getitem__, (full_slice, '9'))
    test_ndvar_index(x, 'cat', '13', 2)
    test_ndvar_index(x, 'cat', ['8', '13'], [0, 2])
    test_ndvar_index(x, 'cat', slice('8', '13'), slice(0, 2))
    test_ndvar_index(x, 'cat', slice('8', None, 2), slice(0, None, 2))

    # SourceSpace
    x = datasets.get_mne_stc(True)
    assert_raises(TypeError, x.__getitem__, slice('insula-rh'))
    assert_raises(TypeError, x.__getitem__, slice('insula-lh', 'insula-rh'))
    assert_raises(TypeError, x.__getitem__, ('insula-lh', 'insula-rh'))
    test_ndvar_index(x, 'source', 'L90', 90)
    test_ndvar_index(x, 'source', 'R90', 642 + 90)
    test_ndvar_index(x, 'source', ['L90', 'R90'], [90, 642 + 90])
    test_ndvar_index(x, 'source', slice('L90', 'R90'), slice(90, 642 + 90))
    test_ndvar_index(x, 'source', 90, 90, False)
    test_ndvar_index(x, 'source', [90, 95], [90, 95], False)
    test_ndvar_index(x, 'source', slice(90, 95), slice(90, 95), False)
    test_ndvar_index(x, 'source', 'insula-lh', x.source.parc == 'insula-lh', False)
    test_ndvar_index(x, 'source', ('insula-lh', 'insula-rh'),
                     x.source.parc.isin(('insula-lh', 'insula-rh')), False)
    n_lh = x.source.parc.endswith('lh').sum()
    test_ndvar_index(x, 'source', 'lh', slice(n_lh), False)
    test_ndvar_index(x, 'source', 'rh', slice(n_lh, None), False)

    # argmax
    x.x[10, 10] = 20
    eq_(x.argmax(), ('L10', 0.1))
    eq_(x[('L10', 0.1)], 20)
    eq_(x.sub(source='L10').argmax(), 0.1)
    eq_(x.sub(time=0.1).argmax(), 'L10')


def test_ndvar_summary_methods():
    "Test NDVar methods for summarizing data over axes"
    ds = datasets.get_uts(utsnd=True)
    x = ds['utsnd']

    dim = 'sensor'
    axis = x.get_axis(dim)
    dims = ('case', 'sensor')
    axes = tuple(x.get_axis(d) for d in dims)
    idx = x > 0
    x0 = x[0]
    idx0 = idx[0]
    xsub = x.sub(time=(0, 0.5))
    idxsub = xsub > 0
    idx1d = x.mean(('case', 'time')) > 0

    # info inheritance
    eq_(x.any(('sensor', 'time')).info, x.info)

    # numpy functions
    eq_(x.any(), x.x.any())
    assert_array_equal(x.any(dim), x.x.any(axis))
    assert_array_equal(x.any(dims), x.x.any(axes))
    assert_array_equal(x.any(idx0), [x_[idx0.x].any() for x_ in x.x])
    assert_array_equal(x.any(idx), [x_[i].any() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.any(idx0), x0.x[idx0.x].any())
    assert_array_equal(x.any(idxsub), xsub.any(idxsub))
    assert_array_equal(x.any(idx1d), x.x[:, idx1d.x].any(1))

    eq_(x.max(), x.x.max())
    assert_array_equal(x.max(dim), x.x.max(axis))
    assert_array_equal(x.max(dims), x.x.max(axes))
    assert_array_equal(x.max(idx0), [x_[idx0.x].max() for x_ in x.x])
    assert_array_equal(x.max(idx), [x_[i].max() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.max(idx0), x0.x[idx0.x].max())
    assert_array_equal(x.max(idxsub), xsub.max(idxsub))
    assert_array_equal(x.max(idx1d), x.x[:, idx1d.x].max(1))

    eq_(x.mean(), x.x.mean())
    assert_array_equal(x.mean(dim), x.x.mean(axis))
    assert_array_equal(x.mean(dims), x.x.mean(axes))
    assert_array_equal(x.mean(idx0), [x_[idx0.x].mean() for x_ in x.x])
    assert_array_equal(x.mean(idx), [x_[i].mean() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.mean(idx0), x0.x[idx0.x].mean())
    assert_array_equal(x.mean(idxsub), xsub.mean(idxsub))
    assert_array_equal(x.mean(idx1d), x.x[:, idx1d.x].mean(1))

    eq_(x.min(), x.x.min())
    assert_array_equal(x.min(dim), x.x.min(axis))
    assert_array_equal(x.min(dims), x.x.min(axes))
    assert_array_equal(x.min(idx0), [x_[idx0.x].min() for x_ in x.x])
    assert_array_equal(x.min(idx), [x_[i].min() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.min(idx0), x0.x[idx0.x].min())
    assert_array_equal(x.min(idxsub), xsub.min(idxsub))
    assert_array_equal(x.min(idx1d), x.x[:, idx1d.x].min(1))

    eq_(x.var(), x.x.var())
    eq_(x.var(ddof=1), x.x.var(ddof=1))
    assert_array_equal(x.var(dim), x.x.var(axis))
    assert_array_equal(x.var(dims, ddof=1), x.x.var(axes, ddof=1))
    assert_array_equal(x.var(idx0), [x_[idx0.x].var() for x_ in x.x])
    assert_array_equal(x.var(idx), [x_[i].var() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.var(idx0), x0.x[idx0.x].var())
    assert_array_equal(x.var(idxsub), xsub.var(idxsub))
    assert_array_equal(x.var(idx1d), x.x[:, idx1d.x].var(1))

    eq_(x.std(), x.x.std())
    assert_array_equal(x.std(dim), x.x.std(axis))
    assert_array_equal(x.std(dims), x.x.std(axes))
    assert_array_equal(x.std(idx0), [x_[idx0.x].std() for x_ in x.x])
    assert_array_equal(x.std(idx), [x_[i].std() for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.std(idx0), x0.x[idx0.x].std())
    assert_array_equal(x.std(idxsub), xsub.std(idxsub))
    assert_array_equal(x.std(idx1d), x.x[:, idx1d.x].std(1))

    # non-numpy
    eq_(x.rms(), rms(x.x))
    assert_array_equal(x.rms(dim), rms(x.x, axis))
    assert_array_equal(x.rms(dims), rms(x.x, axes))
    assert_array_equal(x.rms(idx0), [rms(x_[idx0.x]) for x_ in x.x])
    assert_array_equal(x.rms(idx), [rms(x_[i]) for x_, i in zip(x.x, idx.x)])
    assert_array_equal(x0.rms(idx0), rms(x0.x[idx0.x]))
    assert_array_equal(x.rms(idxsub), xsub.rms(idxsub))
    assert_array_equal(x.rms(idx1d), rms(x.x[:, idx1d.x], 1))


def test_ndvar_timeseries_methods():
    "Test NDVar time-series methods"
    ds = datasets.get_uts(True)
    x = ds['utsnd']
    xs = NDVar(x.x.swapaxes(1, 2), ('case', x.dims[2], x.dims[1]),
               x.info.copy(), x.name)

    # envelope
    env = x.envelope()
    assert_array_equal(env.x >= 0, True)
    envs = xs.envelope()
    assert_array_equal(env.x, envs.x.swapaxes(1,2))

    # indexing
    eq_(len(ds[0, 'uts'][0.01:0.1].time), 9)

    # FFT
    x = ds['uts'].mean('case')
    np.sin(2 * np.pi * x.time.times, x.x)
    f = x.fft()
    assert_array_almost_equal(f.x, (f.frequency.x == 1) * (len(f) - 1))
    np.sin(4 * np.pi * x.time.times, x.x)
    f = x.fft()
    assert_array_almost_equal(f.x, (f.frequency.x == 2) * (len(f) - 1))

    # update tmin
    eq_(x.time.times[0], -0.2)
    x.time.set_tmin(3.2)
    eq_(x.time.times[0], 3.2)


def test_nested_effects():
    """Test nested effects"""
    ds = datasets.get_uv(nrm=True)

    nested = ds.eval("nrm(B)")
    eq_(nested.cells, ds['nrm'].cells)

    # interaction
    i = ds.eval("A % nrm(B)")
    expected_cells = tuple((case['A'], case['nrm']) for case in ds.itercases())
    eq_(i.cells, expected_cells)

    assert_has_no_empty_cells(ds.eval('A * B + nrm(B) + A % nrm(B)'))

    i = ds.eval("nrm(B) % A")
    expected_cells = sorted((case['nrm'], case['A']) for case in ds.itercases())
    eq_(i.cells, tuple(expected_cells))


@skip_on_windows  # uses R
def test_ols():
    "Test NDVar.ols() method"
    from rpy2.robjects import r

    # simulate data
    ds = datasets.get_uts(True)
    n_times = len(ds['uts'].time)
    x = np.zeros(n_times)
    x[20:40] = np.hanning(20)
    utsc = ds.eval("uts.copy()")
    utsc.x += ds['Y'].x[:, None] * x[None, :]
    ds_ = Dataset()
    ds_['x'] = Var(ds['Y'].x)
    ds_['x2'] = ds_['x'] + np.random.normal(0, 1, ds.n_cases)

    # ols regression
    m1 = ds_['x']
    b1 = utsc.ols(m1)
    res1 = utsc.residuals(m1)
    t1 = utsc.ols_t(m1)
    m2 = ds_.eval("x + x2")
    b2 = utsc.ols(m2)
    res2 = utsc.residuals(m2)
    t2 = utsc.ols_t(m2)
    # compare with R
    for i in range(n_times):
        ds_['y'] = Var(utsc.x[:, i])
        ds_.to_r('ds')
        # 1 predictor
        r('lm1 <- lm(y ~ x, ds)')
        beta = r('coef(lm1)')[1]
        assert_almost_equal(b1.x[0, i], beta)
        res = r('residuals(lm1)')
        assert_array_almost_equal(res1.x[:, i], res)
        t = r('coef(summary(lm1))')[5]
        assert_almost_equal(t1.x[0, i], t)
        # 2 predictors
        r('lm2 <- lm(y ~ x + x2, ds)')
        beta = r('coef(lm2)')[1:]
        assert_array_almost_equal(b2.x[:, i], beta)
        res = r('residuals(lm2)')
        assert_array_almost_equal(res2.x[:, i], res)
        lm2_coefs = r('coef(summary(lm2))')
        t = [lm2_coefs[7], lm2_coefs[8]]
        assert_array_almost_equal(t2.x[:, i], t)

    # 3d
    utsnd = ds['utsnd']
    ds_['utsnd'] = utsnd
    b1 = ds_.eval("utsnd.ols(x)")
    res1 = ds_.eval("utsnd.residuals(x)")
    t1 = ds_.eval("utsnd.ols_t(x)")
    for i in range(len(b1.time)):
        ds_['y'] = Var(utsnd.x[:, 1, i])
        ds_.to_r('ds')
        # 1 predictor
        r('lm1 <- lm(y ~ x, ds)')
        beta = r('coef(lm1)')[1]
        assert_almost_equal(b1.x[0, 1, i], beta)
        res = r('residuals(lm1)')
        assert_array_almost_equal(res1.x[:, 1, i], res)
        t = r('coef(summary(lm1))')[5]
        assert_almost_equal(t1.x[0, 1, i], t)


def test_io_pickle():
    "Test io by pickling"
    ds = datasets.get_uts()
    ds.info['info'] = "Some very useful information about the Dataset"
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.pickled')
        with open(dest, 'wb') as fid:
            pickle.dump(ds, fid, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dest, 'rb') as fid:
            ds2 = pickle.load(fid)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds2)


def test_io_txt():
    "Test Dataset io as text"
    ds = datasets.get_uv()

    # Var that has integer values as float
    ds['intflt'] = ds.eval('intvar * 1.')
    ds['intflt'].name = 'intflt'

    # io test
    tempdir = tempfile.mkdtemp()
    try:
        dest = os.path.join(tempdir, 'test.txt')
        ds.save_txt(dest)
        ds2 = load.tsv(dest)
    finally:
        shutil.rmtree(tempdir)

    assert_dataset_equal(ds, ds2, decimal=6)


@skip_on_windows  # uses R
def test_r():
    "Test interaction with R through rpy2"
    from rpy2.robjects import r

    r("data(sleep)")
    ds = Dataset.from_r("sleep")
    eq_(ds.name, 'sleep')
    extra = (0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0, 1.9, 0.8,
             1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4)
    assert_array_equal(ds.eval('extra'), extra)
    assert_array_equal(ds.eval('ID'), list(map(str, range(1, 11))) * 2)
    assert_array_equal(ds.eval('group'), ['1'] * 10 + ['2'] * 10)

    # test putting
    ds.to_r('sleep_copy')
    ds_copy = Dataset.from_r('sleep_copy')
    assert_dataset_equal(ds_copy, ds)


def test_sensor():
    "Test Sensor dimension"
    locs = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
    names = ['1', '2', '3']
    sensor = Sensor(locs, names, 'test')
    s1 = sensor[[0, 1]]
    s2 = sensor[[1, 2]]
    eq_(tuple(s1.names), ('1', '2'))
    eq_(tuple(s2.names), ('2', '3'))
    eq_(s1, sensor[[0, 1]])
    assert_not_equal(s1, s2)
    eq_(s1.intersect(s2), sensor[[1]])
    eq_(sensor._index_repr(np.array([0, 1, 1], bool)), ['2', '3'])


def test_shuffle():
    x = Factor('aabbaa')
    for _ in range(3):
        i = shuffled_index(6, x)
        eq_(sorted(i[2:4]), [2, 3])
        eq_(sorted(i), list(range(6)))

@requires_mne_sample_data
def test_source_space():
    "Test SourceSpace Dimension"
    subject = 'fsaverage'
    data_path = mne.datasets.sample.data_path()
    mri_sdir = os.path.join(data_path, 'subjects')
    mri_dir = os.path.join(mri_sdir, subject)
    label_dir = os.path.join(mri_dir, 'label')
    label_ba1 = mne.read_label(os.path.join(label_dir, 'lh.BA1.label'))
    label_v1 = mne.read_label(os.path.join(label_dir, 'lh.V1.label'))
    label_mt = mne.read_label(os.path.join(label_dir, 'lh.MT.label'))
    label_ba1_v1 = label_ba1 + label_v1
    label_v1_mt = label_v1 + label_mt

    src = datasets._mne_source_space(subject, 'ico-5', mri_sdir)
    source = SourceSpace.from_mne_source_spaces(src, 'ico-5', mri_sdir)
    source_v1 = source[source.dimindex(label_v1)]
    eq_(source_v1, SourceSpace.from_mne_source_spaces(src, 'ico-5', mri_sdir,
                                                      label=label_v1))
    source_ba1_v1 = source[source.dimindex(label_ba1_v1)]
    source_v1_mt = source[source.dimindex(label_v1_mt)]
    source_v1_intersection = source_ba1_v1.intersect(source_v1_mt)
    assert_source_space_equal(source_v1, source_v1_intersection)

    # index from label
    index = source.index_for_label(label_v1)
    assert_array_equal(index.source[index.x].vertno[0],
                       np.intersect1d(source.lh_vertno, label_v1.vertices, 1))

    # parcellation and cluster localization
    parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=mri_sdir)
    indexes = [source.index_for_label(label) for label in parc
               if len(label) > 10]
    x = np.vstack([index.x for index in indexes])
    ds = source._cluster_properties(x)
    for i in range(ds.n_cases):
        eq_(ds[i, 'location'], parc[i].name)

    # multiple labels
    lingual_index = source.dimindex('lingual-lh')
    cuneus_index = source.dimindex('cuneus-lh')
    assert_array_equal(source.dimindex(('cuneus-lh', 'lingual-lh')),
                       np.logical_or(cuneus_index, lingual_index))
    lingual_source = source[lingual_index]
    cuneus_source = source[cuneus_index]
    assert_raises(IndexError, lingual_source.dimindex, cuneus_source)
    sub_source = source[source.dimindex(('cuneus-lh', 'lingual-lh'))]
    eq_(sub_source[sub_source.dimindex('lingual-lh')], lingual_source)
    eq_(sub_source[sub_source.dimindex('cuneus-lh')], cuneus_source)
    eq_(len(sub_source), len(lingual_source) + len(cuneus_source))

    # indexing
    tgt = np.hstack(sub_source.vertno)
    assert_array_equal([i for i in sub_source], tgt)
    assert_array_equal([sub_source[i] for i in range(len(sub_source))], tgt)
    # hemisphere indexing
    lh = source.dimindex('lh')
    source_lh = source[lh]
    eq_(source_lh.dimindex('rh'), slice(0, 0))
    eq_(source_lh.dimindex('lh'), slice(len(source_lh)))


def test_var():
    "Test Var objects"
    base = Factor('aabbcde')

    # initialization
    x = np.arange(4)
    y = Var(x)
    assert_array_equal(y, x)
    y = Var(x, repeat=2)
    assert_array_equal(y, x.repeat(2))
    y = Var(x, repeat=x)
    assert_array_equal(y, x.repeat(x))
    y = Var.from_dict(base, {'a': 5, 'e': 8}, default=0)
    assert_array_equal(y.x, [5, 5, 0, 0, 0, 0, 8])
    assert_raises(TypeError, Var, x, info=1)

    # basic operations
    info = {'a': 1}
    v = Var([1., 2., 3., -4.], 'v', info=info)
    c = 2
    v2 = Var([2., 2., 3., 3.], 'w', info=info)
    eq_(v.info, info)
    for op, iop, desc in OPERATORS:
        target = op(v.x, c)
        vtarget = op(v.x, v2.x)
        # op
        if desc == '+':
            w = v.copy()
            w.x = iop(w.x, c)
        else:
            w = op(v, c)
            eq_(w.info, {'a': 1, 'longname': 'v %s %s' % (desc, c)})
            assert_array_equal(w, target)
            # with Var
            w = op(v, v2)
            eq_(w.info, {'a': 1, 'longname': 'v %s w' % desc})
            assert_array_equal(w, vtarget)
        # i-op
        w = v.copy()
        w = iop(w, c)
        assert_array_equal(w, target)
        # i-op with Var
        w = v.copy()
        w = iop(w, v2)
        assert_array_equal(w, vtarget)

    # methods
    w = v.abs()
    eq_(w.info, {'a': 1, 'longname': 'abs(v)'})
    assert_array_equal(w, np.abs(v.x))
    x = w.log()
    eq_(x.info, {'a': 1, 'longname': 'log(abs(v))'})
    assert_array_equal(x, np.log(w.x))

    # assignment
    tgt1 = np.arange(10)
    tgt2 = np.tile(np.arange(5), 2)
    v = Var(np.arange(10))
    v[v > 4] = np.arange(5)
    assert_array_equal(v, tgt2)
    v[5:] = np.arange(5, 10)
    assert_array_equal(v, tgt1)
    v = Var(np.arange(10))
    v[v > 4] = Var(np.arange(5))
    assert_array_equal(v, tgt2)
    v[5:] = Var(np.arange(5, 10))
    assert_array_equal(v, tgt1)

    # .count()
    v = Var([1., 2., 1.11, 2., 1.11, 4.])
    assert_array_equal(v.count(), [0, 0, 0, 1, 1, 0])

    # .split()
    y = Var(np.arange(16))
    for i in range(1, 9):
        split = y.split(i)
        eq_(len(split.cells), i)

    # .as_factor()
    v = Var(np.arange(4))
    assert_dataobj_equal(v.as_factor(), Factor('0123'))
    assert_dataobj_equal(v.as_factor({0: 'a'}), Factor(['a', '', '', '']))
    assert_dataobj_equal(v.as_factor({(0, 1): 'a', (2, 3): 'b'}), Factor('aabb'))
    assert_dataobj_equal(v.as_factor({(0, 1): 'a', 2: 'b', 'default': 'c'}),
                         Factor('aabc'))
    assert_dataobj_equal(v.as_factor({(0, 1): 'a', (2, 'default'): 'b'}),
                         Factor('aabb'))
