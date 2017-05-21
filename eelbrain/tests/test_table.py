# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from nose.tools import eq_, ok_, assert_is_instance, assert_raises
from eelbrain._utils.testing import assert_dataobj_equal
from numpy.testing import assert_array_equal

from eelbrain import Factor, NDVar, Var, datasets, table, combine


def test_difference():
    "Test table.difference"
    ds = datasets.get_uv()
    print(table.difference('fltvar', 'A', 'a1', 'a2', 'rm', ds=ds))
    print(table.difference('fltvar', 'A', 'a1', 'a2', 'rm', by='B', ds=ds))
    print(table.difference('fltvar', 'A%B', ('a1', 'b1'), ('a2', 'b2'), 'rm',
                           ds=ds))

    # create bigger dataset
    ds['C', :] = 'c1'
    ds2 = datasets.get_uv()
    ds2['C', :] = 'c2'
    ds = combine((ds, ds2))
    print(table.difference('fltvar', 'A', 'a1', 'a2', 'rm', 'B%C', ds=ds))
    print(table.difference('fltvar', 'A%B', ('a1', 'b1'), ('a2', 'b2'), 'rm',
                           'C', ds=ds))


def test_frequencies():
    "Test table.frequencies"
    ds = datasets.get_uts()
    freq = table.frequencies('YCat', 'A', ds=ds)
    assert_array_equal(freq['A'], ['a0', 'a1'])
    ok_(all(c in freq for c in ds['YCat'].cells))
    print(freq)
    freq = table.frequencies('YCat', 'A % B', ds=ds)
    assert_array_equal(freq['A'], ['a0', 'a0', 'a1', 'a1'])
    assert_array_equal(freq['B'], ['b0', 'b1', 'b0', 'b1'])
    print(freq)
    freq = table.frequencies('YCat % A', 'B', ds=ds)
    print(freq)


def test_melt_ndvar():
    "Test table.melt_ndvar()"
    ds = datasets.get_uts(True)
    ds = ds.sub("A == 'a1'")

    lds = table.melt_ndvar('uts', ds=ds)
    ok_('time' in lds)
    assert_is_instance(lds['time'], Var)
    eq_(set(lds['time'].x), set(ds['uts'].time.x))

    # no ds
    lds2 = table.melt_ndvar(ds['uts'])
    assert_dataobj_equal(lds2['uts'], lds['uts'])

    # sensor
    lds = table.melt_ndvar("utsnd.summary(time=(0.1, 0.2))", ds=ds, varname='summary')
    eq_(set(lds['sensor'].cells), set(ds['utsnd'].sensor.names))

    # NDVar out
    lds = table.melt_ndvar("utsnd", 'sensor', ds=ds)
    ok_('utsnd' in lds)
    assert_is_instance(lds['utsnd'], NDVar)
    assert_dataobj_equal(lds[:ds.n_cases, 'utsnd'], ds.eval("utsnd.sub(sensor='0')"))

    # more than one dimensions
    assert_raises(ValueError, table.melt_ndvar, 'utsnd', ds=ds)


def test_repmeas():
    "Test table.repmeas (repeated measures table)"
    ds = datasets.get_uv()
    print(table.repmeas('fltvar', 'A', 'rm', ds=ds))
    print(table.repmeas('fltvar', 'A%B', 'rm', ds=ds))
    print(table.repmeas('fltvar', 'A', 'B%rm', ds=ds))

    # with int model
    ds['Bv'] = ds['B'].as_var({'b1': 1, 'b2': 2})
    print(table.repmeas('fltvar', 'A', 'Bv%rm', ds=ds))

    # test naturalization of cellnames
    ds['ANum'] = Factor(ds['A'], labels={'a1': '1', 'a2': '2'})
    print(table.repmeas('fltvar', 'ANum', 'rm', ds=ds))

    # with empty cell name
    ds['A'].update_labels({'a1': ''})
    print(table.repmeas('fltvar', 'A', 'rm', ds=ds))
