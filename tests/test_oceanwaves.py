from nose.tools import *
import numpy as np
from datetime import datetime
from oceanwaves import *


DIMS = [('time',      [datetime(1970,1,1,0),
                       datetime(1970,1,1,1)]),
        ('location',  [(0,0),
                       (1,0),
                       (0,1),
                       (.5,.5)]),
        ('frequency', [.025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]),
        ('direction', [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])]


### CHECK SHAPE

def test_shape_dim1():
    for dim, arr in DIMS:
        yield check_shape, {dim:arr}, (len(arr),)


def test_shape_dim2():
    for i, (dim1, arr1) in enumerate(DIMS):
        for dim2, arr2 in DIMS[i+1:]:
            yield check_shape, {dim1:arr1, dim2:arr2}, (len(arr1), len(arr2))


def test_shape_dim3():
    for i, (dim1, arr1) in enumerate(DIMS):
        for j, (dim2, arr2) in enumerate(DIMS[i+1:]):
            for dim3, arr3 in DIMS[i+j+2:]:
                yield check_shape, {dim1:arr1, dim2:arr2, dim3:arr3}, (len(arr1), len(arr2), len(arr3))

                
def test_shape_dim4():
    yield check_shape, dict(DIMS), tuple([len(x[1]) for x in DIMS])


def check_shape(init, shape):
    ow = OceanWaves(**init)
    assert_equals(ow.shape, shape)
    
    
### CHECK ENERGY

def test_energy_tospectral():
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    assert_almost_equals(np.sum(np.abs(ow['_energy'].values - ow_spc.Hm0().values)), 0.)


def test_energy_todirectional():
    ow = _generate_nonspectral()
    ow_dir = ow.as_directional(direction=dict(DIMS)['direction'])
    assert_almost_equals(np.sum(np.abs(ow['_energy'].values - ow_dir.as_omnidirectional()['_energy'].values)), 0.)

    
def test_energy_todirectionalspectrum():
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    ow_dir = ow_spc.as_directional(direction=dict(DIMS)['direction'])

    E1 = ow_dir['_energy'].values
    E2 = np.trapz(E1, ow_dir.coords['direction'].values, axis=-1)
    E3 = np.trapz(E2, ow_dir.coords['frequency'].values, axis=-1)

    assert_almost_equals(np.sum(np.abs(ow['_energy'].values**2./16. - E3)), 0.)


### STANDARD TEST OBJECTS

def _generate_nonspectral():
    dims = dict(DIMS)
    ow = OceanWaves(time=dims['time'],
                    energy=[1.5] * len(dims['time']),
                    energy_units='m')
    return ow


