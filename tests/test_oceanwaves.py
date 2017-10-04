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
        ('direction', [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])]

PARAMETERIZATIONS = [('Tm01', 3.61435704),
                     ('Tm02', 3.53313111),
                     ('Tp', 4.),
                     ('peak_direction', 0.),
                     ('peak_period', 4.)]

### CHECK SHAPE

def test_shape_dim1():
    '''Test one-dimensional object initialization'''
    for dim, arr in DIMS:
        yield check_shape, {dim:arr}, (len(arr),)


def test_shape_dim2():
    '''Test two-dimensional object initialization'''
    for i, (dim1, arr1) in enumerate(DIMS):
        for dim2, arr2 in DIMS[i+1:]:
            yield check_shape, {dim1:arr1, dim2:arr2}, (len(arr1), len(arr2))


def test_shape_dim3():
    '''Test three-dimensional object initialization'''
    for i, (dim1, arr1) in enumerate(DIMS):
        for j, (dim2, arr2) in enumerate(DIMS[i+1:]):
            for dim3, arr3 in DIMS[i+j+2:]:
                yield check_shape, {dim1:arr1, dim2:arr2, dim3:arr3}, \
                    (len(arr1), len(arr2), len(arr3))

                
def test_shape_dim4():
    '''Test four-dimensional object initialization'''
    yield check_shape, dict(DIMS), tuple([len(x[1]) for x in DIMS])


def check_shape(init, shape):
    ow = OceanWaves(**init)
    assert_equals(ow.shape, shape)
    
    
### CHECK ENERGY

def test_energy_tospectral():
    '''Test conversion from non-spectral to spectral'''
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    assert_almost_equals(np.sum(np.abs(ow['_energy'].values -
                                       ow_spc.Hm0().values)), 0.)


def test_energy_todirectional():
    '''Test conversion from omnidirectional to directional'''
    ow = _generate_nonspectral()
    ow_dir = ow.as_directional(direction=dict(DIMS)['direction'])
    assert_almost_equals(np.sum(np.abs(ow['_energy'].values -
                                       ow_dir.as_omnidirectional()['_energy'].values)),
                         0.)

    
def test_energy_todirectionalspectrum():
    '''Test conversion from non-spectral and omnidirectional to directional spectrum'''
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    ow_dir = ow_spc.as_directional(direction=dict(DIMS)['direction'])

    E1 = ow_dir['_energy'].values
    E2 = np.trapz(E1, ow_dir.coords['direction'].values, axis=-1)
    E3 = np.trapz(E2, ow_dir.coords['frequency'].values, axis=-1)

    assert_almost_equals(np.sum(np.abs(ow['_energy'].values**2./16. - E3)), 0.)


### CHECK PARAMETERIZATIONS

def test_params():
    '''Test computation of spectral parameters'''
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    ow_dir = ow_spc.as_directional(direction=dict(DIMS)['direction'])
    for p, v in PARAMETERIZATIONS:
        yield check_param, ow_dir, p, v


def check_param(ow, p, v):
    val = (getattr(ow, p)().values - v).max()
    assert_almost_equals(val, 0)
    

### STANDARD TEST OBJECTS

def _generate_nonspectral():
    dims = dict(DIMS)
    ow = OceanWaves(time=dims['time'],
                    energy=[1.5] * len(dims['time']),
                    energy_units='m')
    return ow


