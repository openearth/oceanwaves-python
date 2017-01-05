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


### CHECK UNITS

def test_units_tospectral():
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    assert_equals(ow_spc.energy.units, 'm^2 Hz^-1')

    
def test_units_todirectional():
    ow = _generate_nonspectral()
    ow_spc = ow.as_spectral(frequency=dict(DIMS)['frequency'])
    ow_dir = ow_spc.as_directional(direction=dict(DIMS)['direction'])
    assert_equals(ow_dir.energy.units, 'm^2 Hz^-1 deg^-1')

    
### STANDARD TEST OBJECTS

def _generate_nonspectral():
    dims = dict(DIMS)
    ow = OceanWaves(time=dims['time'],
                    energy=[1.] * len(dims['time']),
                    energy_units='m')
    return ow

