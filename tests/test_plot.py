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


### CHECK PLOTTING

def test_plot_polar():
    ow = OceanWaves(**dict(DIMS))
    ow['_energy'] = [d[0] for d in DIMS], np.random.rand(*ow.shape)
    ow.plot(col='time', row='location',
            subplot_kws=dict(projection='polar'), sharex=False, sharey=False)


