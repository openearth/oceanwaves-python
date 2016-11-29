import functools
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
from xarray.plot.plot import _PlotMethods, _plot2d, plot


@_plot2d
def spectrum2d(theta, r, z, ax, **kwargs):
    '''Polar plot

    '''
    
    primitive = ax.pcolormesh(theta, r, z, **kwargs)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    return ax, primitive

