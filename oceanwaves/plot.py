import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt


def polar(f, d, E, figsize=(10,10), ax=None, **kwargs):
    '''Create polar plot for single 2D spectrum

    '''
    
    ax = create_figure(ax=ax, figsize=figsize, subplot_kw=dict(projection='polar'))
    ax.pcolormesh(f.values, d.values, E.values.T, **kwargs)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    return ax


def spectrum(f, E, figsize=(10,4), ax=None, **kwargs):
    '''Create spectrum plot for single 1D spectrum

    '''
    
    ax = create_figure(ax=ax, figsize=figsize)
    ax.plot(f.values, E.values, **kwargs)
    ax.set_xlabel('frequency [$\mathrm{%s}$]' % f.attrs['units'])
    ax.set_ylabel('energy [$\mathrm{%s}$]' % E.attrs['units'])

    return ax


def create_figure(ny=1, nx=1, ax=None, extent=None, **kwargs):
    '''Create figure object

    '''
    
    if ax is None:
        fig, axs = plt.subplots(ny, nx, sharex=True, sharey=True, **kwargs)
    else:
        axs = ax

    if extent is not None:
        x, y = extent
        
        o = 10.**np.ceil(np.log10(x.max()))
        axs.set_xlim((np.floor(x.min()/o)*o, np.ceil(x.max()/o)*o))
            
        o = 10.**np.ceil(np.log10(y.max()))
        axs.set_ylim((np.floor(y.min()/o)*o, np.ceil(y.max()/o)*o))

    return axs
