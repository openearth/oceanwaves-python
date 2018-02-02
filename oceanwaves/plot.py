import functools
import numpy as np
from xarray.plot.plot import _PlotMethods


try:
    import matplotlib.pyplot as plt
    from matplotlib.projections import PolarAxes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def spatial_map(darray, x, y, ax=None, scale=.1, dim='location',
                add_colorbar=True, figure_kw={}, subplot_kw={},
                **kwargs):
    '''Plot data on a spatial map

    Creates a subplot for each location in the DataArray and positions
    the subplot on a map. Connects events to both map axes and figure
    to keep the subplots positioned.

    Parameters
    ----------
    darray : xarray.DataArray
        DataArray with spatial information
    x : list
        Array with x-coordinates of locations
    y : list
        Array with y-coordinates of locations
    scale : float
        Size of subplots in axes coordinates
    dim : str
        Name of spatial dimensions
    figure_kw : dict
        Options passed to :func:`matplotlib.pyplot.subplots`
    subplot_kw : dict
        Options passed to :func:`matplotlib.pyplot.add_axes`
    kwargs : dict
        Options passed to :meth:`xarray.DataArray.plot`

    Returns
    -------
    ax_map : AxesSubplot
        Subplot containing the map
    axs : list of AxesSubplot
        Positionsed subplots visualizing data

    '''

    if not HAS_MATPLOTLIB:
        raise ImportError('Matplotlib not available')

    # create map axis
    if ax is None:
        fig, ax_map = plt.subplots(**figure_kw)
        ax_map.set_aspect('equal')
    else:
        ax_map = ax
        fig = ax.get_figure()

    # plot locations to set axis limits
    ax_map.scatter(x, y, c='none', edgecolor='none')

    # set default plot options
    plot_kw = {}
    if darray.ndim == 2:
        pass
    elif darray.ndim == 3:
        plot_kw.update(dict(add_colorbar=False,
                            vmin = np.min(darray.values),
                            vmax = np.max(darray.values)))
    elif darray.ndim == 4:
        pass
    plot_kw.update(**kwargs)

    # create a subplot for each location
    axs = []
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax = fig.add_axes([0, 0, 1, 1], label=i, **subplot_kw)
        darray[{dim:i}].plot(ax=ax, **plot_kw)
        ax.coords = (xi, yi)
        ax.set_axis_off()

        axs.append(ax)

    # add colorbar
    if add_colorbar:
        meshes = [c
                  for ax in axs
                  for c in ax.get_children()
                  if hasattr(c, 'autoscale')]
        if len(meshes) > 0:
            fig.colorbar(meshes[0], ax=ax_map)

    # set figure events to update subplot positions
    ax_map.callbacks.connect('xlim_changed', lambda x: position_subplots(ax_map, axs, scale=scale))
    ax_map.callbacks.connect('ylim_changed', lambda x: position_subplots(ax_map, axs, scale=scale))

    fig.canvas.mpl_connect('resize_event', lambda x: position_subplots(ax_map, axs, scale=scale))
    fig.canvas.mpl_connect('draw_event', lambda x: position_subplots(ax_map, axs, scale=scale))

    # set initial subplot positions
    position_subplots(ax_map, axs, scale=scale)

    return ax_map, axs

        
def position_subplots(ax_map, axs, scale=.1):
    '''Updates subplot positions in map

    Parameters
    ----------
    ax_map : AxesSubplot
        Map axes object
    axs : list of AxesSubplot
        List of subplots to be positioned. Each axes should have a
        property ``coords`` specifying the target position in data
        coordinates of the map axes.
    scale : float
        Size of subplots in axes coordinates

    '''

    fig = ax_map.get_figure()
    
    for i, ax in enumerate(axs):
        if not hasattr(ax, 'coords'):
            continue

        # get current subplot position in axes coordinates
        pos = ax.get_position()
        pos_scn = ax_map.transData.transform(ax.coords)
        pos_axs = ax_map.transAxes.inverted().transform(pos_scn)

        # get position of subplot corners in figure coordinates
        pos_crn = [(pos_axs[0] - scale/2.,
                    pos_axs[1] - scale/2.),
                   (pos_axs[0] + scale/2.,
                    pos_axs[1] + scale/2.)]
        pos_scn = ax_map.transAxes.transform(pos_crn)
        pos_fig = fig.transFigure.inverted().transform(pos_scn)

        # update subplot position
        pos.x0 = pos_fig[0,0]
        pos.x1 = pos_fig[1,0]
        pos.y0 = pos_fig[0,1]
        pos.y1 = pos_fig[1,1]
        
        ax.set_position(pos)


class OceanWavesPlotMethods(_PlotMethods):

    '''Inheritence class to add map plotting functionality to xarray.DataArray objects'''
    
    def __init__(self, darray, x=None, y=None, **kwargs):
        '''Class initilisation

        Parameters
        ----------
        x : list
            Array with x-coordinates
        y : list
            Array with y-coordinates
        args, kwargs
            Arguments passed to super class

        '''

        if not HAS_MATPLOTLIB:
            raise ImportError('Matplotlib not available')
        
        self._x = x
        self._y = y
        super(OceanWavesPlotMethods, self).__init__(darray, **kwargs)


    def __call__(self, **kwargs):

        # if data is directional, faceted and not yet polar, make it polar
        if 'direction' in self._da.dims:
            if 'col' in kwargs.keys() or 'row' in kwargs.keys():
                if not 'subplot_kws' in kwargs.keys():
                    kwargs.update(dict(subplot_kws = dict(projection = 'polar'),
                                       sharex = False,
                                       sharey = False))

        r = super(OceanWavesPlotMethods, self).__call__(**kwargs)

        self.find_axes(r)
        self.rotate_axes()
        
        return r


    @functools.wraps(spatial_map)
    def spatial_map(self, ax=None, **kwargs):
        '''Plot wave data on map'''
        if self._x is None or self._y is None:
            raise ValueError('Cannot plot map if locations are not defined')
        self._axes = spatial_map(self._da, self._x, self._y, ax=ax, **kwargs)[1]
        self.rotate_axes()
        return self._axes


    def find_axes(self, r):

        # find axes
        try:
            self._axes = r.axes.flatten()
        except:
            try:
                self._axes = r.axes
            except:
                self._axes = r

        try:
            iter(self._axes)
        except:
            self._axes = [self._axes]
            
                
    def rotate_axes(self):

        # rotate polars
        try:
            for ax in self._axes:
                if isinstance(ax, PolarAxes):
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
        except:
            pass


