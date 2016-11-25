import pyproj
import logging
import numpy as np
import xarray as xr
import scipy.integrate
import scipy.interpolate
from collections import OrderedDict

from units import simplify
import plot


# initialize logger
logger = logging.getLogger(__name__)


class OceanWaves(xr.Dataset):
    '''Class to store (spectral) data of ocean waves

    The class is a specific variant of an xarray.Dataset that defines
    any selection of the following dimensions: time, location,
    frequency and direction. The dependent variable is some form of
    wave energy.

    The class has methods to compute spectral moments and derived wave
    properties, like the significant wave height and various spectral
    wave periods. In addition the peak wave period and peak wave
    directions can be computed using dedicated methods.

    The class automatically converts locations from a local coordinate
    reference level to lat/lon coordinates, if the local coordinate
    reference level is specified.

    The class interpretates combined variable units and simplifies the
    result to practical entities.

    The class provides two plotting routines: 1) plotting of spectral
    wave data in a raster of subplots and 2) plotting of spectral wabe
    data on a map.

    The class supports all convenient properties of an xarray.Dataset,
    like writing to netCDF or converting to pandas.DataFrame.

    '''


    crs = None
    

    def __init__(self,
                 times=None, locations=None, frequencies=None, directions=None, energy=None,
                 times_units='s', locations_units='m', frequencies_units='Hz', directions_units='rad', energy_units='m^2/Hz',
                 attrs=None, crs=None, **kwargs):
        '''Initialize class

        Sets dimensions, converts coordinates and fills the dataset,
        if data is provided.

        Parameters
        ----------
        times : iterable, optional
            Time coordinates, each item can be a datetime object or
            float
        locations : iterable of 2-tuples, optional
            Location coordinates, each item is a 2-tuple with x- and
            y-coordinates
        frequencies : iterable, optional
            Frequency cooridinates
        directions : iterable, optional
            Direction coordinates
        energy : matrix, optional
            Wave energy
        times_units : str, optional
            Units of time coordinates (default: s)
        locations_units : str, optional
            Units of location coordinates (default: m)
        frequencies_units : str, optional
            Units of frequency coordinates (default: Hz)
        directions_units : str, optional
            Units of direction coordinates (default: rad)
        energy_units : str, optional
            Units of wave energy (default: m^2/Hz)
        attrs : dict-like, optional
            Global attributes
        crs : str, optional
            Proj4 specification of local coordinate reference system
        kwargs : dict, optional
            Additional options passed to the xarray.Dataset
            initialization method

        '''
        
        coords = OrderedDict()
        variables = OrderedDict()

        # simplify units
        times_units = simplify(times_units)
        locations_units = simplify(locations_units)
        frequencies_units = simplify(frequencies_units)
        directions_units = simplify(directions_units)
        energy_units = simplify(energy_units)
        
        # determine object dimensions
        if times is not None:
            coords['time']      = xr.Variable('time',
                                              times,
                                              attrs=dict(units=times_units))

        if locations is not None:
            coords['location']  = xr.Variable('location',
                                              np.arange(len(locations)))
            
            x, y = zip(*locations)
            variables['x']      = xr.Variable('location',
                                              np.asarray(x),
                                              attrs=dict(units=locations_units))
            variables['y']      = xr.Variable('location',
                                              np.asarray(y),
                                              attrs=dict(units=locations_units))
                
            variables['lat']    = xr.Variable('location',
                                              np.asarray(x) + np.nan,
                                              attrs=dict(units='degN'))
            variables['lon']    = xr.Variable('location',
                                              np.asarray(y) + np.nan,
                                              attrs=dict(units='degE'))

        if frequencies is not None:
            coords['frequency'] = xr.Variable('frequency',
                                              frequencies,
                                              attrs=dict(units=frequencies_units))
            
        if directions is not None:
            coords['direction'] = xr.Variable('direction',
                                               directions,
                                               attrs=dict(units=directions_units))

        # determine object shape
        shp = tuple([len(c) for c in coords.itervalues()])

        # initialize energy variable
        variables['energy']     = xr.DataArray(np.nan + np.zeros(shp),
                                               coords=coords,
                                               attrs=dict(units=energy_units))
        
        # initialize empty object
        super(OceanWaves, self).__init__(data_vars=variables, coords=coords, attrs=attrs, **kwargs)

        # set wave energy
        if energy is not None:
            self.energy = energy

        # convert coordinates
        self.convert_coordinates(crs)
        

    def Hm0(self, f_min=0, f_max=np.inf):
        '''Compute significant wave height based on zeroth order moment

        Parameters
        ----------
        f_min : float
            Minimum frequency to include in moment
        f_max : float
            Maximum frequency to include in moment

        Returns
        -------
        H : xarray.DataArray
            Significant wave height at each point in time and location
            in the dataset

        '''

        # compute moments
        m0 = self.moment(0, f_min=f_min, f_max=f_max)

        # compute wave height
        H = 4. * np.sqrt(m0)

        # determine units
        units = '(%s)^0.5' % m0.attrs['units']
        H.attrs['units'] = simplify(units)

        return H

    
    def Tm01(self):
        '''Compute wave period based on first order moment

        Returns
        -------
        T : xarray.DataArray
            Spectral wave period at each point in time and location in
            the dataset

        '''
    
        # compute moments
        m0 = self.moment(0)
        m1 = self.moment(1)

        # compute wave period
        T = m0/m1

        # determine units
        units = '(%s)/(%s)' % (m0.attrs['units'], m1.attrs['units'])
        T.attrs['units'] = simplify(units)

        return T


    def Tm02(self):
        '''Compute wave period based on second order moment

        Returns
        -------
        T : xarray.DataArray
            Spectral wave period at each point in time and location in
            the dataset

        '''

        # compute moments
        m0 = self.moment(0)
        m2 = self.moment(2)

        # compute wave period
        T = np.sqrt(m0/m2)

        # determine units
        units = '((%s)/(%s))^0.5' % (m0.attrs['units'], m2.attrs['units'])
        T.attrs['units'] = simplify(units)

        return T


    def Tp(self):
        '''Alias for :func:`peak_period`

        '''

        return self.peak_period()

    
    def peak_period(self):
        '''Compute peak wave period
        
        Returns
        -------
        T : xarray.DataArray
            Peak wave period at each point in time and location in the
            dataset

        '''
    
        if self.has_dimension('frequency', raise_error=True):

            coords = OrderedDict(self.coords)
            E = self.variables['energy']

            # determine peak frequencies
            if self.has_dimension('direction'):
                coords.pop('direction')
                E = E.max(dim='direction')

            # determine peak directions
            f = coords.pop('frequency').values
            ix = E.argmax(dim='frequency').values
            peaks = 1. / f[ix.flatten()].reshape(ix.shape)

            # determine units
            units = '1/(%s)' % self.variables['frequency'].attrs['units']
            units = simplify(units)
            
            return xr.DataArray(peaks, coords=coords, attrs=dict(units=units))

        
    def peak_direction(self):
        '''Compute peak wave direction

        Returns
        -------
        theta : xarray.DataArray
            Peak wave direction at each point in time and location in
            the dataset

        '''

        if self.has_dimension('direction', raise_error=True):

            coords = OrderedDict(self.coords)
            E = self.variables['energy']

            # determine peak frequencies
            if self.has_dimension('frequency'):
                coords.pop('frequency')
                E = E.max(dim='frequency')

            # determine peak directions
            theta = coords.pop('direction').values
            ix = E.argmax(dim='direction').values
            peaks = theta[ix.flatten()].reshape(ix.shape)

            # determine units
            units = self.variables['direction'].attrs['units']
            units = simplify(units)
            
            return xr.DataArray(peaks, coords=coords, attrs=dict(units=units))
            

    def moment(self, n, f_min=0., f_max=np.inf):
        '''Compute nth order moment of wave spectrum

        Parameters
        ----------
        n : int
            Order of moment
        f_min : float
            Minimum frequency to include in moment
        f_max : float
            Maximum frequency to include in moment

        Returns
        -------
        m : xarray.DataArray
            nth order moment of the wave spectrum at each point in
            time and location in the dataset

        '''

        if self.has_dimension('frequency', raise_error=True):

            coords = OrderedDict(self.coords)
            E = self.variables['energy'].values
        
            # integrate directions
            if self.has_dimension('direction'):
                theta = coords.pop('direction').values
                E = np.abs(np.trapz(E, theta, axis=-1))

            # integrate frequencies
            f = coords.pop('frequency').values
            if f_min == 0. and f_max == np.inf:
                
                m = np.trapz(E * f**n, f, axis=-1)
                
            else:

                if n != 0:
                    logger.warn('Computing %d-order moment using a frequency range; Are you sure what you are doing?', n)
                    
                # integrate range of frequencies
                f_min = np.maximum(f_min, np.min(f))
                f_max = np.minimum(f_max, np.max(f))
                
                m = scipy.integrate.cumtrapz(E * f**n, f, axis=-1, initial=0)

                dims = []
                if self.has_dimension('time'):
                    dims.append(coords['time'].values.flatten().astype(np.float))
                if self.has_dimension('location'):
                    dims.append(coords['location'].values.flatten().astype(np.float))
                    
                points = tuple(dims + [f.flatten()])
                
                xi_min = zip(*[x.flatten() for x in np.meshgrid(*(dims + [f_min]))])
                xi_max = zip(*[x.flatten() for x in np.meshgrid(*(dims + [f_max]))])

                m_min = scipy.interpolate.interpn(points, m, xi_min)
                m_max = scipy.interpolate.interpn(points, m, xi_max)
                
                m = (m_max - m_min).reshape([len(x) for x in dims])

            # determine units
            E_units = self.variables['energy'].attrs['units']
            f_units = self.variables['frequency'].attrs['units']
            units = E_units + ('*%s^%d' % (f_units, n+1))
            units = simplify(units)
            
            return xr.DataArray(m, coords=coords, attrs=dict(units=units))


    def plot(self, figsize=None, **kwargs):
        '''Plot data in raster of subplots

        '''

        # determine number of subplots
        nx, ny = 1, 1
        
        if self.has_dimension('time'):
            nx = len(self.variables['time'])
        if self.has_dimension('location'):
            ny = len(self.variables['location'])

        # create figure object
        if self.has_dimension('direction'):
            axs = plot.create_figure(ny, nx, figsize=figsize,
                                     subplot_kw=dict(projection='polar'), squeeze=False)
        else:
            axs = plot.create_figure(ny, nx, figsize=figsize, squeeze=False)

        # plot
        for i in range(ny):
            for j in range(nx):

                # select energy subset
                ix = {}
                if self.has_dimension('time'):
                    ix['time'] = j
                if self.has_dimension('location'):
                    ix['location'] = i

                E = self.variables['energy'][ix]

                if self.has_dimension('direction'):

                    # polar plot
                    plot.polar(self.variables['frequency'],
                               self.variables['direction'],
                               E, ax=axs[i,j], **kwargs)
                    
                else:

                    # spectral plot
                    plot.spectrum(self.variables['frequency'],
                                  E, ax=axs[i,j], **kwargs)

                # set labels
                if i < ny-1:
                    axs[i,j].set_xlabel('')
                if j > 0:
                    axs[i,j].set_ylabel('')

                # set titles
                if self.has_dimension('time') and i == 0:
                    axs[i,j].set_title(self.variables['time'][j].values)
                if self.has_dimension('location') and j == nx-1:
                    axs[i,j].set_title(self.variables['location'][i].values, loc='right')

        return axs
                    
        
    def plot_map(self, size=.1, time=0, ax=None, figsize=None, **kwargs):
        '''Plot data on map

        '''

        if self.has_dimension('location', raise_error=True):

            x = self.variables['x']
            y = self.variables['y']

            ax = plot.create_figure(ax=ax, figsize=figsize, extent=(x, y))
            fig = ax.get_figure()
            axs = [ax]

            # select energy subset
            ix = {}
            if self.has_dimension('time'):
                ix['time'] = time
                
            for i in range(len(self.variables['location'])):

                ix['location'] = i
                E = self.variables['energy'][ix]

                # determine subplot position
                pos = axs[0].transData.transform((x[i], y[i]))
                pos = fig.transFigure.inverted().transform(pos)
                rect = list(pos-.5*size) + [size, size]

                if self.has_dimension('direction'):

                    # polar plot
                    ax = fig.add_axes(rect, projection='polar')
                    plot.polar(self.variables['frequency'],
                               self.variables['direction'],
                               E, ax=ax, **kwargs)
                    
                else:

                    # spectral plot
                    ax = fig.add_axes(rect)
                    plot.spectrum(self.variables['frequency'],
                                  E, ax=ax, **kwargs)

                axs.append(ax)

            return axs

        
    @property
    def energy(self):

        return self.variables['energy']

    
    @energy.setter
    def energy(self, energy):
        '''Convenience function to set wave energy with arbitrary dimension order

        '''
        
        self.variables['energy'].values = energy

        
    @property
    def shape(self):

        return self.variables['energy'].shape


    def has_dimension(self, dim, raise_error=False):
        '''Checks if dimension is present

        Parameters
        ----------
        dim : str
            Name of dimension
        raise_error : bool, optional
            Raise error if dimension is absent (default: False)

        Returns
        -------
        bool
            Boolean indicating whether dimension is present

        Raises
        ------
        KeyError

        '''

        has_dim = dim in self.dims.keys()

        if raise_error and not has_dim:
            raise KeyError('Object has no dimension "%s"' % dim)

        return has_dim


    def convert_coordinates(self, crs):
        '''Convert coordinates from local coordinate reference system to lat/lon

        Parameters
        ----------
        crs : str
            Proj4 specification of local coordinate reference system

        '''

        if self.has_dimension('location'):
            
            self.crs = crs

            if crs is not None:
                p1 = pyproj.Proj(init=crs)
                p2 = pyproj.Proj(proj='latlong', datum='WGS84')
                x = self._variables['x'].values
                y = self._variables['y'].values
                lat, lon = pyproj.transform(p1, p2, x, y)
                self.variables['lat'].values = lat
                self.variables['lon'].values = lon


