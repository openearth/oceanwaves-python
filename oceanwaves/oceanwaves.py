import copy
import pyproj
import logging
import numpy as np
import xarray as xr
import scipy.integrate
import scipy.interpolate
from collections import OrderedDict

from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataset import DataVariables

from .units import simplify
from .plot import OceanWavesPlotMethods
from .spectral import *
from .swan import *
from .datawell import *
from .wavedroid import *


# initialize logger
logger = logging.getLogger(__name__)


# initialize readers
from_swan = SwanSpcReader()
from_swantable = SwanTableReader()
from_datawell = DatawellReader()
from_wavedroid = WaveDroidReader()


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
    reference system to lat/lon coordinates, if the local coordinate
    reference system is specified.

    The class interpretates combined variable units and simplifies the
    result to practical entities.

    The class provides two plotting routines: 1) plotting of spectral
    wave data in a raster of subplots and 2) plotting of spectral wabe
    data on a map.

    The class supports all convenient properties of an xarray.Dataset,
    like writing to netCDF or converting to pandas.DataFrame.

    TODO:

    * improve plotting routines

    * add phase functions to use with tides: phase estimates, phase
      interpolation, etc.

    '''


    def __init__(self, *args, **kwargs):
        '''Initialize class

        Initializes class from another OceanWaves or xarray.Dataset
        object or from scratch.

        See Also
        --------
        oceanwaves.OceanWaves.initialize

        '''

        args = list(args)
        if len(args) > 0:
            obj = args[0]
            if isinstance(obj, OceanWaves):
                args.pop(0)
                kwargs = obj._extract_initialization_args(**kwargs)
            elif isinstance(obj, xr.Dataset):
                args.pop(0)
                obj_ow = OceanWaves(*args, **kwargs).merge(obj)
                kwargs = obj_ow._extract_initialization_args(**kwargs)
                                    
            self.initialize(*args, **kwargs)
            self.merge(obj, inplace=True)
        else:
            self.initialize(*args, **kwargs)
                        
        
    def initialize(self, time=None, location=None, frequency=None,
                   direction=None, energy=None, spreading=None,
                   time_units='s', location_units='m',
                   frequency_units='Hz', direction_units='deg',
                   energy_units='m^2/Hz', time_var='time',
                   location_var='location', frequency_var='frequency',
                   direction_var='direction', energy_var='energy',
                   spreading_var='spreading',
                   frequency_convention='absolute',
                   direction_convention='nautical',
                   spreading_convention='cosine', attrs={}, crs=None,
                   **kwargs):
        '''Initialize class

        Sets dimensions, converts coordinates and fills the dataset,
        if data is provided.

        Parameters
        ----------
        time : iterable, optional
            Time coordinates, each item can be a datetime object or
            float
        location : iterable of 2-tuples, optional
            Location coordinates, each item is a 2-tuple with x- and
            y-coordinates
        frequency : iterable, optional
            Frequency cooridinates
        direction : iterable, optional
            Direction coordinates
        energy : matrix, optional
            Wave energy
        time_units : str, optional
            Units of time coordinates (default: s)
        location_units : str, optional
            Units of location coordinates (default: m)
        frequency_units : str, optional
            Units of frequency coordinates (default: Hz)
        direction_units : str, optional
            Units of direction coordinates (default: deg)
        energy_units : str, optional
            Units of wave energy (default: m^2/Hz)
        time_var : str, optional
            Name of time variable (default: time)
        location_var : str, optional
            Name of location variable (default: location)
        frequency_var : str, optional
            Name of frequency variable (default: frequency)
        direction_var : str, optional
            Name of direction variable (default: direction)
        energy_var : str, optional
            Name of wave energy variable (default: energy)
        frequency_convention : str, optional
            Convention of frequency definition (default: absolute)
        direction_convention : str, optional
            Convention of direction definition (default: nautical)
        attrs : dict-like, optional
            Global attributes
        crs : str, optional
            Proj4 specification of local coordinate reference system
        kwargs : dict, optional
            Additional options passed to the xarray.Dataset
            initialization method

        See Also
        --------
        oceanwaves.OceanWaves.reinitialize

        '''

        dims = []
        coords = OrderedDict()
        data_vars = OrderedDict()

        # simplify dimensions
        time = np.asarray(time)
        location = np.asarray(location)
        frequency = np.asarray(frequency, dtype=np.float)
        direction = np.asarray(direction, dtype=np.float)
        energy = np.asarray(energy, dtype=np.float)
        
        # simplify units
        time_units = simplify(time_units)
        location_units = simplify(location_units)
        frequency_units = simplify(frequency_units)
        direction_units = simplify(direction_units)
        energy_units = simplify(energy_units)
        
        # determine object dimensions
        if self._isvalid(time):
            dims.append(time_var)
            coords[time_var]      = xr.Variable(time_var,
                                                time,
                                                attrs=dict(units=time_units))

        if self._isvalid(location):
            dims.append(location_var)
            coords[location_var]  = xr.Variable(location_var,
                                                np.arange(len(location)))
            
            x, y = zip(*location)
            coords['%s_x' % location_var]   = xr.Variable(location_var,
                                                          np.asarray(x),
                                                          attrs=dict(units=location_units))
            coords['%s_y' % location_var]   = xr.Variable(location_var,
                                                          np.asarray(y),
                                                          attrs=dict(units=location_units))
                
            coords['%s_lat' % location_var] = xr.Variable(location_var,
                                                          np.asarray(x) + np.nan,
                                                          attrs=dict(units='degN'))
            coords['%s_lon' % location_var] = xr.Variable(location_var,
                                                          np.asarray(y) + np.nan,
                                                          attrs=dict(units='degE'))

        if self._isvalid(frequency, mask=frequency>0):
            dims.append(frequency_var)
            coords[frequency_var] = xr.Variable(frequency_var,
                                                frequency[frequency>0],
                                                attrs=dict(units=frequency_units))
            
        if self._isvalid(direction):
            dims.append(direction_var)
            coords[direction_var] = xr.Variable(direction_var,
                                                direction,
                                                attrs=dict(units=direction_units))

        # determine object shape
        shp = tuple([len(c) for k, c in coords.items() if k in dims])

        # initialize energy variable
        data_vars[energy_var]     = xr.DataArray(np.nan + np.zeros(shp),
                                                 dims=dims,
                                                 coords=coords,
                                                 attrs=dict(units=energy_units))
        
        # collect global attributes
        attrs.update(dict(
            _init=kwargs.copy(),
            _crs=crs,
            _names=dict(time = time_var,
                        location = location_var,
                        frequency = frequency_var,
                        direction = direction_var,
                        energy = energy_var),
            _units=dict(time = time_units,
                        location = location_units,
                        frequency = frequency_units,
                        direction = direction_units,
                        energy = energy_units),
            _conventions=dict(frequency = frequency_convention,
                              direction = direction_convention)))
        
        # initialize empty object
        super(OceanWaves, self).__init__(data_vars=data_vars,
                                         coords=coords,
                                         attrs=attrs,
                                         **kwargs)

        # set wave energy
        if self._isvalid(energy):
            self['_energy'] = dims, energy.squeeze()

        # convert coordinates
        self.convert_coordinates(crs)
        

    def reinitialize(self, **kwargs):
        '''Reinitializes current object with modified parameters

        Gathers current object's initialization settings and updates
        them with the given initialization options. Then initializes a
        new object with the resulting option set. See for all
        supported options the initialization method of this class.

        Parameters
        ----------
        kwargs : dict
            Keyword/value pairs with initialization options that need
            to be overwritten

        Returns
        -------
        OceanWaves
            New OceanWaves object

        '''

        settings = self._extract_initialization_args(**kwargs)
        return OceanWaves(**settings).restore(self)


    def _extract_initialization_args(self, **kwargs):
        '''Return updated initialization settings

        Parameters
        ----------
        kwargs : dict
            Keyword/value pairs with initialization options that need
            to be overwritten

        Returns
        -------
        dict
            Dictionary with initialization arguments

        '''

        settings = dict(crs = self.attrs['_crs'],
                        attrs = dict([(k, v)
                                      for k, v in self.attrs.items()
                                      if not k.startswith('_')]))

        # add dimensions
        for dim in ['direction', 'frequency', 'time']:
            if self.has_dimension(dim):
                k = self._key_lookup('_%s' % dim)
                v = self.coords[k]
                settings[dim] = v.values
                if 'units' in v.attrs:
                    settings['%s_units' % dim] = v.attrs['units']

        # add locations
        if self.has_dimension('location'):
            k = self._key_lookup('_location')
            x = self.variables['%s_x' % k].values
            y = self.variables['%s_y' % k].values
            settings['location'] = zip(x, y)
            settings['location_units'] = self.variables['%s_x' % k].attrs['units']

        # add energy
        k = self._key_lookup('_energy')
        v = self.variables[k]
        settings['energy'] = v.values
        if 'units' in v.attrs:
            settings['energy_units'] = v.attrs['units']

        # add variable names
        for k, v in self.attrs['_names'].items():
            settings['%s_var' % k] = v

        # add additional arguments
        settings.update(self.attrs['_init'])
        settings.update(kwargs)

        return settings

        
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
        '''Alias for :meth:`oceanwaves.OceanWaves.peak_period`

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
            k = self._key_lookup('_energy')
            E = self.variables[k]

            # determine peak frequencies
            if self.has_dimension('direction'):
                k = self._key_lookup('_direction')
                coords.pop(k)
                E = E.max(dim=k)

            # determine peak directions
            k = self._key_lookup('_frequency')
            f = coords.pop(k).values
            ix = E.argmax(dim=k).values
            peaks = 1. / f[ix.flatten()].reshape(ix.shape)

            # determine units
            units = '1/(%s)' % self.variables[k].attrs['units']
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
            k = self._key_lookup('_energy')
            E = self.variables['energy']

            # determine peak frequencies
            if self.has_dimension('frequency'):
                k = self._key_lookup('_frequency')
                coords.pop(k)
                E = E.max(dim=k)

            # determine peak directions
            k = self._key_lookup('_direction')
            theta = coords.pop(k).values
            ix = E.argmax(dim=k).values
            peaks = theta[ix.flatten()].reshape(ix.shape)

            # determine units
            units = self.variables[k].attrs['units']
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

        if self.has_dimension('direction'):
            self = self.as_omnidirectional()

        if self.has_dimension('frequency', raise_error=True):

            dims = OrderedDict(self.dims)
            coords = OrderedDict(self.coords)
            k = self._key_lookup('_energy')
            E = self.variables[k].values
        
            # integrate directions
            if self.has_dimension('direction'):
                k = self._key_lookup('_direction')
                theta = coords.pop(k).values
                E = np.abs(np.trapz(E, theta, axis=-1))
                dims.pop(k)

            # integrate frequencies
            k = self._key_lookup('_frequency')
            f = coords.pop(k).values
            dims.pop(k)
            if f_min == 0. and f_max == np.inf:
                
                m = np.trapz(E * f**n, f, axis=-1)
                
            else:

                if n != 0:
                    logger.warn('Computing %d-order moment using a frequency range; Are you sure what you are doing?', n)
                    
                # integrate range of frequencies
                f_min = np.maximum(f_min, np.min(f))
                f_max = np.minimum(f_max, np.max(f))
                
                m = scipy.integrate.cumtrapz(E * f**n, f, axis=-1, initial=0)

                vals = []
                if self.has_dimension('time'):
                    k = self._key_lookup('_time')
                    vals.append(coords[k].values.flatten().astype(np.float))
                if self.has_dimension('location'):
                    k = self._key_lookup('_location')
                    vals.append(coords[k].values.flatten().astype(np.float))
                    
                points = tuple(vals + [f.flatten()])
                
                xi_min = zip(*[x.flatten() for x in np.meshgrid(*(vals + [f_min]))])
                xi_max = zip(*[x.flatten() for x in np.meshgrid(*(vals + [f_max]))])

                m_min = scipy.interpolate.interpn(points, m, xi_min)
                m_max = scipy.interpolate.interpn(points, m, xi_max)
                
                m = (m_max - m_min).reshape([len(x) for x in vals])

            # determine units
            k = self._key_lookup('_energy')
            E_units = self.variables[k].attrs['units']
            k = self._key_lookup('_frequency')
            f_units = self.variables[k].attrs['units']
            units = E_units + ('*((%s)^%d)' % (f_units, n+1))
            units = simplify(units)
            
            return xr.DataArray(m, dims=dims, coords=coords,
                                attrs=dict(units=units))


    def as_spectral(self, frequency, frequency_units='Hz', Tp=4.,
                    gamma=3.3, sigma_low=.07, sigma_high=.09,
                    shape='jonswap', method='yamaguchi', g=9.81,
                    normalize=True):
        '''Convert wave energy to spectrum

        Spreads total wave energy over a given set of frequencies
        according to the JONSWAP spectrum shape.

        See :func:`oceanwaves.spectral.jonswap` for options.

        Returns
        -------
        OceanWaves
            New OceanWaves object

        '''

        if self.has_dimension('frequency'):
            return self

        frequency = np.asarray(frequency, dtype=np.float)
        frequency = frequency[frequency>0]
        
        k = self._key_lookup('_energy')
        energy = self.variables[k].values
        energy_units = self.variables[k].attrs['units']

        # convert to energy
        if self.units == 'm':
            energy = energy**2. / 16.
            energy_units = '(%s)^2' % energy_units
            
        # compute spectrum shape
        if shape.lower() == 'jonswap':
            spectrum = jonswap(frequency, Hm0=1., Tp=Tp, gamma=gamma,
                               sigma_low=sigma_low,
                               sigma_high=sigma_high, g=g,
                               method=method, normalize=normalize)

            # normalize shape
            spectrum /= np.trapz(spectrum, frequency)
        else:
            raise ValueError('Unknown spectrum shape: %s', shape)

        # expand energy matrix
        if self.has_dimension('direction'):
            energy = energy[...,np.newaxis,:].repeat(len(frequency), axis=-2)
            spectrum = spectrum[...,np.newaxis].repeat(energy.shape[-1], axis=-1)
        else:
            energy = energy[...,np.newaxis].repeat(len(frequency), axis=-1)

        if self.has_dimension('location'):
            spectrum = spectrum[np.newaxis,...].repeat(len(self['_location']), axis=0)

        if self.has_dimension('time'):
            spectrum = spectrum[np.newaxis,...].repeat(len(self['_time']), axis=0)

        energy = np.multiply(energy, spectrum)

        # determine units
        units = '(%s)/(%s)' % (energy_units,
                               frequency_units)

        # reinitialize object with new dimensions
        return self.reinitialize(frequency=frequency,
                                 frequency_units=frequency_units,
                                 energy=energy,
                                 energy_units=simplify(units))


    def as_directional(self, direction, direction_units='deg',
                       theta_peak=0., s=20., normalize=True):
        '''Convert omnidirectional spectrum to a directional spectrum

        Spreads total wave energy over a given set of directions
        according to a spreading factor ``s``.

        See :func:`oceanwaves.spectral.directional_spreading` for options.

        Returns
        -------
        OceanWaves
            New OceanWaves object

        '''

        if self.has_dimension('direction'):
            return self

        direction = np.asarray(direction, dtype=np.float)
        
        # expand energy matrix
        k = self._key_lookup('_energy')
        energy = self.variables[k].values
        energy = energy[...,np.newaxis].repeat(len(direction), axis=-1)

        # compute directional spreading
        spreading = directional_spreading(direction,
                                          units=direction_units,
                                          theta_peak=theta_peak, s=s,
                                          normalize=normalize)
        for n in energy.shape[::-1][1:]:
            spreading = spreading[np.newaxis,...].repeat(n, axis=0)

        energy = np.multiply(energy, spreading)

        # determine units
        units = '(%s)/(%s)' % (self.variables[k].attrs['units'],
                               direction_units)

        # reinitialize object with new dimensions
        return self.reinitialize(direction=direction,
                                 direction_units=direction_units,
                                 energy=energy,
                                 energy_units=simplify(units))


    def as_omnidirectional(self):
        '''Convert directional spectrum to an omnidirectional spectrum

        Integrate spectral energy over the directions.

        Returns
        -------
        OceanWaves
            New OceanWaves object

        '''

        if not self.has_dimension('direction'):
            return self

        # expand energy matrix
        k1 = self._key_lookup('_energy')
        k2 = self._key_lookup('_direction')
        energy = np.abs(np.trapz(self.variables[k1].values,
                                 self.coords[k2].values, axis=-1))

        # determine units
        units = '(%s)*(%s)' % (self.variables[k1].attrs['units'],
                               self.variables[k2].attrs['units'])
        
        # reinitialize object with new dimensions
        return self.reinitialize(direction=None,
                                 energy=energy,
                                 energy_units=simplify(units))


    def as_radians(self):
        '''Convert directions to radians'''

        if self.has_dimension('direction'):
            k = self._key_lookup('_direction')
            d = self.coords[k]
            if d.attrs['units'].lower().startswith('deg'):
                return self.reinitialize(direction=np.radians(d.values),
                                         direction_units='rad')

        return self
    

    def as_degrees(self):
        '''Convert directions to degrees'''

        if self.has_dimension('direction'):
            k = self._key_lookup('_direction')
            d = self.coords[k]
            if d.attrs['units'].lower().startswith('rad'):
                return self.reinitialize(direction=np.degrees(d.values),
                                         direction_units='deg')

        return self


    @property
    def from_swan(self):

        return SwanSpcReader()

    
    @property
    def from_swantable(self):

        return SwanTableReader()


    @property
    def from_datawell(self):

        return DatawellReader()


    @property
    def from_wavedroid(self):

        return WaveDroidReader()


    @property
    def to_swan(self):

        return SwanSpcWriter(self)
        
        
    def to_netcdf(self, *args, **kwargs):

        obj = self.copy()
        obj.attrs = {k:v for k, v in obj.attrs.items() if not k.startswith('_')}
                
        return super(OceanWaves, obj).to_netcdf(*args, **kwargs)

    
    @property
    def plot(self):

        obj = self.as_radians()
        k1 = self._key_lookup('_energy')

        if self.has_dimension('location'):
            k2 = self._key_lookup('_location')
            return OceanWavesPlotMethods(obj.data_vars[k1],
                                         obj.variables['%s_x' % k2].values,
                                         obj.variables['%s_y' % k2].values)
        else:
            return OceanWavesPlotMethods(obj.data_vars[k1])

        
    @property
    def shape(self):

        k = self._key_lookup('_energy')
        return self.variables[k].shape


    @property
    def units(self):

        k = self._key_lookup('_energy')
        return self.variables[k].attrs['units']


    def restore(self, other, **kwargs):
        if '_names' in self.attrs:
            for k in self.attrs['_names'].iterkeys():
                if k in other.variables.keys():
                    other = other.drop(k)

        return super(OceanWaves, self).merge(other, **kwargs)

    
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
        ValueError

        '''

        if dim in self.attrs['_names']:
            dim = self.attrs['_names'][dim]
        if dim in self.dims.keys():
            return True
            #if len(self.variables[dim].values) > 1:
            #    return True
            #elif raise_error:
            #    raise ValueError('Object has dimension "%s", but it has a length unity' % dim)
        elif raise_error:
            raise ValueError('Object has no dimension "%s"' % dim)

        return False


    def convert_coordinates(self, crs):
        '''Convert coordinates from local coordinate reference system to lat/lon

        Parameters
        ----------
        crs : str
            Proj4 specification of local coordinate reference system

        '''

        if self.has_dimension('location'):
            
            if crs is not None:
                k = self._key_lookup('_location')
                p1 = pyproj.Proj(init=crs)
                p2 = pyproj.Proj(proj='latlong', datum='WGS84')
                x = self._variables['%s_x' % k].values
                y = self._variables['%s_y' % k].values
                lat, lon = pyproj.transform(p1, p2, x, y)
                self.variables['%s_lat' % k].values = lat
                self.variables['%s_lon' % k].values = lon
                self.attrs['_crs'] = crs


    def __getitem__(self, key):
        k = self._key_lookup(key)
        return super(OceanWaves, self).__getitem__(k)


    def __setitem__(self, key, value):
        k = self._key_lookup(key)
        super(OceanWaves, self).__setitem__(k, value)
        if 'units' not in self[k].attrs:
            if key[1:] in self.attrs['_units']:
                self[k].attrs['units'] = self.attrs['_units'][key[1:]]

    
    def _key_lookup(self, key):
        if type(key) is dict:
            key0 = key.copy()
            key = {}
            for k, v in key0.items():
                key[self._key_lookup(k)] = v
        else:
            if key.startswith('_'):
                keys = self.attrs['_names']
                if key[1:] in keys:
                    key = keys[key[1:]]
        return key


    @staticmethod
    def _isvalid(arr, mask=None):

        # check if not None
        if arr is None:
            return False

        # check if iterable
        try:
            itr = iter(arr)
        except TypeError:
            return False

        # apply mask
        if mask is not None:
            arr = arr[mask]

        # check if non-zero
        if len(arr) == 0:
            return False

        # check if all invalid
        if arr.dtype == 'float':
            if ~np.any(np.isfinite(arr)):
                return False

        return True
