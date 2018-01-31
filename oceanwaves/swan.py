from __future__ import absolute_import

import os
import re
import glob
import json
import pyproj
import logging
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict

import oceanwaves.oceanwaves


TABLE_UNITS_FILE = 'table_units.json'
SWAN_TIME_FORMAT = '%Y%m%d.%H%M%S'


class SwanSpcReader:


    def __init__(self):

        self.stationairy = True
        self.directional = False

        self.crs = None
        self.frequency_convention = None
        self.direction_convention = None

        self.reset()


    def __call__(self, fpath):

        self.reset()
        return self.read(fpath)

        
    def reset(self):
        
        self.stationary = True
        self.directional = False

        self.version = None
        self.timecoding = None
        self.comments = []

        self.time = []
        self.locations = []
        self.frequencies = []
        self.directions = []
        self.specs = OrderedDict()
        self.quantities = []

        self.l = 0 # location counter
        

    def read(self, fpath):

        for fname in glob.glob(fpath):
            self.readfile(fname)

        return self.to_oceanwaves()

        
    def readfile(self, fpath):

        with open(fpath, 'r') as fp:
            self.lines = fp.readlines()

        self.n = 0 # line counter
        while self.n < len(self.lines):

            line = self._currentline()

            if line.startswith('$'):
                self.parse_comments()
            elif line.startswith('SWAN'):
                self.parse_version()
            elif line.startswith('TIME'):
                self.parse_time()
            elif line.startswith('LOCATIONS'):
                self.parse_locations()
            elif line.startswith('LONLAT'):
                self.crs = 'epsg:4326'
                self.parse_locations()
            elif line.startswith('AFREQ'):
                self.frequency_convention = 'absolute'
                self.parse_frequencies()
            elif line.startswith('RFREQ'):
                self.frequency_convention = 'relative'
                self.parse_frequencies()
            elif line.startswith('NDIR'):
                self.direction_convention = 'nautical'
                self.parse_directions()
            elif line.startswith('CDIR'):
                self.direction_convention = 'cartesian'
                self.parse_directions()
            elif line.startswith('QUANT'):
                self.parse_quantities()
            elif line.startswith('FACTOR'):
                self.parse_data()
            elif line.startswith('LOCATION'):
                self.parse_data()
            elif line.startswith('NODATA'):
                self.parse_nodata()
            elif re.match('\s*[\d\.]+', line):
                self.parse_timestamp()
            else:
                logging.warn('Line not parsed: %s' % line)

            self.n += 1


    def to_oceanwaves(self):

        energy_units = '1'
        for var, specs in self.specs.items():
            if 'units' in specs.keys():
                energy_units = specs['units']
                break
            
        kwargs = dict(
            location=self.locations,
            location_units='m' if self.crs is None else 'deg',
            frequency=self.frequencies,
            frequency_units='Hz',
            frequency_convention=self.frequency_convention,
            energy_units=energy_units,
            attrs=dict(comments='\n'.join(self.comments)),
            crs=self.crs
        )

        if self.directional:
            kwargs.update(dict(
                direction=self.directions,
                direction_units='deg',
                direction_convention=self.direction_convention,
                energy=self.quantities
            ))
            if not self.stationary:
                kwargs.update(dict(
                    time=self.time,
                    time_units='s'
                ))
        else:
            if not self.stationary:
                kwargs.update(dict(
                    time=self.time,
                    time_units='s',
                    energy=[[q2[:,0] for q2 in q1] for q1 in self.quantities],
                    direction=[[q2[:,1] for q2 in q1] for q1 in self.quantities],
                    spreading=[[q2[:,2] for q2 in q1] for q1 in self.quantities]
                ))
            else:
                kwargs.update(dict(
                    energy=[q[:,0] for q in self.quantities],
                    direction=[q[:,1] for q in self.quantities],
                    spreading=[q[:,2] for q in self.quantities]
                ))
        kwargs.update(dict(directional=self.directional))

        return oceanwaves.oceanwaves.OceanWaves(**kwargs)

            
    def parse_comments(self):
        line = self._currentline()
        self.comments.append(line[1:].strip())


    def parse_version(self):
        line = self._currentline()
        m = re.match('SWAN\s+([^\s]+)', line)
        if m:
            version = m.groups()[0]
            self._check_if_matches(self.version, version,
                                   errormsg='Version mismatch')
            self.version = version


    def parse_time(self):
        lines = self._currentlines()
        m = re.match('\s*([^\s]+)', lines[1])
        if m:
            timecoding = m.groups()[0]
            self._check_if_matches(self.timecoding, timecoding,
                                   errormsg='Timecoding mismatch')
            self.timecoding = timecoding
            self.stationary = False
            self.n += 1


    def parse_locations(self):
        lines = self._currentlines()
        locations = self._parse_block(lines[1:])
        if not self.stationary:
            self._check_if_matches(self.locations, locations,
                                   errormsg='Location dimension mismatch')
            self.locations = locations
        else:
            self.locations.extend(locations)


    def parse_frequencies(self):
        lines = self._currentlines()
        frequencies = np.asarray(self._parse_block(lines[1:])).flatten()
        self._check_if_matches(self.frequencies, frequencies,
                               errormsg='Frequency dimension mismatch')
        self.frequencies = frequencies


    def parse_directions(self):
        lines = self._currentlines()
        directions = np.asarray(self._parse_block(lines[1:])).flatten()
        self._check_if_matches(self.directions, directions,
                               errormsg='Direction dimension mismatch')
        self.directions = directions
        self.directional = True


    def parse_quantities(self):
        lines = self._currentlines()

        m = re.match('\s*(\d+)', lines[1])
        if m:
            n = int(m.groups()[0])
        else:
            raise ValueError('Number of quantities not understood: %s' % lines[1])

        self.specs = OrderedDict()
        for i in range(n):
            q = []
            for j in range(3):
                m = re.match('\s*([^\s]+)', lines[2+3*i+j])
                if m:
                    q.append(m.groups()[0])

            if len(q) == 3:
                self.specs[q[0]] = dict(zip(('units', 'fill_value'), q[1:]))
            else:
                logging.warn('Skipped invalid quantity definiton: %s' % ' '.join(q))

        self.n += 1 + 3*n


    def parse_data(self):
        lines = self._currentlines()

        key = lines.pop(0)
        if key.startswith('FACTOR'):
            factor = lines.pop(0)
            m = re.match('\s*([\+\-\d\.Ee]+)\s*$', factor)
            if m:
                f = float(m.groups()[0])
            else:
                raise ValueError('Factor not understood: %s' % factor)

            self.n += 1
        else:
            f = 1.

        n = len(self.frequencies)
        q = np.asarray(self._parse_blockbody(lines, n)) * f
        if self.stationary:
            self.quantities.append(q)
        else:
            self.quantities[-1].append(q)


    def parse_nodata(self):
        if self.directional:
            q = np.zeros((len(self.frequencies),
                          len(self.directions)))
        else:
            q = np.zeros((len(self.frequencies), 3))

        if self.stationary:
            self.quantities.append(q)
        else:
            self.quantities[-1].append(q)


    def parse_timestamp(self):
        line = self._currentline()

        m = re.match('\s*([\d\.]+)', line)
        if m:
            self.time.append(datetime.strptime(m.groups()[0], SWAN_TIME_FORMAT))
            self.quantities.append([])
        else:
            raise ValueError('Time definition not understood: %s' % line)


    def _parse_block(self, lines):

        m = re.match('\s*(\d+)', lines[0])
        if m:
            n = int(m.groups()[0])
        else:
            raise ValueError('Length of block not understood: %s' % lines[0])

        self.n += 1

        return self._parse_blockbody(lines[1:], n)


    def _parse_blockbody(self, lines, n):

        block = []
        for i in range(n):
            arr = re.split('\s+', lines[i].strip())
            arr = tuple([float(x) for x in arr])
            block.append(arr)

        self.n += n

        return block


    def _check_if_matches(self, current, new, errormsg='Dimension mismatch'):
        if current is None:
            return True
        elif type(current) is list:
            if len(current) == 0:
                return
            else:
                try:
                    if all([a == b for a, b in zip(current, new)]):
                        return
                except:
                    pass
        else:
            if current == new:
                return

        raise ValueError(errormsg)


    def _currentline(self):
        return self.lines[self.n]


    def _currentlines(self):
        return self.lines[self.n:]


class SwanSpcWriter:


    def __init__(self, obj):

        self.obj0 = obj.as_degrees()
        self.obj = self.obj0
        

    def __call__(self, fpath):

        self.write(fpath)

        
    def write(self, fpath):

        if self.obj.has_dimension('time'):
            fpath, fext = os.path.splitext(fpath)
            k = self._key_lookup('_time')
            for ix in range(len(self.obj.coords[k])):
                self.obj = self.obj0[dict(_time=ix)]
                self.writefile('%s_%03d%s' % (fpath, ix, fext))
        else:
            self.obj = self.obj0
            self.writefile(fpath)
        

    def writefile(self, fpath):

        self.fp = open(fpath, 'w')
        self.fp.write('SWAN %4d\n' % 1)

        self.write_comments()
        self.write_time()
        self.write_locations()
        self.write_frequencies()
        self.write_directions()
        self.write_quantities()
        self.write_timestamp()
        self.write_data()

        self.fp.close()


    def write_comments(self):

        comments = self._get_attr('comments', default=[])
        for c in comments:
            self.fp.write('$ %s\n' % c)

    
    def write_time(self):

        if self.obj.has_dimension('time'):
            self.fp.write('TIME\n')
            self.fp.write('%4d\n' % 1)

        
    def write_locations(self, latlon=False):
        
        if self.obj.has_dimension('location'):

            crs = self._get_attr('_crs')
            if crs is not None:
                latlon = pyproj.Proj(init=crs).is_latlong()

            if latlon:
                self.fp.write('LONLAT\n')
            else:
                self.fp.write('LOCATIONS\n')

            k = self.obj._key_lookup('_location')
            x = self.obj.variables['%s_x' % k].values
            y = self.obj.variables['%s_y' % k].values
            self.fp.write('%4d\n' % len(x))
            for coords in zip(x, y):
                self.fp.write('%10.2f %10.2f\n' % coords)

    
    def write_frequencies(self, convention='absolute'):

        if self.obj.has_dimension('frequency'):

            convention = self._get_convention('frequency', convention)
            fmt = '%10.4f'

            if convention.lower() == 'relative':
                self._write_block('RFREQ', self.obj['_frequency'], fmt=fmt)
            else:
                self._write_block('AFREQ', self.obj['_frequency'], fmt=fmt)

    
    def write_directions(self, convention='nautical'):

        if self.obj.has_dimension('direction'):

            convention = self._get_convention('direction', convention)
            fmt = '%10.2f'
            
            if convention.lower() == 'cartesian':
                self._write_block('CDIR', self.obj['_direction'], fmt=fmt)
            else:
                self._write_block('NDIR', self.obj['_direction'], fmt=fmt)

                
    def write_quantities(self):

        self.fp.write('QUANT\n')
        
        if self.obj.has_dimension('direction'):
            
            self.fp.write('%4d\n' % 1)
            self.fp.write('VaDens\n')
            self.fp.write('%s\n' % self._get_units('_energy', 'm^2 s'))
            self.fp.write('-99.0\n') # TODO: replace NaN with fill value

        else:

            convention = self._get_convention('direction', 'nautical').lower()

            self.fp.write('%4d\n' % 3)
            self.fp.write('VaDens\n')
            self.fp.write('%s\n' % self._get_units('_energy', 'm^2 s'))
            self.fp.write('-99.0\n')
            self.fp.write('%s\n' % ('CDIR' if convention == 'cartesion' else 'NDIR'))
            self.fp.write('%s\n' % self._get_units('_direction', 'deg'))
            self.fp.write('-999\n')
            self.fp.write('DSPRDEGR\n')
            self.fp.write('%s\n' % self._get_units('_spreading', 'deg'))
            self.fp.write('-9\n')


    def write_timestamp(self):

        if self.obj.has_dimension('time'):
            time = self.obj['_time'].values[0].strftime(SWAN_TIME_FORMAT)
            self.fp.write('%s\n' % time)


    def write_data(self):
        
        E = self.obj['_energy'].values
        
        try:
            D = self.obj['_direction'].values
        except:
            D = np.zeros(E.shape)

        try:
            S = self.obj['_spreading'].values
        except:
            S = np.zeros(E.shape)
        
        if self.obj.has_dimension('direction'):

            if E.ndim == 2:
                E = E[np.newaxis,:,:]

            n = E.shape[2]
            for i in range(E.shape[0]):
                
                if E[i,:,:].max() == 0:
                    f = 1.
                else:
                    f = E[i,:,:].max() / 999999.

                self.fp.write('FACTOR\n')
                self.fp.write('%4e\n' % f)

                fmt = '%s\n' % ('%8d ' * n)
                for j in range(E.shape[1]):
                    self.fp.write(fmt % tuple(E[i,j,:] / f))

        else:

            if E.ndim == 1:
                E = E[np.newaxis,:]
                D = D[np.newaxis,:]
                S = S[np.newaxis,:]

            n = E.shape[1]
            for i in range(E.shape[0]):
                self.fp.write('LOCATION %4d\n' % i)
                fmt = '%8e %8e %8e\n'
                for j in range(E.shape[1]):
                    self.fp.write(fmt % (E[i,j], D[i,j], S[i,j]))


    def _write_block(self, header, data, fmt='%10.4f'):

        self.fp.write('%s\n' % header.upper())
        self.fp.write('%4d\n' % len(data))
            
        for x in data.values:
            self.fp.write(('%s\n' % fmt) % x)

                
    def _get_convention(self, convention, default=None):

        conventions = self._get_attr('_conventions', default={})
        if convention in conventions.keys():
            return conventions[convention]
        else:
            return default

        
    def _get_attr(self, attr, default=None):

        if attr in self.obj.attrs.keys():
            return self.obj.attrs[attr]
        else:
            return default

        
    def _get_units(self, variable, default=None):

        if variable in self.obj.variables.keys():
            attrs = self.obj.variables[variable].attrs
            if 'units' in attrs.keys():
                return attrs['units']
        return default


class SwanTableReader:
    
    
    def __init__(self):
        
        pass
    
    
    def __call__(self, fpath, columns=[], time_var='Time',
                 location_vars=['Xp', 'Yp'], frequency_var='RTpeak',
                 direction_var='Dir', energy_var='Hsig', **kwargs):

        # clear variables
        self.headers = []
        self.columns = []
        self.units = []
        self.data = []
        self.attrs = {}
        
        # assume columns names and units
        self.columns = columns
        self.get_units()
        
        # read data, column names and units from file
        self.read(fpath)
        self.parse_headers()
        self.check_integrity()
        
        return self.to_oceanwaves(
            energy_var=energy_var,
            time_var=time_var,
            **kwargs
        )
    
    
    def to_oceanwaves(self, time_var='Time',
                      location_vars=['Xp','Yp'], period_var='RTpeak',
                      frequency_var=None, direction_var='Dir',
                      energy_var='Hsig', **kwargs):
        '''Converts raw data in OceanWaves object

        Converts raw data and column names into Pandas
        DataFrame. Groups the DataFrame by time and location. Converts
        the MultiIndex DataFrame into an xarray Dataset. Adds unit
        information as attributes and uses the resulting Dataset to
        initialize a OceanWaves object.

        See for possible initialization arguments the
        :class:`OceanWaves` class.

        Returns
        -------
        OceanWaves
            OceanWaves object.

        '''
        
        df = pd.DataFrame(self.data, columns=self.columns)
        
        # group by time
        if time_var in df.columns:
            df[time_var] = ([datetime.strptime('%15.6f' % t, SWAN_TIME_FORMAT)
                             for t in df[time_var].values])
            dfs1 = []
            grouped = df.groupby(time_var)
            for t, group in grouped:
                group.set_index(time_var, drop=True, append=True, inplace=True)
                dfs1.append(group)
        else:
            dfs1 = [df]
            
        # group by location
        if all([v in df.columns for v in location_vars]):
            dfs2 = []
            for df in dfs1:
                grouped = df.groupby(location_vars)
                for (x, y), group in grouped:
                    group = group.copy()
                    group['Location'] = [(x,y)] * len(group)
                    group.drop(location_vars, axis=1, inplace=True)
                    group.set_index('Location', drop=True, append=True, inplace=True)
                    dfs2.append(group)
        else:
            dfs2 = dfs1
        
        # concatenate per time/location dataframes and reset index
        df = pd.concat(dfs2, axis=0)
        df = df.reset_index(0, drop=True)

        # convert period to frequency
        if frequency_var is None:
            frequency_var = 'Freq'
            df[frequency_var] = 1./df[period_var]
        
        # convert dataframe to dataset and add units
        xa = df.to_xarray()
        for k in xa.variables.keys():
            if k in self.columns:
                ix = self.columns.index(k)
                xa.variables[k].attrs['units'] = self.units[ix]
            elif k == 'Location':
                units = set()
                for v in location_vars:
                    ix = self.columns.index(v)
                    units.add(self.units[ix])
                if len(units) == 1:
                    xa.variables[k].attrs['units'] = units.pop()
                else:
                    raise ValueError('Inconsistent units for location coordinates.')
            else:
                xa.variables[k].attrs['units'] = '1'
               
        # convert dataset to oceanwaves object
        return oceanwaves.OceanWaves.from_dataset(
            xa,
            time_var=time_var,
            location_var='Location',
            frequency_var=frequency_var,
            direction_var=direction_var,
            energy_var=energy_var,
            **kwargs
        )    

    
    def read(self, fpath):
        '''Read headers and data seperately'''
        
        with open(fpath, 'r') as fp:
            for line in fp:
                if line.startswith('%'):
                    self.headers.append(line[1:].strip())
                else:
                    self.data.append([float(x) for x in line.split()])
                    
        self.data = np.asarray(self.data)
        
        
    def parse_headers(self):
        '''Parse headers into attributes, units and column names'''
        
        for line in self.headers:
            if len(line) == 0:
                continue
            elif ':' in line:
                self.attrs.update(dict([re.split('\s*:\s*', x)
                                        for x in re.split('\s{2,}', line)]))
            elif re.search('\[\S*\]', line):
                self.units = re.findall('\[\s*(\S*)\s*\]', line)
            else:
                self.columns = re.split('\s+', line)
                
                
    def check_integrity(self):
        '''Check integrity of parsed data

        Raises
        ------
        ValueError
            If no columns are specified when using NOHEAD option or if
            the number of column names or units do not match the
            number of data columns.

        '''
        
        if not self.columns:
            raise ValueError('Column names must be specified '
                             'when using \'NOHEAD\' option.')
        if self.data.shape[1] != len(self.columns):
            raise ValueError('Number of column names (%d) does not match '
                             'number of data columns (%d).' % (self.data.shape[1], 
                                                               len(self.columns)))
        if self.data.shape[1] != len(self.units):
            raise ValueError('Number of units (%d) does not match '
                             'number of data columns (%d).' % (self.data.shape[1], 
                                                               len(self.units)))

        
    def get_units(self):
        '''Read relevant units from JSON file'''
        
        jsonpath = os.path.join(os.path.split(__file__)[0], TABLE_UNITS_FILE)
        if os.path.exists(jsonpath):
            with open(jsonpath, 'r') as fp:
                self.units = [u for c, u in json.load(fp).items() if c in self.columns]
