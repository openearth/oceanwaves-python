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

import oceanwaves


TABLE_UNITS_FILE = 'table_units.json'
SWAN_TIME_FORMAT = '%Y%m%d.%H%M%S'


class SwanSpcReader:


    stationairy = True
    directional = False

    crs = None
    frequency_convention = None
    direction_convention = None


    def __init__(self):

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

        if self.specs.has_key('VaDens'):
            energy_units = self.specs['VaDens']['units']
        else:
            energy_units = None
            
        kwargs = dict(location=self.locations,
                      location_units='m' if self.crs is None else 'deg',
                      frequency=self.frequencies,
                      frequency_units='Hz',
                      frequency_convention=self.frequency_convention,
                      energy_units=energy_units,
                      attrs=dict(comments=self.comments),
                      crs=self.crs)

        if self.directional:
            kwargs.update(dict(direction=self.directions,
                               direction_units='deg',
                               direction_convention=self.direction_convention,
                               energy=self.quantities))
            if not self.stationary:
                kwargs.update(dict(time=self.time,
                                   time_units='s'))
        else:
            if not self.stationary:
                kwargs.update(dict(time=self.time,
                                   time_units='s',
                                   energy=[[q2[:,0] for q2 in q1] for q1 in self.quantities]))
            else:
                kwargs.update(dict(energy=[q[:,0] for q in self.quantities]))

        return oceanwaves.OceanWaves(**kwargs)

            
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
            m = re.match('\s*([\d\.]+)', factor)
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
        self.quantities.append(None)


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
            self.fp.write('VarDens\n')
            self.fp.write('m2/Hz/degr\n') # TODO: read units from OceanWaves object
            self.fp.write('-99.0\n') # TODO: replace NaN with fill value

        else:

            self.fp.write('%4d\n' % 3)
            self.fp.write('VarDens\n')
            self.fp.write('m2/Hz/degr\n')
            self.fp.write('-99.0\n')
            self.fp.write('NDIR\n') # TODO: read convention from OceanWaves object
            self.fp.write('degr\n')
            self.fp.write('-999\n')
            self.fp.write('DSPRDEGR\n')
            self.fp.write('degr\n')
            self.fp.write('-9\n')


    def write_timestamp(self):

        if self.obj.has_dimension('time'):
            self.fp.write('%s\n' % self.obj['_time'].values[0].strftime(SWAN_TIME_FORMAT))


    def write_data(self):
        
        E = self.obj['_energy'].values
        
        if self.obj.has_dimension('direction'):

            if E.ndim == 2:
                E = E[np.newaxis,:,:]

            n = E.shape[2]
            for i in range(E.shape[0]):
                
                f = E[i,:,:].max() / 99999.
                self.fp.write('FACTOR\n')
                self.fp.write('%4e\n' % f)

                fmt = '%s\n' % ('%8d ' * n)
                for j in range(E.shape[1]):
                    self.fp.write(fmt % tuple(E[i,j,:] / f))

        else:

            if E.ndim == 1:
                E = E[np.newaxis,:]

            n = E.shape[1]
            for i in range(E.shape[0]):
                self.fp.write('LOCATION %4d\n' % i)
                fmt = '%8e 0.0 0.0\n' # TODO: set peak direction and directional spreading
                for j in range(E.shape[1]):
                    self.fp.write(fmt % E[i,j])


    def _write_block(self, header, data, fmt='%10.4f'):

        self.fp.write('%s\n' % header.upper())
        self.fp.write('%4d\n' % len(data))
            
        for x in data.values:
            self.fp.write(('%s\n' % fmt) % x)

                
    def _get_convention(self, convention, default=None):

        conventions = self._get_attr('_conventions', default={})
        if conventions.has_key(convention):
            return conventions[convention]
        else:
            return default

        
    def _get_attr(self, attr, default=None):

        if self.obj.attrs.has_key(attr):
            return self.obj.attrs[attr]
        else:
            return default


class SwanTableReader:


    stationary = True
    
    run = None
    table = None
    variables = []
    units = []
    

    def __init__(self):

        pass


    def __call__(self, fpath, headers=[], energy_var='Hsig', **kwargs):

        ds = self.read(fpath, headers=headers)

        return oceanwaves.OceanWaves(ds,
                                     location=zip(ds['Xp'], ds['Yp']),
                                     energy_var=energy_var,
                                     **kwargs)

        
    def read(self, fpath, headers=[]):
        ''' Read swan table

        Parameters
        ----------
        fpath : str
            Input file name (ex: /data/swan/table.tbl)
        headers : list of strings
            If using NOHEAD, use headers to inform the reader the names of the
            variables being rad.
        
        Returns
        -------
        ds : xr.Dataset()
            dataset with the parsed data
        
        '''

        # list of all table variables
        swanoutputs = {}
        jsonpath = os.path.join(os.path.split(__file__)[0], TABLE_UNITS_FILE)
        if os.path.exists(jsonpath):
            with open(jsonpath, 'r') as fp:
                swanoutputs = json.load(fp)

        # loop over lines
        with open(fpath, 'r') as fp:
            self.lines = fp.readlines()

            self.n = 0 # line counter
            self.l = 0 # location counter

            data = []
            while self.n < len(self.lines):
                line = self._currentline()

                # read the headers
                if line.startswith('%'):
                    # read swan metadata
                    if "SWAN" in line:
                        self.version = line.split('version')[1].strip(":").split(" ")[0].strip()
                    # read metadata
                    if "Run" in line:
                        self.run = line.split('Run')[1].strip(":").split(" ")[0].strip()
                    if "Table" in line:
                        self.table = line.split('Table')[1].strip(":").split(" ")[0].strip()
                    # read variable names
                    _vars = []; _units=[]
                    for tvar in line.split():
                        for svar,sunit in swanoutputs.iteritems():
                            if tvar == svar:
                                _vars.append(svar)
                                _units.append(sunit)
                    if _vars:
                        self.variables = _vars
                        self.units = _units

                # read data
                else:
                    data.append(line.split())
                # update line
                self.n += 1

            # update stationary variable
            if "Time" in self.variables:
                self.stationary = False

            # stationary cases
            if self.stationary:

                if self.variables:
                    df = pd.DataFrame(np.array(data).astype(float),
                                      columns=self.variables)
                else:
                    if headers:
                        df = pd.DataFrame(np.array(data).astype(float),
                                        columns=headers)
                    else:
                        raise ValueError("When using \'NOHEAD\' option, the user must inform the variable names")

                # figure out locations
                if "Xp" and "Yp" in self.variables:
                        x = np.unique(df["Xp"])
                        y = np.unique(df["Yp"])
                        self.locations = zip(x,y)

                # group data by geographic location
                grouped = df.groupby(["Xp","Yp"])
                # create the final dataset
                dss = []
                for group in grouped:
                    ds_at_xy = xr.Dataset()
                    # extract one dataframe at each location
                    df_at_xy = group[1].copy()
                    # organize dataframe
                    x = df_at_xy["Xp"].values[0]; df_at_xy.drop("Xp",axis=1,inplace=True)
                    y = df_at_xy["Yp"].values[0]; df_at_xy.drop("Yp",axis=1,inplace=True)
                    # organize dataset
                    for var in df_at_xy.columns:
                        ds_at_xy[var] = df_at_xy[var].values[0]
                    # set locations
                    ds_at_xy.coords["Xp"] = x
                    ds_at_xy.coords["Yp"] = y
                    # append
                    dss.append(ds_at_xy)
                # concatenate
                ds = xr.concat(dss,dim='location')

            # non stationary cases
            else:
                # could figure out variable names
                if self.variables:
                    # read raw data into the dataframe
                    df = pd.DataFrame(np.array(data),columns=self.variables)
                    # update time
                    self.time = np.unique([datetime.strptime(t, SWAN_TIME_FORMAT)
                                           for t in df["Time"].values])
                    # update dataframe
                    df.drop("Time",axis=1,inplace=True)
                    df = df.astype(float)
                    # figure out locations
                    if "Xp" and "Yp" in self.variables:
                        x = np.unique(df["Xp"])
                        y = np.unique(df["Yp"])
                        self.locations = zip(x,y)
                    # group data by geographic location
                    grouped = df.groupby(["Xp","Yp"])
                    # create the final dataset
                    dss = []
                    for group in grouped:
                        # extract one dataframe at each location
                        df_at_xy = group[1].copy()
                        # organize dataframe
                        x = df_at_xy["Xp"].values[0]
                        y = df_at_xy["Yp"].values[0]
                        df_at_xy.index = self.time
                        df_at_xy.index.name = "time"
                        df_at_xy.drop("Xp",axis=1,inplace=True)
                        df_at_xy.drop("Yp",axis=1,inplace=True)
                        # organize dataset
                        ds_at_xy = df_at_xy.to_xarray()
                        # set location
                        ds_at_xy["Xp"] = x
                        ds_at_xy["Yp"] = y
                        # append
                        dss.append(ds_at_xy)
                    # concatenate
                    ds = xr.concat(dss,dim='location')

                else:
                    raise ValueError("When reading a non-stationary table, please use \'HEAD\' option.")

            # Update metatada
            ds.attrs["version"] = self.version
            ds.attrs["run"] = self.run
            ds.attrs["table"] =  self.run

            for var, units in zip(self.variables, self.units):
                ds[var].attrs['units'] = units

            return ds


    def _currentline(self):
        return self.lines[self.n]


    def _currentlines(self):
        return self.lines[self.n:]
