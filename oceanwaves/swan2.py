import re
import glob
import logging
import xarray as xr
import numpy as np
from datetime import datetime

from .oceanwaves import OceanWaves


class SwanIO:

    
    stationairy = True
    directional = False

    
    def __init__(self):

        self.stationary = True
        self.directional = False
        
        self.version = None
        self.timecoding = None
        self.comments = []

        self.time = []
        self.locations = []
        self.frequencies = []
        self.directions = []
        self.specs = []
        self.quantities = []


#    def read_multi(self, fpath):
#
#        objs = []
#        for fname in glob.glob(fpath):
#            objs.append(self._read(fname))
#            
#        return xr.merge(objs)
    
            
    def read(self, fpath):

        with open(fpath, 'r') as fp:
            self.lines = fp.readlines()

        self.n = 0 # line counter
        self.l = 0 # location counter
        self.quantities = []
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
                self.parse_locations()
            elif line.startswith('AFREQ'):
                self.parse_frequencies()
            elif line.startswith('RFREQ'):
                self.parse_frequencies()
            elif line.startswith('NDIR'):
                self.parse_directions()
            elif line.startswith('CDIR'):
                self.parse_directions()
            elif line.startswith('QUANT'):
                self.parse_quantities()
            elif line.startswith('FACTOR'):
                self.parse_location()
            elif line.startswith('LOCATION'):
                self.parse_location()
            elif line.startswith('NODATA'):
                self.parse_nodata()
            elif re.match('\s*[\d\.]+', line):
                self.parse_timestamp()
            else:
                logging.warn('Line not parsed: %s' % line)

            self.n += 1

        if self.directional:
            return OceanWaves(time=[self.time],
                              location=self.locations,
                              frequency=self.frequencies,
                              direction=self.directions,
                              energy=self.quantities)
        else:
            return OceanWaves(time=[self.time],
                              location=self.locations,
                              frequency=self.frequencies,
                              energy=[q[:,0] for q in self.quantities])


    def write(self, obj, fpath):

        with open(fpath, 'w') as fp:

            fp.write('SWAN %4d\n' % 1)

            if obj.has_dimension('time'):
                fp.write('TIME\n')
                fp.write('%4d\n' % 1)

            if obj.has_dimension('location'):
                fp.write('LOCATIONS\n')

                x = obj.variables['x'].values
                y = obj.variables['y'].values
                for coords in zip(x, y):
                    fp.write('%10.2f %10.2f\n' % coords)

            if obj.has_dimension('frequency'):
                fp.write('AFREQ\n')
                fp.write('%4d\n' % len(obj.coords['frequency']))

                for f in obj.coords['frequency'].values:
                    fp.write('%10.2f\n' % f)

            if obj.has_dimension('direction'):
                fp.write('NDIR\n')
                fp.write('%4d\n' % len(obj.coords['direction']))

                for f in obj.coords['direction'].values:
                    fp.write('%10.2f\n' % f)

                fp.write('QUANT\n')
                fp.write('%4d\n' % 1)
                fp.write('VarDens\n')
                fp.write('m2/Hz/degr\n')
                fp.write('-99.0\n')
                
                fp.write('FACTOR\n')
                fp.write('%4e\n' % 1e-5)

                fp.write(obj.variables['energy'].values)

                    
    def parse_comments(self):
        line = self._currentline()
        self.comments.append(line.strip())

        
    def parse_version(self):
        line = self._currentline()
        m = re.match('SWAN\s+([^\s]+)', line)
        if m:
            self.version = m.groups()[0]


    def parse_time(self):
        lines = self._currentlines()
        m = re.match('\s*([^\s]+)', lines[1])
        if m:
            self.timecoding = m.groups()[0]
            self.stationary = False
            self.n += 1


    def parse_locations(self):
        lines = self._currentlines()
        self.locations = self._parse_block(lines[1:])
        

    def parse_frequencies(self):
        lines = self._currentlines()
        self.frequencies = np.asarray(self._parse_block(lines[1:])).flatten()


    def parse_directions(self):
        lines = self._currentlines()
        self.directions = np.asarray(self._parse_block(lines[1:])).flatten()
        self.directional = True


    def parse_quantities(self):
        lines = self._currentlines()
        
        m = re.match('\s*(\d+)', lines[1])
        if m:
            n = int(m.groups()[0])
        else:
            raise ValueError('Number of quantities not understood: %s' % lines[1])

        self.specs = []
        for i in range(n):
            q = []
            for j in range(3):
                m = re.match('\s*([^\s]+)', lines[2+3*i+j])
                if m:
                    q.append(m.groups()[0])
            self.specs.append(tuple(q))

        self.n += 1 + 3*n


    def parse_location(self):
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
        self.quantities.append(q)


    def parse_nodata(self):
        self.quantities.append(None)

    
    def parse_timestamp(self):
        line = self._currentline()

        m = re.match('\s*([\d\.]+)', line)
        if m:
            self.time = datetime.strptime(m.groups()[0], '%Y%m%d.%H%M%S')
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
                

    def _currentline(self):
        return self.lines[self.n]


    def _currentlines(self):
        return self.lines[self.n:]
