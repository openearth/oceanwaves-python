import re
import glob
import logging
import xarray as xr
import numpy as np
import pandas as pd
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

        # TABLE parameters
        self.run = None
        self.table = None
        self.variables = []
        self.units = []

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


    def read_spc(self, fpath):

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


    def write_spc(self, obj, fpath):

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


    def read_table(self,fpath,headers=[]):
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
        swanoutputs = self._swan_tableoutputs()

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
                        for svar,sunit in zip(swanoutputs["output"],swanoutputs["unit"]):
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
                ds = xr.concat(dss,dim='locations')

            # non stationary cases
            else:
                # could figure out variable names
                if self.variables:
                    # read raw data into the dataframe
                    df = pd.DataFrame(np.array(data),columns=self.variables)
                    # update time
                    self.time = np.unique(swantime2datetime(df["Time"].values))
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
                    ds = xr.concat(dss,dim='locations')

                else:
                    raise ValueError("When reading a non-stationary table, please use \'HEAD\' option.")

            # Update metatada
            ds.attrs["version"] = self.version
            ds.attrs["run"] = self.run
            ds.attrs["table"] =  self.run
            ds.attrs["variables"] =  self.variables
            ds.attrs["units"] =  self.units

            return ds


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


    def _swan_tableoutputs(self):
        ''' SWAN's TABLE output options and units """.
        
        Parameters
        ----------
        
        Returns
        -------
        swantable : Dict
            Dictonary of output names and units.
        '''
        output = [  'Hsig','Tsec','Smud','WlPT10','WfPT10',
                    'Hswell','Xp','HsPT01','DrPT01','StPT01',
                    'Dir','Yp','HsPT02','DrPT02','StPT02',
                    'PkDir','Dist','HsPT03','DrPT03','StPT03',
                    'TDir','Setup','HsPT04','DrPT04','StPT04',
                    'Tm01','Tm_10','HsPT05','DrPT05','StPT05',
                    'RTm01','RTm_10','HsPT06','DrPT06','StPT06',
                    'RTpeak','Depth','HsPT07','DrPT07','StPT07',
                    'Tm02','TmBot','HsPT08','DrPT08','StPT08',
                    'FSpr','Qp','HsPT09','DrPT09','StPT09',
                    'Dspr','BFI','HsPT10','DrPT10','StPT10',
                    'X-Vel','Watlev','TpPT01','DsPT01',
                    'Y-Vel','Botlev','TpPT02','DsPT02',
                    'FrCoef','TPsmoo','TpPT03','DsPT03',
                    'X-Windv','Sfric','TpPT04','DsPT04',
                    'Y-Windv','Ssurf','TpPT05','DsPT05',
                    'Dissip','Swcap','TpPT06','DsPT',
                    'Qb','Genera','TpPT07','DsPT07',
                    'X-Transp','Swind','TpPT08','DsPT08',
                    'Y-Transp','Redist','TpPT09','DsPT09',
                    'X-WForce','Snl4','TpPT10','DsPT10',
                    'Y-WForce','Snl3','WlPT01','WfPT01',
                    'Ubot','Propag','WlPT02','WfPT02',
                    'Urms','Propxy','WlPT03','WfPT03',
                    'Wlen','Propth','WlPT04','WfPT04',
                    'Steepn','Propsi','WlPT05','WfPT05',
                    'dHs','Radstr','WlPT06','WfPT06',
                    'dTm','Lwavp','WlPT07','WfPT07',
                    'Leak','Stur','WlPT08','WfPT08',
                    'Time','Turb','WlPT09','WfPT09'  ]
        # units
        unit = [  'm','s','m2/s','m',None,
                  'm','m','m','degr',None,
                  'degr','m','m','degr',None,
                  'degr','m','m','degr',None,
                  'degr','m','m','degr',None,
                  'sec','sec','m','degr',None,
                  'sec','sec','m','degr',None,
                  'sec','m','m','degr',None,
                  'sec','sec','m','degr',None,
                  None,None,'m','degr',None,
                  'degr',None,'m','degr',None,
                  'm/s','m','sec','degr',
                  'm/s','m','sec','degr',
                  None,'sec','sec','degr',
                  'm/s','m2/s','sec','degr',
                  'm/s','m2/s','sec','degr',
                  'm2/s','m2/s','sec','degr',
                  None,'m2/s','sec','degr',
                  'm3/s','m2/s','sec','degr',
                  'm3/s','m2/s','sec','degr',
                  'N/m2','m2/s','sec','degr',
                  'N/m2','m2/s','m',None,
                  'm/s','m2/s','m',None,
                  'm/s','m2/s','m',None,
                  'm','m2/s','m',None,
                  None,'m2/s','m',None,
                  'm','m2/s','m',None,
                  'sec','m','m',None,
                  'm2/s','m2/s','m',None,
                  None,'m2/s','m',None  ]

        swantable = {"output":output,"unit":unit}

        return swantable


def swantime2datetime(time,inverse=False,fmt="%Y%m%d.%H%M%S"):
    ''' Translate SWANS's time string to datetimes and vice-versa.
    
    Parameters
    ----------
    time : numpy.ndarray
        Array of SWAN's time strings
    inverse : bool
        If True, translate datetimes into SWAN's time strings
    fmt : str
        Which string format to use. Default is Ymd.HMS
    
    Returns
    -------
    times : numpy.ndarray
        Array of converted times
    '''
    times = []

    if inverse:
        for date in time:
            times.append(datetime.strftime(date,fmt))
        return np.array(times)
    else:
        for date in time:
            times.append(datetime.strptime(date,fmt))
        return np.array(times)
