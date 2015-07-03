import os.path

import datetime

import scipy.io

import numpy  as np
import pandas as pd

import netCDF4

from .. utils import swantime2datetime

class SwanIO:

	def __init__(self):
		pass

	@classmethod
	def iocheck(self,fname):
		io = os.path.isfile(fname)
		if io:
			pass
		else:
			raise IOError('File {0} not found.'.format(fname))

	def read_swantable(self,fname,headers=[]):

		"""
		    Use this function to read data generated with the command table.
		    Both NOHEAD and HEAD options can be read here.

		    If using NOHEAD,
		    the user must specify with variables are being read, for example:

		    R  = swantools.io.SwanIO()
			headers = ["TIME","HSIGN","HSWELL","PDIR","DIR","TPS","PER","WINDX","WINDY","PROPAGAT"]
			df = R.read_swantable('file.txt',headers=headers)

			If usind HEAD option, just do:

			R  = swantools.io.SwanIO()
			df = R.read_swantable('file_with_headers.txt')

			The function will return a pandas DataFrame.
		"""

		# I/O check
		self.iocheck(fname)

		f=open(fname,'r').readlines()

		dates=[]

		if headers:
			# Handle times
			for line in f: dates.append(line.split()[0])
			times        = swantime2datetime(dates)
			# print times
			# Read the table file
			rawdata = np.genfromtxt(fname)
			data    = rawdata[:,1:]
			# rawdata[:,0] = times
			# index        = np.arange(0,len(rawdata[:,0]),1)
			df           = pd.DataFrame(data,index=times,columns=headers[1:])
			return df
		else:
			# Handle times
			for i,line in enumerate(f):
				if i > 6: dates.append(line.split()[0])
			times        = swantime2datetime(dates)
			# Handle headers
			headers     = []
			for i,h in enumerate(f[4].split()):
				if i >0: headers.append(h)
			# Read the table file
			rawdata = np.genfromtxt(fname,skip_header=7)
			data    = rawdata[:,1:]
			df      = pd.DataFrame(data[:,:],index=times,columns=headers[1:])
			return df


	def read_swanspc(self,fname):

		"""
		    Use this function to read data generated with the SPECOUT command.

		    The sixtase MUST be :
		    'SPECOUT 'Location' SPEC2D ABS 'name.spc'

		    Read the documentation in http://swanmodel.sourceforge.net to more details on spectral output.

		    Inputs
		    fname:    the name of the file
		    swantime: a date and time string in swans's format

		    Outputs
		    lon:     longitude of the point
		    lat:     latitude of the point
		    freqs:   list of frequenciess
		    dirs:    list of directions
			times:   list of all times
		    spectra: array with spectral data [time,frequencies,directions]
		"""

		# I/O check
		self.iocheck(fname)

		f = open(fname,'r').readlines()
		# Heading the headers
		for l,line in enumerate(f):
			if "TIME" in line:
				time = f[l+1].split()[0]
			if "LONLAT" in line:
				lon = float(f[l+2].split()[0])
				lat = float(f[l+2].split()[1])
			if "AFREQ" in line:
				nfreqs = int(f[l+1].split()[0])
				start  = l+2
				end    = l+nfreqs+1
				freqs  = []
				for i,l in enumerate(f):
					if i >= start and i <= end:
						fq = l.split()[0]
						freqs.append(float(fq))
			if "NDIR" in line:
				ndirs = int(f[l+1].split()[0])
				start  = l+2
				end    = l+ndirs+1
				dirs  = []
				for i,l in enumerate(f):
					if i >= start and i <= end:
						ds = l.split()[0]
						dirs.append(float(ds))
			# if l > 300: break

		# Dates
		dates = []
		with open(fname, 'r') as f:
		    for line in f:
		        if 'date and time' in line:
					dates.append(line.split()[0][0:15])
		times  = swantime2datetime(dates)
		ntimes = len(times)
		#
		factors  = []
		spectrum = np.ones([ntimes,nfreqs,ndirs])
		#
		f = open(fname,'r').readlines()
		for l,line in enumerate(f):
			for t,date in enumerate(dates):
				if date in line:
					factor = float(f[l+2])
					start  = l+3
					end    = l+nfreqs+2
					spclines=[]
					for i,lines in enumerate(f):
						if i >= start and i <= end:
							il  = lines.split()
							spclines.append(il)
					values=[]
					for block in spclines:
						for strs in block:
							values.append(float(strs))
					factors.append(float(factor))
					spectra = np.reshape(values,(nfreqs,ndirs))
					spectrum[t,:,:] = spectra

		return lat,lon,freqs,dirs,times,factors,spectrum

	def read_swanblock(self,fname,var,stat=False):

		"""
			Function to read SWAN's BLOCK output statement. Both stationary
			and non-stationary versions can be handled there. If requesting
			a non-stationary output a proper date string should be given.
			In case of non-stationary data with no dates, such as XP or YP,
			just call the function as if it were stationary.
			will return a numpy array with [Xp,Yp] dimension.
		"""

		# Reading this data causes to some useless warnings to be printed,
		# removing it using "brute-force"
		import warnings
		warnings.filterwarnings("ignore")

		# I/O check
		self.iocheck(fname)

		block = scipy.io.loadmat(fname)
		keys  = block.keys()

		if stat:
			times = 0
			for key in keys:
				if var in keys:
					x = block["Xp"]
					y = block["Yp"]
					x = np.linspace(x.min(),x.max(),x.shape[1])
					y = np.linspace(y.min(),y.max(),y.shape[0])
					z   = scipy.io.loadmat(fname)[var]
					return x,y,times,z
				else:
					raise ValueError('It seems the variable requested is \
						                  not present in the file.')

		else:
			# digging the data
			dates = []
			for key in keys:
				if var in key.split("_")[0]:
					date = key.split("_")[1]+"."+key.split("_")[2]
					dates.append(date)

			times  = np.sort(swantime2datetime(dates))
			ntimes = len(times)

			# geometry
			x = block["Xp"]
			y = block["Yp"]
			x = np.linspace(x.min(),x.max(),x.shape[1])
			y = np.linspace(y.min(),y.max(),y.shape[0])

			# output
			blockout = np.ones([ntimes,len(y),len(x)])
			fmt       = "_%Y%m%d_%H%M%S"
			for t,time in enumerate(times):
				keyname = var+time.strftime(fmt)
				z       = block[keyname]
				blockout[t,:,:] = z
			return x,y,times,blockout


	def write_spectrum(self,fname,lat,lon,times,freqs,dirs,facs,spc):


		if isinstance(lat, (list, tuple, np.ndarray)):
			raise ValueError ("Only single location are supported at this moment")
		else:
			pass

		# To SWAN time convention
		stimes = swantime2datetime(times,inverse=True)

		f = open(fname,"w")

		f.write("SWAN   1          Swan standard spectral file, version \n")
		f.write("$ Data produced by swantools version 0.2 \n")
		f.write("TIME              time-dependent data \n")
		f.write("1                 time coding option \n")
		f.write("LONLAT            locations in spherical coordinates \n")
		f.write("1                 number of locations \n")
		f.write(" {} {} \n".format(lon,lat))
		f.write("AFREQ             absolute frequencies in Hz \n")
		f.write("     {}           number of frequencies \n".format(len(freqs)))
		for freq in freqs:
			f.write(" {} \n".format(str(freq).ljust(6,"0")))
		f.write("NDIR              spectral nautical directions in degr \n")
		f.write("    {}            number of directions \n".format(len(dirs)))
		for dir in dirs:
			f.write(" {} \n".format(str(dir).ljust(8,"0")))
		f.write("QUANT \n")
		f.write("     1            number of quantities in table \n")
		f.write("VaDens            variance densities in m2/Hz/degr \n")
		f.write("m2/Hz/degr        unit \n")
		f.write("   -0.9900E+02    exception value \n")
		for t,time in enumerate(stimes):
			f.write("{}  date and time \n".format(time))
			f.write("FACTOR \n")
			f.write(" {} \n".format(facs[t]))
			np.savetxt(f,spc[t,:,:],'%.4i')
		# f.write(" \n")
		# f.write(" \n")
		# f.write(" \n")





		# # c1 = lat.size
		# try:
		# 	c1 = lat[1]
		# except Exception as e:
		# 	raise
		#
		# try
		# 	c1 = lat[1]



	def write_tpar(self,data,fname='tpar'):

		"""
			Given an array Nx5, where N is the number of timestamps and
			the columns are respecvely, TIME, HS, TP, DP, SP, this function
			will write TPAR formated  file named *fname*.
		"""

		# check data dimensions
		if data.shape[1] == 5:
			pass
		else:
			raise ValueError('You MUST give HS, TP, DP and SP \
			                          (in this order) information.')

		f = open(fname,'w')

		# Write header
		f.write("TPAR \n")

		# Timeloop
		times = swantime2datetime(data[:,0],inverse=True)
		for t,time in enumerate(times):

			hs = str(abs(np.round(data[t,1],2)))
			tp = str(abs(np.round(data[t,2],2)))
			dp = str(abs(np.round(data[t,3],2)))
			sp = str(abs(np.round(data[t,4],2)))

			tpar = time+" {0} {1} {2} {3} \n".format(hs,tp,dp,sp)

			f.write(tpar)

class Converters:

	def __init__(self):
		pass

	def np2nc(self,fname,lat,lon,ts,z,var):

		ncout = netCDF4.Dataset(fname, 'w', format='NETCDF4')

		t   = ncout.createDimension('time',len(ts))
		x   = ncout.createDimension('longitude',len(lon))
		y   = ncout.createDimension('latitude', len(lat))

		time = ncout.createVariable('times','f8',('time',))
		lats = ncout.createVariable('lat', 'f8', ('latitude',))
		lons = ncout.createVariable('lon', 'f8', ('longitude',))

		lats[:] = lat
		lons[:] = lon
		time[:] = netCDF4.date2num(ts,units="hours since 1900-01-01 00:00:0.0")

		lats.units    = 'degrees_north'
		lons.units    = 'degrees_east'
		time.units    = 'hours since 1900-01-01 00:00:0.0'
		time.calendar = 'gregorian'

		out = ncout.createVariable(var, 'f4', ('time', 'latitude', 'longitude',))

		out[:,:,:] = z[:,:,:]

		ncout.close()

	def spc2nc(sefl,fname,lat,lon,freq,dirr,time,facs,spc):

		ncout = netCDF4.Dataset(fname, 'w', format='NETCDF4')

		ncout.createDimension('time',len(time))         # Temporal  dimension
		ncout.createDimension('frequency',len(freq))    # Frequency dimension
		ncout.createDimension('direction', len(dirr))   # Direction dimension

		times = ncout.createVariable('times','f8',('time',))
		freqs = ncout.createVariable('frequencies', 'f8', ('frequency',))
		dirrs = ncout.createVariable('directions', 'f8', ('direction',))

		times[:] = netCDF4.date2num(time,units="hours since 1900-01-01 00:00:0.0")
		freqs[:] = freq
		dirrs[:] = dirr

		freqs.units    = 'hertz'
		dirrs.units    = 'degrees'
		times.units    = 'hours since 1900-01-01 00:00:0.0'
		times.calendar = 'gregorian'

		name   = "Spectral Energy"
		spcout = ncout.createVariable(name, 'f4', ('time', 'frequency', 'direction',))

		spcout.units = "m2/Hz/degr"

		for t,time in enumerate(times):
			spcout[t,:,:] = spc[t,:,:]*facs[t]

		ncout.close()
