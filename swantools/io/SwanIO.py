import sys,os
import os.path
import datetime

import scipy.io

import numpy  as np
import pandas as pd

from .. utils import swantime2datetime

class SwanIO:

	def __init__(self):
		pass

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

		    reader = SwanIO()
			headers = ["TIME","HSIGN","HSWELL","PDIR","DIR","TPS","PER","WINDX","WINDY","PROPAGAT"]
			table = reader.read_swantable('file.txt',headers=headers)

			If usind HEAD option, just do:

			reader = SwanIO()
			table  = reader.read_swantable('file_with_headers.txt')

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
			# Read the table file
			rawdata      = np.genfromtxt(fname)
			rawdata[:,0] = times
			index        = np.arange(0,len(rawdata[:,0]),1)
			df           = pd.DataFrame(rawdata[:,:],index=index,columns=headers)
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
			rawdata      = np.genfromtxt(fname,skip_header=7)
			rawdata[:,0] = times
			index        = np.arange(0,len(rawdata[:,0]),1)
			df           = pd.DataFrame(rawdata[:,:],index=index,columns=headers)
			return df


	def read_swanspc(self,fname,swantime):

		"""
		    Use this function to read data generated with the SPECOUT command.

		    The sixtase MUST be :
		    'SPECOUT 'Location' SPEC2D ABS 'name.spc'

		    Read the documentation in http://swanmodel.sourceforge.net to more details on spectral output.

		    Inputs
		    fname:    the name of the file
		    swantime: a date and time string in swans's format

		    Outputs
		    lon:    longitude of the point
		    lat:    latitude of the point
		    nfreqs: number of frequencies
		    freqs:  list of frequencies
		    ndirs:  number of directions
		    dirs:   list of directions
		    spectra: array with spectral data (frequencies,directions)
		"""

		# I/O check
		self.iocheck(fname)

		f = open(fname,'r').readlines()

		# Time check
		check = False
		for line in f:
			if swantime in line:
				check = True
				break
		if check:
			pass
		else:
			raise ValueError('It seems the date requested is not present in the file.')

		for l,line in enumerate(f):

			# Heading the headers
			if "TIME" in line:
				time = f[l+1].split()[0]
			elif "LONLAT" in line:
				lon = float(f[l+2].split()[0])
				lat = float(f[l+2].split()[1])
			elif "AFREQ" in line:
				nfreqs = int(f[l+1].split()[0])
				start  = l+2
				end    = l+nfreqs+1
				freqs  = []
				for i,l in enumerate(f):
					if i >= start and i <= end:
						fq = l.split()[0]
						freqs.append(float(fq))
			elif "NDIR" in line:
				ndirs = int(f[l+1].split()[0])
				start  = l+2
				end    = l+ndirs+1
				dirs  = []
				for i,l in enumerate(f):
					if i >= start and i <= end:
						ds = l.split()[0]
						dirs.append(float(ds))

			# Read the spectrum for a given date
			elif swantime in line:
				factor = float(f[l+2])
				start  = l+3
				end    = l+nfreqs+2
				LINES=[]
				for i,lines in enumerate(f):
					if i >= start and i <= end:
						LINE  = lines.split()
						LINES.append(LINE)
				VALUES=[]
				for block in LINES:
					for strs in block:
						VALUES.append(float(strs))
				spectra=np.reshape(VALUES,(nfreqs,ndirs))*factor

		return lon,lat,freqs,dirs,spectra

	def read_swanblock(self,fname,basename,time=False,stat=False):

		"""
			Function to read SWAN's BLOCK output statment. Both stationary
			and non-stationary versions can be handled there. If requesting
			a non-stationary output a proper date string shoudl be given.
			In case of non-stationary data with no dates, such as XP or YP,
			just call the function as if it were stationary.
			will return a numpy array with [Xp,Yp] dimension.
		"""

		# Reading ths data causes to some useless warnigs to be printed,
		# removing it using "brute-force"
		import warnings
		warnings.filterwarnings("ignore")

		# I/O check
		self.iocheck(fname)

		block = scipy.io.loadmat(fname)
		keys  = block.keys()

		if stat:
			var = basename
			for key in keys:
				if var in keys:
					z   = scipy.io.loadmat(fname)[var]
					return z
				else:
					raise ValueError('It seems the variable requested is \
						                  not present in the file.')


		else:

			if time:

				var = basename+"_"+time.replace('.','_')+"00"

				for k in keys:
					if var in keys:
						z   = scipy.io.loadmat(fname)[var]
						return z
						break
					else:
						raise ValueError('It seems the variable requested is \
						                  not present in the file.')
			else:

				var = basename

				for k in keys:
					if var in keys:
						z   = scipy.io.loadmat(fname)[var]
						return z
						break
					else:
						raise ValueError('It seems the variable requested is \
							              not present in the file.')

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
