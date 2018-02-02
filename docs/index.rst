.. oceanwaves documentation master file, created by
   sphinx-quickstart on Wed Dec  7 12:28:55 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to oceanwaves's documentation!
======================================

Oceanwaves is a Python package that provides a generic data storage
object for ocean wave data (time series and/or spectral). The package
provides a series of I/O functions to use with various file formats,
like `SWAN <http://swanmodel.sourceforge.net>`_, `Waverider buoys
<http://datawell.nl>`_ and `WaveDroid <http://www.wavedroid.net>`_.

Oceanwaves is based on the `xarray DataSet object
<https://pypi.python.org/pypi/xarray>`_, but defines special variables
for time, location, frequency and direction. Many of its
functionalities are obtained from the `pyswan
<https://github.com/openearth/pyswan>`_ toolbox, originally developed
by Gerben de Boer, and the `swantools
<https://pypi.python.org/pypi/swantools/>`_ toolbox, originally
developed by Caio Eadi Stringari.

The source code of the oceanwaves package can be found at
`<https://github.com/openearth/oceanwaves-python>`_.

Usage examples can be found in this notebook
`<https://github.com/openearth/oceanwaves-python/blob/master/notebooks/oceanwaves.ipynb>`_.

The OceanWaves object supports various standard conversions, like:

* From significant wave height to spectral
* From omnidirectional to directional
* From directional to omnidirectional
* From spectral to significant wave height
* From spectral to spectral wave period
* From spectral to peak wave period
* From directional to peak wave direction
* From degrees to radians

The OceanWaves object supports various standard plotting methods, like:

* Polar subplots for directional data on multiple locations/times
* Polar subplots on a map
* Plots supported by `xarray.Dataset
  <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_
  and `Seaborn <http://seaborn.pydata.org>`_

The OceanWaves object can be instantiated from:

* Raw data
* SWaN 1D/2D spectral or table files
* An `xarray.Dataset
  <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_
  object
* Another OceanWaves object

The OceanWaves object can be written to:

* SWaN 1D/2D spectral files
* Output supported by `xarray.Dataset
  <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_
  (e.g. netcdf)

Contents:

.. toctree::
   :maxdepth: 2

   sourcecode
   whatsnew


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

