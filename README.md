# oceanwaves-python

[![CircleCI](https://circleci.com/gh/openearth/oceanwaves-python.svg?style=svg)](https://circleci.com/gh/openearth/oceanwaves-python)

This toolbox provides a generic data storage object for ocean waves data (OceanWaves). OceanWaves is built upon the xarray.Dataset data storage object, but defines special variables for time, location, frequency and direction. Many of its functionalities are obtained from the pyswan toolbox, originally developed by Gerben de Boer, and the swantools toolbox, originally developed by Caio Eadi Stringari.

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
* Plots supported by xarray.Dataset and Seaborn

The OceanWaves object can be instantiated from:
* Raw data
* SWaN 1D/2D spectral or table files
* An xarray.Dataset object
* Another OceanWaves object

The OceanWaves object can be written to:
* SWaN 1D/2D spectral files
* Output supported by xarray.Dataset (e.g. netcdf)

Usage examples can be found in the IPython notebook notebooks/oceanwaves.ipynb
