What's New
==========

v1.0.1 (unreleased)
-------------------

Breaking changes
^^^^^^^^^^^^^^^^

None.

Improvements
^^^^^^^^^^^^

* Also read units from quantities other than VarDens (e.g. EnDens and
  AcDens)

* Package `pyproj` is not an optional dependency. Coordinate
  conversion is disabled if `pyproj` is not installed. Instead a
  warning is given in that situation.

* Raise NotImplemented exception if SWAN test file with ITER keyword
  is read.

New functions/methods
^^^^^^^^^^^^^^^^^^^^^

* Added `SwanBlockReader` for memory efficient reading of large
  files. The `SwanSpcReader` now uses this class to read spectrum
  files with a minimum required number of lines in memory at each
  point in the reading procedure.

* Added support for the `ZERO` keyword in SWAN spectrum files. `ZERO`
  now results in zeros, while `NODATA` results in NaN values.

Bug fixes
^^^^^^^^^

* Store comments from SWAN spectrum files as single string intead of a
  list of strings as the scipy netcdf I/O cannot cope with lists of
  strings.

* Do not set units attribute on time coordinate if not given, as the
  time is likely to be given by a list of datetime objects that are
  automatically encoded by xarray. Setting the units attribute
  manually would raise an exception if the dataset is written to
  netCDF.

* Do not squeeze energy matrix as that may cause conflicts with
  dimensions of length unity. Instead, reshape the energy matrix to
  the approriate size.

Tests
^^^^^

* Added tests for converting SWAN files to netCDF.

* Added to extra example input files: a *.hot hotstart file, a *.sp2
  file without data and a *.sp2 file with a single location.

* Test SWAN I/O not only for reading the proper shapes, but also the
  proper values.

v1.0.0 (15 November 2017)
-------------------------

Initial release
