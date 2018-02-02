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

New functions/methods
^^^^^^^^^^^^^^^^^^^^^

None.

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

Tests
^^^^^

* Added tests for converting SWAN files to netCDF.

* Added to extra example input files: a *.hot hotstart file and a
  *.sp2 file without data.

v1.0.0 (15 November 2017)
-------------------------

Initial release
