from nose.tools import *
import oceanwaves as ow


SPECTRA = [
    'data/swan/P1.SP1',
    'data/swan/P1.SP2',
    'data/swan/P1b.SP1',
    'data/swan/a11.hot',
    'data/swan/a111uref01.sp2'
]

TABLES = [
    'data/swan/P1.TAB',
    'data/swan/TBL1.tbl',
    'data/swan/TBL2.tbl',
    'data/swan/TBL2NS.tbl',
]


def test_netcdf_spectra():
    '''Test writing of swan spectra to netcdf'''
    for s in SPECTRA:
        yield readwrite_spectrum, s


def test_netcdf_tables():
    '''Test writing of swan tables to netcdf'''
    for t in TABLES:
        yield readwrite_table, t


def readwrite_spectrum(spcfile):
    ow_spc = ow.OceanWaves.from_swan(spcfile)
    ow_spc.to_netcdf('%s.nc' % spcfile)


def readwrite_table(tabfile):
    ow_tab = ow.OceanWaves.from_swantable(
        tabfile,
        columns=['Xp','Yp','Botlev','Hsig',
                 'RTpeak','TPsmoo','Tm01','Tm02'])
    ow_tab.to_netcdf('%s.nc' % tabfile)
