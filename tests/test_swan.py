from nose.tools import *
import numpy as np
import oceanwaves as ow


def test_swan_1():
    '''Test reading of one-dimensional swan spectrum'''
    ow_sp1 = ow.OceanWaves.from_swan('data/swan/P1.SP1')
    assert_equal(ow_sp1.shape, (4,47))


def test_swan_2():
    '''Test reading of two-dimensional swan spectrum'''
    ow_sp2 = ow.OceanWaves.from_swan('data/swan/P1.SP2')
    assert_equal(ow_sp2.shape, (4,47,42))


def test_swan_3():
    '''Test reading of swan table'''
    ow_tab = ow.OceanWaves.from_swantable('data/swan/P1.TAB')
    assert_equal(ow_tab.shape, (35,))

    
def test_swan_4():
    '''Test reading of swan table without header'''
    ow_tab = ow.OceanWaves.from_swantable('data/swan/TBL1.tbl',
                               columns=['Xp','Yp','Botlev','Hsig',
                                        'RTpeak','TPsmoo','Tm01','Tm02'])
    assert_equal(ow_tab.shape, (8,))


def test_swan_5():
    '''Test reading of swan table'''
    ow_tab = ow.OceanWaves.from_swantable('data/swan/TBL2.tbl')
    assert_equal(ow_tab.shape, (8,))


def test_swan_6():
    '''Test reading of instationary swan table'''
    ow_tab = ow.OceanWaves.from_swantable('data/swan/TBL2NS.tbl')
    assert_equal(ow_tab.shape, (7,8))


def test_swan_7():
    '''Test writing of one-dimensional swan spectrum'''
    fname1 = 'data/swan/P1.SP1'
    fname2 = 'data/swan/P1.SP1.copy'
    ow_sp1_1 = ow.OceanWaves.from_swan(fname1)
    ow_sp1_1.to_swan(fname2)
    ow_sp1_2 = ow.OceanWaves.from_swan(fname2)
    assert_almost_equals(np.sum(ow_sp1_1['_energy'].values - 
                                ow_sp1_2['_energy'].values), 0.)


def test_swan_8():
    '''Test writing of two-dimensional swan spectrum'''
    fname1 = 'data/swan/P1.SP2'
    fname2 = 'data/swan/P1.SP2.copy'
    ow_sp2_1 = ow.OceanWaves.from_swan(fname1)
    ow_sp2_1.to_swan(fname2)
    ow_sp2_2 = ow.OceanWaves.from_swan(fname2)
    assert_almost_equals(np.sum(ow_sp2_1['_energy'].values - 
                                ow_sp2_2['_energy'].values), 0., places=3)
