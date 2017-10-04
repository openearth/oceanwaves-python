from nose.tools import *
import numpy as np
from oceanwaves.spectral import *

FREQ = np.arange(0, 1, .05)[1:]
THETA = np.arange(0, 180, 5)

def test_jonswap_1():
    '''Test computation of jonswap spectrum following Yamaguchi'''
    E = jonswap(FREQ, Hm0=1., Tp=4., method='yamaguchi',
                normalize=True)
    assert_almost_equals(np.trapz(E, FREQ), 1/16.)
    assert_almost_equals(FREQ[np.argmax(E)], 1/4.)


def test_jonswap_2():
    '''Test computation of jonswap spectrum following Goda'''
    E = jonswap(FREQ, Hm0=1., Tp=4., method='goda', normalize=True)
    assert_almost_equals(np.trapz(E, FREQ), 1/16.)
    assert_almost_equals(FREQ[np.argmax(E)], 1/4.)


@raises(ValueError)
def test_jonswap_exception1():
    '''Test exception if unsupported spectrum method is requested'''
    E = jonswap(FREQ, Hm0=1., Tp=4., method='pm', normalize=True)


def test_spreading_1():
    '''Test computation of directional spreading in degrees'''
    D = directional_spreading(THETA, theta_peak=90., s=20.,
                              units='deg', normalize=True)
    assert_almost_equals(np.trapz(D, THETA), 1.)
    assert_almost_equals(THETA[np.argmax(D)], 90.)


def test_spreading_2():
    '''Test computation of directional spreading in radians'''
    D = directional_spreading(np.radians(THETA),
                              theta_peak=np.radians(90.), s=20.,
                              units='rad', normalize=True)
    assert_almost_equals(np.trapz(D, np.radians(THETA)), 1.)
    assert_almost_equals(THETA[np.argmax(D)], 90.)


@raises(ValueError)
def test_spreading_exception1():
    '''Test exception if unsupported units are used'''
    D = directional_spreading(THETA, theta_peak=90., s=20.,
                              units='gon', normalize=True)
