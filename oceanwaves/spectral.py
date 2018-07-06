from __future__ import absolute_import

import numpy as np

from oceanwaves.utils import *

import logging
logger = logging.getLogger(__name__)


def jonswap(f, Hm0, Tp, gamma=3.3, sigma_low=.07, sigma_high=.09,
            g=9.81, method='yamaguchi', normalize=True):
    '''Generate JONSWAP spectrum

    Parameters
    ----------
    f : numpy.ndarray
        Array of frequencies
    Hm0 : float, numpy.ndarray
        Required zeroth order moment wave height
    Tp : float, numpy.ndarray
        Required peak wave period
    gamma : float
        JONSWAP peak-enhancement factor (default: 3.3)
    sigma_low : float
        Sigma value for frequencies <= ``1/Tp`` (default: 0.07)
    sigma_high : float
        Sigma value for frequencies > ``1/Tp`` (default: 0.09)
    g : float
        Gravitational constant (default: 9.81)
    method : str
        Method to compute alpha (default: yamaguchi)
    normalize : bool
        Normalize resulting spectrum to match ``Hm0``

    Returns
    -------
    E : numpy.ndarray
        Array of shape ``f, Hm0.shape`` with wave energy densities

    '''

    # C Stringari - 04/06/2018
    # check input data types to avoid the following error:
    # ValueError: Integers to negative integer powers are not allowed.

    # raise an warning if the frequency array starts with zero. if the
    # user gives an array with zeros, the output will be inf at that
    # frequency
    if 0.0 in f:
        logger.warn('Frequency array contains zeros.')

    # get the input dtypes and promote to float, if needed
    f = ensure_float(f)
    Hm0 = ensure_float(Hm0)
    Tp = ensure_float(Tp)

    # check shapes of Hm0 and Tp, raise an error if the don't match
    if isinstance(Hm0, np.ndarray):
        if isinstance(Tp, np.ndarray):
            if Hm0.shape != Tp.shape:
                raise ValueError("Dimensions of Hm0 and Tp should match.")

    # This is a very naive implementation to deal with array inputs,
    # but will work if Hm0 and Tp are vectors.
    if isinstance(Hm0, np.ndarray):
        f = f[:, np.newaxis].repeat(len(Hm0), axis=1)
        Hm0 = Hm0[np.newaxis, :].repeat(len(f), axis=0)
        Tp = Tp[np.newaxis, :].repeat(len(f), axis=0)

    # Pierson-Moskowitz
    if method.lower() == 'yamaguchi':
        alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.
    elif method.lower() == 'goda':
        alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
    else:
        raise ValueError('Unknown method: %s' % method)

    E_pm = alpha * Hm0**2 * Tp**-4 * f**-5 * np.exp(-1.25 * (Tp * f)**-4)

    # JONSWAP
    sigma = np.ones(f.shape) * sigma_low
    sigma[f > 1./Tp] = sigma_high

    E_js = E_pm * gamma**np.exp(-0.5 * (Tp * f - 1)**2. / sigma**2.)

    if normalize:
        # axis=0 seems to work fine with all kinds of inputs
        E_js *= Hm0**2. / (16. * trapz_and_repeat(E_js, f, axis=0))

    return E_js


def directional_spreading(theta, theta_peak=0., s=20., units='deg',
                          normalize=True):
    '''Generate wave spreading

    Parameters
    ----------
    theta : numpy.ndarray
        Array of mean bin directions
    theta_peak : float
        Peak direction (default: 0)
    s : float
        Exponent in cosine law (default: 20)
    units : str
        Directional units (deg or rad, default: deg)
    normalize : bool
        Normalize resulting spectrum to unity

    Returns
    -------
    p_theta : numpy.ndarray
       Array of directional weights
    '''

    from math import gamma

    theta = np.asarray(theta, dtype=np.float)

    # convert units to radians
    if units.lower().startswith('deg'):
        theta = np.radians(theta)
        theta_peak = np.radians(theta_peak)
    elif units.lower().startswith('rad'):
        pass
    else:
        raise ValueError('Unknown units: %s')

    # compute directional spreading
    # A1 = (2.**s) * (gamma(s / 2 + 1))**2. / (np.pi * gamma(s + 1))
    # p_theta = A1 * np.maximum(0., np.cos(theta - theta_peak))
    p_theta = np.maximum(0., np.cos(theta - theta_peak))**s

    # convert to original units
    if units.lower().startswith('deg'):
        theta = np.degrees(theta)
        theta_peak = np.degrees(theta_peak)
        p_theta = np.degrees(p_theta)

    # normalize directional spreading
    if normalize:
        p_theta /= trapz_and_repeat(p_theta, theta - theta_peak, axis=-1)

    return p_theta


def ensure_float(var):
    '''
    Auxiliary function to detect and fix dtypes, if needed

    Parameters
    ----------
    var : anything
        Array to check

    Returns
    -------
    var : numpy.ndarray
       The same as the input but either as a float or an array of floats
    '''
    # fist, if it's a list, convert to a numpy.ndarray
    if isinstance(var, list):
        var = np.array(var)
    # if it's an np.ndarray(), make sure it's a array of floats
    elif isinstance(var, np.ndarray):
        if var.dtype != np.dtype('float'):
            var = var.astype(np.float)
    # if it's a float, well, do nothing
    elif isinstance(var, float):
        var = var
    # unknown data type
    else:
        raise ValueError("Data type could not be detected automatically.")
    # allways return a float or array of floats
    return var
