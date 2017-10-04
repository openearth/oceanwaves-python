from __future__ import absolute_import

import numpy as np

from oceanwaves.utils import *


def jonswap(f, Hm0, Tp, gamma=3.3, sigma_low=.07, sigma_high=.09,
            g=9.81, method='yamaguchi', normalize=True):
    '''Generate JONSWAP spectrum

    Parameters
    ----------
    f : numpy.ndarray
        Array of frequencies
    Hm0 : float
        Required zeroth order moment wave height
    Tp : float
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
        Array of shape ``f`` with wave energy densities 
    
    '''

    # Pierson-Moskowitz
    if method.lower() == 'yamaguchi':
        alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.
    elif method.lower() == 'goda':
        alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
    else:
        raise ValueError('Unknown method: %s' % method)

    E_pm  = alpha * Hm0**2 * Tp**-4 * f**-5 * np.exp(-1.25 * (Tp * f)**-4)
        
    # JONSWAP
    sigma = np.ones(f.shape) * sigma_low
    sigma[f > 1./Tp] = sigma_high

    E_js = E_pm * gamma**np.exp(-0.5 * (Tp * f - 1)**2. / sigma**2.);
    
    if normalize:
        E_js *= Hm0**2. / (16. * trapz_and_repeat(E_js, f, axis=-1))
        
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
    #A1 = (2.**s) * (gamma(s / 2 + 1))**2. / (np.pi * gamma(s + 1))
    #p_theta = A1 * np.maximum(0., np.cos(theta - theta_peak))
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

