"""Module containing various special functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from numpy.fft import fft, ifft
from numpy import sign
from scipy import pi
from scipy import log
from scipy import log2
from scipy import exp
from scipy import sqrt
from scipy.special import psi as digamma
from scipy.integrate import quad

from .mytypes import complexnp
from .mytypes import doublenp
from .mytypes import intnp

def fermi_func(x):
    """Fermi function."""
    return 1/(exp(x)+1)

def func_pauli(E, T, D, itype):
    """
    Function used when generating Pauli master equation kernel.

    Parameters
    ----------
    E : float
        Energy.
    T : float
        Temperature.
    D : float
        Bandwidth.

    Returns
    -------
    float
    """
    alpha = E/T
    R = D/T
    if itype == 1:
        return 2*pi*fermi_func(-alpha)
    else:
        return 2*pi*fermi_func(-alpha) if -alpha < R and -alpha > -R else 0

def func_1vN(E, T, D, eta, itype, limit):
    """
    Function used when generating 1vN, Redfield approach kernel.

    Parameters
    ----------
    E : float
        Energy.
    T : float
        Temperature.
    D : float
        Bandwidth.
    itype : int
        Type of integral for first order method calculations.
        itype=0: the principal parts are neglected.
        itype=1: the principal parts are kept, but approximated by digamma function valid for large bandwidht D.
        itype=2: the principal parts are evaluated using Fortran integration package QUADPACK routine dqawc through SciPy.
    limit : int
        For itype=2 dqawc_limit determines the maximum number of subintervals
        in the partition of the given integration interval.

    Returns
    -------
    complex
    """
    alpha = E/T
    R = D/T
    if itype == 0:
        (rez, err) = quad(fermi_func, -R, +R, weight='cauchy', wvar=-alpha, epsabs=1.0e-6, epsrel=1.0e-6, limit=limit)
    elif itype == 1:
        rez = digamma(0.5-1.0j*alpha/(2*pi)).real - log(R/(2*pi))
    elif itype == 2:
        rez = 0.0
    rez = rez - ( 1.0j*pi*fermi_func(-alpha)*sign(eta) if -alpha < R and -alpha > -R else 0 )
    #-------------------------
    return rez

def kernel_fredriksen(n):
    """
    Generates kernel for Hilbert transform using FFT.

    Parameters
    ----------
    n : int
        Number of equidistant grid points.

    Returns
    -------
    array
        Kernel used when performing Hilbert transform using FFT.
    """
    aux = np.zeros(n+1, dtype=doublenp)
    for i in range(1,n+1):
        aux[i] = i*log(i)
    m = 2*n
    ker = np.zeros(m, dtype=doublenp)
    for i in range(1,n):
        ker[i] = aux[i+1]-2*aux[i]+aux[i-1]
        ker[m-i] = -ker[i]
    return fft(ker)/pi

def hilbert_fredriksen(f, ker=None):
    """
    Performs Hilbert transform of f.

    Parameters
    ----------
    f : array
        Values of function on a equidistant grid.
    ker : array
        Kernel used when performing Hilbert transform using FFT.

    Returns
    -------
    array
        Hilbert transform of f.
    """
    if ker is None:
        ker = kernel_fredriksen(len(f))
    n = len(f)
    fpad = fft(np.concatenate( (f,np.zeros(len(ker)-n)) ))
    r = ifft(fpad*ker)
    return r[0:n]
