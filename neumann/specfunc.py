"""Module containing various special functions."""

from __future__ import division
import numpy as np
from numpy.fft import fft, ifft
from numpy import sign
from scipy import pi
from scipy import log
from scipy import log2
from scipy import exp
from scipy import sqrt
from scipy.special import psi as digamma

from mytypes import complexnp
from mytypes import doublenp
from mytypes import intnp

def fermi_func(x):
    """Fermi function."""
    return 1/(exp(x)+1)

def func_pauli(E, T, D):
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
    #NOTE: Also finite bandwidth can be included as a cut-off
    alpha = E/T
    return 2*pi*fermi_func(-alpha)

def func_1vN(E, T, D, eta):
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

    Returns
    -------
    complex
    """
    alpha = E/T
    R = D/T
    rez = digamma(1/2+alpha/(2*pi*1j)).real - log(R/(2*pi)) - 1j*pi*fermi_func(-alpha)*sign(eta)
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
