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

def func_pauli(Ecb, mu, T, Dm, Dp, itype):
    """
    Function used when generating Pauli master equation kernel.

    Parameters
    ----------
    Ecb : float
        Energy.
    mu : float
        Chemical potential.
    T : float
        Temperature.
    Dm,Dp : float
        Bandwidth.

    Returns
    -------
    array
        | Array of two float numbers [cur0, cur1] containing
          momentum-integrated current amplitudes.
        | cur0 - particle current amplitude.
        | cur1 - hole current amplitude.
    """
    alpha = (Ecb-mu)/T
    Rm, Rp = (Dm-mu)/T, (Dp-mu)/T
    if itype == 1 or itype == 3 or (alpha < Rp and alpha > Rm):
        cur0 = fermi_func(alpha)
        cur1 = 1-cur0
        rez = 2*pi*np.array([cur0, cur1])
    else:
        rez = np.zeros(2)
    return rez

def func_1vN(Ecb, mu, T, Dm, Dp, itype, limit):
    """
    Function used when generating 1vN, Redfield approach kernel.

    Parameters
    ----------
    Ecb : float
        Energy.
    mu : float
        Chemical potential.
    T : float
        Temperature.
    Dm,Dp : float
        Bandwidth.
    itype : int
        | Type of integral for first order approach calculations.
        | itype=0: the principal parts are evaluated using Fortran integration package QUADPACK \
                   routine dqawc through SciPy.
        | itype=1: the principal parts are kept, but approximated by digamma function valid for \
                   large bandwidht D.
        | itype=2: the principal parts are neglected.
        | itype=3: the principal parts are neglected and infinite bandwidth D is assumed.
    limit : int
        For itype=0 dqawc_limit determines the maximum number of subintervals
        in the partition of the given integration interval.

    Returns
    -------
    array
        | Array of four complex numbers [cur0, cur1, en0, en1] containing
          momentum-integrated current amplitudes.
        | cur0 - particle current amplitude.
        | cur1 - hole current amplitude.
        | en0 - particle energy current amplitude.
        | en1 - hol energy current amplitude.
    """
    if itype == 0:
        alpha, Rm, Rp = (Ecb-mu)/T, (Dm-mu)/T, (Dp-mu)/T
        cur0, err = quad(fermi_func, Rm, Rp, weight='cauchy', wvar=alpha, epsabs=1.0e-6,
                                             epsrel=1.0e-6, limit=limit)
        cur0 = cur0 + (-1.0j*pi*fermi_func(alpha) if alpha < Rp and alpha > Rm else 0)
        cur1 = cur0 + log(abs((Rm-alpha)/(Rp-alpha)))
        cur1 = cur1 + (1.0j*pi if alpha < Rp and alpha > Rm else 0)
        #
        const0 = T*((-Rm if Rm < -40 else log(1+exp(-Rm)))
                   -(-Rp if Rp < -40 else log(1+exp(-Rp))))
        const1 = const0 + Dm-Dp
        #
        en0 = const0 + Ecb*cur0
        en1 = const1 + Ecb*cur1
    elif itype == 1:
        alpha, Rm, Rp = (Ecb-mu)/T, Dm/T, Dp/T
        cur0 = digamma(0.5+1.0j*alpha/(2*pi)).real - log(abs(Rm)/(2*pi))
        cur0 = cur0 - 1.0j*pi*fermi_func(alpha)
        cur1 = cur0 + log(abs(Rm/Rp))
        cur1 = cur1 + 1.0j*pi
        #
        en0 = -T*Rm + Ecb*cur0
        en1 = -T*Rp + Ecb*cur1
    elif itype == 2:
        alpha, Rm, Rp = (Ecb-mu)/T, (Dm-mu)/T, (Dp-mu)/T
        cur0 = -1.0j*pi*fermi_func(alpha) if alpha < Rp and alpha > Rm else 0
        cur1 = cur0 + (1.0j*pi if alpha < Rp and alpha > Rm else 0)
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    elif itype == 3:
        alpha = (Ecb-mu)/T
        cur0 = -1.0j*pi*fermi_func(alpha)
        cur1 = cur0 + 1.0j*pi
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    #-------------------------
    return np.array([cur0, cur1, en0, en1])

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
