"""Module containing various special functions."""

from functools import lru_cache

import numpy as np
from numpy.fft import fft, ifft
from numpy import pi
from numpy.lib.scimath import log
from numpy import exp
from scipy import linalg
from scipy.special import psi
from scipy.integrate import quad

from ..wrappers.mytypes import doublenp

MAX_CACHE = 100

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
    itype : int
        Type of function calculation.

    Returns
    -------
    ndarray
        | Array of two float numbers [cur0, cur1] containing
          momentum-integrated current amplitudes.
        | cur0 - particle current amplitude.
        | cur1 - hole current amplitude.
    """
    alpha = (Ecb-mu)/T
    Rm, Rp = (Dm-mu)/T, (Dp-mu)/T
    if itype == 1 or itype == 3 or (Rm < alpha < Rp):
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
                   large bandwidth D.
        | itype=2: the principal parts are neglected.
        | itype=3: the principal parts are neglected and infinite bandwidth D is assumed.
    limit : int
        For itype=0 dqawc_limit determines the maximum number of sub-intervals
        in the partition of the given integration interval.

    Returns
    -------
    ndarray
        | Array of four complex numbers [cur0, cur1, en0, en1] containing
          momentum-integrated current amplitudes.
        | cur0 - particle current amplitude.
        | cur1 - hole current amplitude.
        | en0 - particle energy current amplitude.
        | en1 - hol energy current amplitude.
    """
    if itype == 0:
        alpha, Rm, Rp = (Ecb-mu)/T, (Dm-mu)/T, (Dp-mu)/T
        cur0, err = quad(fermi_func, Rm, Rp,
                         weight='cauchy', wvar=alpha, epsabs=1.0e-6,
                         epsrel=1.0e-6, limit=limit)
        cur0 = cur0 + (-1.0j*pi*fermi_func(alpha) if Rm < alpha < Rp else 0)
        cur1 = cur0 + log(abs((Rm-alpha)/(Rp-alpha)))
        cur1 = cur1 + (1.0j*pi if Rm < alpha < Rp else 0)
        #
        const0 = T*((-Rm if Rm < -40 else log(1+exp(-Rm))) -
                    (-Rp if Rp < -40 else log(1+exp(-Rp))))
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
        cur0 = -1.0j*pi*fermi_func(alpha) if Rm < alpha < Rp else 0
        cur1 = cur0 + (1.0j*pi if Rm < alpha < Rp else 0)
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    elif itype == 3:
        alpha = (Ecb-mu)/T
        cur0 = -1.0j*pi*fermi_func(alpha)
        cur1 = cur0 + 1.0j*pi
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    else:
        cur0, cur1, en0, en1 = 0, 0, 0, 0
    # -------------------------
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
    ndarray
        Kernel used when performing Hilbert transform using FFT.
    """
    aux = np.zeros(n+1, dtype=doublenp)
    for i in range(1, n+1):
        aux[i] = i*log(i)
    m = 2*n
    ker = np.zeros(m, dtype=doublenp)
    for i in range(1, n):
        ker[i] = aux[i+1]-2*aux[i]+aux[i-1]
        ker[m-i] = -ker[i]
    return fft(ker)/pi


def hilbert_fredriksen(f, ker=None):
    """
    Performs Hilbert transform of f.

    Parameters
    ----------
    f : ndarray
        Values of function on a equidistant grid.
    ker : ndarray
        Kernel used when performing Hilbert transform using FFT.

    Returns
    -------
    ndarray
        Hilbert transform of f.
    """
    if ker is None:
        ker = kernel_fredriksen(len(f))
    n = len(f)
    fpad = fft(np.concatenate((f, np.zeros(len(ker)-n))))
    r = ifft(fpad*ker)
    return r[0:n]

@lru_cache(MAX_CACHE)
def fermi_func(x):
    """The Fermi function.

    Parameters
    ----------
    x : double
        Energy

    Returns
    -------
    double
        The function value

    """
    if x > 709.77:  # To avoid overflow in the denominator
        return 0.0
    return 1 / (exp(x) + 1)

@lru_cache(MAX_CACHE)
def diff_fermi(x, sign=1):
    """
    Calculates the derivative of the Fermi function.

    Parameters
    ----------
    x : double
        Energy
    sign: int
        sign to be multiplied with the exponent

    Returns
    -------
    double
        The function value
    """
    if x > 709.77 / 2:  # To avoid overflow in the denominator
        return 0.0
    e = np.exp(sign * x)
    return -sign * e / ((e + 1.0) ** 2)

@lru_cache(MAX_CACHE)
def bose(x, sign=1):
    """Bose distribution function.

    Parameters
    ----------
    x : double
        Energy
    sign : int
        Sign of the exponent, exp(sign*x)

    Returns
    -------
    double
        The function value
    """
    if x > 709.77:  # To avoid overflow in the denominator
        return 0.0
    e = np.exp(sign * x)
    return 1 / (e - 1)

@lru_cache(MAX_CACHE)
def digamma(z):
    return psi(z)

@lru_cache(MAX_CACHE)
def phi(x, Dp, Dm, sign=1):
    """
    Calculates the phi function, i.e. eq. C4 in PRB 78, 235424, 2008.

    Parameters
    ----------
    x : float
        Energy.
    Dp : float
        Bandwidth (positive energy) over temperature
    Dm : float
        Bandwidth (negative energy) over temperature
    sign : int
        Sign factor to be multiplied with the function

    Returns
    -------
    double
        real part of the function value
    """
    Z = 0.5 + x / (2 * np.pi) * 1j
    ret = sign * (-digamma(Z).real + log(0.5*(abs(Dp)+abs(Dm))/(2.0*pi)))
    return ret

@lru_cache(MAX_CACHE)
def diff_phi(x, sign=1):
    """
    Calculates the first derivative of the phi function

    Parameters
    ----------
    x : float
        Energy
    sign : int
        Sign factor to be multiplied with the function

    Returns
    -------
    double
        imaginary part of the function value
    """
    Z = 0.5 + x / (2 * np.pi) * 1j
    ret = sign * 1 / (2 * np.pi) * polygamma(Z, 1).imag
    return ret

@lru_cache(MAX_CACHE)
def diff2_phi(x, sign=1):
    """
    Calculates the second derivative of the phi function

    Parameters
    ----------
    x : float
        Energy
    sign : int
        Sign factor to be multiplied with the function

    Returns
    -------
    double
        real part of the function value
    """
    Z = 0.5 + x / (2 * np.pi) * 1j
    ret = sign * 1 / (2 * np.pi * 2 * np.pi) * polygamma(Z, 2).real
    return ret

@lru_cache(MAX_CACHE)
def delta_phi(x1, x2, Dp, Dm, sign=1):
    """
    Calculates the difference of the phi function evaluated at two energies.

    Parameters
    ----------
    x1 : float
        First energy
    x2 : float
        Second energy
    Dp : float
        Bandwidth (positive energy) over temperature
    Dm : float
        Bandwidth (negative energy) over temperature
    sign : int
        Sign factor to be multiplied with the phi function

    Returns
    -------
    double
        The function value
    """
    return phi(x1, Dp, Dm, sign=sign) - phi(x2, Dp, Dm, sign=sign)

@lru_cache(MAX_CACHE)
def polygamma(U, K):
    """
    Calculates the Kth derivative (up to order K=4) of the complex valued
    digamma function.

    Parameters
    ----------
    K : int
        Order of the derivative
    U : complex
        Argument

    Returns
    -------
    complex
        The function value
    """

    PI = np.pi

    X = U.real
    A = np.abs(X)

    SGN = [-1, 1, -1, 1, -1]
    FCT = [0, 1, 1, 2, 6, 24]

    C = np.zeros([6, 5])

    C[0, 0] = 8.33333333333333333e-2
    C[1, 0] = -8.33333333333333333e-3
    C[2, 0] = 3.96825396825396825e-3
    C[3, 0] = -4.16666666666666667e-3
    C[4, 0] = 7.57575757575757576e-3
    C[5, 0] = -2.10927960927960928e-2

    C[0, 1] = 1.66666666666666667e-1
    C[1, 1] = -3.33333333333333333e-2
    C[2, 1] = 2.38095238095238095e-2
    C[3, 1] = -3.33333333333333333e-2
    C[4, 1] = 7.57575757575757576e-2
    C[5, 1] = -2.53113553113553114e-1

    C[0, 2] = 5.00000000000000000e-1
    C[1, 2] = -1.66666666666666667e-1
    C[2, 2] = 1.66666666666666667e-1
    C[3, 2] = -3.00000000000000000e-1
    C[4, 2] = 8.33333333333333333e-1
    C[5, 2] = -3.29047619047619048e+0

    C[0, 3] = 2.00000000000000000e+0
    C[1, 3] = -1.00000000000000000e+0
    C[2, 3] = 1.33333333333333333e+0
    C[3, 3] = -3.00000000000000000e+0
    C[4, 3] = 1.00000000000000000e+1
    C[5, 3] = -4.60666666666666667e+1

    C[0, 4] = 10
    C[1, 4] = -7
    C[2, 4] = 12
    C[3, 4] = -33
    C[4, 4] = 130
    C[5, 4] = -691

    H = 0 + 0j
    if K < 0 or K > 4:
        return H

    K1 = K + 1
    if X < 0:
        U = - U
    V = U.real + 1j * U.imag

    if A < 15:
        H = 1 / (V ** K1)
        for i in range(15 - int(A)):
            V = V + 1.0
            H = H + 1 / (V ** K1)

        V = V + 1.0

    R = 1 / (V ** 2)
    P = R * C[5, K]

    for i in reversed(range(1, 6)):
        hh1 = P + C[i - 1, K]
        P = R * hh1

    H *= FCT[K + 1]

    hh1 = 1 / (V ** K1)
    hh2 = P + FCT[K]
    hh3 = V * hh2 + 0.5 * FCT[K + 1]
    H += hh1 * hh3
    H *= SGN[K]

    if K == 0:
        H = H + np.log(V)

    if X < 0:
        V = V * PI
        X, Y = V.real, V.imag
        A = np.sin(X)
        B = np.cos(X)
        T = np.tanh(X)
        P = B - 1j * A * T / (A + 1j * A * T)

        if K == 0:
            H += 1 / U + P * PI

        elif K == 1:
            H += 1 / (U ** 2)
            H += (P ** 2 + 1.0) * 9.869604401089358

        elif K == 2:
            H += 1 / (U ** 3)
            H += (P ** 2 + 1.0) * 62.01255336059963

    return H


def integralD(p1, eta1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma):
    """ Evaluates the 'direct' integral in the RTD approach. Picks the appropriate way
    of evaluating the integral based on the temperatures. Assumes that the wide band limits is valid.

    Parameters
    ----------
    p1 : float/int
        Keldysh index of right-most operator
    eta1 : float
        electron-hole index
    E1 : float
        Energy difference (E1+ - E1-).
    E2 : float
        Energy difference (E2+ - E2-).
    E3 : float
        Energy difference (E3+ - E3-).
    T1 : float
        Temperature of lead 1
    T2 : float
        Temperature of lead 2
    mu1 : float
        Chemical potential of lead 1
    mu2 : float
        Chemical potential of lead 2
    Dp : float
        Bandwidth (positive energy) over temperature
    Dm : float
        Bandwidth (negative energy) over temperature
    b_and_R : ndarray
        2xN ndarray containing 1/b in the first row and R in the second row. b and R are the poles and residues
        of the Ozaki expansion of tanh(z), respectively.

    Returns
    -------
    double
        The integral value
    """
    TMIN = 1e-5
    if abs(T2-T1) < TMIN and not ImGamma:
        lambda1 = (E1 - mu1) / T1
        lambda2 = (E2 - mu1 - eta1 * mu2) / T1
        lambda3 = (E3 - mu1) /T1
        ret = _D_integral_equal_T(p1, 1, lambda3, lambda2,  lambda1, D/2/T1, D/2/T1)
        return ret*np.pi/T1
    else:
        ret = _D_integral(1, p1, -E1, -E2, -E3, T1, T2, mu1, eta1*mu2, D/2, D/2, b_and_R)
        return -1j*ret

def integralX(p1, eta1, E1, E2, E3, T1, T2, mu1, mu2, D, b_and_R, ImGamma):
    """ Evaluates the 'exchange' integral in the RTD approach. Picks the appropriate way
    of evaluating the integral based on the temperatures. Assumes that the wide band limit is valid.

    Parameters
    ----------
    p1 : float/int
        Keldysh index of right-most operator
    eta1 : float
        electron-hole index
    E1 : float
        Energy difference (E1+ - E1-).
    E2 : float
        Energy difference (E2+ - E2-).
    E3 : float
        Energy difference (E3+ - E3-).
    T1 : float
        Temperature of lead 1
    T2 : float
        Temperature of lead 2
    mu1 : float
        Chemical potential of lead 1
    mu2 : float
        Chemical potential of lead 2
    Dp : float
        Bandwidth (positive energy) over temperature
    Dm : float
        Bandwidth (negative energy) over temperature
    b_and_R : ndarray
        2xN ndarray containing 1/b in the first row and R in the second row. b and R are the poles and residues
        of the Ozaki expansion of tanh(z), respectively.

    Returns
    -------
    double
        The integral value
    """

    TMIN = 1e-10
    if abs(T2-T1) < TMIN and not ImGamma:
        lambda1 = (E1 - mu1) / T1
        lambda2 = (E2 - mu1 - eta1 * mu2) / T1
        lambda3 = (E3 - eta1*mu2) / T1
        ret = _X_integral_equal_T(p1, 1, lambda3, lambda2,  lambda1, D/2/T1, D/2/T1)
        return ret*np.pi/T1
    else:
        ret = _X_integral(1, p1, -E1, -E2, -E3, T1, T2, mu1, eta1*mu2, D/2, D/2, b_and_R)
        return -1j*ret


def _D_integral_equal_T(p1, p2, E1, deltaE, E2, Dp, Dm):
    """
    Evaluates the 'direct' integral for the RTD approach in the wide band limit for
    the special case of equal temperatures (see eq. D3-D4 in PRB 78, 235424, 2008).

    Parameters
    ----------
    p1 : float/int
        Keldysh index of right-most operator
    p2 : float/int
        Keldysh index of the second right-most operator
    z1 : float
        Energy difference (E1+ - E1-) shifted by the chemical potential, normalized by T.
    z2 : float
        Energy difference (E2+ - E2-) shifted by the chemical potential, normalized by T.
    z3 : float
        Energy difference (E3+ - E3-) shifted by the chemical potential, normalized by T.
    Dp : float
        Bandwidth (positive energy) over temperature
    Dm : float
        Bandwidth (negative energy) over temperature

    Returns
    -------
    double
        The integral value
    """
    # See Appendix D of PRB. 78 235424 (2008)
    E_MIN = 1e-10

    if np.abs(E1 - E2) < E_MIN:
        if np.abs(deltaE) < E_MIN:
            ret = (p1 * p2 * (-diff2_phi(E1) + diff_fermi(E1) * phi(-E1, Dp, Dm) - diff_phi(-E1) * fermi_func(E1)
                              - 0.5 * diff_phi(E1)) + 0.5 * p2 * diff_phi(E1))

        else:
            ret = (p1 * p2 * (-fermi_func(E1) * diff_phi(deltaE - E1) - bose(deltaE) * diff_phi(deltaE - E1)
                              + diff_fermi(E1) * phi(deltaE - E1, Dp, Dm) - (bose(deltaE) + 0.5) * diff_phi(E1)) +
                   0.5 * p2 * diff_phi(E1))

    else:
        if np.abs(deltaE) < E_MIN:
            ret = (1.0 / (E1 - E2) * (p1 * p2 * (-(diff_phi(E1) - diff_phi(E2)) + fermi_func(E1) * phi(-E1, Dp, Dm)
                                                 - fermi_func(E2) * phi(-E2, Dp, Dm) - 0.5 * delta_phi(E1, E2, Dp,
                                                                                                       Dm)) +
                                      0.5 * p2 * delta_phi(E1, E2, Dp, Dm)))

        else:
            ret = (1.0 / (E1 - E2) * (p1 * p2 * ((fermi_func(E1) + bose(deltaE)) * phi(deltaE - E1, Dp, Dm) -
                                                 (fermi_func(E2) + bose(deltaE)) * phi(deltaE - E2, Dp, Dm)
                                                 - (bose(deltaE) + 0.5) * delta_phi(E1, E2, Dp,
                                                                                    Dm)) + 0.5 * p2 * delta_phi(E1, E2,
                                                                                                                Dp,
                                                                                                                Dm)))
    return ret


def _X_integral_equal_T(p1, p2, E1, deltaE, E2, Dp, Dm):
    """
    Evaluates the 'exchange' integral for the RTD approach in the wide band limit for
    the special case of equal temperatures (see eq. D3-D4 in PRB 78, 235424, 2008).

    Parameters
    ----------
    p1 : float/int
        Keldysh index of right-most operator
    p2 : float/int
        Keldysh index of the second right-most operator
    z1 : float
        Energy difference (E1+ - E1-) shifted by the chemical potential, normalized by T.
    z2 : float
        Energy difference (E2+ - E2-) shifted by the chemical potential, normalized by T.
    z3 : float
        Energy difference (E3+ - E3-) shifted by the chemical potential, normalized by T.
    Dp : float
        Bandwidth (positive energy) over temperature
    Dm : float
        Bandwidth (negative energy) over temperature

    Returns
    -------
    double
        The integral value
    """
    # See Appendix D of PRB. 78 235424 (2008)
    E_MIN = 1e-10

    if np.abs(deltaE - E1 - E2) < E_MIN:
        if np.abs(deltaE) < E_MIN:
            ret = (diff2_phi(E1) + fermi_func(E1) * diff_phi(-E1) + diff_phi(E1) * fermi_func(-E1))
        else:
            ret = ((fermi_func(E1) + bose(E1 + E2)) * diff_phi(E2) + diff_phi(E1) * (fermi_func(E2)
                                                                                     + bose(E1 + E2)))
    else:
        if np.abs(deltaE) < E_MIN:
            ret = (-1.0 / (E1 + E2) * (fermi_func(E1) * delta_phi(-E1, E2, Dp, Dm) + fermi_func(E2) *
                                       delta_phi(-E2, E1, Dp, Dm) - (diff_phi(E1) + diff_phi(E2))))
        else:
            ret = (1.0 / (deltaE - E1 - E2) * ((fermi_func(E1) + bose(deltaE)) * delta_phi(deltaE - E1, E2, Dp, Dm)
                                               + (fermi_func(E2) + bose(deltaE)) * delta_phi(deltaE - E2, E1, Dp, Dm)))

    return p1 * p2 * ret


def _D_integral(p1, p2, z1, z2, z3, T1, T2, mu1, mu2, Dp, Dm, b_and_R):
    """
    Evaluates the 'direct' integral for the RTD approach using the residue theorem to express
    one integral as a sum of Matsubara frequencies, which yields the
    digamma function in the wide band limit. The remaining integral is solved
    by approximating tanh(z) as a sum of Ozaki frequencies (see Phys. Rev. B
    75, 035123, 2007).

    See section I.B. of the SI of PRL 120, 017701, 2018 for a  similar procedure
    without using the Ozaki approximation.

    Parameters
    ----------
    p1 : float/int
        Keldysh index of right-most operator
    p2 : float/int
        Keldysh index of the second right-most operator
    z1 : float
        Energy difference (E1+ - E1-) shifted by the chemical potential.
    z2 : float
        Energy difference (E2+ - E2-) shifted by the chemical potential.
    z3 : float
        Energy difference (E3+ - E3-) shifted by the chemical potential.
    T1 : float
        Temperature of lead 1.
    T2 : float
        Temperature of lead 2.
    mu1 : float
        chemical potential of lead 1.
    mu2 : float
        chemical potential of lead 2.
    Dp : float
        Bandwidth (positive energy)
    Dm : float
        Bandwidth (negative energy)

    Returns
    -------
    complex
        The integral value
    """

    PI = np.pi
    E_MIN = 1e-10
    D = (np.abs(Dp) + np.abs(Dm)) * 0.5

    temp_f1 = 0 + 0j
    for i, n in enumerate(b_and_R[:, 0]):
        if 1 / n > D / T2:
            continue

        x = T1 / n

        A = np.log(D / (2 * PI * T1)) + 0j - digamma(0.5 + x / (2 * np.pi * T2) - (z2 + mu1 + mu2) / (2 * PI * T2) * 1j)
        B = x - (z3 + mu1) * 1j
        C = x - (z1 + mu1) * 1j

        temp_f1 += A * 1.0 / B * 1.0 / C * b_and_R[i, 1]
    temp_f1 *= - 8 * PI * T1 * 1j

    if (abs(z3/T1 - z1/T1) > E_MIN):
        temp_f2 = (digamma(0.5 - (z3 + mu1) / (2 * PI * T1) * 1j) - digamma(0.5 - (z1 + mu1) / (2 * PI * T1) * 1j)) / (
                    z3/T1 - z1/T1)
        temp_f2 = temp_f2 * (-2 * PI * 1j)/T1
    else:
        temp_f2 = polygamma(0.5 - (z3 + mu1) / (2 * PI * T1) * 1j, 1) * (-1.0 / T1)

    ret = 0.25 * (p1 * p2 * temp_f1 - p1 * temp_f2)
    return ret


def _X_integral(p1, p2, z1, z2, z3, T1, T2, mu1, mu2, Dp, Dm, b_and_R):
    """Evaluates the 'direct' integral for the RTD approach using the residue theorem to express
    one integral as a sum of Matsubara frequencies, which is equal to the
    digamma function in the wide band limit. The remaining integral is solved
    by approximating tanh(z) as a sum of Ozaki frequencies (see Phys. Rev. B
    75, 035123, 2007).

    See section I.B. of the SI of PRL 120, 017701, 2018 for a  similar procedure
    without using the Ozaki approximation.

    Parameters
    ----------
    p1 : float/int
        Keldysh index of right-most operator
    p2 : float/int
        Keldysh index of the second right-most operator
    z1 : float
        Energy difference (E1+ - E1-) shifted by the chemical potential.
    z2 : float
        Energy difference (E2+ - E2-) shifted by the chemical potential.
    z3 : float
        Energy difference (E3+ - E3-) shifted by the chemical potential.
    T1 : float
        Temperature of lead 1.
    T2 : float
        Temperature of lead 2.
    mu1 : float
        chemical potential of lead 1.
    mu2 : float
        chemical potential of lead 2.
    Dp : float
        Bandwidth (positive energy)
    Dm : float
        Bandwidth (negative energy)

    Returns
    -------
    complex
        The integral value
    """
    PI = np.pi
    D = (np.abs(Dp) + np.abs(Dm)) * 0.5

    ret = 0 + 0j
    for i, n in enumerate(b_and_R[:, 0]):
        if 1 / n > D / T2:
            continue

        x = T1 / n

        A = digamma((0.5 + x / (2 * np.pi * T2)) - (z2 + mu1 + mu2) / (2 * PI * T2) * 1j)
        B = digamma(0.5 - (z3 + mu2) / (2 * PI * T2) * 1j)
        C = x + (-z2 + z3 - mu1) * 1j
        E = x - (z1 + mu1) * 1j

        ret += (A - B) / C * 1.0 / E * (b_and_R[i, 1] + 0.0j)

    ret *= (0 - 8 * PI * T1 * 1j)
    ret *= (p1 * p2 * 0.25 + 0j)

    return ret


def Ozaki(N):
    """
    Calculates N/2 poles and residues for the Ozaki expansion
    of tanh(z) along the imaginary axis. See Phys. Rev. B 75, 035123, 2007
    and Phys. Rev. B 82, 125114, 2010 for details.

    Parameters
    ----------
    N : int
        Size of the matrix to be diagonalized. Number of poles is N/2. N must be
        an even number.

    Returns
    -------
         b_and_R : np.ndarray
            a ndarray where the first column contains the reciprocal of the poles (1/b)
            and the second columns contains the residues R.
    """
    B = np.zeros([N - 1], dtype=float)
    for n in range(1, N + 0):
        t = 1 / (2 * np.sqrt((2 * n - 1) * (2 * n + 1)))
        B[n - 1] = t

    b, v = linalg.eigh_tridiagonal(np.zeros(N), B)
    b_and_R = np.zeros([N // 2, 2], dtype=doublenp)

    count = 0
    for i in range(N):
        val = b[i]
        if val > 0:
            r = v[0, i] ** 2 / (4 * val ** 2)
            b_and_R[count, 0] = val
            b_and_R[count, 1] = r
            count += 1
    return b_and_R


def BW_Ozaki(D):
    """ Caluclates the poles and residues for the Ozaki expansion of tanh(z) along the
    imaginary axis up to Im(z) = D. The number of poles is chosen such that
    the largest pole is roughly equal to D. For increased accuracy choose a larger D.

    Parameters
    ----------
    D : double
        Band width

    Returns
    -------
         b_and_R : np.ndarray
            a ndarray where the first column contains the reciprocal of the poles (1/b)
            and the second columns contains the residues R.
    """
    # Calculate the number of poles such that max(b)~D. Based on empirical testing.
    p = [2.54647909, 1.27324094, 0.52325856 - D]
    N = int(max(np.roots(p)))

    b_and_R = Ozaki(N * 2 + 2)
    return b_and_R