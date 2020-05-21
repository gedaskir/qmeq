"""Module containing various special functions, cython implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import psi as digamma
from scipy.integrate import quad

from ..mytypes import doublenp
from ..mytypes import complexnp

cimport numpy as np
cimport cython

# These definitions are already specified in specfuncc.pxd
# as well as 'import numpy as np' and 'cimport numpy as np'
'''
ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t
'''

cdef double_t pi = 3.14159265358979323846

from libc.math cimport exp
# cdef extern from "math.h":
#     double_t exp(double_t)

from libc.math cimport log
# cdef extern from "math.h":
#     double_t log(double_t)


@cython.cdivision(True)
cdef double_t fermi_func(double_t x):
    return 1./(exp(x)+1.)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef int_t func_pauli(double_t Ecb, double_t mu, double_t T,
                      double_t Dm, double_t Dp, int_t itype,
                      double_t [:] rez):
    cdef double_t alpha, Rm, Rp, cur0, cur1
    alpha = (Ecb-mu)/T
    Rm, Rp = (Dm-mu)/T, (Dp-mu)/T
    if itype == 1 or itype == 3 or (Rm < alpha < Rp):
        cur0 = fermi_func(alpha)
        cur1 = 1.0-cur0
        rez[0] = 2*pi*cur0
        rez[1] = 2*pi*cur1
    else:
        rez[0], rez[1] = 0.0, 0.0
    return 0


@cython.boundscheck(False)
@cython.cdivision(True)
cdef int_t func_1vN(double_t Ecb, double_t mu, double_t T,
                    double_t Dm, double_t Dp,
                    int_t itype, int_t limit,
                    complex_t [:] rez):
    cdef double_t alpha, Rm, Rp, err
    cdef complex_t cur0, cur1, en0, en1, const0, const1
    # -------------------------
    if itype == 0:
        alpha, Rm, Rp = (Ecb-mu)/T, (Dm-mu)/T, (Dp-mu)/T
        cur0, err = quad(fermi_func, Rm, Rp,
                         weight='cauchy', wvar=alpha,
                         epsabs=1.0e-6, epsrel=1.0e-6, limit=limit)
        cur0 = cur0 + (-1.0j*pi*fermi_func(alpha) if Rm < alpha < Rp else 0.0j)
        cur1 = cur0 + log(abs((Rm-alpha)/(Rp-alpha)))
        cur1 = cur1 + (1.0j*pi if Rm < alpha < Rp else 0.0j)
        #
        const0 = T*((-Rm if Rm < -40.0 else log(1+exp(-Rm)))
                   -(-Rp if Rp < -40.0 else log(1+exp(-Rp))))
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
        cur0 = -1.0j*pi*fermi_func(alpha) if Rm < alpha < Rp else 0.0j
        cur1 = cur0 + (1.0j*pi if Rm < alpha < Rp else 0.0j)
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    elif itype == 3:
        alpha = (Ecb-mu)/T
        cur0 = -1.0j*pi*fermi_func(alpha)
        cur1 = cur0 + 1.0j*pi
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    # -------------------------
    rez[0], rez[1], rez[2], rez[3] = cur0, cur1, en0, en1
    return 0


def c_fermi_func(double_t x):
    return fermi_func(x)


def c_func_pauli(double_t Ecb, double_t mu, double_t T,
                 double_t Dm, double_t Dp, int_t itype):
    rez = np.zeros(2, dtype=doublenp)
    func_pauli(Ecb, mu, T, Dm, Dp, itype, rez)
    return rez


def c_func_1vN(double_t Ecb, double_t mu, double_t T,
               double_t Dm, double_t Dp,
               int_t itype, int_t limit):
    rez = np.zeros(4, dtype=complexnp)
    func_1vN(Ecb, mu, T, Dm, Dp, itype, limit, rez)
    return rez
