"""Module containing various special functions, cython implementation."""

import numpy as np
from scipy.integrate import quad

from ..wrappers.mytypes import doublenp
from ..wrappers.mytypes import complexnp

cimport numpy as np
cimport cython

cdef double_t pi = 3.14159265358979323846
from libc.math cimport exp
from libc.math cimport log


cdef class Func:
    cpdef double_t eval(self, double_t x):
        return 1.


@cython.cdivision(True)
cdef class FuncBose:
    """Bose function."""
    cdef double_t eval(self, double_t x):
        return 1./(exp(x)-1.)


@cython.cdivision(True)
@cython.boundscheck(False)
cdef class FuncPauliElPh:

    def __init__(self, double_t [:] tlst,
                       double_t [:, :] dlst,
                       bath_func,
                       double_t eps):
        self.tlst, self.dlst = tlst, dlst
        self.bath_func = bath_func
        self.bath_func_q = False if bath_func is None else True
        self.eps = eps
        #
        self.bose, self.dos = FuncBose(), Func()
        self.val = 0.

    cpdef void eval(self, double_t Ebbp, int_t l):
        cdef double_t T, omm, omp
        cdef double_t alpha, Rm, Rp
        T, omm, omp = self.tlst[l], self.dlst[l,0], self.dlst[l,1]
        # alpha, Rm, Rp = Ebbp/T, omm/T, omp/T
        alpha = max(abs(Ebbp/T), self.eps) * (1 if Ebbp >= 0 else -1)
        Rm, Rp = max(omm/T, 0.9*self.eps), omp/T
        if self.bath_func_q:
            self.dos = self.bath_func[l]
        if Rm < alpha < Rp:
            # Absorption
            self.val = 2*pi*self.bose.eval(alpha)*self.dos.eval(T*alpha)
        elif Rm < -alpha < Rp:
            # Emission
            self.val = 2*pi*(1+self.bose.eval(-alpha))*self.dos.eval(-T*alpha)
        else:
            self.val = 0.


@cython.cdivision(True)
@cython.boundscheck(False)
cdef class Func1vNElPh:

    def __init__(self, double_t [:] tlst,
                       double_t [:, :] dlst,
                       int_t itype,
                       long_t limit,
                       bath_func,
                       double_t eps):
        self.tlst, self.dlst = tlst, dlst
        self.itype, self.limit = itype, limit
        self.bath_func = bath_func
        self.bath_func_q = False if bath_func is None else True
        self.eps = eps
        #
        self.bose, self.dos = FuncBose(), Func()
        self.val0, self.val1 = 0., 0.

    cpdef double_t iplus(self, double_t x):
        return +self.dos.eval(self.T*x)*self.bose.eval(x)

    cpdef double_t iminus(self, double_t x):
        return -self.dos.eval(self.T*x)*(1.+self.bose.eval(x))

    cpdef void eval(self, double_t Ebbp, int_t l):
        cdef double_t T, omm, omp
        cdef double_t alpha, Rm, Rp, err
        cdef complex_t val0, val1
        T, omm, omp = self.tlst[l], self.dlst[l,0], self.dlst[l,1]
        # alpha, Rm, Rp = Ebbp/T, omm/T, omp/T
        alpha = max(abs(Ebbp/T), self.eps) * (1 if Ebbp >= 0 else -1)
        Rm, Rp = max(omm/T, 0.9*self.eps), omp/T
        self.T = T
        if self.bath_func_q:
            self.dos = self.bath_func[l]
        if self.itype is 0:
            self.val0, err = quad(self.iplus, Rm, Rp,
                                  weight='cauchy', wvar=alpha,
                                  epsabs=1.0e-6, epsrel=1.0e-6, limit=self.limit)
            self.val0 = self.val0 + (-1.0j*pi*self.iplus(alpha) if Rm < alpha < Rp else 0)
            self.val1, err = quad(self.iminus, Rm, Rp,
                                  weight='cauchy', wvar=alpha,
                                  epsabs=1.0e-6, epsrel=1.0e-6, limit=self.limit)
            self.val1 = self.val1 + (-1.0j*pi*self.iminus(alpha) if Rm < alpha < Rp else 0)
        elif self.itype is 2:
            self.val0 = -1.0j*pi*self.iplus(alpha) if Rm < alpha < Rp else 0
            self.val1 = -1.0j*pi*self.iminus(alpha) if Rm < alpha < Rp else 0
