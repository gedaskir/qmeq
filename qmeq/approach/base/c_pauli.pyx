# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp
from ...mytypes import complexnp

# Cython imports

cimport numpy as np
cimport cython

from ...specfunc.c_specfunc cimport func_pauli
from ..c_aprclass cimport Approach
from ..c_aprclass cimport KernelHandler

# ---------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------
cdef class ApproachPauli(Approach):

    kerntype = 'Pauli'
    no_coherences = True

    def get_kern_size(self):
        return self.si.npauli

    cpdef void generate_fct(self):
        cdef double_t [:] E = self.qd.Ea
        cdef complex_t [:, :, :] Tba = self.leads.Tba
        cdef double_t [:] mulst = self.leads.mulst
        cdef double_t [:] tlst = self.leads.tlst
        cdef double_t [:, :] dlst = self.leads.dlst

        cdef KernelHandler kh = self._kernel_handler
        cdef long_t nleads = kh.nleads

        cdef long_t itype = self.funcp.itype

        cdef long_t c, b, bcharge, cb, l
        cdef double_t Ecb, xcb

        cdef double_t [:, :, :] paulifct = np.zeros((nleads, kh.ndm1, 2), dtype=doublenp)
        cdef np.ndarray[double_t, ndim=1] rez = np.zeros(2, dtype=doublenp)

        for i in range(kh.ndm1):
            c = kh.all_ba[i, 0]
            b = kh.all_ba[i, 1]
            bcharge = kh.all_ba[i, 2]
            cb = kh.get_ind_dm1(c, b, bcharge)
            Ecb = E[c]-E[b]
            for l in range(nleads):
                xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, rez)
                paulifct[l, cb, 0] = xcb*rez[0]
                paulifct[l, cb, 1] = xcb*rez[1]

        self.paulifct = paulifct

    cdef void set_coupling(self):
        self._paulifct = self.paulifct

    cdef void generate_coupling_terms(self,
            long_t b, long_t bp, long_t bcharge,
            KernelHandler kh) nogil:

        cdef long_t a, c, aa, bb, cc, ba, cb
        cdef double_t fctm, fctp

        cdef long_t i, l
        cdef long_t nleads = kh.nleads
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef double_t [:, :, :] paulifct = self._paulifct

        cdef long_t acharge = bcharge-1
        cdef long_t ccharge = bcharge+1

        cdef long_t acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        cdef long_t ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        bb = kh.get_ind_dm0(b, b, bcharge)

        for i in range(acount):
            a = statesdm[acharge, i]
            aa = kh.get_ind_dm0(a, a, acharge)
            ba = kh.get_ind_dm1(b, a, acharge)
            fctm, fctp = 0, 0
            for l in range(nleads):
                fctm -= paulifct[l, ba, 1]
                fctp += paulifct[l, ba, 0]
            kh.set_matrix_element_pauli(fctm, fctp, bb, aa)

        for i in range(ccount):
            c = statesdm[ccharge, i]
            cc = kh.get_ind_dm0(c, c, ccharge)
            cb = kh.get_ind_dm1(c, b, bcharge)
            fctm, fctp = 0, 0
            for l in range(nleads):
                fctm -= paulifct[l, cb, 0]
                fctp += paulifct[l, cb, 1]
            kh.set_matrix_element_pauli(fctm, fctp, bb, cc)

    cpdef void generate_current(self):
        cdef double_t [:] phi0 = self.phi0
        cdef double_t [:] E = self.qd.Ea
        cdef double_t [:, :, :] paulifct = self._paulifct

        cdef long_t i, j, l
        cdef long_t bcharge, ccharge, bcount, ccount
        cdef long_t b, c, bb, cc, cb
        cdef double_t fct1, fct2

        cdef KernelHandler kh = self._kernel_handler
        cdef long_t [:, :] statesdm = kh.statesdm
        cdef long_t nleads = kh.nleads

        cdef np.ndarray[double_t, ndim=1] current = np.zeros(nleads, dtype=doublenp)
        cdef np.ndarray[double_t, ndim=1] energy_current = np.zeros(nleads, dtype=doublenp)

        for i in range(kh.ndm1):
            c = kh.all_ba[i, 0]
            b = kh.all_ba[i, 1]

            bcharge = kh.all_ba[i, 2]
            ccharge = bcharge+1

            bcount = kh.statesdm_count[bcharge]
            ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

            bb = kh.get_ind_dm0(b, b, bcharge)
            cc = kh.get_ind_dm0(c, c, ccharge)
            cb = kh.get_ind_dm1(c, b, bcharge)

            for l in range(nleads):
                fct1 = +phi0[bb]*paulifct[l, cb, 0]
                fct2 = -phi0[cc]*paulifct[l, cb, 1]
                current[l] = current[l] + fct1 + fct2
                energy_current[l] = energy_current[l] - (E[b]-E[c])*(fct1 + fct2)

        self.current = current
        self.energy_current = energy_current
        self.heat_current = energy_current - current*self.leads.mulst
# ---------------------------------------------------------------------------------------------------
