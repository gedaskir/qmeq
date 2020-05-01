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


@cython.boundscheck(False)
def generate_norm_vec(self, length):
    si, symq, norm_row = (self.si, self.funcp.symq, self.funcp.norm_row)

    self.bvec_ext = np.zeros(length+1, dtype=doublenp)
    self.bvec_ext[-1] = 1

    self.bvec = self.bvec_ext[0:-1]
    self.bvec[norm_row] = 1 if symq else 0

    self.norm_vec = np.zeros(length, dtype=doublenp)
    cdef np.ndarray[double_t, ndim=1] norm_vec = self.norm_vec

    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0

    cdef int_t charge, b, bb
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            norm_vec[bb] += 1

    return 0


@cython.boundscheck(False)
def generate_paulifct(self):
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Tba = self.leads.Tba
    si = self.si
    cdef np.ndarray[double_t, ndim=1] mulst = self.leads.mulst
    cdef np.ndarray[double_t, ndim=1] tlst = self.leads.tlst
    cdef np.ndarray[double_t, ndim=2] dlst = self.leads.dlst
    #
    cdef long_t c, b, cb
    cdef int_t bcharge, ccharge, charge, l
    cdef double_t xcb, Ecb
    cdef int_t nleads = si.nleads
    cdef int_t itype = self.funcp.itype
    #
    cdef np.ndarray[double_t, ndim=3] paulifct = np.zeros((nleads, si.ndm1, 2), dtype=doublenp)
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    #
    cdef np.ndarray[double_t, ndim=1] rez = np.zeros(2, dtype=doublenp)
    #
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            Ecb = E[c]-E[b]
            for l in range(nleads):
                xcb = (Tba[l, b, c]*Tba[l, c, b]).real
                func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, rez)
                paulifct[l, cb, 0] = xcb*rez[0]
                paulifct[l, cb, 1] = xcb*rez[1]
    self.paulifct = paulifct
    return 0


# ---------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def generate_kern_pauli(self):
    cdef np.ndarray[double_t, ndim=3] paulifct = self.paulifct
    si = self.si
    #
    cdef bool_t bb_bool
    cdef long_t b, bb, a, aa, c, cc, ba, cb
    cdef int_t acharge, bcharge, ccharge, charge, l
    cdef int_t nleads = si.nleads
    cdef int_t npauli = si.npauli
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    #
    self.kern_ext = np.zeros((npauli+1, npauli), dtype=doublenp)
    self.kern = self.kern_ext[0:-1, :]

    generate_norm_vec(self, npauli)
    cdef np.ndarray[double_t, ndim=2] kern = self.kern
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
            bb_bool = booldm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
            if bb_bool:
                for a in si.statesdm[charge-1]:
                    aa = mapdm0[lenlst[acharge]*dictdm[a] + dictdm[a] + shiftlst0[acharge]]
                    ba = lenlst[acharge]*dictdm[b] + dictdm[a] + shiftlst1[acharge]
                    for l in range(nleads):
                        kern[bb, bb] = kern[bb, bb] - paulifct[l, ba, 1]
                        kern[bb, aa] = kern[bb, aa] + paulifct[l, ba, 0]
                for c in si.statesdm[charge+1]:
                    cc = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[c] + shiftlst0[ccharge]]
                    cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                    for l in range(nleads):
                        kern[bb, bb] = kern[bb, bb] - paulifct[l, cb, 0]
                        kern[bb, cc] = kern[bb, cc] + paulifct[l, cb, 1]
    return 0


@cython.boundscheck(False)
def generate_current_pauli(self):
    cdef np.ndarray[double_t, ndim=1] phi0 = self.phi0
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    cdef np.ndarray[double_t, ndim=3] paulifct = self.paulifct
    si = self.si
    #
    cdef long_t c, cc, b, bb, cb
    cdef int_t bcharge, ccharge, charge, l, nleads
    cdef double_t fct1, fct2
    nleads = si.nleads
    #
    cdef np.ndarray[double_t, ndim=1] current = np.zeros(nleads, dtype=doublenp)
    cdef np.ndarray[double_t, ndim=1] energy_current = np.zeros(nleads, dtype=doublenp)
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    #
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c in si.statesdm[ccharge]:
            cc = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[c] + shiftlst0[ccharge]]
            for b in si.statesdm[bcharge]:
                bb = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
                cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                for l in range(nleads):
                    fct1 = +phi0[bb]*paulifct[l, cb, 0]
                    fct2 = -phi0[cc]*paulifct[l, cb, 1]
                    current[l] = current[l] + fct1 + fct2
                    energy_current[l] = energy_current[l] - (E[b]-E[c])*(fct1 + fct2)
    self.current = current
    self.energy_current = energy_current
    self.heat_current = energy_current - current*self.leads.mulst
    return 0


@cython.boundscheck(False)
def generate_vec_pauli(self, np.ndarray[double_t, ndim=1] phi0):
    cdef np.ndarray[double_t, ndim=3] paulifct = self.paulifct
    si = self.si
    cdef long_t norm_row = self.funcp.norm_row
    #
    cdef bool_t bb_bool
    cdef long_t b, bb, a, aa, c, cc, ba, cb
    cdef int_t acharge, bcharge, ccharge, charge, l
    cdef int_t nleads = si.nleads
    cdef double_t norm = 0
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    #
    cdef np.ndarray[double_t, ndim=1] dphi0_dt = np.zeros(si.npauli, dtype=doublenp)
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
            bb_bool = booldm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
            norm = norm + phi0[bb]
            if bb_bool:
                for a in si.statesdm[charge-1]:
                    aa = mapdm0[lenlst[acharge]*dictdm[a] + dictdm[a] + shiftlst0[acharge]]
                    ba = lenlst[acharge]*dictdm[b] + dictdm[a] + shiftlst1[acharge]
                    for l in range(nleads):
                        dphi0_dt[bb] = dphi0_dt[bb] - paulifct[l, ba, 1]*phi0[bb]
                        dphi0_dt[bb] = dphi0_dt[bb] + paulifct[l, ba, 0]*phi0[aa]
                for c in si.statesdm[charge+1]:
                    cc = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[c] + shiftlst0[ccharge]]
                    cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                    for l in range(nleads):
                        dphi0_dt[bb] = dphi0_dt[bb] - paulifct[l, cb, 0]*phi0[bb]
                        dphi0_dt[bb] = dphi0_dt[bb] + paulifct[l, cb, 1]*phi0[cc]
    dphi0_dt[norm_row] = norm-1
    return dphi0_dt


cdef class ApproachPauli(Approach):

    kerntype = 'Pauli'

    cpdef generate_fct(self):
        generate_paulifct(self)

    cpdef generate_kern(self):
        generate_kern_pauli(self)

    cpdef generate_current(self):
        generate_current_pauli(self)

    cpdef generate_vec(self, phi0):
        return generate_vec_pauli(self, phi0)

# ---------------------------------------------------------------------------------------------------
