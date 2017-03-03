"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

# cython: profile=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ..mytypes import doublenp
from ..mytypes import complexnp

from ..specfuncc cimport func_pauli

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

@cython.boundscheck(False)
def c_generate_paulifct(sys):
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Tba = sys.leads.Tba
    si = sys.si
    cdef np.ndarray[double_t, ndim=1] mulst = sys.leads.mulst
    cdef np.ndarray[double_t, ndim=1] tlst = sys.leads.tlst
    cdef np.ndarray[double_t, ndim=2] dlst = sys.leads.dlst
    #
    cdef long_t c, b, cb
    cdef int_t bcharge, ccharge, charge, l
    cdef double_t xcb, Ecb
    cdef int_t nleads = si.nleads
    cdef int_t itype = sys.funcp.itype
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
                func_pauli(Ecb, mulst[l], tlst[l], dlst[l,0], dlst[l,1], itype, rez)
                paulifct[l, cb, 0] = xcb*rez[0]
                paulifct[l, cb, 1] = xcb*rez[1]
    return paulifct

#---------------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_pauli(sys):
    cdef np.ndarray[double_t, ndim=3] paulifct = sys.paulifct
    si = sys.si
    cdef bint symq = sys.funcp.symq
    cdef long_t norm_rowp = sys.funcp.norm_row
    #
    cdef bool_t bb_bool
    cdef long_t b, bb, a, aa, c, cc, ba, cb
    cdef int_t acharge, bcharge, ccharge, charge, l
    cdef int_t norm_row, last_row
    cdef int_t nleads = si.nleads
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    #
    norm_row = norm_rowp if symq else si.npauli
    last_row = si.npauli-1 if symq else si.npauli
    #
    cdef np.ndarray[double_t, ndim=2] kern = np.zeros((last_row+1, si.npauli), dtype=doublenp)
    cdef np.ndarray[double_t, ndim=1] bvec = np.zeros(last_row+1, dtype=doublenp)
    bvec[norm_row] = 1
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
            bb_bool = booldm0[lenlst[bcharge]*dictdm[b] + dictdm[b] + shiftlst0[bcharge]]
            kern[norm_row, bb] = kern[norm_row, bb] + 1
            if not (symq and bb == norm_row) and bb_bool:
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
    return kern, bvec

@cython.boundscheck(False)
def c_generate_current_pauli(sys):
    cdef np.ndarray[double_t, ndim=1] phi0 = sys.phi0
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[double_t, ndim=3] paulifct = sys.paulifct
    si = sys.si
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
    return current, energy_current

@cython.boundscheck(False)
def c_generate_vec_pauli(np.ndarray[double_t, ndim=1] phi0, sys):
    cdef np.ndarray[double_t, ndim=3] paulifct = sys.paulifct
    si = sys.si
    cdef long_t norm_row = sys.funcp.norm_row
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
#---------------------------------------------------------------------------------------------------------
