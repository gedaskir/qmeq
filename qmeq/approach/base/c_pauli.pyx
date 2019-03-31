"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp
from ...mytypes import complexnp

from ...specfunc.specfuncc cimport func_pauli
from ...aprclass import Approach

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

@cython.boundscheck(False)
def c_generate_norm_vec(sys, length):
    si, symq, norm_row = (sys.si, sys.funcp.symq, sys.funcp.norm_row)

    sys.bvec_ext = np.zeros(length+1, dtype=doublenp)
    sys.bvec_ext[-1] = 1

    sys.bvec = sys.bvec_ext[0:-1]
    sys.bvec[norm_row] = 1 if symq else 0

    sys.norm_vec = np.zeros(length, dtype=doublenp)
    cdef np.ndarray[double_t, ndim=1] norm_vec = sys.norm_vec

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
    sys.paulifct = paulifct
    return 0

#---------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_pauli(sys):
    cdef np.ndarray[double_t, ndim=3] paulifct = sys.paulifct
    si = sys.si
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
    sys.kern_ext = np.zeros((npauli+1, npauli), dtype=doublenp)
    sys.kern = sys.kern_ext[0:-1, :]

    c_generate_norm_vec(sys, npauli)
    cdef np.ndarray[double_t, ndim=2] kern = sys.kern
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
    sys.current = current
    sys.energy_current = energy_current
    sys.heat_current = energy_current - current*sys.leads.mulst
    return 0

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

class Approach_Pauli(Approach):

    kerntype = 'Pauli'
    generate_fct = c_generate_paulifct
    generate_kern = c_generate_kern_pauli
    generate_current = c_generate_current_pauli
    generate_vec = c_generate_vec_pauli
#---------------------------------------------------------------------------------------------------
