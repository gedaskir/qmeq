"""Module containing cython functions, which generate first order Lindblad kernel.
   For docstrings see documentation of module lindblad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp
from ...mytypes import complexnp

from ...specfunc.specfuncc cimport func_pauli
from ...aprclass import Approach
from .c_pauli import c_generate_norm_vec

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

from libc.math cimport sqrt

#---------------------------------------------------------------------------------------------------
# Lindblad approach
#---------------------------------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
def c_generate_tLba(sys):
    cdef np.ndarray[complex_t, ndim=3] Tba = sys.leads.Tba
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    si = sys.si
    cdef np.ndarray[double_t, ndim=1] mulst = sys.leads.mulst
    cdef np.ndarray[double_t, ndim=1] tlst = sys.leads.tlst
    cdef np.ndarray[double_t, ndim=2] dlst = sys.leads.dlst
    #
    #mtype = sys.leads.mtype
    cdef int_t itype = sys.funcp.itype
    #
    cdef long_t b, a
    cdef int_t bcharge, acharge, charge, l
    cdef double_t Eba
    cdef int_t nleads = si.nleads
    #
    cdef np.ndarray[double_t, ndim=1] rez = np.zeros(2, dtype=doublenp)
    #
    cdef np.ndarray[complex_t, ndim=3] tLba = np.zeros((nleads, si.nmany, si.nmany), dtype=complexnp)
    for charge in range(si.ncharge-1):
        bcharge = charge+1
        acharge = charge
        for b, a in itertools.product(si.statesdm[bcharge], si.statesdm[acharge]):
            Eba = E[b]-E[a]
            for l in range(nleads):
                #fct1 = fermi_func((E[b]-E[a]-mulst[l])/tlst[l])
                #fct2 = 1-fct1
                func_pauli(Eba, mulst[l], tlst[l], dlst[l,0], dlst[l,1], itype, rez)
                tLba[l, b, a] = sqrt(rez[0])*Tba[l, b, a]
                tLba[l, a, b] = sqrt(rez[1])*Tba[l, a, b]
    sys.tLba = tLba
    return 0

@cython.boundscheck(False)
def c_generate_kern_lindblad(sys):
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] tLba = sys.tLba
    si = sys.si
    #
    cdef bool_t bbp_bool, bbpi_bool
    cdef int_t charge, acharge, bcharge, ccharge, l, nleads, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef long_t b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi, \
                c, cp, ccp, ccpi
    cdef long_t ndm0, ndm0r, npauli,
    cdef complex_t fct_aap, fct_bppbp, fct_bbpp, fct_ccp
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[bool_t, ndim=1] conjdm0 = si.conjdm0
    #
    ndm0r, ndm0, npauli, nleads = si.ndm0r, si.ndm0, si.npauli, si.nleads

    sys.kern_ext = np.zeros((ndm0r+1, ndm0r), dtype=doublenp)
    sys.kern = sys.kern_ext[0:-1, :]

    c_generate_norm_vec(sys, ndm0r)
    cdef np.ndarray[double_t, ndim=2] kern = sys.kern
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b, bp in itertools.combinations_with_replacement(si.statesdm[bcharge], 2):
            bbp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bp] + shiftlst0[bcharge]]
            bbp_bool = booldm0[lenlst[bcharge]*dictdm[b] + dictdm[bp] + shiftlst0[bcharge]]
            if bbp != -1 and bbp_bool:
                bbpi = ndm0 + bbp - npauli
                bbpi_bool = True if bbpi >= ndm0 else False
                if bbpi_bool:
                    kern[bbp, bbpi] = kern[bbp, bbpi] + E[b]-E[bp]
                    kern[bbpi, bbp] = kern[bbpi, bbp] + E[bp]-E[b]
                #--------------------------------------------------
                for a, ap in itertools.product(si.statesdm[acharge], si.statesdm[acharge]):
                    aap = mapdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]]
                    if aap != -1:
                        fct_aap = 0
                        for l in range(nleads):
                            fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.real                              # kern[bbp, aap]   += fct_aap.real
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] - fct_aap.imag*aap_sgn                # kern[bbp, aapi]  -= fct_aap.imag*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.real*aap_sgn          # kern[bbpi, aapi] += fct_aap.real*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] + fct_aap.imag                        # kern[bbpi, aap]  += fct_aap.imag
                #--------------------------------------------------
                for bpp in si.statesdm[bcharge]:
                    bppbp = mapdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[acharge]:
                            for l in range(nleads):
                                fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                        for c in si.statesdm[ccharge]:
                            for l in range(nleads):
                                fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.real                        # kern[bbp, bppbp] += fct_bppbp.real
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] - fct_bppbp.imag*bppbp_sgn        # kern[bbp, bppbpi] -= fct_bppbp.imag*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.real*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.real*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] + fct_bppbp.imag                  # kern[bbpi, bppbp] += fct_bppbp.imag
                    #--------------------------------------------------
                    bbpp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[acharge]:
                            for l in range(nleads):
                                fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                        for c in si.statesdm[ccharge]:
                            for l in range(nleads):
                                fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.real                           # kern[bbp, bbpp] += fct_bbpp.real
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] - fct_bbpp.imag*bbpp_sgn            # kern[bbp, bbppi] -= fct_bbpp.imag*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.real*bbpp_sgn      # kern[bbpi, bbppi] += fct_bbpp.real*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] + fct_bbpp.imag                     # kern[bbpi, bbpp] += fct_bbpp.imag
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[ccharge], si.statesdm[ccharge]):
                    ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                    if ccp != -1:
                        fct_ccp = 0
                        for l in range(nleads):
                            fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                        ccpi = ndm0 + ccp - npauli
                        ccp_sgn = +1 if conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]] else -1
                        kern[bbp, ccp] = kern[bbp, ccp] + fct_ccp.real                              # kern[bbp, ccp] += fct_ccp.real
                        if ccpi >= ndm0:
                            kern[bbp, ccpi] = kern[bbp, ccpi] - fct_ccp.imag*ccp_sgn                # kern[bbp, ccpi] -= fct_ccp.imag*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] = kern[bbpi, ccpi] + fct_ccp.real*ccp_sgn          # kern[bbpi, ccpi] += fct_ccp.real*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] = kern[bbpi, ccp] + fct_ccp.imag                        # kern[bbpi, ccp] += fct_ccp.imag
                #--------------------------------------------------
    return 0

@cython.boundscheck(False)
def c_generate_current_lindblad(sys):
    cdef np.ndarray[double_t, ndim=1] phi0p = sys.phi0
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] tLba = sys.tLba
    si = sys.si
    #
    cdef bool_t bbp_conj
    cdef int_t acharge, bcharge, ccharge, charge, l, nleads,
    cdef long_t c, a, b, bp, bbp
    cdef long_t ndm0, ndm1, npauli
    cdef complex_t fcta, fctc, phi0bbp
    ndm0, ndm1, npauli, nleads = si.ndm0, si.ndm1, si.npauli, si.nleads
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[complex_t, ndim=1] current = np.zeros(nleads, dtype=complexnp)
    cdef np.ndarray[complex_t, ndim=1] energy_current = np.zeros(nleads, dtype=complexnp)
    cdef np.ndarray[complex_t, ndim=1] phi0 = np.zeros(ndm0, dtype=complexnp)
    #
    phi0[0:npauli] = phi0p[0:npauli]
    phi0[npauli:ndm0] = phi0p[npauli:ndm0] + 1j*phi0p[ndm0:]
    #
    for charge in range(si.ncharge):
        ccharge = charge+1
        bcharge = charge
        acharge = charge-1
        for b, bp in itertools.product(si.statesdm[bcharge], si.statesdm[bcharge]):
            bbp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bp] + shiftlst0[bcharge]]
            if bbp != -1:
                bbp_conj = conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bp] + shiftlst0[bcharge]]
                phi0bbp = phi0[bbp] if bbp_conj else phi0[bbp].conjugate()
                for l in range(nleads):
                    for a in si.statesdm[acharge]:
                        fcta = tLba[l, a, b]*phi0bbp*tLba[l, a, bp].conjugate()
                        current[l] = current[l] - fcta
                        energy_current[l] = energy_current[l] + (E[a]-0.5*(E[b]+E[bp]))*fcta
                    for c in si.statesdm[ccharge]:
                        fctc = tLba[l, c, b]*phi0bbp*tLba[l, c, bp].conjugate()
                        current[l] = current[l] + fctc
                        energy_current[l] = energy_current[l] + (E[c]-0.5*(E[b]+E[bp]))*fctc
    #
    sys.current = np.array(current.real, dtype=doublenp)
    sys.energy_current = np.array(energy_current.real, dtype=doublenp)
    sys.heat_current = sys.energy_current - sys.current*sys.leads.mulst
    return 0

@cython.boundscheck(False)
def c_generate_vec_lindblad(np.ndarray[double_t, ndim=1] phi0p, sys):
    #cdef np.ndarray[double_t, ndim=1] phi0p = sys.phi0
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] tLba = sys.tLba
    si = sys.si
    cdef long_t norm_row = sys.funcp.norm_row
    #
    cdef bool_t bbp_bool
    cdef int_t charge, acharge, bcharge, ccharge, l, nleads, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef long_t b, bp, bbp, bb, \
                a, ap, aap, \
                bpp, bppbp, bbpp, \
                c, cp, ccp
    cdef long_t ndm0, npauli
    cdef complex_t fct_aap, fct_bppbp, fct_bbpp, fct_ccp, norm
    cdef complex_t phi0aap, phi0bppbp, phi0bbpp, phi0ccp
    ndm0, npauli, nleads = si.ndm0, si.npauli, si.nleads
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[bool_t, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[complex_t, ndim=1] phi0 = np.zeros(ndm0, dtype=complexnp)
    cdef np.ndarray[complex_t, ndim=1] i_dphi0_dt = np.zeros(ndm0, dtype=complexnp)
    #
    phi0[0:npauli] = phi0p[0:npauli]
    phi0[npauli:ndm0] = phi0p[npauli:ndm0] + 1j*phi0p[ndm0:]
    norm = 0
    for charge in range(si.ncharge):
        acharge = charge-1
        bcharge = charge
        ccharge = charge+1
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bp] + shiftlst0[bcharge]]
            if bbp != -1:
                if b == bp: norm = norm + phi0[bbp]
                bbp_bool = booldm0[lenlst[bcharge]*dictdm[b] + dictdm[bp] + shiftlst0[bcharge]]
                if bbp_bool:
                    i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + (E[b]-E[bp])*phi0[bbp]
                    #--------------------------------------------------
                    for a, ap in itertools.product(si.statesdm[charge-1], si.statesdm[charge-1]):
                        aap = mapdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]]
                        if aap != -1:
                            fct_aap = 0
                            for l in range(nleads):
                                fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                            phi0aap = phi0[aap] if conjdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]] else phi0[aap].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + 1.0j*fct_aap*phi0aap
                    #--------------------------------------------------
                    for bpp in si.statesdm[charge]:
                        bppbp = mapdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]]
                        if bppbp != -1:
                            fct_bppbp = 0
                            for a in si.statesdm[charge-1]:
                                for l in range(nleads):
                                    fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                            for c in si.statesdm[charge+1]:
                                for l in range(nleads):
                                    fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                            phi0bppbp = phi0[bppbp] if conjdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]] else phi0[bppbp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + 1.0j*fct_bppbp*phi0bppbp
                        #--------------------------------------------------
                        bbpp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]]
                        if bbpp != -1:
                            fct_bbpp = 0
                            for a in si.statesdm[charge-1]:
                                for l in range(nleads):
                                    fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                            for c in si.statesdm[charge+1]:
                                for l in range(nleads):
                                    fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                            phi0bbpp = phi0[bbpp] if conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]] else phi0[bbpp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + 1.0j*fct_bbpp*phi0bbpp
                    #--------------------------------------------------
                    for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                        ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                        if ccp != -1:
                            fct_ccp = 0
                            for l in range(nleads):
                                fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                            phi0ccp = phi0[ccp] if conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]] else phi0[ccp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + 1.0j*fct_ccp*phi0ccp
                    #--------------------------------------------------
    i_dphi0_dt[norm_row] = 1j*(norm-1)
    return np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[npauli:ndm0].real))

class Approach_Lindblad(Approach):

    kerntype = 'Lindblad'
    generate_fct = c_generate_tLba
    generate_kern = c_generate_kern_lindblad
    generate_current = c_generate_current_lindblad
    generate_vec = c_generate_vec_lindblad
#---------------------------------------------------------------------------------------------------
