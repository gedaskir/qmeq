"""Module containing cython functions, which generate first order kernels (Pauli, 1vN, Redfield).
   For docstrings see documentation of module neumannpy."""

# cython: profile=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.special import psi as digamma
from scipy.integrate import quad
import itertools
#import mytypes

cimport numpy as np
cimport cython

# These definitions are already specified in neumannc.pxd
# as well as 'import numpy as np' and 'cimport numpy as np'
'''
ctypedef np.uint8_t boolnp
#ctypedef bint boolnp
ctypedef np.int_t intnp
ctypedef np.long_t longnp
ctypedef np.double_t doublenp
#ctypedef double doublenp
ctypedef np.complex128_t complexnp
#ctypedef complex complexnp
'''

cdef doublenp pi = 3.14159265358979323846

from libc.math cimport exp
#cdef extern from "math.h":
#    doublenp exp(doublenp)

from libc.math cimport log
#cdef extern from "math.h":
#    doublenp log(doublenp)

@cython.cdivision(True)
cdef doublenp fermi_func(doublenp x):
    return 1/(exp(x)+1)

@cython.cdivision(True)
cdef doublenp func_pauli(doublenp E, doublenp T, doublenp D):
    cdef doublenp alpha
    alpha = E/T
    R = D/T
    #return 2*pi*fermi_func(-alpha)
    return 2*pi*1/(exp(-alpha)+1) * (1.0 if alpha < R and alpha > -R else 0.0)

@cython.cdivision(True)
cdef complexnp func_1vN(doublenp E, doublenp T, doublenp D, doublenp eta, intnp itype, intnp limit):
    cdef doublenp alpha, R, err
    cdef complexnp rez
    alpha = E/T
    R = D/T
    #-------------------------
    if itype == 0:
        rez.real = 0
    elif itype == 1:
        rez.real = digamma(0.5-1.0j*alpha/(2*pi)).real - log(R/(2*pi))
    elif itype == 2:
        (rez.real, err) = quad(fermi_func, -R, +R, weight='cauchy', wvar=-alpha, epsabs=1.0e-6, epsrel=1.0e-6, limit=limit)
    #-------------------------
    rez.imag = -pi*1/(exp(-alpha)+1)*eta * (1.0 if alpha < R and alpha > -R else 0.0)
    return rez

@cython.boundscheck(False)
def c_generate_phi1fct(sys):
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    si = sys.si
    cdef np.ndarray[doublenp, ndim=1] mulst = sys.leads.mulst
    cdef np.ndarray[doublenp, ndim=1] tlst = sys.leads.tlst
    cdef np.ndarray[doublenp, ndim=1] dlst = sys.leads.dlst
    #
    cdef longnp c, b, cb
    cdef intnp bcharge, ccharge, charge, l
    #
    cdef intnp nleads = si.nleads
    cdef intnp itype = sys.funcp.itype
    cdef intnp dqawc_limit = sys.funcp.dqawc_limit
    #
    cdef np.ndarray[complexnp, ndim=3] phi1fct = np.zeros((nleads, si.ndm1, 2), dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=3] phi1fct_energy = np.zeros((nleads, si.ndm1, 2), dtype=np.complex)
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    #
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                phi1fct[l, cb, 0] = +func_1vN(+(E[b]-E[c]+mulst[l]), tlst[l], dlst[l], +1, itype, dqawc_limit)
                phi1fct[l, cb, 1] = +func_1vN(-(E[b]-E[c]+mulst[l]), tlst[l], dlst[l], -1, itype, dqawc_limit)
                phi1fct_energy[l, cb, 0] = +dlst[l]-(E[b]-E[c])*phi1fct[l, cb, 0] # (E[b]-E[c]+mulst[l])
                phi1fct_energy[l, cb, 1] = -dlst[l]-(E[b]-E[c])*phi1fct[l, cb, 1] # (E[b]-E[c]+mulst[l])
    return phi1fct, phi1fct_energy

@cython.boundscheck(False)
def c_generate_paulifct(sys):
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    si = sys.si
    cdef np.ndarray[doublenp, ndim=1] mulst = sys.leads.mulst
    cdef np.ndarray[doublenp, ndim=1] tlst = sys.leads.tlst
    cdef np.ndarray[doublenp, ndim=1] dlst = sys.leads.dlst
    #
    cdef longnp c, b, cb
    cdef intnp bcharge, ccharge, charge, l
    cdef doublenp xcb
    cdef intnp nleads = si.nleads
    #
    cdef np.ndarray[doublenp, ndim=3] paulifct = np.zeros((nleads, si.ndm1, 2), dtype=np.double)
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    #
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                xcb = (Xba[l, b, c]*Xba[l, c, b]).real
                paulifct[l, cb, 0] = xcb*func_pauli(+(E[b]-E[c]+mulst[l]), tlst[l], dlst[l])
                paulifct[l, cb, 1] = xcb*func_pauli(-(E[b]-E[c]+mulst[l]), tlst[l], dlst[l]) #2*pi*xcb - paulifct[l, cb, 0]
    return paulifct

#---------------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_pauli(sys):
    cdef np.ndarray[doublenp, ndim=3] paulifct = sys.paulifct
    si = sys.si
    cdef bint symq = sys.funcp.symq
    cdef longnp norm_rowp = sys.funcp.norm_row
    #
    cdef bb_bool
    cdef longnp b, bb, a, aa, c, cc, ba, cb
    cdef intnp acharge, bcharge, ccharge, charge, l
    cdef intnp norm_row, last_row
    cdef intnp nleads = si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    #
    norm_row = norm_rowp if symq else si.npauli
    last_row = si.npauli-1 if symq else si.npauli
    #
    cdef np.ndarray[doublenp, ndim=2] kern = np.zeros((last_row+1, si.npauli), dtype=np.double)
    cdef np.ndarray[doublenp, ndim=1] bvec = np.zeros(last_row+1, dtype=np.double)
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
    cdef np.ndarray[doublenp, ndim=1] phi0 = sys.phi0
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[doublenp, ndim=3] paulifct = sys.paulifct
    si = sys.si
    #
    cdef longnp c, cc, b, bb, cb
    cdef intnp bcharge, ccharge, charge, l, nleads
    cdef doublenp fct1, fct2
    nleads = si.nleads
    #
    cdef np.ndarray[doublenp, ndim=1] current = np.zeros(nleads, dtype=np.double)
    cdef np.ndarray[doublenp, ndim=1] energy_current = np.zeros(nleads, dtype=np.double)
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
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

#---------------------------------------------------------------------------------------------------------
# Redfield approach
#---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_redfield(sys):
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    cdef np.ndarray[complexnp, ndim=3] phi1fct = sys.phi1fct
    si = sys.si
    cdef bint symq = sys.funcp.symq
    cdef longnp norm_rowp = sys.funcp.norm_row
    #
    cdef boolnp bbp_bool, bbpi_bool
    cdef intnp charge, acharge, bcharge, ccharge, l, nleads, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef longnp b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi, \
                c, cp, ccp, ccpi, \
                bpap, ba, bppa, cbpp, cpbp, cb
    cdef longnp norm_row, last_row, ndm0, npauli,
    cdef complexnp fct_aap, fct_bppbp, fct_bbpp, fct_ccp
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[boolnp, ndim=1] conjdm0 = si.conjdm0
    #
    norm_row = norm_rowp if symq else si.ndm0r
    last_row = si.ndm0r-1 if symq else si.ndm0r
    ndm0, npauli, nleads = si.ndm0, si.npauli, si.nleads
    #
    cdef np.ndarray[doublenp, ndim=2] kern = np.zeros((last_row+1, si.ndm0r), dtype=np.double)
    cdef np.ndarray[doublenp, ndim=1] bvec = np.zeros(last_row+1, dtype=np.double)
    bvec[norm_row] = 1
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
                        bpap = lenlst[acharge]*dictdm[bp] + dictdm[ap] + shiftlst1[acharge]
                        ba = lenlst[acharge]*dictdm[b] + dictdm[a] + shiftlst1[acharge]
                        fct_aap = 0
                        for l in range(nleads):
                            fct_aap += (+Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate()
                                        -Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, ba, 0])
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.imag                              # kern[bbp, aap]   += fct_aap.imag
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] + fct_aap.real*aap_sgn                # kern[bbp, aapi]  += fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.imag*aap_sgn          # kern[bbpi, aapi] += fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] - fct_aap.real                        # kern[bbpi, aap]  -= fct_aap.real
                #--------------------------------------------------
                for bpp in si.statesdm[bcharge]:
                    bppbp = mapdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[acharge]:
                            bppa = lenlst[acharge]*dictdm[bpp] + dictdm[a] + shiftlst1[acharge]
                            for l in range(nleads):
                                fct_bppbp += +Xba[l, b, a]*Xba[l, a, bpp]*phi1fct[l, bppa, 1].conjugate()
                        for c in si.statesdm[ccharge]:
                            cbpp = lenlst[bcharge]*dictdm[c] + dictdm[bpp] + shiftlst1[bcharge]
                            for l in range(nleads):
                                fct_bppbp += +Xba[l, b, c]*Xba[l, c, bpp]*phi1fct[l, cbpp, 0]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.imag                        # kern[bbp, bppbp] += fct_bppbp.imag
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] + fct_bppbp.real*bppbp_sgn        # kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.imag*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] - fct_bppbp.real                  # kern[bbpi, bppbp] -= fct_bppbp.real
                    #--------------------------------------------------
                    bbpp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[acharge]:
                            bppa = lenlst[acharge]*dictdm[bpp] + dictdm[a] + shiftlst1[acharge]
                            for l in range(nleads):
                                fct_bbpp += -Xba[l, bpp, a]*Xba[l, a, bp]*phi1fct[l, bppa, 1]
                        for c in si.statesdm[ccharge]:
                            cbpp = lenlst[bcharge]*dictdm[c] + dictdm[bpp] + shiftlst1[bcharge]
                            for l in range(nleads):
                                fct_bbpp += -Xba[l, bpp, c]*Xba[l, c, bp]*phi1fct[l, cbpp, 0].conjugate()
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.imag                           # kern[bbp, bbpp] += fct_bbpp.imag
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] + fct_bbpp.real*bbpp_sgn            # kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.imag*bbpp_sgn      # kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] - fct_bbpp.real                     # kern[bbpi, bbpp] -= fct_bbpp.real
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[ccharge], si.statesdm[ccharge]):
                    ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                    if ccp != -1:
                        cpbp = lenlst[bcharge]*dictdm[cp] + dictdm[bp] + shiftlst1[bcharge]
                        cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                        fct_ccp = 0
                        for l in range(nleads):
                            fct_ccp += (+Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cpbp, 1]
                                        -Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cb, 1].conjugate())
                        ccpi = ndm0 + ccp - npauli
                        ccp_sgn = +1 if conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]] else -1
                        kern[bbp, ccp] = kern[bbp, ccp] + fct_ccp.imag                              # kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= ndm0:
                            kern[bbp, ccpi] = kern[bbp, ccpi] + fct_ccp.real*ccp_sgn                # kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] = kern[bbpi, ccpi] + fct_ccp.imag*ccp_sgn          # kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] = kern[bbpi, ccp] - fct_ccp.real                        # kern[bbpi, ccp] -= fct_ccp.real
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=np.double)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            kern[norm_row, bb] += 1
    return kern, bvec

@cython.boundscheck(False)
def c_generate_phi1_redfield(sys):
    cdef np.ndarray[doublenp, ndim=1] phi0p = sys.phi0
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    cdef np.ndarray[complexnp, ndim=3] phi1fct = sys.phi1fct
    cdef np.ndarray[complexnp, ndim=3] phi1fct_energy = sys.phi1fct_energy
    si = sys.si
    #
    cdef boolnp bpb_conj, ccp_conj
    cdef intnp bcharge, ccharge, charge, l, nleads,
    cdef longnp c, b, cb, bp, bpb, cp, ccp, cbp, cpb
    cdef longnp ndm0, ndm1, npauli
    cdef complexnp fct1, fct2, fct1h, fct2h, phi0bpb, phi0ccp
    ndm0, ndm1, npauli, nleads = si.ndm0, si.ndm1, si.npauli, si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[boolnp, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[complexnp, ndim=2] phi1 = np.zeros((nleads, ndm1), dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] current = np.zeros(nleads, dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] energy_current = np.zeros(nleads, dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] phi0 = np.zeros(ndm0, dtype=np.complex)
    #
    phi0[0:npauli] = phi0p[0:npauli]
    phi0[npauli:ndm0] = phi0p[npauli:ndm0] + 1j*phi0p[ndm0:]
    #
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                for bp in si.statesdm[bcharge]:
                    bpb = mapdm0[lenlst[bcharge]*dictdm[bp] + dictdm[b] + shiftlst0[bcharge]]
                    if bpb != -1:
                        cbp = lenlst[bcharge]*dictdm[c] + dictdm[bp] + shiftlst1[bcharge]
                        fct1 = phi1fct[l, cbp, 0]
                        fct1h = phi1fct_energy[l, cbp, 0]
                        bpb_conj = conjdm0[lenlst[bcharge]*dictdm[bp] + dictdm[b] + shiftlst0[bcharge]]
                        phi0bpb = phi0[bpb] if bpb_conj else phi0[bpb].conjugate()
                        phi1[l, cb] = phi1[l, cb] + Xba[l, c, bp]*phi0bpb*fct1
                        current[l] = current[l] + Xba[l, b, c]*Xba[l, c, bp]*phi0bpb*fct1
                        energy_current[l] = energy_current[l] + Xba[l, b, c]*Xba[l, c, bp]*phi0bpb*fct1h
                for cp in si.statesdm[ccharge]:
                    ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                    if ccp != -1:
                        cpb = lenlst[bcharge]*dictdm[cp] + dictdm[b] + shiftlst1[bcharge]
                        fct2 = phi1fct[l, cpb, 1]
                        fct2h = phi1fct_energy[l, cpb, 1]
                        ccp_conj = conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                        phi0ccp = phi0[ccp] if ccp_conj else phi0[ccp].conjugate()
                        phi1[l, cb] = phi1[l, cb] + Xba[l, cp, b]*phi0ccp*fct2
                        current[l] = current[l] + Xba[l, b, c]*phi0ccp*Xba[l, cp, b]*fct2
                        energy_current[l] = energy_current[l] + Xba[l, b, c]*phi0ccp*Xba[l, cp, b]*fct2h
    for l in range(nleads):
        current[l] = -2*current[l].imag
        energy_current[l] = -2*energy_current[l].imag
    return phi1, current, energy_current

@cython.boundscheck(False)
def c_generate_vec_redfield(np.ndarray[doublenp, ndim=1] phi0p, sys):
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    cdef np.ndarray[complexnp, ndim=3] phi1fct = sys.phi1fct
    si = sys.si
    cdef longnp norm_row = sys.funcp.norm_row
    #
    cdef boolnp bbp_bool
    cdef intnp charge, acharge, bcharge, ccharge, l, nleads, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef longnp b, bp, bbp, bb, \
                a, ap, aap, \
                bpp, bppbp, bbpp, \
                c, cp, ccp, \
                bpap, ba, bppa, cbpp, cpbp, cb
    cdef longnp ndm0, npauli
    cdef complexnp fct_aap, fct_bppbp, fct_bbpp, fct_ccp, norm
    cdef complexnp phi0aap, phi0bppbp, phi0bbpp, phi0ccp
    ndm0, npauli, nleads = si.ndm0, si.npauli, si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[boolnp, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[complexnp, ndim=1] phi0 = np.zeros(ndm0, dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] i_dphi0_dt = np.zeros(ndm0, dtype=np.complex)
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
                            bpap = lenlst[acharge]*dictdm[bp] + dictdm[ap] + shiftlst1[acharge]
                            ba = lenlst[acharge]*dictdm[b] + dictdm[a] + shiftlst1[acharge]
                            fct_aap = 0
                            for l in range(nleads):
                                fct_aap += (+Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, bpap, 0].conjugate()
                                            -Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, ba, 0])
                            phi0aap = phi0[aap] if conjdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]] else phi0[aap].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_aap*phi0aap
                    #--------------------------------------------------
                    for bpp in si.statesdm[charge]:
                        bppbp = mapdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]]
                        if bppbp != -1:
                            fct_bppbp = 0
                            for a in si.statesdm[charge-1]:
                                bppa = lenlst[acharge]*dictdm[bpp] + dictdm[a] + shiftlst1[acharge]
                                for l in range(nleads):
                                    fct_bppbp += +Xba[l, b, a]*Xba[l, a, bpp]*phi1fct[l, bppa, 1].conjugate()
                            for c in si.statesdm[charge+1]:
                                cbpp = lenlst[bcharge]*dictdm[c] + dictdm[bpp] + shiftlst1[bcharge]
                                for l in range(nleads):
                                    fct_bppbp += +Xba[l, b, c]*Xba[l, c, bpp]*phi1fct[l, cbpp, 0]
                            phi0bppbp = phi0[bppbp] if conjdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]] else phi0[bppbp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_bppbp*phi0bppbp
                        #--------------------------------------------------
                        bbpp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]]
                        if bbpp != -1:
                            fct_bbpp = 0
                            for a in si.statesdm[charge-1]:
                                bppa = lenlst[acharge]*dictdm[bpp] + dictdm[a] + shiftlst1[acharge]
                                for l in range(nleads):
                                    fct_bbpp += -Xba[l, bpp, a]*Xba[l, a, bp]*phi1fct[l, bppa, 1]
                            for c in si.statesdm[charge+1]:
                                cbpp = lenlst[bcharge]*dictdm[c] + dictdm[bpp] + shiftlst1[bcharge]
                                for l in range(nleads):
                                    fct_bbpp += -Xba[l, bpp, c]*Xba[l, c, bp]*phi1fct[l, cbpp, 0].conjugate()
                            phi0bbpp = phi0[bbpp] if conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]] else phi0[bbpp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_bbpp*phi0bbpp
                    #--------------------------------------------------
                    for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                        ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                        if ccp != -1:
                            cpbp = lenlst[bcharge]*dictdm[cp] + dictdm[bp] + shiftlst1[bcharge]
                            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                            fct_ccp = 0
                            for l in range(nleads):
                                fct_ccp += (+Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cpbp, 1]
                                            -Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cb, 1].conjugate())
                            phi0ccp = phi0[ccp] if conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]] else phi0[ccp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_ccp*phi0ccp
                    #--------------------------------------------------
    i_dphi0_dt[norm_row] = 1j*(norm-1)
    #print(np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[npauli:ndm0].real)))
    return np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[npauli:ndm0].real))

#---------------------------------------------------------------------------------------------------------
# 1 von Neumann approach
#---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_1vN(sys):
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    cdef np.ndarray[complexnp, ndim=3] phi1fct = sys.phi1fct
    si = sys.si
    cdef bint symq = sys.funcp.symq
    cdef longnp norm_rowp = sys.funcp.norm_row
    #
    cdef boolnp bbp_bool, bbpi_bool
    cdef intnp charge, acharge, bcharge, ccharge, l, nleads, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef longnp b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi, \
                c, cp, ccp, ccpi, \
                bpa, bap, cbp, ba, cb, cpb
    cdef longnp norm_row, last_row, ndm0, npauli,
    cdef complexnp fct_aap, fct_bppbp, fct_bbpp, fct_ccp
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[boolnp, ndim=1] conjdm0 = si.conjdm0
    #
    norm_row = norm_rowp if symq else si.ndm0r
    last_row = si.ndm0r-1 if symq else si.ndm0r
    ndm0, npauli, nleads = si.ndm0, si.npauli, si.nleads
    #
    cdef np.ndarray[doublenp, ndim=2] kern = np.zeros((last_row+1, si.ndm0r), dtype=np.double)
    cdef np.ndarray[doublenp, ndim=1] bvec = np.zeros(last_row+1, dtype=np.double)
    bvec[norm_row] = 1
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
                        bpa = lenlst[acharge]*dictdm[bp] + dictdm[a] + shiftlst1[acharge]
                        bap = lenlst[acharge]*dictdm[b] + dictdm[ap] + shiftlst1[acharge]
                        fct_aap = 0
                        for l in range(nleads):
                            fct_aap += (+Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                        -Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, bap, 0])
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.imag                              # kern[bbp, aap]   += fct_aap.imag
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] + fct_aap.real*aap_sgn                # kern[bbp, aapi]  += fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.imag*aap_sgn          # kern[bbpi, aapi] += fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] - fct_aap.real                        # kern[bbpi, aap]  -= fct_aap.real
                #--------------------------------------------------
                for bpp in si.statesdm[bcharge]:
                    bppbp = mapdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[acharge]:
                            bpa = lenlst[acharge]*dictdm[bp] + dictdm[a] + shiftlst1[acharge]
                            for l in range(nleads):
                                fct_bppbp += +Xba[l, b, a]*Xba[l, a, bpp]*phi1fct[l, bpa, 1].conjugate()
                        for c in si.statesdm[ccharge]:
                            cbp = lenlst[bcharge]*dictdm[c] + dictdm[bp] + shiftlst1[bcharge]
                            for l in range(nleads):
                                fct_bppbp += +Xba[l, b, c]*Xba[l, c, bpp]*phi1fct[l, cbp, 0]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.imag                        # kern[bbp, bppbp] += fct_bppbp.imag
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] + fct_bppbp.real*bppbp_sgn        # kern[bbp, bppbpi] += fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.imag*bppbp_sgn  # kern[bbpi, bppbpi] += fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] - fct_bppbp.real                  # kern[bbpi, bppbp] -= fct_bppbp.real
                    #--------------------------------------------------
                    bbpp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[acharge]:
                            ba = lenlst[acharge]*dictdm[b] + dictdm[a] + shiftlst1[acharge]
                            for l in range(nleads):
                                fct_bbpp += -Xba[l, bpp, a]*Xba[l, a, bp]*phi1fct[l, ba, 1]
                        for c in si.statesdm[ccharge]:
                            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                            for l in range(nleads):
                                fct_bbpp += -Xba[l, bpp, c]*Xba[l, c, bp]*phi1fct[l, cb, 0].conjugate()
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.imag                           # kern[bbp, bbpp] += fct_bbpp.imag
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] + fct_bbpp.real*bbpp_sgn            # kern[bbp, bbppi] += fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.imag*bbpp_sgn      # kern[bbpi, bbppi] += fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] - fct_bbpp.real                     # kern[bbpi, bbpp] -= fct_bbpp.real
                #--------------------------------------------------
                for c, cp in itertools.product(si.statesdm[ccharge], si.statesdm[ccharge]):
                    ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                    if ccp != -1:
                        cbp = lenlst[bcharge]*dictdm[c] + dictdm[bp] + shiftlst1[bcharge]
                        cpb = lenlst[bcharge]*dictdm[cp] + dictdm[b] + shiftlst1[bcharge]
                        fct_ccp = 0
                        for l in range(nleads):
                            fct_ccp += (+Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cbp, 1]
                                        -Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                        ccpi = ndm0 + ccp - npauli
                        ccp_sgn = +1 if conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]] else -1
                        kern[bbp, ccp] = kern[bbp, ccp] + fct_ccp.imag                              # kern[bbp, ccp] += fct_ccp.imag
                        if ccpi >= ndm0:
                            kern[bbp, ccpi] = kern[bbp, ccpi] + fct_ccp.real*ccp_sgn                # kern[bbp, ccpi] += fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] = kern[bbpi, ccpi] + fct_ccp.imag*ccp_sgn          # kern[bbpi, ccpi] += fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] = kern[bbpi, ccp] - fct_ccp.real                        # kern[bbpi, ccp] -= fct_ccp.real
                #--------------------------------------------------
    # Normalisation condition
    kern[norm_row] = np.zeros(si.ndm0r, dtype=np.double)
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            kern[norm_row, bb] += 1
    return kern, bvec

@cython.boundscheck(False)
def c_generate_phi1_1vN(sys):
    cdef np.ndarray[doublenp, ndim=1] phi0p = sys.phi0
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    cdef np.ndarray[complexnp, ndim=3] phi1fct = sys.phi1fct
    cdef np.ndarray[complexnp, ndim=3] phi1fct_energy = sys.phi1fct_energy
    si = sys.si
    #
    cdef boolnp bpb_conj, ccp_conj
    cdef intnp bcharge, ccharge, charge, l, nleads,
    cdef longnp c, b, cb, bp, bpb, cp, ccp
    cdef longnp ndm0, ndm1, npauli
    cdef complexnp fct1, fct2, fct1h, fct2h, phi0bpb, phi0ccp
    ndm0, ndm1, npauli, nleads = si.ndm0, si.ndm1, si.npauli, si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[boolnp, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[complexnp, ndim=2] phi1 = np.zeros((nleads, ndm1), dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] current = np.zeros(nleads, dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] energy_current = np.zeros(nleads, dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] phi0 = np.zeros(ndm0, dtype=np.complex)
    #
    phi0[0:npauli] = phi0p[0:npauli]
    phi0[npauli:ndm0] = phi0p[npauli:ndm0] + 1j*phi0p[ndm0:]
    #
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
            for l in range(nleads):
                fct1 = phi1fct[l, cb, 0]
                fct2 = phi1fct[l, cb, 1]
                fct1h = phi1fct_energy[l, cb, 0]
                fct2h = phi1fct_energy[l, cb, 1]
                for bp in si.statesdm[bcharge]:
                    bpb = mapdm0[lenlst[bcharge]*dictdm[bp] + dictdm[b] + shiftlst0[bcharge]]
                    if bpb != -1:
                        bpb_conj = conjdm0[lenlst[bcharge]*dictdm[bp] + dictdm[b] + shiftlst0[bcharge]]
                        phi0bpb = phi0[bpb] if bpb_conj else phi0[bpb].conjugate()
                        phi1[l, cb] = phi1[l, cb] + Xba[l, c, bp]*phi0bpb*fct1
                        current[l] = current[l] + Xba[l, b, c]*Xba[l, c, bp]*phi0bpb*fct1
                        energy_current[l] = energy_current[l] + Xba[l, b, c]*Xba[l, c, bp]*phi0bpb*fct1h
                for cp in si.statesdm[ccharge]:
                    ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                    if ccp != -1:
                        ccp_conj = conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                        phi0ccp = phi0[ccp] if ccp_conj else phi0[ccp].conjugate()
                        phi1[l, cb] = phi1[l, cb] + Xba[l, cp, b]*phi0ccp*fct2
                        current[l] = current[l] + Xba[l, b, c]*phi0ccp*Xba[l, cp, b]*fct2
                        energy_current[l] = energy_current[l] + Xba[l, b, c]*phi0ccp*Xba[l, cp, b]*fct2h
    for l in range(nleads):
        current[l] = -2*current[l].imag
        energy_current[l] = -2*energy_current[l].imag
    return phi1, current, energy_current

@cython.boundscheck(False)
def c_generate_vec_1vN(np.ndarray[doublenp, ndim=1] phi0p, sys):
    #cdef np.ndarray[doublenp, ndim=1] phi0p = sys.phi0
    cdef np.ndarray[doublenp, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complexnp, ndim=3] Xba = sys.leads.Xba
    cdef np.ndarray[complexnp, ndim=3] phi1fct = sys.phi1fct
    si = sys.si
    cdef longnp norm_row = sys.funcp.norm_row
    #
    cdef boolnp bbp_bool
    cdef intnp charge, acharge, bcharge, ccharge, l, nleads, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef longnp b, bp, bbp, bb, \
                a, ap, aap, \
                bpp, bppbp, bbpp, \
                c, cp, ccp, \
                bpa, bap, cbp, ba, cb, cpb
    cdef longnp ndm0, npauli
    cdef complexnp fct_aap, fct_bppbp, fct_bbpp, fct_ccp, norm
    cdef complexnp phi0aap, phi0bppbp, phi0bbpp, phi0ccp
    ndm0, npauli, nleads = si.ndm0, si.npauli, si.nleads
    #
    cdef np.ndarray[longnp, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[longnp, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[longnp, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[longnp, ndim=1] shiftlst1 = si.shiftlst1
    cdef np.ndarray[longnp, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[boolnp, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[boolnp, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[complexnp, ndim=1] phi0 = np.zeros(ndm0, dtype=np.complex)
    cdef np.ndarray[complexnp, ndim=1] i_dphi0_dt = np.zeros(ndm0, dtype=np.complex)
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
                            bpa = lenlst[acharge]*dictdm[bp] + dictdm[a] + shiftlst1[acharge]
                            bap = lenlst[acharge]*dictdm[b] + dictdm[ap] + shiftlst1[acharge]
                            fct_aap = 0
                            for l in range(nleads):
                                fct_aap += (+Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                            -Xba[l, b, a]*Xba[l, ap, bp]*phi1fct[l, bap, 0])
                            phi0aap = phi0[aap] if conjdm0[lenlst[acharge]*dictdm[a] + dictdm[ap] + shiftlst0[acharge]] else phi0[aap].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_aap*phi0aap
                    #--------------------------------------------------
                    for bpp in si.statesdm[charge]:
                        bppbp = mapdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]]
                        if bppbp != -1:
                            fct_bppbp = 0
                            for a in si.statesdm[charge-1]:
                                bpa = lenlst[acharge]*dictdm[bp] + dictdm[a] + shiftlst1[acharge]
                                for l in range(nleads):
                                    fct_bppbp += +Xba[l, b, a]*Xba[l, a, bpp]*phi1fct[l, bpa, 1].conjugate()
                            for c in si.statesdm[charge+1]:
                                cbp = lenlst[bcharge]*dictdm[c] + dictdm[bp] + shiftlst1[bcharge]
                                for l in range(nleads):
                                    fct_bppbp += +Xba[l, b, c]*Xba[l, c, bpp]*phi1fct[l, cbp, 0]
                            phi0bppbp = phi0[bppbp] if conjdm0[lenlst[bcharge]*dictdm[bpp] + dictdm[bp] + shiftlst0[bcharge]] else phi0[bppbp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_bppbp*phi0bppbp
                        #--------------------------------------------------
                        bbpp = mapdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]]
                        if bbpp != -1:
                            fct_bbpp = 0
                            for a in si.statesdm[charge-1]:
                                ba = lenlst[acharge]*dictdm[b] + dictdm[a] + shiftlst1[acharge]
                                for l in range(nleads):
                                    fct_bbpp += -Xba[l, bpp, a]*Xba[l, a, bp]*phi1fct[l, ba, 1]
                            for c in si.statesdm[charge+1]:
                                cb = lenlst[bcharge]*dictdm[c] + dictdm[b] + shiftlst1[bcharge]
                                for l in range(nleads):
                                    fct_bbpp += -Xba[l, bpp, c]*Xba[l, c, bp]*phi1fct[l, cb, 0].conjugate()
                            phi0bbpp = phi0[bbpp] if conjdm0[lenlst[bcharge]*dictdm[b] + dictdm[bpp] + shiftlst0[bcharge]] else phi0[bbpp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_bbpp*phi0bbpp
                    #--------------------------------------------------
                    for c, cp in itertools.product(si.statesdm[charge+1], si.statesdm[charge+1]):
                        ccp = mapdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]]
                        if ccp != -1:
                            cbp = lenlst[bcharge]*dictdm[c] + dictdm[bp] + shiftlst1[bcharge]
                            cpb = lenlst[bcharge]*dictdm[cp] + dictdm[b] + shiftlst1[bcharge]
                            fct_ccp = 0
                            for l in range(nleads):
                                fct_ccp += (+Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cbp, 1]
                                            -Xba[l, b, c]*Xba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                            phi0ccp = phi0[ccp] if conjdm0[lenlst[ccharge]*dictdm[c] + dictdm[cp] + shiftlst0[ccharge]] else phi0[ccp].conjugate()
                            i_dphi0_dt[bbp] = i_dphi0_dt[bbp] + fct_ccp*phi0ccp
                    #--------------------------------------------------
    i_dphi0_dt[norm_row] = 1j*(norm-1)
    #print(np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[npauli:ndm0].real)))
    return np.concatenate((i_dphi0_dt.imag, i_dphi0_dt[npauli:ndm0].real))
#---------------------------------------------------------------------------------------------------------
