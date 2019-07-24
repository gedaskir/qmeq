"""Module containing cython functions, which generate first order Redfield kernel.
   For docstrings see documentation of module neumann1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from .c_neumann1 import generate_w1fct_elph
from ...aprclass import ApproachElPh

from ...mytypes import doublenp
from ...mytypes import complexnp

from ..base.c_neumann1 import generate_phi1fct
from ..base.c_redfield import generate_kern_redfield
from ..base.c_redfield import generate_current_redfield
from ..base.c_redfield import generate_vec_redfield
from ..base.c_pauli import generate_norm_vec

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t


# ---------------------------------------------------------------------------------------------------------
# Redfield approach
# ---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def generate_kern_redfield_elph(self):
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Vbbp = self.baths.Vbbp
    cdef np.ndarray[complex_t, ndim=3] w1fct = self.w1fct
    si, si_elph = self.si, self.si_elph
    #
    cdef bool_t bbp_bool, bbpi_bool
    cdef int_t charge, l, nbaths, \
               aap_sgn, bppbp_sgn, bbpp_sgn, ccp_sgn
    cdef long_t b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi, \
                c, cp, ccp, ccpi, \
                bpap, ba, bppa, cbpp, cpbp, cb
    cdef long_t ndm0, ndm0r, npauli,
    cdef complex_t fct_aap, fct_bppbp, fct_bbpp, fct_ccp, \
                   gamma_ba_bpap, gamma_ba_bppa, gamma_bc_bppc, \
                   gamma_abpp_abp, gamma_cbpp_cbp, gamma_bc_bpcp
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[bool_t, ndim=1] conjdm0 = si.conjdm0
    #
    cdef np.ndarray[long_t, ndim=1] mapdm0_ = si_elph.mapdm0
    #
    ndm0r, ndm0, npauli, nbaths = si.ndm0r, si.ndm0, si.npauli, si.nbaths
    #
    if self.kern is None:
        self.kern_ext = np.zeros((ndm0r+1, ndm0r), dtype=doublenp)
        self.kern = self.kern_ext[0:-1, :]
        generate_norm_vec(self, ndm0r)

    cdef np.ndarray[double_t, ndim=2] kern = self.kern
    for charge in range(si.ncharge):
        for b, bp in itertools.combinations_with_replacement(si.statesdm[charge], 2):
            bbp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            bbp_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            if bbp != -1 and bbp_bool:
                bbpi = ndm0 + bbp - npauli
                bbpi_bool = True if bbpi >= ndm0 else False
                # --------------------------------------------------
                for a, ap in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    aap = mapdm0[lenlst[charge]*dictdm[a] + dictdm[ap] + shiftlst0[charge]]
                    if aap != -1:
                        bpap = mapdm0_[lenlst[charge]*dictdm[bp] + dictdm[ap] + shiftlst0[charge]]
                        ba = mapdm0_[lenlst[charge]*dictdm[b] + dictdm[a] + shiftlst0[charge]]
                        fct_aap = 0
                        for l in range(nbaths):
                            gamma_ba_bpap = 0.5*(Vbbp[l, b, a]*Vbbp[l, bp, ap].conjugate() +
                                                 Vbbp[l, a, b].conjugate()*Vbbp[l, ap, bp])
                            fct_aap += gamma_ba_bpap*(w1fct[l, bpap, 0].conjugate() - w1fct[l, ba, 0])
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[charge]*dictdm[a] + dictdm[ap] + shiftlst0[charge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.imag
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] + fct_aap.real*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.imag*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] - fct_aap.real
                # --------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = mapdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            bppa = mapdm0_[lenlst[charge]*dictdm[bpp] + dictdm[a] + shiftlst0[charge]]
                            for l in range(nbaths):
                                gamma_ba_bppa = 0.5*(Vbbp[l, b, a]*Vbbp[l, bpp, a].conjugate() +
                                                     Vbbp[l, a, b].conjugate()*Vbbp[l, a, bpp])
                                fct_bppbp += +gamma_ba_bppa*w1fct[l, bppa, 1].conjugate()
                        for c in si.statesdm[charge]:
                            cbpp = mapdm0_[lenlst[charge]*dictdm[c] + dictdm[bpp] + shiftlst0[charge]]
                            for l in range(nbaths):
                                gamma_bc_bppc = 0.5*(Vbbp[l, b, c]*Vbbp[l, bpp, c].conjugate() +
                                                     Vbbp[l, c, b].conjugate()*Vbbp[l, c, bpp])
                                fct_bppbp += +gamma_bc_bppc*w1fct[l, cbpp, 0]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.imag
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] + fct_bppbp.real*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.imag*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] - fct_bppbp.real
                    # --------------------------------------------------
                    bbpp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            bppa = mapdm0_[lenlst[charge]*dictdm[bpp] + dictdm[a] + shiftlst0[charge]]
                            for l in range(nbaths):
                                gamma_abpp_abp = 0.5*(Vbbp[l, a, bpp].conjugate()*Vbbp[l, a, bp] +
                                                      Vbbp[l, bpp, a]*Vbbp[l, bp, a].conjugate())
                                fct_bbpp += -gamma_abpp_abp*w1fct[l, bppa, 1]
                        for c in si.statesdm[charge]:
                            cbpp = mapdm0_[lenlst[charge]*dictdm[c] + dictdm[bpp] + shiftlst0[charge]]
                            for l in range(nbaths):
                                gamma_cbpp_cbp = 0.5*(Vbbp[l, c, bpp].conjugate()*Vbbp[l, c, bp] +
                                                      Vbbp[l, bpp, c]*Vbbp[l, bp, c].conjugate())
                                fct_bbpp += -gamma_cbpp_cbp*w1fct[l, cbpp, 0].conjugate()
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.imag
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] + fct_bbpp.real*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.imag*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] - fct_bbpp.real
                # --------------------------------------------------
                for c, cp in itertools.product(si.statesdm[charge], si.statesdm[charge]):
                    ccp = mapdm0[lenlst[charge]*dictdm[c] + dictdm[cp] + shiftlst0[charge]]
                    if ccp != -1:
                        cpbp = mapdm0_[lenlst[charge]*dictdm[cp] + dictdm[bp] + shiftlst0[charge]]
                        cb = mapdm0_[lenlst[charge]*dictdm[c] + dictdm[b] + shiftlst0[charge]]
                        fct_ccp = 0
                        for l in range(nbaths):
                            gamma_bc_bpcp = 0.5*(Vbbp[l, b, c]*Vbbp[l, bp, cp].conjugate() +
                                                 Vbbp[l, c, b].conjugate()*Vbbp[l, cp, bp])
                            fct_ccp += gamma_bc_bpcp*(w1fct[l, cpbp, 1] - w1fct[l, cb, 1].conjugate())
                        ccpi = ndm0 + ccp - npauli
                        ccp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[c] + dictdm[cp] + shiftlst0[charge]] else -1
                        kern[bbp, ccp] = kern[bbp, ccp] + fct_ccp.imag
                        if ccpi >= ndm0:
                            kern[bbp, ccpi] = kern[bbp, ccpi] + fct_ccp.real*ccp_sgn
                            if bbpi_bool:
                                kern[bbpi, ccpi] = kern[bbpi, ccpi] + fct_ccp.imag*ccp_sgn
                        if bbpi_bool:
                            kern[bbpi, ccp] = kern[bbpi, ccp] - fct_ccp.real
                # --------------------------------------------------
    return 0


class ApproachRedfield(ApproachElPh):

    kerntype = 'Redfield'
    generate_fct = generate_phi1fct
    generate_kern = generate_kern_redfield
    generate_current = generate_current_redfield
    generate_vec = generate_vec_redfield
    #
    generate_kern_elph = generate_kern_redfield_elph
    generate_fct_elph = generate_w1fct_elph
