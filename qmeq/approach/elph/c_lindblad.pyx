"""Module containing cython functions, which generate first order Lindblad kernel.
   For docstrings see documentation of module lindblad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...aprclass import ApproachElPh
from ...specfunc.c_specfunc_elph cimport FuncPauliElPh

from ...mytypes import doublenp
from ...mytypes import complexnp

from ..base.c_lindblad import generate_tLba
from ..base.c_lindblad import generate_kern_lindblad
from ..base.c_lindblad import generate_current_lindblad
from ..base.c_lindblad import generate_vec_lindblad
from ..base.c_pauli import generate_norm_vec

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

from libc.math cimport sqrt


# ---------------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------------
@cython.cdivision(True)
@cython.boundscheck(False)
def generate_tLbbp_elph(self):
    cdef np.ndarray[complex_t, ndim=3] Vbbp = self.baths.Vbbp
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    si = self.si
    #
    cdef long_t b, bp
    cdef int_t  charge, l
    cdef double_t Ebbp
    cdef int_t nbaths = si.nbaths
    #
    cdef np.ndarray[complex_t, ndim=4] tLbbp = np.zeros((nbaths, si.nmany, si.nmany, 2), dtype=complexnp)
    #
    func_pauli = FuncPauliElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                               self.baths.bath_func, self.funcp.eps_elph)
    # Diagonal elements
    for l in range(nbaths):
        func_pauli.eval(0., l)
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                tLbbp[l, b, b, 0] = sqrt(0.5*func_pauli.val)*Vbbp[l, b, b]
                tLbbp[l, b, b, 1] = tLbbp[l, b, b, 0].conjugate()
    # Off-diagonal elements
    for charge in range(si.ncharge):
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            Ebbp = E[b]-E[bp]
            for l in range(nbaths):
                func_pauli.eval(Ebbp, l)
                tLbbp[l, b, bp, 0] = sqrt(0.5*func_pauli.val)*Vbbp[l, b, bp]
                tLbbp[l, b, bp, 1] = sqrt(0.5*func_pauli.val)*Vbbp[l, bp, b].conjugate()
    self.tLbbp = tLbbp
    return 0


@cython.boundscheck(False)
def generate_kern_lindblad_elph(self):
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    cdef np.ndarray[complex_t, ndim=4] tLbbp = self.tLbbp
    si = self.si
    #
    cdef bool_t bbp_bool, bbpi_bool
    cdef int_t charge, l, q, nbaths, \
               aap_sgn, bppbp_sgn, bbpp_sgn
    cdef long_t b, bp, bbp, bbpi, bb, \
                a, ap, aap, aapi, \
                bpp, bppbp, bppbpi, bbpp, bbppi
    cdef long_t ndm0, ndm0r, npauli,
    cdef complex_t fct_aap, fct_bppbp, fct_bbpp
    #
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    cdef np.ndarray[bool_t, ndim=1] conjdm0 = si.conjdm0
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
                        fct_aap = 0
                        for l in range(nbaths):
                            for q in range(2):
                                fct_aap += tLbbp[l, b, a, q]*tLbbp[l, bp, a, q].conjugate()
                        aapi = ndm0 + aap - npauli
                        aap_sgn = +1 if conjdm0[lenlst[charge]*dictdm[a] + dictdm[ap] + shiftlst0[charge]] else -1
                        kern[bbp, aap] = kern[bbp, aap] + fct_aap.real
                        if aapi >= ndm0:
                            kern[bbp, aapi] = kern[bbp, aapi] - fct_aap.imag*aap_sgn
                            if bbpi_bool:
                                kern[bbpi, aapi] = kern[bbpi, aapi] + fct_aap.real*aap_sgn
                        if bbpi_bool:
                            kern[bbpi, aap] = kern[bbpi, aap] + fct_aap.imag
                # --------------------------------------------------
                for bpp in si.statesdm[charge]:
                    bppbp = mapdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]]
                    if bppbp != -1:
                        fct_bppbp = 0
                        for a in si.statesdm[charge]:
                            for l in range(nbaths):
                                for q in range(2):
                                    fct_bppbp += -0.5*tLbbp[l, a, b, q].conjugate()*tLbbp[l, a, bpp, q]
                        bppbpi = ndm0 + bppbp - npauli
                        bppbp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[bpp] + dictdm[bp] + shiftlst0[charge]] else -1
                        kern[bbp, bppbp] = kern[bbp, bppbp] + fct_bppbp.real
                        if bppbpi >= ndm0:
                            kern[bbp, bppbpi] = kern[bbp, bppbpi] - fct_bppbp.imag*bppbp_sgn
                            if bbpi_bool:
                                kern[bbpi, bppbpi] = kern[bbpi, bppbpi] + fct_bppbp.real*bppbp_sgn
                        if bbpi_bool:
                            kern[bbpi, bppbp] = kern[bbpi, bppbp] + fct_bppbp.imag
                    # --------------------------------------------------
                    bbpp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]]
                    if bbpp != -1:
                        fct_bbpp = 0
                        for a in si.statesdm[charge]:
                            for l in range(nbaths):
                                for q in range(2):
                                    fct_bbpp += -0.5*tLbbp[l, a, bpp, q].conjugate()*tLbbp[l, a, bp, q]
                        bbppi = ndm0 + bbpp - npauli
                        bbpp_sgn = +1 if conjdm0[lenlst[charge]*dictdm[b] + dictdm[bpp] + shiftlst0[charge]] else -1
                        kern[bbp, bbpp] = kern[bbp, bbpp] + fct_bbpp.real
                        if bbppi >= ndm0:
                            kern[bbp, bbppi] = kern[bbp, bbppi] - fct_bbpp.imag*bbpp_sgn
                            if bbpi_bool:
                                kern[bbpi, bbppi] = kern[bbpi, bbppi] + fct_bbpp.real*bbpp_sgn
                        if bbpi_bool:
                            kern[bbpi, bbpp] = kern[bbpi, bbpp] + fct_bbpp.imag
                # --------------------------------------------------
    return 0


class ApproachLindblad(ApproachElPh):

    kerntype = 'Lindblad'
    generate_fct = generate_tLba
    generate_kern = generate_kern_lindblad
    generate_current = generate_current_lindblad
    generate_vec = generate_vec_lindblad
    #
    generate_kern_elph = generate_kern_lindblad_elph
    generate_fct_elph = generate_tLbbp_elph
# ---------------------------------------------------------------------------------------------------------
