"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

# cython: profile=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...aprclass import Approach_elph
from ...specfunc.specfuncc_elph cimport Func_pauli_elph

from ...mytypes import complexnp
from ...mytypes import doublenp

from ..base.c_pauli import c_generate_paulifct
from ..base.c_pauli import c_generate_kern_pauli
from ..base.c_pauli import c_generate_current_pauli
from ..base.c_pauli import c_generate_vec_pauli

cimport numpy as np
cimport cython

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

@cython.boundscheck(False)
def c_generate_paulifct_elph(sys):
    cdef np.ndarray[double_t, ndim=1] E = sys.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Vbbp = sys.baths.Vbbp
    si = sys.si_elph
    #
    cdef bool_t bbp_bool
    cdef long_t b, bp, bbp
    cdef int_t charge, l
    cdef double_t xbbp, Ebbp
    cdef int_t nbaths = si.nbaths
    #
    cdef np.ndarray[double_t, ndim=2] paulifct = np.zeros((nbaths, si.ndm0), dtype=doublenp)
    cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
    cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
    cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
    cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0
    cdef np.ndarray[bool_t, ndim=1] booldm0 = si.booldm0
    #
    func_pauli = Func_pauli_elph(sys.baths.tlst_ph, sys.baths.dlst_ph,
                                 sys.baths.bath_func, sys.funcp.eps_elph)
    #
    for charge in range(si.ncharge):
        # The diagonal elements b=bp are excluded, because they do not contribute
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            bbp_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            if bbp_bool:
                bbp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    xbbp = 0.5*(Vbbp[l, b, bp]*Vbbp[l, b, bp].conjugate()
                               +Vbbp[l, bp, b].conjugate()*Vbbp[l, bp, b]).real
                    func_pauli.eval(Ebbp, l)
                    paulifct[l, bbp] = xbbp*func_pauli.val
    sys.paulifct_elph = paulifct
    return 0

#---------------------------------------------------------------------------------------------------------
# Pauli master equation
#---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def c_generate_kern_pauli_elph(sys):
    cdef np.ndarray[double_t, ndim=2] paulifct = sys.paulifct_elph
    si, si_elph = sys.si, sys.si_elph
    cdef bint symq = sys.funcp.symq
    cdef long_t norm_rowp = sys.funcp.norm_row
    #
    cdef bool_t bb_bool, ba_conj
    cdef long_t b, bb, a, aa, ab, ba
    cdef int_t charge, l
    cdef int_t norm_row, last_row
    cdef int_t nbaths = si.nbaths
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
    norm_row = norm_rowp if symq else si.npauli
    last_row = si.npauli-1 if symq else si.npauli
    #
    cdef np.ndarray[double_t, ndim=2] kern
    #
    if sys.kern is None:
        kern = np.zeros((last_row+1, si.npauli), dtype=doublenp)
    else:
        kern = sys.kern
    #
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            bb_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
            if not (symq and bb == norm_row) and bb_bool:
                for a in si.statesdm[charge]:
                    aa = mapdm0[lenlst[charge]*dictdm[a] + dictdm[a] + shiftlst0[charge]]
                    ab = mapdm0_[lenlst[charge]*dictdm[a] + dictdm[b] + shiftlst0[charge]]
                    ba = mapdm0_[lenlst[charge]*dictdm[b] + dictdm[a] + shiftlst0[charge]]
                    if aa != -1 and ba != -1:
                        for l in range(nbaths):
                            kern[bb, bb] = kern[bb, bb] - paulifct[l, ab]
                            kern[bb, aa] = kern[bb, aa] + paulifct[l, ba]
    sys.kern = kern
    return 0

class Approach_Pauli(Approach_elph):

    kerntype = 'Pauli'
    generate_fct = c_generate_paulifct
    generate_kern = c_generate_kern_pauli
    generate_current = c_generate_current_pauli
    generate_vec = c_generate_vec_pauli
    #
    generate_kern_elph = c_generate_kern_pauli_elph
    generate_fct_elph = c_generate_paulifct_elph
#---------------------------------------------------------------------------------------------------------
