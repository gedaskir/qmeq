"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import complexnp
from ...mytypes import doublenp

# Cython imports

cimport numpy as np
cimport cython

from ...specfunc.c_specfunc_elph cimport FuncPauliElPh

from ..c_aprclass cimport ApproachElPh

from ..base.c_pauli cimport ApproachPauli as Approach


@cython.boundscheck(False)
def generate_paulifct_elph(self):
    cdef np.ndarray[double_t, ndim=1] E = self.qd.Ea
    cdef np.ndarray[complex_t, ndim=3] Vbbp = self.baths.Vbbp
    si = self.si_elph
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
    func_pauli = FuncPauliElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                                 self.baths.bath_func, self.funcp.eps_elph)
    #
    for charge in range(si.ncharge):
        # The diagonal elements b=bp are excluded, because they do not contribute
        for b, bp in itertools.permutations(si.statesdm[charge], 2):
            bbp_bool = booldm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
            if bbp_bool:
                bbp = mapdm0[lenlst[charge]*dictdm[b] + dictdm[bp] + shiftlst0[charge]]
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    xbbp = 0.5*(Vbbp[l, b, bp]*Vbbp[l, b, bp].conjugate() +
                                Vbbp[l, bp, b].conjugate()*Vbbp[l, bp, b]).real
                    func_pauli.eval(Ebbp, l)
                    paulifct[l, bbp] = xbbp*func_pauli.val
    self.paulifct_elph = paulifct
    return 0


# ---------------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
def generate_kern_pauli_elph(self):
    cdef np.ndarray[double_t, ndim=2] paulifct = self.paulifct_elph
    si, si_elph = self.si, self.si_elph
    cdef bint symq = self.funcp.symq
    cdef long_t norm_rowp = self.funcp.norm_row
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
    if self.kern is None:
        kern = np.zeros((last_row+1, si.npauli), dtype=doublenp)
    else:
        kern = self.kern
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
    self.kern = kern
    return 0


cdef class ApproachPauli(ApproachElPh):

    kerntype = 'Pauli'

    cpdef generate_fct(self):
        Approach.generate_fct(self)
        generate_paulifct_elph(self)

    cpdef generate_kern(self):
        Approach.generate_kern(self)
        generate_kern_pauli_elph(self)

    cpdef generate_current(self):
        Approach.generate_current(self)

    cpdef generate_vec(self, phi0):
        return Approach.generate_vec(self, phi0)
