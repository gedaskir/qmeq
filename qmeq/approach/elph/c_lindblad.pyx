# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order Lindblad kernel.
   For docstrings see documentation of module lindblad."""

# Python imports

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

# Cython imports

cimport numpy as np
cimport cython

from libc.math cimport sqrt

from ...specfunc.c_specfunc_elph cimport FuncPauliElPh

from ..c_aprclass cimport ApproachElPh
from ..c_kernel_handler cimport KernelHandler

from ..base.c_lindblad cimport ApproachLindblad as ApproachLindbladBase


# ---------------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------------
cdef class ApproachLindblad(ApproachElPh):

    kerntype = 'Lindblad'

    cdef void prepare_arrays(self):
        ApproachElPh.prepare_arrays(self)
        ApproachLindbladBase.prepare_arrays(self)

        Vbbp, mtype = self.baths.Vbbp, self.baths.mtype
        tLbbp_shape = Vbbp.shape + (2,)
        self.tLbbp = np.zeros(tLbbp_shape, dtype=mtype)
        self.func_pauli_at_zero = np.zeros(self.si.nbaths, dtype=doublenp)

        self._tLbbp = self.tLbbp
        self._func_pauli_at_zero = self.func_pauli_at_zero

    cdef void clean_arrays(self):
        ApproachElPh.clean_arrays(self)
        ApproachLindbladBase.clean_arrays(self)
        self._tLbbp[::1] = 0.0
        self._func_pauli_at_zero[::1] = 0.0

    cpdef void generate_fct(self):
        ApproachLindbladBase.generate_fct(self)

        cdef complex_t [:, :, :] Vbbp = self._Vbbp
        cdef double_t [:] E = self._Ea

        cdef KernelHandler kh = self._kernel_handler.elph
        cdef long_t nbaths = kh.nbaths

        cdef long_t b, bp, bcharge, l, i
        cdef double_t Ebbp

        func_pauli = FuncPauliElPh(self._tlst_ph, self._dlst_ph,
                                   self.baths.bath_func, self.funcp.eps_elph)

        cdef complex_t [:, :, :, :] tLbbp = self._tLbbp

        cdef double_t [:] func_pauli_at_zero = self._func_pauli_at_zero
        for l in range(nbaths):
            func_pauli.eval(0., l)
            func_pauli_at_zero[l] = func_pauli.val

        for i in range(kh.ndm0):
            b = kh.all_bbp[i, 0]
            bp = kh.all_bbp[i, 1]

            if b == bp:
                # Diagonal elements
                for l in range(nbaths):
                    tLbbp[l, b, b, 0] = sqrt(0.5*func_pauli_at_zero[l])*Vbbp[l, b, b]
                    tLbbp[l, b, b, 1] = tLbbp[l, b, b, 0].conjugate()
            else:
                # Off-diagonal elements
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    func_pauli.eval(Ebbp, l)
                    tLbbp[l, b, bp, 0] = sqrt(0.5*func_pauli.val)*Vbbp[l, b, bp]
                    tLbbp[l, b, bp, 1] = sqrt(0.5*func_pauli.val)*Vbbp[l, bp, b].conjugate()

    cdef void generate_coupling_terms(self,
                long_t b, long_t bp, long_t bcharge,
                KernelHandler kh) nogil:

        ApproachLindbladBase.generate_coupling_terms(self, b, bp, bcharge, kh)

        cdef long_t bpp, a, ap
        cdef complex_t fct_aap, fct_bppbp, fct_bbpp

        cdef long_t i, j, l, q
        cdef long_t nbaths = kh.nbaths
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef complex_t [:, :, :, :] tLbbp = self._tLbbp

        cdef long_t acharge = bcharge

        cdef long_t acount = kh.statesdm_count[acharge]
        cdef long_t bcount = kh.statesdm_count[bcharge]

        # --------------------------------------------------
        for i in range(acount):
            for j in range(acount):
                a = statesdm[acharge, i]
                ap = statesdm[acharge, j]
                if not kh.is_included(a, ap, acharge):
                    continue
                fct_aap = 0
                for l in range(nbaths):
                    for q in range(2):
                        fct_aap += tLbbp[l, b, a, q]*tLbbp[l, bp, a, q].conjugate()
                kh.set_matrix_element(1j*fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for i in range(bcount):
            bpp = statesdm[bcharge, i]
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    for l in range(nbaths):
                        for q in range(2):
                            fct_bppbp += -0.5*tLbbp[l, a, b, q].conjugate()*tLbbp[l, a, bpp, q]
                kh.set_matrix_element(1j*fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    for l in range(nbaths):
                        for q in range(2):
                            fct_bbpp += -0.5*tLbbp[l, a, bpp, q].conjugate()*tLbbp[l, a, bp, q]
                kh.set_matrix_element(1j*fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------

    cpdef void generate_current(self):
        ApproachLindbladBase.generate_current(self)
