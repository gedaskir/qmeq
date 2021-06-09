# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order 1vN kernel.
   For docstrings see documentation of module neumann1."""

# Python imports

import numpy as np
import itertools

from ...wrappers.mytypes import doublenp
from ...wrappers.mytypes import complexnp

# Cython imports

cimport numpy as np
cimport cython

from ...specfunc.c_specfunc_elph cimport Func1vNElPh

from ..c_aprclass cimport ApproachElPh
from ..c_kernel_handler cimport KernelHandler

from ..base.c_neumann1 cimport Approach1vN as Approach1vNBase

# ---------------------------------------------------------------------------------------------------------
# 1 von Neumann approach
# ---------------------------------------------------------------------------------------------------------
cdef class Approach1vN(ApproachElPh):

    kerntype = '1vN'

    cdef void prepare_arrays(self):
        ApproachElPh.prepare_arrays(self)
        Approach1vNBase.prepare_arrays(self)

        nbaths, ndm0 = self.si_elph.nbaths, self.si_elph.ndm0
        self.w1fct = np.zeros((nbaths, ndm0, 2), dtype=complexnp)
        self.func_1vN_elph_at_zero = np.zeros((nbaths, 2), dtype=complexnp)

        self._w1fct = self.w1fct
        self._func_1vN_elph_at_zero = self.func_1vN_elph_at_zero

    cdef void clean_arrays(self):
        ApproachElPh.clean_arrays(self)
        Approach1vNBase.clean_arrays(self)
        self._w1fct[::1] = 0.0
        self._func_1vN_elph_at_zero[::1] = 0.0

    cpdef void generate_fct(self):
        Approach1vNBase.generate_fct(self)

        cdef double_t [:] E = self._Ea

        cdef KernelHandler kh = self._kernel_handler.elph
        cdef long_t nbaths = kh.nbaths

        cdef long_t b, bp, bbp, bb, bcharge, l, i
        cdef double_t Ebbp

        func_1vN_elph = Func1vNElPh(self._tlst_ph, self._dlst_ph,
                                    self.funcp.itype_ph, self.funcp.dqawc_limit,
                                    self.baths.bath_func,
                                    self.funcp.eps_elph)

        cdef complex_t [:, :, :] w1fct = self._w1fct

        cdef complex_t [:, :] func_1vN_elph_at_zero = self._func_1vN_elph_at_zero
        for l in range(nbaths):
            func_1vN_elph.eval(0., l)
            func_1vN_elph_at_zero[l, 0] = func_1vN_elph.val0 - 0.5j*func_1vN_elph.val0.imag
            func_1vN_elph_at_zero[l, 1] = func_1vN_elph.val1 - 0.5j*func_1vN_elph.val1.imag

        for i in range(kh.ndm0):
            b = kh.all_bbp[i, 0]
            bp = kh.all_bbp[i, 1]
            bcharge = kh.all_bbp[i, 2]
            bbp = kh.get_ind_dm0(b, bp, bcharge)

            if b == bp:
                for l in range(nbaths):
                    w1fct[l, bbp, 0] = func_1vN_elph_at_zero[l, 0]
                    w1fct[l, bbp, 1] = func_1vN_elph_at_zero[l, 1]
            else:
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    func_1vN_elph.eval(Ebbp, l)
                    w1fct[l, bbp, 0] = func_1vN_elph.val0
                    w1fct[l, bbp, 1] = func_1vN_elph.val1

    cdef void generate_coupling_terms(self,
                long_t b, long_t bp, long_t bcharge,
                KernelHandler kh) nogil:

        Approach1vNBase.generate_coupling_terms(self, b, bp, bcharge, kh)

        cdef long_t a, ap, bpp, c, cp, \
                    ba, bap, bpa, cb, cbp, cpb

        cdef complex_t fct_aap, fct_bppbp, fct_bbpp, fct_ccp, \
                       gamma_ba_bpap, gamma_ba_bppa, gamma_bc_bppc, \
                       gamma_abpp_abp, gamma_cbpp_cbp, gamma_bc_bpcp

        cdef long_t i, j, l
        cdef long_t nbaths = kh.nbaths
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef complex_t [:, :, :] Vbbp = self._Vbbp
        cdef complex_t [:, :, :] w1fct = self._w1fct

        cdef long_t acharge = bcharge
        cdef long_t ccharge = bcharge

        cdef long_t bcount = kh.statesdm_count[bcharge]
        cdef long_t acount = bcount
        cdef long_t ccount = bcount

        # --------------------------------------------------
        for i in range(acount):
            for j in range(acount):
                a = statesdm[acharge, i]
                ap = statesdm[acharge, j]
                if not kh.is_included(a, ap, acharge):
                    continue
                bpa = kh.elph.get_ind_dm0(bp, a, acharge)
                bap = kh.elph.get_ind_dm0(b, ap, acharge)
                if bpa == -1 or bap == -1:
                    continue
                fct_aap = 0
                for l in range(nbaths):
                    gamma_ba_bpap = 0.5*(Vbbp[l, b, a]*Vbbp[l, bp, ap].conjugate() +
                                         Vbbp[l, a, b].conjugate()*Vbbp[l, ap, bp])
                    fct_aap += gamma_ba_bpap*(w1fct[l, bpa, 0].conjugate() - w1fct[l, bap, 0])
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for i in range(bcount):
            bpp = statesdm[bcharge, i]
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    bpa = kh.elph.get_ind_dm0(bp, a, acharge)
                    if bpa == -1:
                        continue
                    for l in range(nbaths):
                        gamma_ba_bppa = 0.5*(Vbbp[l, b, a]*Vbbp[l, bpp, a].conjugate() +
                                             Vbbp[l, a, b].conjugate()*Vbbp[l, a, bpp])
                        fct_bppbp += gamma_ba_bppa*w1fct[l, bpa, 1].conjugate()
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    cbp = kh.elph.get_ind_dm0(c, bp, bcharge)
                    if cbp == -1:
                        continue
                    for l in range(nbaths):
                        gamma_bc_bppc = 0.5*(Vbbp[l, b, c]*Vbbp[l, bpp, c].conjugate() +
                                             Vbbp[l, c, b].conjugate()*Vbbp[l, c, bpp])
                        fct_bppbp += gamma_bc_bppc*w1fct[l, cbp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    ba = kh.elph.get_ind_dm0(b, a, acharge)
                    if ba == -1:
                        continue
                    for l in range(nbaths):
                        gamma_abpp_abp = 0.5*(Vbbp[l, a, bpp].conjugate()*Vbbp[l, a, bp] +
                                              Vbbp[l, bpp, a]*Vbbp[l, bp, a].conjugate())
                        fct_bbpp += -gamma_abpp_abp*w1fct[l, ba, 1]
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    cb = kh.elph.get_ind_dm0(c, b, bcharge)
                    if cb == -1:
                        continue
                    for l in range(nbaths):
                        gamma_cbpp_cbp = 0.5*(Vbbp[l, c, bpp].conjugate()*Vbbp[l, c, bp] +
                                              Vbbp[l, bpp, c]*Vbbp[l, bp, c].conjugate())
                        fct_bbpp += -gamma_cbpp_cbp*w1fct[l, cb, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for i in range(ccount):
            for j in range(ccount):
                c = statesdm[ccharge, i]
                cp = statesdm[ccharge, j]
                if not kh.is_included(c, cp, ccharge):
                    continue
                cbp = kh.elph.get_ind_dm0(c, bp, bcharge)
                cpb = kh.elph.get_ind_dm0(cp, b, bcharge)
                if cbp == -1 or cpb == -1:
                    continue
                fct_ccp = 0
                for l in range(nbaths):
                    gamma_bc_bpcp = 0.5*(Vbbp[l, b, c]*Vbbp[l, bp, cp].conjugate() +
                                         Vbbp[l, c, b].conjugate()*Vbbp[l, cp, bp])
                    fct_ccp += gamma_bc_bpcp*(w1fct[l, cbp, 1] - w1fct[l, cpb, 1].conjugate())
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    cpdef void generate_current(self):
        Approach1vNBase.generate_current(self)
