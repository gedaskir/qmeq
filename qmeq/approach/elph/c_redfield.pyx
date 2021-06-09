# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order Redfield kernel.
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
from .c_neumann1 cimport Approach1vN

from ..base.c_redfield cimport ApproachRedfield as ApproachRedfieldBase


# ---------------------------------------------------------------------------------------------------------
# Redfield approach
# ---------------------------------------------------------------------------------------------------------
cdef class ApproachRedfield(ApproachElPh):

    kerntype = 'Redfield'

    cdef void prepare_arrays(self):
        Approach1vN.prepare_arrays(self)

    cdef void clean_arrays(self):
        Approach1vN.clean_arrays(self)

    cpdef void generate_fct(self):
        Approach1vN.generate_fct(self)

    cdef void generate_coupling_terms(self,
                long_t b, long_t bp, long_t bcharge,
                KernelHandler kh) nogil:

        ApproachRedfieldBase.generate_coupling_terms(self, b, bp, bcharge, kh)

        cdef long_t a, ap, bpp, c, cp, \
                    ba, bpap, bppa, cb, cpbp, cbpp

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

        for i in range(acount):
            for j in range(acount):
                a = statesdm[acharge, i]
                ap = statesdm[acharge, j]
                if not kh.is_included(a, ap, acharge):
                    continue
                bpap = kh.elph.get_ind_dm0(bp, ap, acharge)
                ba = kh.elph.get_ind_dm0(b, a, acharge)
                if bpap == -1 or ba == -1:
                    continue
                fct_aap = 0
                for l in range(nbaths):
                    gamma_ba_bpap = 0.5*(Vbbp[l, b, a]*Vbbp[l, bp, ap].conjugate() +
                                         Vbbp[l, a, b].conjugate()*Vbbp[l, ap, bp])
                    fct_aap += gamma_ba_bpap*(w1fct[l, bpap, 0].conjugate() - w1fct[l, ba, 0])
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for i in range(bcount):
            bpp = statesdm[bcharge, i]
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    bppa = kh.elph.get_ind_dm0(bpp, a, acharge)
                    if bppa == -1:
                        continue
                    for l in range(nbaths):
                        gamma_ba_bppa = 0.5*(Vbbp[l, b, a]*Vbbp[l, bpp, a].conjugate() +
                                             Vbbp[l, a, b].conjugate()*Vbbp[l, a, bpp])
                        fct_bppbp += +gamma_ba_bppa*w1fct[l, bppa, 1].conjugate()
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    cbpp = kh.elph.get_ind_dm0(c, bpp, bcharge)
                    if cbpp == -1:
                        continue
                    for l in range(nbaths):
                        gamma_bc_bppc = 0.5*(Vbbp[l, b, c]*Vbbp[l, bpp, c].conjugate() +
                                             Vbbp[l, c, b].conjugate()*Vbbp[l, c, bpp])
                        fct_bppbp += +gamma_bc_bppc*w1fct[l, cbpp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    bppa = kh.elph.get_ind_dm0(bpp, a, acharge)
                    if bppa == -1:
                        continue
                    for l in range(nbaths):
                        gamma_abpp_abp = 0.5*(Vbbp[l, a, bpp].conjugate()*Vbbp[l, a, bp] +
                                              Vbbp[l, bpp, a]*Vbbp[l, bp, a].conjugate())
                        fct_bbpp += -gamma_abpp_abp*w1fct[l, bppa, 1]
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    cbpp = kh.elph.get_ind_dm0(c, bpp, bcharge)
                    if cbpp == -1:
                        continue
                    for l in range(nbaths):
                        gamma_cbpp_cbp = 0.5*(Vbbp[l, c, bpp].conjugate()*Vbbp[l, c, bp] +
                                              Vbbp[l, bpp, c]*Vbbp[l, bp, c].conjugate())
                        fct_bbpp += -gamma_cbpp_cbp*w1fct[l, cbpp, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for i in range(ccount):
            for j in range(ccount):
                c = statesdm[ccharge, i]
                cp = statesdm[ccharge, j]
                if not kh.is_included(c, cp, ccharge):
                    continue
                cpbp = kh.elph.get_ind_dm0(cp, bp, bcharge)
                cb = kh.elph.get_ind_dm0(c, b, bcharge)
                if cpbp == -1 or cb == -1:
                    continue
                fct_ccp = 0
                for l in range(nbaths):
                    gamma_bc_bpcp = 0.5*(Vbbp[l, b, c]*Vbbp[l, bp, cp].conjugate() +
                                         Vbbp[l, c, b].conjugate()*Vbbp[l, cp, bp])
                    fct_ccp += gamma_bc_bpcp*(w1fct[l, cpbp, 1] - w1fct[l, cb, 1].conjugate())
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    cpdef void generate_current(self):
        ApproachRedfieldBase.generate_current(self)
