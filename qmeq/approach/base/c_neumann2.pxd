import numpy as np
cimport numpy as np

from ...wrappers.c_mytypes cimport bool_t
from ...wrappers.c_mytypes cimport int_t
from ...wrappers.c_mytypes cimport long_t
from ...wrappers.c_mytypes cimport double_t
from ...wrappers.c_mytypes cimport complex_t

from ..c_kernel_handler cimport KernelHandler


cdef class TermsCalculator2vN:

    cdef appr

    cdef KernelHandler kernel_handler

    cdef long_t kpnt_left

    cdef double_t [:] Ek_grid
    cdef double_t [:] Ek_grid_ext

    cdef double_t [:] Ea
    cdef complex_t [:, :, :] Tba

    cdef double_t [:, :] fkp
    cdef double_t [:, :] fkm
    cdef complex_t [:, :] hfkp
    cdef complex_t [:, :] hfkm

    cdef complex_t [:, :, :, :] phi1k_delta
    cdef complex_t [:, :, :, :] phi1k_delta_old
    cdef complex_t [:, :, :, :] kern1k_inv
    cdef complex_t [:, :, : ,:] hphi1k_delta


    cdef void retrieve_approach_variables(self)

    cdef void phi1k_local(self, long_t k, long_t l, KernelHandler kh)

    cdef void phi1k_iterate(self, long_t k, long_t l, KernelHandler kh)

    cdef complex_t func_2vN(self, double_t Ek, long_t l, int_t eta, complex_t [:, :] hfk)

    cdef void get_at_k1(self, double_t Ek, long_t l, long_t cb, bint conj,
                              complex_t fct, int_t eta, complex_t [:] term)
