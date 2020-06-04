import numpy as np
cimport numpy as np

from ..wrappers.c_lapack cimport LapackSolver

from ..wrappers.c_mytypes cimport bool_t
from ..wrappers.c_mytypes cimport int_t
from ..wrappers.c_mytypes cimport long_t
from ..wrappers.c_mytypes cimport double_t
from ..wrappers.c_mytypes cimport complex_t

from .c_kernel_handler cimport KernelHandler
from .c_kernel_handler cimport KernelHandlerMatrixFree


cdef class Approach:
    cdef dict __dict__

    cdef KernelHandler _kernel_handler
    cdef LapackSolver _solver

    cdef double_t [:] _rez_real
    cdef complex_t [:] _rez_complex

    cdef double_t [:] _tlst
    cdef double_t [:] _mulst
    cdef double_t [:, :] _dlst

    cdef double_t [:] _Ea
    cdef complex_t [:, :, :] _Tba
    cdef complex_t [:, :, :] _tLba
    cdef double_t [:, :, :] _paulifct
    cdef complex_t [:, :, :] _phi1fct
    cdef complex_t [:, :, :] _phi1fct_energy

    cdef double_t [:, :] _kern
    cdef double_t [:] _bvec
    cdef double_t [:] _norm_vec

    cdef double_t [:] _phi0
    cdef double_t [:] _dphi0_dt
    cdef double_t [:] _current
    cdef double_t [:] _energy_current
    cdef double_t [:] _heat_current

    cdef bool_t _success
    cdef bool_t _make_kern_copy
    cdef bool_t _mfreeq
    cdef bool_t _symq
    cdef long_t _norm_row

    cpdef void generate_fct(self)
    cpdef void generate_kern(self)
    cpdef void generate_current(self)
    cpdef generate_vec(self, phi0)

    cdef void generate_coupling_terms(self,
                long_t b, long_t bp, long_t bcharge,
                KernelHandler kh) nogil

    cpdef void prepare_kern(self)

    cdef void prepare_arrays(self)

    cdef void clean_arrays(self)

    cpdef void solve_kern(self)


cdef class ApproachElPh(Approach):

    cdef double_t [:] _func_pauli_at_zero
    cdef complex_t [:, :] _func_1vN_elph_at_zero

    cdef double_t [:] _tlst_ph
    cdef double_t [:, :] _dlst_ph

    cdef complex_t [:, :, :] _Vbbp
    cdef complex_t [:, :, :, :] _tLbbp
    cdef double_t [:, :] _paulifct_elph
    cdef complex_t [:, :, :] _w1fct
