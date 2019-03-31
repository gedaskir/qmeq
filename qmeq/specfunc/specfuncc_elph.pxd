import numpy as np
cimport numpy as np

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

cdef class Func:
    cpdef double_t eval(self, double_t x)

cdef class Func_bose:
    cdef double_t eval(self, double_t x)

cdef class Func_pauli_elph:
    cdef Func dos
    cdef Func_bose bose
    cdef double_t T, eps
    cdef double_t [:] tlst
    cdef double_t [:, :] dlst
    cdef bath_func
    cdef bool_t bath_func_q
    cdef public double_t val
    cpdef void eval(self, double_t Ebbp, int_t l)

cdef class Func_1vN_elph:
    cdef Func dos
    cdef Func_bose bose
    cdef double_t T, eps
    cdef double_t [:] tlst
    cdef double_t [:, :] dlst
    cdef int_t itype
    cdef long_t limit
    cdef bath_func
    cdef bool_t bath_func_q
    cdef public complex_t val0, val1
    cpdef double_t iplus(self, double_t x)
    cpdef double_t iminus(self, double_t x)
    cpdef void eval(self, double_t Ebbp, int_t l)
