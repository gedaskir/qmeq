import numpy as np
cimport numpy as np

from ..wrappers.c_mytypes cimport bool_t
from ..wrappers.c_mytypes cimport int_t
from ..wrappers.c_mytypes cimport long_t
from ..wrappers.c_mytypes cimport double_t
from ..wrappers.c_mytypes cimport complex_t


cdef class Func:
    cpdef double_t eval(self, double_t x)


cdef class FuncBose:
    cdef double_t eval(self, double_t x)


cdef class FuncPauliElPh:
    cdef Func dos
    cdef FuncBose bose
    cdef double_t T, eps
    cdef double_t [:] tlst
    cdef double_t [:, :] dlst
    cdef bath_func
    cdef bool_t bath_func_q
    cdef public double_t val
    cpdef void eval(self, double_t Ebbp, int_t l)


cdef class Func1vNElPh:
    cdef Func dos
    cdef FuncBose bose
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
