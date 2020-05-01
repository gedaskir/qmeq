import numpy as np
cimport numpy as np

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t


cdef class Approach:
    cdef dict __dict__

    cpdef generate_fct(self)
    cpdef generate_kern(self)
    cpdef generate_current(self)
    cpdef generate_vec(self, phi0)


cdef class ApproachElPh(Approach):
    pass
