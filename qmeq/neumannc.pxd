import numpy as np
cimport numpy as np

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t

cdef double_t fermi_func(double_t)
cdef double_t func_pauli(double_t, double_t, double_t)
cdef complex_t func_1vN(double_t, double_t, double_t, double_t, int_t, int_t)
