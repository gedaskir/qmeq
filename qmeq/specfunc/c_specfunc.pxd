import numpy as np
cimport numpy as np

from ..wrappers.c_mytypes cimport bool_t
from ..wrappers.c_mytypes cimport int_t
from ..wrappers.c_mytypes cimport long_t
from ..wrappers.c_mytypes cimport double_t
from ..wrappers.c_mytypes cimport complex_t

cdef double_t fermi_func(double_t)

cdef int_t func_pauli(double_t, double_t, double_t, double_t, double_t,
                      int_t, double_t [:])

cdef int_t func_1vN(double_t, double_t, double_t, double_t, double_t,
                    int_t, int_t, complex_t [:])
