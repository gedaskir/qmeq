import numpy as np
cimport numpy as np

from ..approach.c_aprclass cimport bool_t
from ..approach.c_aprclass cimport int_t
from ..approach.c_aprclass cimport long_t
from ..approach.c_aprclass cimport double_t
from ..approach.c_aprclass cimport complex_t

cdef double_t fermi_func(double_t)

cdef int_t func_pauli(double_t, double_t, double_t, double_t, double_t,
                      int_t, double_t [:])

cdef int_t func_1vN(double_t, double_t, double_t, double_t, double_t,
                    int_t, int_t, complex_t [:])
