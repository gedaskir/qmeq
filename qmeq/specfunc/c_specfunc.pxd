import numpy as np
cimport numpy as np

from ..wrappers.c_mytypes cimport bool_t
from ..wrappers.c_mytypes cimport int_t
from ..wrappers.c_mytypes cimport long_t
from ..wrappers.c_mytypes cimport double_t
from ..wrappers.c_mytypes cimport complex_t

cdef double_t pi
cdef double_t cabs(complex_t z) nogil
cdef double_t fermi_func(double_t) nogil

cdef int_t func_pauli(double_t, double_t, double_t, double_t, double_t,
                      int_t, double_t [:])

cdef int_t func_1vN(double_t, double_t, double_t, double_t, double_t,
                    int_t, int_t, complex_t [:])

cdef complex_t digamma(complex_t) nogil
cdef complex_t polygamma(complex_t, long_t) nogil
cdef double_t diff_fermi(double_t, double_t sign=*) nogil
cdef double_t phi(double_t, double_t, double_t, double_t sign=*) nogil
cdef double_t bose(double_t, double_t sign=*) nogil
cdef double_t diff_phi(double_t, double_t sign=*) nogil
cdef double_t diff2_phi(double_t, double_t sign=*) nogil
cdef double_t delta_phi(double_t, double_t, double_t, double_t, double_t sign=*) nogil
cdef complex_t integralD(double_t, double_t, double_t, double_t, double_t, double_t, 
            double_t, double_t, double_t, double_t, double_t[:,:], bint) nogil
cdef complex_t integralX(double_t, double_t, double_t, double_t, double_t, double_t, 
            double_t, double_t, double_t, double_t, double_t[:,:], bint) nogil
cdef double_t[:,:] BW_Ozaki(double_t BW)
cdef void diag_matrix_multiply(double_t[:], double_t[:,:])