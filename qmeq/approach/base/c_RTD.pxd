import numpy as np
cimport numpy as np

from ..c_aprclass cimport Approach
from ..c_kernel_handler cimport KernelHandlerRTD

from ...wrappers.c_mytypes cimport bool_t
from ...wrappers.c_mytypes cimport int_t
from ...wrappers.c_mytypes cimport long_t
from ...wrappers.c_mytypes cimport double_t
from ...wrappers.c_mytypes cimport complex_t


cdef class ApproachRTD(Approach):

    cdef double_t[:,:] Ozaki_poles_and_residues
    cdef complex_t [:,:] _tleads_array
    cdef double_t BW_Ozaki_expansion
    cdef bool_t OffDiagCorrections
    cdef bool_t ImGamma
    cdef bool_t printed_warning_ImGamma
    cdef long_t nbr_Wdd2_copies

    cpdef long_t get_kern_size(self)
    cdef void add_off_diag_corrections(self, KernelHandlerRTD)
    cdef void generate_row_1st_order_kernel(self, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_row_1st_energy_kernel(self, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_row_2nd_energy_kernel(self, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_matrix_element_2nd_order(self, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_col_nondiag_kern_1st_order_dn(self, long_t, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_col_nondiag_kern_1st_order_nd(self, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_row_inverse_Liouvillian(self, long_t, long_t, long_t, KernelHandlerRTD) nogil
    cdef void generate_LN(self)
    cdef void set_Ozaki_params(self)