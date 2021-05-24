from ..wrappers.c_mytypes cimport bool_t
from ..wrappers.c_mytypes cimport int_t
from ..wrappers.c_mytypes cimport long_t
from ..wrappers.c_mytypes cimport double_t
from ..wrappers.c_mytypes cimport complex_t


cdef class KernelHandler:

    cdef long_t bbp, bbpi
    cdef bool_t bbp_bool, bbpi_bool

    cdef long_t aap, aapi
    cdef int_t aap_sgn

    cdef long_t nmany
    cdef long_t ndm0
    cdef long_t ndm0r
    cdef long_t ndm1
    cdef long_t npauli
    cdef long_t nleads
    cdef long_t nbaths
    cdef long_t ncharge

    cdef bool_t no_coherences
    cdef bool_t no_conjugates
    cdef long_t nelements

    cdef long_t [:] lenlst
    cdef long_t [:] dictdm
    cdef long_t [:] shiftlst0
    cdef long_t [:] shiftlst1
    cdef long_t [:] mapdm0
    cdef bool_t [:] booldm0
    cdef bool_t [:] conjdm0

    cdef double_t [:, :] kern
    cdef double_t [:] phi0

    cdef long_t [:, :] statesdm
    cdef long_t [:] statesdm_count

    cdef long_t [:, :] all_bbp
    cdef long_t [:, :] all_ba

    cdef KernelHandler elph

    cpdef void set_kern(self, kern)

    cpdef void set_phi0(self, phi0)

    cdef void set_statesdm(self, si)

    cdef void set_all_bbp(self)

    cdef void set_all_ba(self)

    cdef bool_t is_included(self, long_t b, long_t bp, long_t bcharge) nogil

    cdef bool_t is_unique(self, long_t b, long_t bp, long_t bcharge) nogil

    cdef void set_energy(self, double_t energy, long_t b, long_t bp, long_t bcharge) nogil

    cdef void set_matrix_element(self,
                complex_t fct,
                long_t b, long_t bp, long_t bcharge,
                long_t a, long_t ap, long_t acharge) nogil

    cdef void set_matrix_element_pauli(self,
                double_t fctm, double_t fctp,
                long_t bb, long_t aa) nogil

    cdef complex_t get_phi0_element(self, long_t b, long_t bp, long_t bcharge) nogil

    cdef long_t get_ind_dm0(self, long_t b, long_t bp, long_t bcharge) nogil

    cdef bool_t get_ind_dm0_conj(self, long_t b, long_t bp, long_t bcharge) nogil

    cdef bool_t get_ind_dm0_bool(self, long_t b, long_t bp, long_t bcharge) nogil

    cdef long_t get_ind_dm1(self, long_t b, long_t a, long_t acharge) nogil


cdef class KernelHandlerMatrixFree(KernelHandler):

    cdef double_t [:] dphi0_dt

    cpdef void set_dphi0_dt(self, double_t [:] dphi0_dt)

    cdef double_t get_phi0_norm(self)


cdef class KernelHandlerRTD(KernelHandler):

    cdef long_t nsingle

    cdef double_t[:,:,:] Wdd
    cdef double_t[:,:,:,:] Wdd2
    cdef double_t[:,:,:] WE1
    cdef double_t[:,:,:] WE2
    cdef double_t[:,:] ReWnd
    cdef double_t[:,:] ImWnd
    cdef double_t[:,:,:] ReWdn
    cdef double_t[:,:,:] ImWdn
    cdef double_t[:] Lnn
    cdef double_t[:,:,:] LE
    cdef double_t[:,:,:] LN

    cdef void add_matrix_element(self, double_t, long_t, long_t, long_t,
        long_t, long_t, long_t, long_t, int_t) nogil

    cdef void set_matrix_element_dd(self, long_t, double_t, double_t, long_t, long_t,
        long_t) nogil

    cdef void add_element_2nd_order(self, long_t, long_t, double_t, long_t, long_t, long_t,
                             long_t, long_t, long_t) nogil

    cdef void set_matrix_list(self)