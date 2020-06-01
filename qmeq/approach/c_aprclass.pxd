import numpy as np
cimport numpy as np

from ..wrappers.c_lapack cimport LapackSolver

ctypedef np.uint8_t bool_t
ctypedef np.int_t int_t
ctypedef np.int64_t long_t
ctypedef np.float64_t double_t
ctypedef np.complex128_t complex_t


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

    cpdef void set_kern(self, double_t [:, :] kern)

    cpdef void set_phi0(self, double_t [:] phi0)

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
