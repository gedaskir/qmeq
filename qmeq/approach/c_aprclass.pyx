# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from .aprclass import Approach as ApproachPy
from .aprclass import ApproachElPh as ApproachElPhPy

from ..mytypes import longnp
from ..mytypes import doublenp
from ..mytypes import complexnp

from ..indexing import StateIndexingDMc

# Cython imports

cimport numpy as np
cimport cython

from ..wrappers.c_lapack cimport LapackSolverDGESV
from ..wrappers.c_lapack cimport LapackSolverDGELSD


cdef class Approach:

    kerntype = 'not defined'
    dtype = doublenp
    indexing_class_name = 'StateIndexingDM'
    no_coherences = False

    #region Properties

    @property
    def solmethod(self):
        return self.funcp.solmethod
    @solmethod.setter
    def solmethod(self, value):
        ApproachPy.solmethod.fset(self, value)

    @property
    def mfreeq(self):
        return self.funcp.mfreeq
    @mfreeq.setter
    def mfreeq(self, value):
        ApproachPy.mfreeq.fset(self, value)
        self._mfreeq = value

    @property
    def symq(self):
        return self.funcp.symq
    @symq.setter
    def symq(self, value):
        ApproachPy.symq.fset(self, value)
        self._symq = value

    @property
    def norm_row(self):
        return self.funcp.norm_row
    @norm_row.setter
    def norm_row(self, value):
        ApproachPy.norm_row.fset(self, value)
        self._norm_row = value

    @property
    def itype(self):
        return self.funcp.itype
    @itype.setter
    def itype(self, value):
        ApproachPy.itype.fset(self, value)

    @property
    def success(self):
        return self._success
    @success.setter
    def success(self, value):
        self._success = value

    @property
    def make_kern_copy(self):
        return self._make_kern_copy
    @make_kern_copy.setter
    def make_kern_copy(self, value):
        self._make_kern_copy = value

    #endregion Properties

    def __init__(self, builder):
        ApproachPy.__init__(self, builder)

    cpdef void generate_fct(self):
        pass

    cpdef void generate_kern(self):
        cdef double_t [:] E = self._Ea
        cdef KernelHandler kh = self._kernel_handler

        cdef long_t i, b, bp, bcharge

        for i in range(kh.nelements):
            b = kh.all_bbp[i, 0]
            bp = kh.all_bbp[i, 1]
            bcharge = kh.all_bbp[i, 2]
            kh.set_energy(E[b]-E[bp], b, bp, bcharge)
            self.generate_coupling_terms(b, bp, bcharge, kh)

    cdef void generate_coupling_terms(self,
            long_t b, long_t bp, long_t bcharge,
            KernelHandler kh) nogil:
        pass

    cpdef void generate_current(self):
        pass

    cpdef generate_vec(self, phi0):
        cdef long_t norm_row = self._norm_row

        cdef KernelHandlerMatrixFree kh = self._kernel_handler
        kh.set_phi0(phi0)
        cdef double_t norm = kh.get_phi0_norm()

        self._dphi0_dt[::1] = 0.0

        # Here dphi0_dt and norm will be implicitly calculated by using KernelHandlerMatrixFree
        self.generate_kern()

        self._dphi0_dt[norm_row] = norm-1

        return self._dphi0_dt

    def get_kern_size(self):
        return ApproachPy.get_kern_size(self)

    def restart(self):
        ApproachPy.restart(self)

    def set_phi0_init(self):
        return ApproachPy.set_phi0_init(self)

    cpdef void prepare_kern(self):
        if self.is_prepared and not self.si.states_changed:
            self.clean_arrays()
            return

        self.prepare_kernel_handler()
        self.prepare_arrays()
        self.prepare_solver()

        self._mfreeq = self.funcp.mfreeq
        self._norm_row = self.funcp.norm_row
        self._symq = self.funcp.symq

        self.si.states_changed = False
        self.is_prepared = True

    cdef void clean_arrays(self):

        if not self._mfreeq:
            self._kern[::1] = 0.0
            self._bvec[::1] = 0.0

        self._phi0[::1] = 0.0
        self._current[::1] = 0.0
        self._energy_current[::1] = 0.0
        self._heat_current[::1] = 0.0

    cdef void prepare_arrays(self):
        ApproachPy.prepare_arrays(self)

        self._kern = self.kern
        self._bvec = self.bvec
        self._norm_vec = self.norm_vec

        self._phi0 = self.phi0
        self._dphi0_dt = self.dphi0_dt
        self._current = self.current
        self._energy_current = self.energy_current
        self._heat_current = self.heat_current

        self._tlst = self.leads.tlst
        self._mulst = self.leads.mulst
        self._dlst = self.leads.dlst

        self._Ea = self.qd.Ea
        self._Tba = self.leads.Tba

    def prepare_kernel_handler(self):
        if self.funcp.mfreeq:
            self.kernel_handler = KernelHandlerMatrixFree(self.si, self.no_coherences)
        else:
            self.kernel_handler = KernelHandler(self.si, self.no_coherences)

        self._kernel_handler = self.kernel_handler

    def prepare_solver(self):
        ApproachPy.prepare_solver(self)
        if self.funcp.mfreeq:
            return

        solmethod = self.funcp.solmethod
        if solmethod == 'lsqr':
            self._solver = LapackSolverDGELSD(self.kern, self.bvec, self._make_kern_copy)
        else:
        #else if solmethod == 'solve':
            self._solver = LapackSolverDGESV(self.kern, self.bvec, self._make_kern_copy)

    cpdef void solve_kern(self):
        """Finds the stationary state using least squares or using LU decomposition."""

        # Replace one equation by the normalisation condition
        cdef double_t [:, :] kern = self._kern
        cdef double_t [:] bvec = self._bvec

        cdef long_t norm_row = self._norm_row if self._symq else kern.shape[0] - 1
        kern[norm_row, :] = self._norm_vec
        bvec[norm_row] = 1

        # Try to solve the master equation
        # Solver modifies the arrays kern and bvec
        # The solution is stored in bvec
        self._solver.solve()

        # Copy over the solution
        cdef long_t phi0_size = self._phi0.shape[0]
        self._phi0[:] = self._bvec[0:phi0_size]

        if self._solver.info == 0:
            self._success = True
        else:
            self.funcp.print_error("Singular matrix.")
            self._phi0[::1] = 0.0
            self._success = False

    def solve_matrix_free(self):
        ApproachPy.solve_matrix_free(self)

    def generate_norm_vec(self):
        kh = self._kernel_handler

        cdef double_t [:] norm_vec = self.norm_vec
        cdef int_t bcharge, bcount, b, bb, i
        for bcharge in range(kh.ncharge):
            bcount = kh.statesdm_count[bcharge]
            for i in range(bcount):
                b = kh.statesdm[bcharge, i]
                bb = kh.get_ind_dm0(b, b, bcharge)
                norm_vec[bb] += 1

    def rotate(self):
        ApproachPy.rotate(self)

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        ApproachPy.solve(self, qdq, rotateq, masterq, currentq, args, kwargs)


cdef class ApproachElPh(Approach):

    def __init__(self, builder):
        ApproachElPhPy.__init__(self, builder)

    def prepare_kernel_handler(self):
        Approach.prepare_kernel_handler(self)
        if self.funcp.mfreeq:
            self.kernel_handler_elph = KernelHandlerMatrixFree(self.si_elph)
        else:
            self.kernel_handler_elph = KernelHandler(self.si_elph)

        self._kernel_handler.elph = self.kernel_handler_elph

    cdef void prepare_arrays(self):
        self._tlst_ph = self.baths.tlst_ph
        self._dlst_ph = self.baths.dlst_ph
        self._Vbbp = self.baths.Vbbp

    cdef void clean_arrays(self):
        pass

    def restart(self):
        ApproachElPhPy.restart(self)

    def rotate(self):
        ApproachElPhPy.rotate(self)


cdef class KernelHandler:

    def __init__(self, si, no_coherences=False):
        self.nmany = si.nmany
        self.ndm0 = si.ndm0
        self.ndm0r = si.ndm0r
        self.ndm1 = si.ndm1
        self.npauli = si.npauli
        self.nleads = si.nleads
        self.nbaths = si.nbaths
        self.ncharge = si.ncharge

        self.lenlst = si.lenlst
        self.dictdm = si.dictdm
        self.shiftlst0 = si.shiftlst0
        self.shiftlst1 = si.shiftlst1
        self.mapdm0 = si.mapdm0
        self.booldm0 = si.booldm0
        self.conjdm0 = si.conjdm0

        self.kern = None
        self.phi0 = None
        self.statesdm = None
        self.statesdm_count = None
        self.all_bbp = None
        self.all_ba = None

        self.no_coherences = no_coherences
        self.no_conjugates = not isinstance(si, StateIndexingDMc)
        self.nelements = self.npauli if self.no_coherences else self.ndm0

        self.set_statesdm(si)
        self.set_all_bbp()
        self.set_all_ba()

    cdef void set_statesdm(self, si):
        cdef int_t max_states = 0
        cdef int_t i, j
        statesdm = si.statesdm

        for states in statesdm:
            max_states = max(max_states, len(states))

        statesdm_len = len(statesdm)
        self.statesdm_count = np.zeros(statesdm_len, dtype=longnp)
        self.statesdm = np.zeros((statesdm_len, max_states), dtype=longnp)

        for i in range(statesdm_len):
            self.statesdm_count[i] = len(statesdm[i])
            for j in range(self.statesdm_count[i]):
                self.statesdm[i, j] = statesdm[i][j]

    cdef void set_all_bbp(self):
        self.all_bbp = np.zeros((self.nelements, 3), dtype=longnp)
        cdef long_t bcharge, bcount, i, j, j_lower, b, bp, ind
        ind = 0
        for bcharge in range(self.ncharge):
            bcount = self.statesdm_count[bcharge]
            for i in range(bcount):
                j_lower = i if self.no_conjugates else 0
                for j in range(j_lower, bcount):
                    if self.no_coherences and i != j:
                        continue
                    b = self.statesdm[bcharge, i]
                    bp = self.statesdm[bcharge, j]
                    if self.is_unique(b, bp, bcharge):
                        self.all_bbp[ind, 0] = b
                        self.all_bbp[ind, 1] = bp
                        self.all_bbp[ind, 2] = bcharge
                        ind += 1

    cdef void set_all_ba(self):
        self.all_ba = np.zeros((self.ndm1, 3), dtype=longnp)
        cdef long_t bcharge, acharge, i, j, b, a, ind

        ind = 0
        for bcharge in range(1, self.ncharge):
            acharge = bcharge-1
            for i in range(self.statesdm_count[bcharge]):
                for j in range(self.statesdm_count[acharge]):
                    b = self.statesdm[bcharge, i]
                    a = self.statesdm[acharge, j]
                    self.all_ba[ind, 0] = b
                    self.all_ba[ind, 1] = a
                    self.all_ba[ind, 2] = acharge
                    ind += 1

    cpdef void set_kern(self, double_t [:, :] kern):
        self.kern = kern

    cpdef void set_phi0(self, double_t [:] phi0):
        self.phi0 = phi0

    cdef bool_t is_included(self, long_t b, long_t bp, long_t bcharge) nogil:
        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return False

        return True

    cdef bool_t is_unique(self, long_t b, long_t bp, long_t bcharge) nogil:
        cdef bool_t bbp_bool = self.get_ind_dm0_bool(b, bp, bcharge)
        return bbp_bool

    cdef void set_energy(self,
                double_t energy,
                long_t b, long_t bp, long_t bcharge) nogil:

        if b == bp:
            return

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli

        self.kern[bbp, bbpi] = self.kern[bbp, bbpi] + energy
        self.kern[bbpi, bbp] = self.kern[bbpi, bbp] - energy

    cdef void set_matrix_element(self,
                complex_t fct,
                long_t b, long_t bp, long_t bcharge,
                long_t a, long_t ap, long_t acharge) nogil:

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli
        cdef bool_t bbpi_bool = True if bbpi >= self.ndm0 else False

        cdef long_t aap = self.get_ind_dm0(a, ap, acharge)
        cdef long_t aapi = self.ndm0 + aap - self.npauli
        cdef int_t aap_sgn = +1 if self.get_ind_dm0_conj(a, ap, acharge) else -1

        cdef double_t fct_imag = fct.imag
        cdef double_t fct_real = fct.real

        self.kern[bbp, aap] = self.kern[bbp, aap] + fct_imag
        if aapi >= self.ndm0:
            self.kern[bbp, aapi] = self.kern[bbp, aapi] + fct_real*aap_sgn
            if bbpi_bool:
                self.kern[bbpi, aapi] = self.kern[bbpi, aapi] + fct_imag*aap_sgn
        if bbpi_bool:
            self.kern[bbpi, aap] = self.kern[bbpi, aap] - fct_real

    cdef void set_matrix_element_pauli(self,
                double_t fctm, double_t fctp,
                long_t bb, long_t aa) nogil:

        self.kern[bb, bb] += fctm
        self.kern[bb, aa] += fctp

    cdef complex_t get_phi0_element(self, long_t b, long_t bp, long_t bcharge) nogil:

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return 0.0

        cdef long_t bbpi = self.ndm0 + bbp - self.npauli
        cdef bool_t bbpi_bool = True if bbpi >= self.ndm0 else False

        cdef double_t phi0_real = self.phi0[bbp]
        cdef double_t phi0_imag = 0
        if bbpi_bool:
            phi0_imag = self.phi0[bbpi] if self.get_ind_dm0_conj(b, bp, bcharge) else -self.phi0[bbpi]

        return phi0_real + 1j*phi0_imag

    cdef long_t get_ind_dm0(self, long_t b, long_t bp, long_t bcharge) nogil:
        return self.mapdm0[self.lenlst[bcharge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[bcharge]]

    cdef bool_t get_ind_dm0_conj(self, long_t b, long_t bp, long_t bcharge) nogil:
        return self.conjdm0[self.lenlst[bcharge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[bcharge]]

    cdef bool_t get_ind_dm0_bool(self, long_t b, long_t bp, long_t bcharge) nogil:
        return self.booldm0[self.lenlst[bcharge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[bcharge]]

    cdef long_t get_ind_dm1(self, long_t b, long_t a, long_t acharge) nogil:
        return self.lenlst[acharge]*self.dictdm[b] + self.dictdm[a] + self.shiftlst1[acharge]


cdef class KernelHandlerMatrixFree(KernelHandler):

    def __init__(self, si, no_coherences=False):
        KernelHandler.__init__(self, si, no_coherences)
        self.dphi0_dt = None

    cpdef void set_dphi0_dt(self, double_t [:] dphi0_dt):
        self.dphi0_dt = dphi0_dt

    cdef void set_energy(self,
                double_t energy,
                long_t b, long_t bp, long_t bcharge) nogil:

        if b == bp:
            return

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli

        cdef complex_t phi0bbp = self.get_phi0_element(b, bp, bcharge)
        cdef complex_t dphi0_dt_bbp = -1j*energy*phi0bbp

        self.dphi0_dt[bbp] = self.dphi0_dt[bbp] + dphi0_dt_bbp.real
        self.dphi0_dt[bbpi] = self.dphi0_dt[bbpi] - dphi0_dt_bbp.imag

    cdef void set_matrix_element(self,
                complex_t fct,
                long_t b, long_t bp, long_t bcharge,
                long_t a, long_t ap, long_t acharge) nogil:

        cdef long_t bbp = self.get_ind_dm0(b, bp, bcharge)
        cdef long_t bbpi = self.ndm0 + bbp - self.npauli
        cdef bool_t bbpi_bool = True if bbpi >= self.ndm0 else False
        cdef long_t aap = self.get_ind_dm0(a, ap, acharge)

        cdef complex_t phi0aap = self.get_phi0_element(a, ap, acharge)
        cdef complex_t dphi0_dt_bbp = -1j*fct*phi0aap

        self.dphi0_dt[bbp] = self.dphi0_dt[bbp] + dphi0_dt_bbp.real
        if bbpi_bool:
            self.dphi0_dt[bbpi] = self.dphi0_dt[bbpi] - dphi0_dt_bbp.imag

    cdef void set_matrix_element_pauli(self,
                double_t fctm, double_t fctp,
                long_t bb, long_t aa) nogil:

        self.dphi0_dt[bb] = self.dphi0_dt[bb] + fctm*self.phi0[bb] + fctp*self.phi0[aa]

    cdef double_t get_phi0_norm(self):

        cdef long_t bcharge, bcount, b, bb, i
        cdef double_t norm = 0.0

        for bcharge in range(self.ncharge):
            bcount = self.statesdm_count[bcharge]
            for i in range(bcount):
                b = self.statesdm[bcharge, i]
                bb = self.get_ind_dm0(b, b, bcharge)
                norm += self.phi0[bb]

        return norm
