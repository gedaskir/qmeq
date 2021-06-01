# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

# Python imports

import numpy as np

from .aprclass import Approach as ApproachPy
from .aprclass import ApproachElPh as ApproachElPhPy

from ..wrappers.mytypes import longnp
from ..wrappers.mytypes import doublenp
from ..wrappers.mytypes import complexnp

# Cython imports

cimport numpy as np
cimport cython

from cython.parallel cimport prange
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

    def restart(self):
        ApproachPy.restart(self)

    #region Preparation

    def get_kern_size(self):
        return ApproachPy.get_kern_size(self)

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

    #endregion Preparation

    #region Generation

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

    cpdef void generate_fct(self):
        pass

    cpdef void generate_kern(self):
        cdef double_t [:] E = self._Ea
        cdef KernelHandler kh = self._kernel_handler

        cdef long_t i, b, bp, bcharge

        for i in prange(kh.nelements, nogil=True):
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


    #endregion Generation

    #region Solution

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

    def set_phi0_init(self):
        return ApproachPy.set_phi0_init(self)

    def rotate(self):
        ApproachPy.rotate(self)

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        ApproachPy.solve(self, qdq, rotateq, masterq, currentq, args, kwargs)

    #endregion Solution


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
