"""Module containing sample classes for approach definitions."""

import itertools

import numpy as np
from scipy import optimize

from ..wrappers.mytypes import doublenp
from ..wrappers.mytypes import complexnp

from .kernel_handler import KernelHandler
from .kernel_handler import KernelHandlerMatrixFree


class Approach(object):
    """
    Sample class used to define different approaches.

    Attributes
    ----------
    qd : QuantumDot
        QuantumDot object.
    leads : LeadsTunneling
        LeadsTunneling object.
    si : StateIndexingDM
        StateIndexingDM object.
    kern : array
        Kernel (Liouvillian) representing the master equation.
    bvec : array
        Right hand side column vector for master equation.
        The entry funcp.norm_row is 1 representing normalization condition.
    sol0 : array
        Least squares solution for the master equation.
    phi0 : array
        Values of zeroth order density matrix elements.
    phi1 : array
        Values of first order density matrix elements
        stored in nleads by ndm1 numpy array.
    current : array
        Values of the current having nleads entries.
    energy_current : array
        Values of the energy current having nleads entries.
    heat_current : array
        Values of the heat current having nleads entries.
    funcp : FunctionProperties
        FunctionProperties object.
    paulifct : array
        Factors used for generating Pauli master equation kernel.
    phi1fct : array
        Factors used for generating 1vN, Redfield master equation kernels.
    phi1fct_energy : array
        Factors used to calculate energy and heat currents in 1vN, Redfield approaches.
    tLba : array
        Jump operator matrix in many-body basis for Lindblad approach.
    """

    kerntype = 'not defined'
    dtype = doublenp
    indexing_class_name = 'StateIndexingDM'

    #region Properties

    @property
    def solmethod(self):
        return self.funcp.solmethod
    @solmethod.setter
    def solmethod(self, value):
        self.funcp.solmethod = value
        self.prepare_solver()

    @property
    def mfreeq(self):
        return self.funcp.mfreeq
    @mfreeq.setter
    def mfreeq(self, value):
        self.funcp.mfreeq = value

    @property
    def symq(self):
        return self.funcp.symq
    @symq.setter
    def symq(self, value):
        if value == self.funcp.symq:
            return
        self.is_prepared = False
        self.funcp.symq = value

    @property
    def norm_row(self):
        return self.funcp.norm_row
    @norm_row.setter
    def norm_row(self, value):
        self.funcp.norm_row = value

    @property
    def itype(self):
        return self.funcp.itype
    @itype.setter
    def itype(self, value):
        self.funcp.itype = value

    #endregion Properties

    def __init__(self, builder):
        """
        Initialization of the Approach class.

        Parameters
        ----------
        builder : Builder, Approach, etc. object.
            Any object having qd, leads, si, and funcp attributes.
        """
        self.qd = builder.qd
        self.leads = builder.leads
        self.si = builder.si
        self.funcp = builder.funcp
        self.restart()

    def restart(self):
        """Restart values of some variables."""
        self.is_prepared = False
        self.make_kern_copy = False

        self.kern, self.bvec, self.norm_vec = None, None, None
        self.sol0, self.phi0, self.phi1 = None, None, None
        self.dphi0_dt = None
        self.current = None
        self.energy_current = None
        self.heat_current = None
        self.phi1fct, self.paulifct = None, None
        self.phi1fct_energy = None
        self.tLba = None
        self.kernel_handler = None

    #region Preparation

    def get_kern_size(self):
        return self.si.ndm0r

    def prepare_kern(self):
        if self.is_prepared and not self.si.states_changed:
            self.clean_arrays()
            return

        self.prepare_kernel_handler()
        self.prepare_arrays()
        self.prepare_solver()

        self.si.states_changed = False
        self.is_prepared = True

    def clean_arrays(self):
        if not self.mfreeq:
            self.kern.fill(0.0)
            self.bvec.fill(0.0)

        self.current.fill(0.0)
        self.energy_current.fill(0.0)
        self.heat_current.fill(0.0)

    def prepare_arrays(self):
        kern_size = self.get_kern_size()
        nleads = self.si.nleads

        self.current = np.zeros(nleads, dtype=doublenp)
        self.energy_current = np.zeros(nleads, dtype=doublenp)
        self.heat_current = np.zeros(nleads, dtype=doublenp)

        self.phi0 = np.zeros(kern_size, dtype=self.dtype)
        self.kernel_handler.set_phi0(self.phi0)

        if self.funcp.mfreeq:
            self.dphi0_dt = np.zeros(kern_size, dtype=self.dtype)
            self.kernel_handler.set_dphi0_dt(self.dphi0_dt)
        else:
            kern_size_rows = kern_size if self.funcp.symq else kern_size+1

            self.kern = np.zeros((kern_size_rows, kern_size), dtype=self.dtype, order='F')

            self.kernel_handler.set_kern(self.kern)

            self.bvec = np.zeros(kern_size_rows, dtype=self.dtype)

            self.replaced_eq = np.zeros(kern_size, dtype=self.dtype)

            self.norm_vec = np.zeros(kern_size, dtype=self.dtype)

            self.generate_norm_vec()

    def prepare_kernel_handler(self):
        if self.funcp.mfreeq:
            self.kernel_handler = KernelHandlerMatrixFree(self.si)
        else:
            self.kernel_handler = KernelHandler(self.si)

    def prepare_solver(self):
        solmethod = self.funcp.solmethod
        # Determine the proper solution method
        if self.funcp.mfreeq:
            solmethod = solmethod if solmethod is not None else 'krylov'
            self.funcp.solmethod = solmethod
            return

        symq = self.funcp.symq
        if solmethod is None:
            solmethod = 'solve' if symq else 'lsqr'
        if not symq and solmethod != 'lsqr' and solmethod != 'lsmr':
            print("WARNING: Using solmethod=lsqr, because the kernel is not symmetric, symq=False.")
            solmethod = 'lsqr'

        self.funcp.solmethod = solmethod

    #endregion Preparation

    #region Generation

    def generate_norm_vec(self):
        """
        Generates normalisation condition for 1vN approach.

        Parameters
        ----------
        norm_vec : array
            (Modifies) Left hand side of the normalisation condition.
        """
        si = self.si

        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                bb = si.get_ind_dm0(b, b, charge)
                self.norm_vec[bb] += 1

    def generate_fct(self):
        pass

    def generate_kern(self):
        """
        Generates a kernel (Liouvillian) matrix corresponding to first order von Neumann approach (1vN).

        Parameters
        ----------
        kern : array
            (Modifies) Kernel matrix for 1vN approach.
        """
        E = self.qd.Ea
        si, kh = self.si, self.kernel_handler
        ncharge, statesdm = si.ncharge, si.statesdm

        for bcharge in range(ncharge):
            for b, bp in itertools.combinations_with_replacement(statesdm[bcharge], 2):
                if not (kh.is_included(b, bp, bcharge) and kh.is_unique(b, bp, bcharge)):
                    continue
                kh.set_energy(E[b]-E[bp], b, bp, bcharge)
                self.generate_coupling_terms(b, bp, bcharge)

    def generate_coupling_terms(self, b, bp, bcharge):
        pass

    def generate_current(self):
        pass

    def generate_vec(self, phi0):
        """
        Acts on given phi0 with Liouvillian of first-order approach.

        Parameters
        ----------
        phi0 : array
            Some values of zeroth order density matrix elements.
        self : Approach
            Approach object.

        Returns
        -------
        dphi0_dt : array
            Values of zeroth order density matrix elements
            after acting with Liouvillian, i.e., dphi0_dt=L(phi0p).
        """
        norm_row = self.funcp.norm_row

        kh = self.kernel_handler
        kh.set_phi0(phi0)
        norm = kh.get_phi0_norm()

        self.dphi0_dt.fill(0.0)

        # Here dphi0_dt and norm will be implicitly calculated by using KernelHandlerMatrixFree
        self.generate_kern()

        self.dphi0_dt[norm_row] = norm-1

        return self.dphi0_dt

    #endregion Generation

    #region Solution

    def solve_kern(self):
        """Finds the stationary state using least squares or using LU decomposition."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        norm_row = self.funcp.norm_row
        replaced_eq = self.replaced_eq

        kern = self.kern
        bvec = self.bvec

        # Replace one equation by the normalisation condition
        if symq:
            replaced_eq[:] = kern[norm_row]
            kern[norm_row] = self.norm_vec
            bvec[norm_row] = 1
        else:
            kern[-1] = self.norm_vec
            bvec[-1] = 1

        # Try to solve the master equation
        try:
            if solmethod == 'solve':
                self.sol0 = [np.linalg.solve(kern, bvec)]
            elif solmethod == 'lsqr':
                self.sol0 = np.linalg.lstsq(kern, bvec, rcond=-1)

            self.phi0[:] = self.sol0[0]
            self.success = True
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0.fill(0.0)
            self.success = False

        # Return back the replaced equation
        if symq:
            kern[norm_row] = replaced_eq

    def solve_matrix_free(self):
        """Finds the stationary state using matrix free methods like broyden, krylov, etc."""
        solmethod = self.funcp.solmethod
        phi0_init = self.funcp.phi0_init

        if phi0_init is None:
            self.funcp.print_warning(0, "WARNING: For mfreeq=True no phi0_init is specified. " +
                                     "Using phi0_init[0]=1.0 as a default. " +
                                     "This warning will not be shown again.")
            phi0_init = self.set_phi0_init()

        try:
            self.sol0 = optimize.root(self.generate_vec, phi0_init, method=solmethod)
            self.phi0[:] = self.sol0.x
            self.success = self.sol0.success
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0.fill(0.0)
            self.success = False

    def set_phi0_init(self):
        phi0_init = np.zeros(self.get_kern_size(), dtype=self.dtype)
        phi0_init[0] = 1.0
        return phi0_init

    def rotate(self):
        self.leads.rotate(self.qd.vecslst)

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        """
        Solves the master equation.

        Parameters
        ----------
        qdq : bool
            Diagonalise many-body quantum dot Hamiltonian
            and express the lead matrix Tba in the eigenbasis.
        rotateq : bool
            Rotate the many-body tunneling matrix Tba.
        masterq : bool
            Solve the master equation.
        currentq : bool
            Calculate the current.
        """
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.rotate()
        #
        if masterq:
            self.prepare_kern()
            self.generate_fct()
            if not self.funcp.mfreeq:
                self.generate_kern()
                self.solve_kern()
            else:
                self.solve_matrix_free()
            if currentq:
                self.generate_current()

    #endregion Solution

class ApproachElPh(Approach):

    def __init__(self, builder):
        Approach.__init__(self, builder)
        self.baths = builder.baths
        self.si_elph = builder.si_elph

    def restart(self):
        Approach.restart(self)
        self.w1fct, self.paulifct_elph = None, None
        self.tLbbp = None

    def rotate(self):
        self.leads.rotate(self.qd.vecslst)
        self.baths.rotate(self.qd.vecslst)


class Iterations2vN(object):
    """
    Class for storing some properties of the system after 2vN iteration.
    """

    def __init__(self, appr):
        self.niter = appr.niter
        self.phi0 = np.copy(appr.phi0)
        self.phi1 = np.copy(appr.phi1)
        self.current = np.copy(appr.current)
        self.energy_current = np.copy(appr.energy_current)
        self.heat_current = np.copy(appr.heat_current)


class ApproachBase2vN(Approach):
    """
    Sample class for solving the 2vN approach for stationary state.
    Most of the attributes are the same as in Approach class.
    Here we define the Python implementation of the 2vN approach.

    Attributes
    ----------
    kern : array
        Kernel for zeroth order density matrix elements Phi[0].
    funcp.kpnt : int
        Number of energy grid points on which 2vN approach equations are solved.
    Ek_grid : array
        Energy grid on which 2vN approach equations are solved.
    Ek_grid_ext : array
        Extension of Ek_grid by neumann2py.get_grid_ext.
    niter : int
        Number of iterations performed when solving integral equation.
    iters : Iterations2vN
        Iterations2vN object.
    phi1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        containing energy resolved first order density matrix elements Phi[1](k).
    phi1k_delta : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        Difference from phi1k after performing one iteration.
    hphi1k_delta : array
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0)
        Hilbert transform of phi1k_delta.
    kern1k_inv : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm1)
        corresponding to inverse of energy resolved local kernel for Phi[1](k).
    fkp, fkm : array
        nleads by len(Ek_grid_ext) numpy array containing
        Fermi function (fkp) and 1-Fermi (fkm) values on the grid Ek_grid_ext.
    hfkp, hfkm : array
        Hilbert transform of fkp, fkm.
    """

    kerntype = 'not defined'
    dtype = complexnp
    indexing_class_name = 'StateIndexingDMc'

    def __init__(self, builder):
        """
        Initialization of the Approach2vN class.

        Parameters
        ----------
        builder : Builder, Approach, etc. object.
            Any object having qd, leads, si, and funcp attributes.
        """
        self.qd = builder.qd
        self.leads = builder.leads
        self.si = builder.si
        self.funcp = builder.funcp
        self.fkp, self.fkm = None, None
        self.hfkp, self.hfkm = None, None
        self.restart()
        self.Ek_grid = np.zeros(1, dtype=doublenp)
        self.Ek_grid_ext = np.zeros(0, dtype=doublenp)
        # Some exceptions
        if type(self.si).__name__ != self.indexing_class_name:
            raise TypeError('The state indexing class for 2vN approach has to be StateIndexingDMc')

    def restart(self):
        """Restart values of some variables for new calculations."""
        Approach.restart(self)
        self.is_zeroth_iteration = True

        self.h_phi1 = None

        self.phi1_phi0 = None
        self.e_phi1_phi0 = None

        self.phi1k = None
        self.phi1k_delta = None
        self.hphi1k_delta = None
        self.kern1k_inv = None

        self.niter = -1
        self.iters = []

    #region Preparation

    def get_kern_size(self):
        return self.si.ndm0

    def prepare_arrays(self):
        Approach.prepare_arrays(self)
        nleads, ndm0, ndm1 = self.si.nleads, self.si.ndm0, self.si.ndm1

        self.phi1 = np.zeros((nleads, ndm1), dtype=complexnp)
        self.h_phi1 = np.zeros((nleads, ndm1), dtype=complexnp)

        self.phi1_phi0 = np.zeros((nleads, ndm1, ndm0), dtype=complexnp)
        self.e_phi1_phi0 = np.zeros((nleads, ndm1, ndm0), dtype=complexnp)

        if self.is_zeroth_iteration:
            Eklen = len(self.Ek_grid)
            self.phi1k = np.zeros((Eklen, nleads, ndm1, ndm0), dtype=complexnp)
            self.phi1k_delta = np.zeros((Eklen, nleads, ndm1, ndm0), dtype=complexnp)
            self.kern1k_inv = np.zeros((Eklen, nleads, ndm1, ndm1), dtype=complexnp)

    def clean_arrays(self):
        Approach.clean_arrays(self)

        self.phi1.fill(0.0)
        self.h_phi1.fill(0.0)

        self.phi1_phi0.fill(0.0)
        self.e_phi1_phi0.fill(0.0)

    #endregion Preparation

    #region Generation

    def iterate(self):
        pass

    def determine_phi1_phi0(self):
        pass

    def generate_kern(self):
        pass

    def generate_current(self):
        pass

    #endregion Generation

    def iteration(self):
        """Makes one iteration for solution of the 2vN integral equation."""
        self.prepare_kern()

        self.iterate()
        self.phi1k += self.phi1k_delta
        self.determine_phi1_phi0()
        self.niter += 1

        self.generate_kern()
        self.solve_kern()
        self.generate_current()

        self.iters.append(Iterations2vN(self))

    def solve(self,
              qdq=True, rotateq=True, masterq=True, restartq=True,
              niter=None, func_iter=None, *args, **kwargs):
        """
        Solves the 2vN approach integral equations iteratively.

        Parameters
        ----------
        qdq : bool
            Diagonalise many-body quantum dot Hamiltonian
            and express the lead matrix Tba in the eigenbasis.
        rotateq : bool
            Rotate the many-body tunneling matrix Tba.
        masterq : bool
            Solve the master equation.
        restartq : bool
            Call restart() and erase old values of variables.
            To continue from the last iteration set restartq=False.
        niter : int
            Number of iterations to perform.
        func_iter : function
            User defined function which is performed after every iteration and
            takes Approach2vN object as an input.
        """
        if restartq:
            self.restart()
        if qdq:
            self.qd.diagonalise()
            if rotateq:
                self.leads.rotate(self.qd.vecslst)

        if masterq:
            if niter is None:
                raise ValueError('Number of iterations niter needs to be specified')

            for it in range(niter):
                self.iteration()
                if func_iter is not None:
                    func_iter(self)
