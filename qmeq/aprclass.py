"""Module containing sample classes for approach definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse.linalg
from scipy import optimize

from .mytypes import doublenp
from .mytypes import complexnp

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
    indexing_class_name = 'StateIndexingDM'

    @staticmethod
    def generate_fct(sys): pass
    @staticmethod
    def generate_kern(sys): pass
    @staticmethod
    def generate_current(sys): pass
    @staticmethod
    def generate_vec(sys): pass

    def __init__(self, sys):
        """
        Initialization of the Approach class.

        Parameters
        ----------
        sys : Approach, Builder, etc. object.
            Any object having qd, leads, si, and funcp attributes.
        """
        self.qd = sys.qd
        self.leads = sys.leads
        self.si = sys.si
        self.funcp = sys.funcp
        self.restart()

    def restart(self):
        """Restart values of some variables."""
        self.kern, self.bvec = None, None
        self.sol0, self.phi0, self.phi1 = None, None, None
        self.current = None
        self.energy_current = None
        self.heat_current = None
        self.phi1fct, self.paulifct = None, None

    def set_phi0_init(self):
        if self.kerntype in {'Pauli', 'pyPauli'}:
            phi0_init = np.zeros(self.si.npauli, dtype=doublenp)
        else:
            phi0_init = np.zeros(self.si.ndm0r, dtype=doublenp)
        phi0_init[0] = 1.0
        return phi0_init

    def solve_kern(self):
        """Finds the stationary state using least squares or by inverting the matrix."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        if solmethod == 'n': solmethod = 'solve' if symq else 'lsqr'
        if not symq and solmethod != 'lsqr' and solmethod != 'lsmr':
            print("WARNING: Using solmethod=lsqr, because the kernel is not symmetric, symq=False.")
            solmethod = 'lsqr'
        try:
            if   solmethod == 'solve': self.sol0 = [np.linalg.solve(self.kern, self.bvec)]
            elif solmethod == 'lsqr':  self.sol0 = np.linalg.lstsq(self.kern, self.bvec)
            self.phi0 = self.sol0[0]
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = np.zeros(self.si.ndm0r)
        self.funcp.solmethod = solmethod

    def solve_matrix_free(self):
        """Finds the stationary state using matrix free methods like broyden, krylov, etc."""
        solmethod = self.funcp.solmethod
        #
        phi0_init = self.funcp.phi0_init
        if phi0_init is None:
            self.funcp.print_warning(0, "WARNING: For mfreeq=True no phi0_init is specified. "+
                                        "Using phi0_init[0]=1.0 as a default. "+
                                        "This warning will not be shown again.")
            phi0_init = self.set_phi0_init()
        #
        solmethod = solmethod if solmethod != 'n' else 'krylov'
        try:
            self.sol0 = optimize.root(self.generate_vec, phi0_init, args=(self), method=solmethod)
            self.phi0 = self.sol0.x
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = 0*phi0_init
        self.funcp.solmethod = solmethod

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
                self.leads.rotate(self.qd.vecslst)
        #
        if masterq:
            self.generate_fct(self)
            if self.funcp.mfreeq:
                self.solve_matrix_free()
            else:
                self.generate_kern(self)
                self.solve_kern()
            if currentq:
                self.generate_current(self)


class Iterations2vN(object):
    """
    Class for storing some properties of the system after 2vN iteration.
    """

    def __init__(self, sys):
        self.niter = sys.niter
        self.phi0 = sys.phi0
        self.phi1 = sys.phi1
        self.current = sys.current
        self.energy_current = sys.energy_current
        self.heat_current = sys.heat_current


class Approach2vN(Approach):
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
        Extension of Ek_grid by neumann2py.get_grid_ext(sys).
    niter : int
        Number of iterations performed when solving integral equation.
    iters : Iterations2vN
        Iterations2vN object.
    phi1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        cotaining energy resolved first order density matrix elements Phi[1](k).
    phi1k_delta : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm0)
        Difference from phi1k after performing one iteration.
    hphi1k_delta : array
        Numpy array with dimensions (len(Ek_grid_ext), nleads, ndm1, ndm0)
        Hilbert transform of phi1k_delta.
    kern1k : array
        Numpy array with dimensions (len(Ek_grid), nleads, ndm1, ndm1)
        corresponding to energy resolved local kernel for Phi[1](k).
    fkp, fkm : array
        nleads by len(Ek_grid_ext) numpy array containing
        Fermi function (fkp) and 1-Fermi (fkm) values on the grid Ek_grid_ext.
    hfkp, hfkm : array
        Hilbert transform of fkp, fkm.
    """

    kerntype = 'not defined'
    indexing_class_name = 'StateIndexingDMc'
    @staticmethod
    def iterate(sys): pass
    @staticmethod
    def get_phi1_phi0(sys): pass
    @staticmethod
    def kern_phi0(sys): pass
    @staticmethod
    def generate_current(sys): pass

    def __init__(self, sys):
        """
        Initialization of the Approach2vN class.

        Parameters
        ----------
        sys : Approach, Builder, etc. object.
            Any object having qd, leads, si, and funcp attributes.
        """
        self.qd = sys.qd
        self.leads = sys.leads
        self.si = sys.si
        self.funcp = sys.funcp
        self.fkp, self.fkm = None, None
        self.hfkp, self.hfkm = None, None
        self.restart()
        self.Ek_grid = np.zeros(1, dtype=doublenp)
        self.Ek_grid_ext = np.zeros(0, dtype=doublenp)
        # Some exceptions
        if type(self.si).__name__ != self.indexing_class_name:
            raise TypeError('The state indexing class for 2vN approach has to be StateIndexingDMc')

    def make_Ek_grid(self):
        """Make an energy grid on which 2vN equations are solved. """
        if self.funcp.kpnt is None:
            raise ValueError('kpnt needs to be specified.')
        if self.si.nleads > 0:
            dmin = np.min(self.leads.dlst)
            dmax = np.max(self.leads.dlst)
            Ek_grid, kpnt = self.Ek_grid, self.funcp.kpnt
            if Ek_grid[0] != dmin or Ek_grid[-1] != dmax or Ek_grid.shape[0] != kpnt:
                self.funcp.dmin = dmin
                self.funcp.dmax = dmax
                self.Ek_grid = np.linspace(dmin, dmax, kpnt)
                #
                if self.niter != -1:
                    print("WARNING: Ek_grid has changed. Restarting the calculation.")
                    self.restart()
                #
                if ((dmin*np.ones(self.si.nleads)).tolist() != self.leads.dlst.T[0].tolist() or
                    (dmax*np.ones(self.si.nleads)).tolist() != self.leads.dlst.T[1].tolist()):
                    print("WARNING: The bandwidth and Ek_grid for all leads will be the same: from "+
                          "dmin="+str(dmin)+" to dmax="+str(dmax)+".")

    def restart(self):
        """Restart values of some variables for new calculations."""
        self.kern, self.bvec = None, None
        self.sol0, self.phi0, self.phi1 = None, None, None
        self.current = None
        self.energy_current = None
        self.heat_current = None
        #
        self.niter = -1
        self.phi1k = None
        self.phi1k_delta = None
        self.hphi1k_delta = None
        self.kern1k = None
        #
        self.iters = []

    def iteration(self):
        """Makes one iteration for solution of the 2vN integral equation."""
        self.iterate(self)
        self.phi1k = self.phi1k_delta if self.phi1k is None else self.phi1k + self.phi1k_delta
        self.get_phi1_phi0(self)
        self.niter += 1
        self.kern_phi0(self)
        self.solve_kern()
        self.generate_current(self)
        #
        self.iters.append(Iterations2vN(self))


    def solve(self, qdq=True, rotateq=True, masterq=True, restartq=True,
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
            To continue from the last iteration set restarq=False.
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
        #
        if masterq:
            # Exception
            if niter is None:
                raise ValueError('Number of iterations niter needs to be specified')
            self.make_Ek_grid()
            #
            for it in range(niter):
                self.iteration()
                if not func_iter is None:
                    func_iter(self)
