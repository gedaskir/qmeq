"""Module containing sample classes for approach definitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools

import numpy as np
from scipy import optimize

from ..mytypes import doublenp
from ..mytypes import complexnp


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
    kern_ext : array
        Same as kern, only with one additional row added.
    bvec_ext : array
        Same as bvec, only with one additional entry added.
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
        self.kern, self.bvec, self.norm_vec = None, None, None
        self.kern_ext, self.bvec_ext = None, None
        self.sol0, self.phi0, self.phi1 = None, None, None
        self.current = None
        self.energy_current = None
        self.heat_current = None
        self.phi1fct, self.paulifct = None, None
        self.phi1fct_energy = None
        self.tLba = None
        self.kernel_handler = None

    def generate_fct(self):
        pass

    def generate_kern(self):
        """
        Generates a kernel (Liouvillian) matrix corresponding to first order von Neumann approach (1vN).

        Parameters
        ----------
        self.kern : array
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

        dphi0_dt = np.zeros(phi0.shape, dtype=doublenp)
        kh.set_dphi0_dt(dphi0_dt)

        # Here i_dphi0_dt and norm will be implicitly calculated by using KernelHandlerMatrixFree
        self.generate_kern()

        dphi0_dt[norm_row] = norm-1

        return dphi0_dt

    def get_kern_size(self):
        return self.si.ndm0r

    def set_phi0_init(self):
        phi0_init = np.zeros(self.get_kern_size(), dtype=self.dtype)
        phi0_init[0] = 1.0
        return phi0_init

    def prepare_kern(self):
        self.prepare_kernel_handler()

        if not self.funcp.mfreeq:
            kern_size = self.get_kern_size()
            self.kern_ext = np.zeros((kern_size+1, kern_size), dtype=self.dtype)
            self.kern = self.kern_ext[0:-1, :]
            self.kernel_handler.set_kern(self.kern)

            self.generate_norm_vec(kern_size)

    def solve_kern(self):
        """Finds the stationary state using least squares or by inverting the matrix."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        norm_row = self.funcp.norm_row
        replaced_eq = None

        # Determine the proper solution method
        if solmethod is None:
            solmethod = 'solve' if symq else 'lsqr'
        if not symq and solmethod != 'lsqr' and solmethod != 'lsmr':
            print("WARNING: Using solmethod=lsqr, because the kernel is not symmetric, symq=False.")
            solmethod = 'lsqr'
        self.funcp.solmethod = solmethod

        # Replace one equation by the normalisation condition
        if symq:
            kern, bvec = self.kern, self.bvec
            replaced_eq = np.array(kern[norm_row])
            kern[norm_row] = self.norm_vec
        else:
            kern, bvec = self.kern_ext, self.bvec_ext
            kern[-1] = self.norm_vec

        # Try to solve the master equation
        try:
            if solmethod == 'solve':
                self.sol0 = [np.linalg.solve(kern, bvec)]
            elif solmethod == 'lsqr':
                self.sol0 = np.linalg.lstsq(kern, bvec, rcond=-1)

            self.phi0 = self.sol0[0]
            self.success = True
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = np.zeros(self.get_kern_size())
            self.success = False

        # Return back the replaced equation
        if symq:
            kern[norm_row] = replaced_eq

    def prepare_kernel_handler(self):
        if self.funcp.mfreeq:
            self.kernel_handler = KernelHandlerMatrixFree(self.si)
        else:
            self.kernel_handler = KernelHandler(self.si)
            self.kernel_handler.set_kern(self.kern)

    def solve_matrix_free(self):
        """Finds the stationary state using matrix free methods like broyden, krylov, etc."""
        solmethod = self.funcp.solmethod
        #
        phi0_init = self.funcp.phi0_init
        if phi0_init is None:
            self.funcp.print_warning(0, "WARNING: For mfreeq=True no phi0_init is specified. " +
                                     "Using phi0_init[0]=1.0 as a default. " +
                                     "This warning will not be shown again.")
            phi0_init = self.set_phi0_init()
        #
        solmethod = solmethod if solmethod is not None else 'krylov'
        try:
            self.sol0 = optimize.root(self.generate_vec, phi0_init, method=solmethod)
            self.phi0 = self.sol0.x
            self.success = self.sol0.success
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = 0 * phi0_init
            self.success = False
        self.funcp.solmethod = solmethod

    def generate_norm_vec(self, length):
        """
        Generates normalisation condition for 1vN approach.

        Parameters
        ----------
        length: int
            Length of the normalisation row.

        self.norm_vec : array
            (Modifies) Left hand side of the normalisation condition.
        self.bvec : array
            (Modifies) Right hand side column vector for master equation.
            The entry funcp.norm_row is 1 representing normalization condition.
        """
        si, symq, norm_row = self.si, self.funcp.symq, self.funcp.norm_row

        self.bvec_ext = np.zeros(length+1, dtype=self.dtype)
        self.bvec_ext[-1] = 1

        self.bvec = self.bvec_ext[0:-1]
        self.bvec[norm_row] = 1 if symq else 0

        self.norm_vec = np.zeros(length, dtype=self.dtype)
        norm_vec = self.norm_vec

        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                bb = si.get_ind_dm0(b, b, charge)
                norm_vec[bb] += 1

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
        self.phi0 = appr.phi0
        self.phi1 = appr.phi1
        self.current = appr.current
        self.energy_current = appr.energy_current
        self.heat_current = appr.heat_current


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

    def iterate(self):
        pass

    def get_phi1_phi0(self):
        pass

    def kern_phi0(self):
        pass

    def generate_current(self):
        pass

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
                if ((dmin * np.ones(self.si.nleads)).tolist() != self.leads.dlst.T[0].tolist() or
                        (dmax * np.ones(self.si.nleads)).tolist() != self.leads.dlst.T[1].tolist()):
                    print("WARNING: The bandwidth and Ek_grid for all leads will be the same: from " +
                          "dmin=" + str(dmin) + " to dmax=" + str(dmax) + ".")

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
        self.kern1k_inv = None
        self.phi1_phi0 = None
        self.e_phi1_phi0 = None
        #
        self.iters = []

    def get_kern_size(self):
        return self.si.ndm0

    def iteration(self):
        """Makes one iteration for solution of the 2vN integral equation."""
        self.iterate()
        self.phi1k = self.phi1k_delta if self.phi1k is None else self.phi1k + self.phi1k_delta
        self.get_phi1_phi0()
        self.niter += 1
        self.prepare_kern()
        self.kern_phi0()
        self.solve_kern()
        self.generate_current()
        #
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
        #
        if masterq:
            # Exception
            if niter is None:
                raise ValueError('Number of iterations niter needs to be specified')
            self.make_Ek_grid()
            #
            for it in range(niter):
                self.iteration()
                if func_iter is not None:
                    func_iter(self)


class KernelHandler(object):

    def __init__(self, si):
        self.si = si
        self.ndm0 = si.ndm0
        self.ndm0r = si.ndm0r
        self.npauli = si.npauli
        self.phi0 = None
        self.kern = None

    def set_kern(self, kern):
        self.kern = kern

    def set_phi0(self, phi0):
        self.phi0 = phi0

    def is_included(self, b, bp, bcharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return False

        return True

    def is_unique(self, b, bp, bcharge):
        bbp_bool = self.si.get_ind_dm0(b, bp, bcharge, maptype=2)
        return bbp_bool

    def set_energy(self, energy, b, bp, bcharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False

        if bbpi_bool:
            self.kern[bbp, bbpi] = self.kern[bbp, bbpi] + energy
            self.kern[bbpi, bbp] = self.kern[bbpi, bbp] - energy

    def set_matrix_element(self, fct, b, bp, bcharge, a, ap, acharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False

        aap = self.si.get_ind_dm0(a, ap, acharge)
        aapi = self.ndm0 + aap - self.npauli
        aap_sgn = +1 if self.si.get_ind_dm0(a, ap, acharge, maptype=3) else -1

        fct_imag = fct.imag
        fct_real = fct.real

        self.kern[bbp, aap] += fct_imag
        if aapi >= self.ndm0:
            self.kern[bbp, aapi] += fct_real*aap_sgn
            if bbpi_bool:
                self.kern[bbpi, aapi] += fct_imag*aap_sgn
        if bbpi_bool:
            self.kern[bbpi, aap] += -fct_real

    def set_matrix_element_pauli(self, fctm, fctp, bb, aa):
        self.kern[bb, bb] += fctm
        self.kern[bb, aa] += fctp

    def get_phi0_element(self, b, bp, bcharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        if bbp == -1:
            return 0.0

        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False

        phi0_real = self.phi0[bbp]
        phi0_imag = 0
        if bbpi_bool:
            bbp_conj = self.si.get_ind_dm0(b, bp, bcharge, maptype=3)
            phi0_imag = self.phi0[bbpi] if bbp_conj else -self.phi0[bbpi]

        return phi0_real + 1j*phi0_imag

class KernelHandlerMatrixFree(KernelHandler):

    def __init__(self, si):
        KernelHandler.__init__(self, si)
        self.dphi0_dt = None

    def set_dphi0_dt(self, dphi0_dt):
        self.dphi0_dt = dphi0_dt

    def set_energy(self, energy, b, bp, bcharge):
        if b == bp:
            return

        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli

        phi0bbp = self.get_phi0_element(b, bp, bcharge)
        dphi0_dt_bbp = -1j*energy*phi0bbp

        self.dphi0_dt[bbp] += dphi0_dt_bbp.real
        self.dphi0_dt[bbpi] -= dphi0_dt_bbp.imag

    def set_matrix_element(self, fct, b, bp, bcharge, a, ap, acharge):
        bbp = self.si.get_ind_dm0(b, bp, bcharge)
        bbpi = self.ndm0 + bbp - self.npauli
        bbpi_bool = True if bbpi >= self.ndm0 else False
        aap = self.si.get_ind_dm0(a, ap, acharge)

        phi0aap = self.get_phi0_element(a, ap, acharge)
        dphi0_dt_bbp = -1j*fct*phi0aap

        self.dphi0_dt[bbp] += dphi0_dt_bbp.real
        if bbpi_bool:
            self.dphi0_dt[bbpi] -= dphi0_dt_bbp.imag

    def set_matrix_element_pauli(self, fctm, fctp, bb, aa):
        self.dphi0_dt[bb] += fctm*self.phi0[bb] + fctp*self.phi0[aa]

    def get_phi0_norm(self):
        ncharge, statesdm = self.si.ncharge, self.si.statesdm

        norm = 0.0
        for bcharge in range(ncharge):
            for b in statesdm[bcharge]:
                bb = self.si.get_ind_dm0(b, b, bcharge)
                norm += self.phi0[bb]

        return norm
