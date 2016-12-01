"""Module for solving different master equations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy import optimize
import os
import copy

from .lindbladpy import generate_tLba
from .lindbladpy import generate_kern_lindblad
from .lindbladpy import generate_current_lindblad
from .lindbladpy import generate_vec_lindblad
from .lindbladc import c_generate_tLba
from .lindbladc import c_generate_kern_lindblad
from .lindbladc import c_generate_current_lindblad
from .lindbladc import c_generate_vec_lindblad

from .neumannpy import generate_phi1fct
from .neumannpy import generate_paulifct
from .neumannc import c_generate_phi1fct
from .neumannc import c_generate_paulifct

from .neumannpy import generate_kern_pauli
from .neumannpy import generate_current_pauli
from .neumannpy import generate_kern_1vN
from .neumannpy import generate_phi1_1vN
from .neumannpy import generate_vec_1vN

from .neumannc import c_generate_kern_pauli
from .neumannc import c_generate_current_pauli
from .neumannc import c_generate_kern_redfield
from .neumannc import c_generate_phi1_redfield
from .neumannc import c_generate_vec_redfield
from .neumannc import c_generate_kern_1vN
from .neumannc import c_generate_phi1_1vN
from .neumannc import c_generate_vec_1vN

from .neumann2py import get_phi1_phi0_2vN
from .neumann2py import kern_phi0_2vN
from .neumann2py import generate_current_2vN
from .neumann2py import iterate_2vN

from .neumann2c import c_get_phi1_phi0_2vN
from .neumann2c import c_iterate_2vN

from .mytypes import doublenp
from .mytypes import complexnp

from .indexing import StateIndexingDM
from .indexing import StateIndexingDMc
from .manyham import QuantumDot
from .constructxba import LeadsTunneling

from .various import get_phi0
from .various import get_phi1
from .various import print_state
from .various import print_all_states
from .various import remove_states
from .various import use_all_states

class Builder(object):
    """
    Class for building the system for stationary transport calculations.

    Attributes
    ----------
    nsingle : int
        Number of single-particle states.
    hsingle : dict, list or array
        Dictionary, list or array corresponding to single-particle hopping (tunneling) Hamiltonian.
        On input list or array gets converted to dictionary.
    coulomb : list, dict, or array
        Dictionary, list or array containing coulomb matrix elements.
        For dictionary:    coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels.
        For list or array: coulomb[i] is list of the format [m, n, k, l, U].
        U is the strength of the coulomb interaction between the states (m, n, k, l).
        Note that only the matrix elements k>l, n>m have to be specified.
        On input list or array gets converted to dictionary.
    nleads : int
        Number of the leads.
    tleads : dict, list or array
        Dictionary, list or numpy array defining single particle tunneling amplitudes.
        numpy array has to be nleads by nsingle.
    mulst : dict, list or array
        Dictionary, list or array containing chemical potentials of the leads.
    tlst : dict, list or array
        Dictionary, list or array containing temperatures of the leads.
    dlst : dict, list or array
        Dictionary, list or array containing bandwidths of the leads.
    indexing : str
        String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        Note that 'sz' indexing for Fock states is used for 'ssq' indexing, with
        additional specification of eigensates in Fock basis.
    Ek_grid : array
        Energy grid on which 2vN approach equations are solved.
    kerntype : string
        String describing what master equation method to use.
        For Transport class the possible values are 'Pauli', '1vN', 'Redfield', 'Lindblad', 'pyPauli', 'py1vN', 'pyLindblad'.
        For Transport2vN class the possible values are '2vN' and 'py2vN'.
        The methods with 'py' in front are not compiled using cython.
    symq : bool
        For symq=False keep all equations in the kernel, and the matrix is of size N by N+1.
        For symq=True replace one equation by the normalisation condition, and the matrix is square N by N.
    norm_row : int
        If symq=True this row will be replaced by normalisation condition in the kernel matrix.
    solmethod : string
        String specifying the solution method of the equation L(Phi0)=0.
        The possible values are matrix inversion 'solve' and least squares 'lsqr'.
        Method 'solve' works only when symq=True.
        For matrix free methods (used when mfreeq=True) the possible values are
        'krylov', 'broyden', etc.
    itype : int
        Type of integral for first order method calculations.
        itype=0: the principal parts are neglected.
        itype=1: the principal parts are kept, but approximated by digamma function valid for large bandwidht D.
        itype=2: the principal parts are evaluated using Fortran integration package QUADPACK routine dqawc through SciPy.
    dqawc_limit : int
        For itype=2 dqawc_limit determines the maximum number of subintervals
        in the partition of the given integration interval.
    mfreeq : bool
        If mfreeq=True the matrix free solution method is used for first order methods.
    phi0_init : array
        For mfreeq=True the initial value of zeroth order density matrix elements.
    mtype_qd : float or complex
        Type for the many-body quantum dot Hamiltonian matrix.
    mtype_leads : float or complex
        Type for the many-body tunneling matrix Tba.
    qd : QuantumDot
        QuantumDot object.
    leads : LeadsTunneling
        LeadsTunneling object.
    si : StateIndexingDM
        StateIndexingDM object.
    tt : Transport or Transport2vN.
        Transport or Transport2vN object.
    funcp : FunctionProperties
        FunctionProperties object.
    Ea : array
        nmany by 1 array containing many-body Hamiltonian eigenvalues.
    Tba : array
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix,
        which is used in calculations.
    kern : array
        Kernel (Liouvillian) representing the master equation.
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
    niter : int
        Number of iterations performed when solving integral equation for 2vN approach.
    iters : Iterations2vN
        Iterations2vN object, which is present just for 2vN approach.
    """

    def __init__(self, nsingle, hsingle, coulomb,
                       nleads, tleads, mulst, tlst, dlst,
                       indexing='charge', Ek_grid=None,
                       kerntype='1vN', symq=True, norm_row=0, solmethod='n',
                       itype=2, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                       mtype_qd=complex, mtype_leads=complex):

        # Make copies of initialized parameters.
        hsingle = copy.deepcopy(hsingle)
        coulomb = copy.deepcopy(coulomb)
        tleads = copy.deepcopy(tleads)
        mulst = copy.deepcopy(mulst)
        tlst = copy.deepcopy(tlst)
        dlst = copy.deepcopy(dlst)
        Ek_grid = copy.deepcopy(Ek_grid)
        phi0_init = copy.deepcopy(phi0_init)

        funcp = FunctionProperties(kerntype=kerntype, symq=symq, norm_row=norm_row, solmethod=solmethod,
                                   itype=itype, dqawc_limit=dqawc_limit, mfreeq=mfreeq, phi0_init=phi0_init,
                                   mtype_qd=mtype_qd, mtype_leads=mtype_leads)

        if kerntype in {'py2vN', '2vN'}:
            if Ek_grid is None:
                raise ValueError('For 2vN method Ek_grid needs to be specified.')
                print('WARNING: For 2vN method Ek_grid needs to be specified.')
            if not indexing in ['Lin', 'charge']:
                print('WARNING: For 2vN method indexing needs to be \'Lin\' or \'charge\'. Using \'charge\' as a default.')
                indexing = 'charge'
            si = StateIndexingDMc(nsingle, indexing)
            qd = QuantumDot(hsingle, coulomb, si, mtype_qd)
            leads = LeadsTunneling(nleads, tleads, si, mulst, tlst, dlst, mtype_leads)
            tt = Transport2vN(qd, leads, si, funcp, Ek_grid)
        else:
            si = StateIndexingDM(nsingle, indexing)
            qd = QuantumDot(hsingle, coulomb, si, mtype_qd)
            leads = LeadsTunneling(nleads, tleads, si, mulst, tlst, dlst, mtype_leads)
            tt = Transport(qd, leads, si, funcp)

        if mfreeq and phi0_init is None and not kerntype in {'Pauli', 'pyPauli'}:
            funcp.phi0_init = np.zeros(si.ndm0r, dtype=doublenp)
            funcp.phi0_init[0] = 1.0

        self.funcp, self.si, self.qd, self.leads, self.tt = funcp, si, qd, leads, tt

    def __getattr__(self, item):
        if item in {'solve', 'current', 'energy_current', 'heat_current', 'phi0', 'phi1',
                    'niter', 'iters', 'Ek_grid', 'kern'}:
            return getattr(self.tt, item)
        elif item in {'symq', 'norm_row', 'solmethod', 'itype', 'dqawc_limit', 'mfreeq', 'phi0_init'}:
            return getattr(self.funcp, item)
        elif item in {'tleads', 'mulst', 'tlst', 'dlst'}:
            return getattr(self.leads, item)
        elif item in {'nsingle', 'nleads'}:
            return getattr(self.si, item)
        elif item in {'hsingle', 'coulomb', 'Ea', 'Tba'}:
            return getattr(self.qd, item)
        else:
            #return self.__getattribute__(item)
            return super(Builder, self).__getattribute__(item)

    def __setattr__(self, item, value):
        if item in {'solve', 'current', 'energy_current', 'heat_current', 'phi0', 'phi1',
                    'niter', 'iters', 'Ek_grid', 'kern'}:
            setattr(self.tt, item, value)
        elif item in {'symq', 'norm_row', 'solmethod', 'itype', 'dqawc_limit', 'mfreeq', 'phi0_init'}:
            setattr(self.funcp, item, value)
        elif item in {'tleads', 'mulst', 'tlst', 'dlst'}:
            setattr(self.leads, item, value)
        elif item in {'nsingle', 'nleads'}:
            setattr(self.si, item, value)
        elif item in {'hsingle', 'coulomb', 'Ea', 'Tba'}:
            setattr(self.qd, item, value)
        else:
            super(Builder, self).__setattr__(item, value)

    # kerntype
    def get_kerntype(self):
        return self.funcp.kerntype
    def set_kerntype(self, value):
        k1 = 1 if value in {'2vN', 'py2vN'} else 0
        k2 = 1 if self.funcp.kerntype in {'2vN', 'py2vN'} else 0
        if k1 == k2:
            self.funcp.kerntype = value
        else:
            print('WARNING: Cannot change kernel type from \''+self.funcp.kerntype+'\' to \''+value
                 +'\' Consider contructing a new system using \'kerntype='+value+'\'')
    kerntype = property(get_kerntype, set_kerntype)

    # indexing
    def get_indexing(self):
        return self.si.indexing
    def set_indexing(self, value):
        if self.si.indexing != value:
            print('WARNING: Cannot change indexing from \''+self.si.indexing+'\' to \''+value
                  +'\' Consider contructing a new system using \'indexing='+value+'\'')
    indexing = property(get_indexing, set_indexing)

    def add(self, hsingle=None, coulomb=None, tleads=None, mulst=None, tlst=None, dlst=None):
        """
        Adds the values to the specified dictionaries and correspondingly redefines
        relevant many-body properties of the system.

        Parameters
        ----------
        hsingle, coulomb, tleads, mulst, tlst, dlst : dict
            Dictionaries describing what values to add.
            For example, tleads[(lead, state)] = value to add.
        """
        if not (hsingle is None and coulomb is None):
            self.qd.add(hsingle, coulomb)
        if not (tleads is None and mulst is None and tlst is None and dlst is None):
            self.leads.add(tleads, mulst, tlst, dlst)

    def change(self, hsingle=None, coulomb=None, tleads=None, mulst=None, tlst=None, dlst=None):
        """
        Changes the values of the specified dictionaries and correspondingly redefines
        relevant many-body properties of the system.

        Parameters
        ----------
        hsingle, coulomb, tleads, mulst, tlst, dlst : dict
            Dictionaries describing what values to change.
            For example, tleads[(lead, state)] = value to change.
        """
        if not (hsingle is None and coulomb is None):
            self.qd.change(hsingle, coulomb)
        if not (tleads is None and mulst is None and tlst is None and dlst is None):
            self.leads.change(tleads, mulst, tlst, dlst)

    def get_phi0(self, b, bp):
        '''
        Get the reduced density matrix element corresponding to
        many-body states b and bp.
        '''
        return get_phi0(self, b, bp)

    def get_phi1(self, l, c, b):
        '''
        Get the momentum integrated current amplitudes corresponding to
        lead l and many-body states c and b.
        '''
        return get_phi1(self, l, c, b)

    def print_state(self, b, eps=0.0, prntq=True, filename=None, separator=''):
        '''
        Prints properties of given many-body eigenstate of the quantum dot Hamiltonain
        '''
        print_state(self, b, eps, prntq, filename, separator)

    def print_all_states(self, filename, eps=0.0, separator='', mode='w'):
        '''
        Prints properties of all many-body eigenstates to a file.
        '''
        print_all_states(self, filename, eps, separator, mode)

    def remove_states(self, dE):
        '''
        Remove the states with energy dE larger than the ground state
        for the transport calculations.
        '''
        remove_states(self, dE)

    def use_all_states(self):
        '''
        Use all states for the transport calculations.
        '''
        use_all_states(self)

class FunctionProperties(object):
    """
    Class containing miscellaneous variables for Transport and Transport2vN classes.

    Attributes
    ----------
    kerntype : string
        String describing what master equation method to use.
        For Transport class the possible values are 'Pauli', '1vN', 'Redfield', 'pyPauli', 'py1vN'.
        For Transport2vN class the possible values are '2vN' and 'py2vN'.
        The method with 'py' in front are not compiled using cython.
    symq : bool
        For symq=False keep all equations in the kernel, and the matrix is of size N by N+1.
        For symq=True replace one equation by the normalisation condition, and the matrix is square N by N.
    norm_row : int
        If symq=True this row will be replaced by normalisation condition in the kernel matrix.
    solmethod : string
        String specifying the solution method of the equation L(Phi0)=0.
        The possible values are matrix inversion 'solve' and least squares 'lsqr'.
        Method 'solve' works only when symq=True.
        For matrix free methods (used when mfreeq=True) the possible values are
        'krylov', 'broyden', etc.
    itype : int
        Type of integral for first order method calculations.
        itype=0: the principal parts are neglected.
        itype=1: the principal parts are kept, but approximated by digamma function valid for large bandwidht D.
        itype=2: the principal parts are evaluated using Fortran integration package QUADPACK routine dqawc through SciPy.
    dqawc_limit : int
        For itype=2 dqawc_limit determines the maximum number of subintervals
        in the partition of the given integration interval.
    mfreeq : bool
        If mfreeq=True the matrix free solution method is used for first order methods.
    phi0_init : array
        For mfreeq=True the initial value of zeroth order density matrix elements.
    mtype_qd : float or complex
        Type for the many-body quantum dot Hamiltonian matrix.
    mtype_leads : float or complex
        Type for the many-body tunneling matrix Tba.
    Ek_left, Ek_right : int
        Number of points Ek_grid is extended to the left and the right for '2vN' method.
    ht_ker : array
        Kernel used when performing Hilbert transform using FFT.
        It is generated using specfunc.kernel_fredriksen(n).
    emin, emax : float
        Minimal and maximal energy in the updated Ek_grid generated by neumann2py.get_grid_ext(sys).
        Note that emin<=-D and emax>=+D.
    ext_fct : float
        Multiplication factor used in neumann2py.get_grid_ext(sys), when determining emin and emax.
    suppress_err : bool
        Determines whether to print the warning when the inversion of the kernel failed.
    """

    def __init__(self, kerntype='2vN', symq=True, norm_row=0, solmethod='n',
                       itype=2, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                       mtype_qd=float, mtype_leads=complex):
        self.kerntype = kerntype
        self.symq = symq
        self.norm_row = norm_row
        self.solmethod = solmethod
        #
        self.itype = itype
        self.dqawc_limit = dqawc_limit
        #
        self.mfreeq = mfreeq
        self.phi0_init = phi0_init
        #
        self.mtype_qd = mtype_qd
        self.mtype_leads = mtype_leads
        #
        self.Ek_left = 0
        self.Ek_right = 0
        self.ht_ker = None
        #
        self.emin, self.emax = 0, 0
        self.ext_fct = 1.1
        #
        self.suppress_err = False

    def print_error(self, exept):
        if not self.suppress_err:
            print(str(exept))
            print("WARNING: Could not invert the kernel. All the transport channels may be outside the bandwidth."
                 +"This warning will not be shown again.")
            self.suppress_err = True

class Transport(object):
    """
    Class for solving different first order rate models.

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
        Factors used to calculate energy and heat currents in 1vN, Redfield methods.
    """

    def __init__(self, qd=None, leads=None, si=None, funcp=None):
        """Initialization of the Transport class."""
        self.qd = qd
        self.leads = leads
        self.si = si
        self.paulifct = None
        self.phi1fct = None
        self.phi1fct_energy = None
        self.kerntype = None
        self.kern, self.bvec = None, None
        self.sol0, self.phi0, self.phi1 = None, None, None
        #self.current, self.energy_current, self.heat_current = None, None, None
        self.current = np.zeros(self.si.nleads, dtype=doublenp)
        self.energy_current = np.zeros(self.si.nleads, dtype=doublenp)
        self.heat_current = np.zeros(self.si.nleads, dtype=doublenp)
        #
        self.funcp = funcp

    def set_kern(self):
        """Generates the kernel (Liouvillian)."""
        kerntype = self.funcp.kerntype
        if kerntype == 'Pauli':
            self.paulifct = c_generate_paulifct(self)
            self.kern, self.bvec = c_generate_kern_pauli(self)
        elif kerntype == 'Redfield':
            self.phi1fct, self.phi1fct_energy = c_generate_phi1fct(self)
            self.kern, self.bvec = c_generate_kern_redfield(self)
        elif kerntype == '1vN':
            self.phi1fct, self.phi1fct_energy = c_generate_phi1fct(self)
            self.kern, self.bvec = c_generate_kern_1vN(self)
        elif kerntype == 'Lindblad':
            self.tLba = c_generate_tLba(self)
            self.kern, self.bvec = c_generate_kern_lindblad(self)
        elif kerntype == 'pyPauli':
            self.paulifct = generate_paulifct(self)
            self.kern, self.bvec = generate_kern_pauli(self)
        elif kerntype == 'py1vN':
            self.phi1fct, self.phi1fct_energy = generate_phi1fct(self)
            self.kern, self.bvec = generate_kern_1vN(self)
        elif kerntype == 'pyLindblad':
            self.tLba = generate_tLba(self)
            self.kern, self.bvec = generate_kern_lindblad(self)

    def solve_kern(self):
        """Finds the stationary state using least squares or by inverting the matrix."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        if solmethod == 'n': solmethod = 'solve' if symq else 'lsqr'
        if not symq and solmethod != 'lsqr' and solmethod != 'lsmr':
            print("WARNING: Using solmethod=lsqr, because the kernel is not symmetric, symq=False.")
            solmethod = 'lsqr'
        try:
            self.sol0 = [None]
            if not self.kern is None:
                self.mtrmethod = 'dense'
                if self.mtrmethod == 'dense':
                    if   solmethod == 'solve': self.sol0 = [np.linalg.solve(self.kern, self.bvec)]
                    elif solmethod == 'lsqr':  self.sol0 = np.linalg.lstsq(self.kern, self.bvec)
                '''
                #The sparse solution methods are removed
                elif self.mtrmethod == 'sparse':
                    if   solmethod == 'solve':    self.sol0 = [sp.sparse.linalg.spsolve(self.kern, self.bvec)]
                    elif solmethod == 'lsqr':     self.sol0 = sp.sparse.linalg.lsqr(self.kern, self.bvec)
                    elif solmethod == 'lsmr':     self.sol0 = sp.sparse.linalg.lsmr(self.kern, self.bvec)
                    elif solmethod == 'bicg':     self.sol0 = sp.sparse.linalg.bicg(self.kern, self.bvec)
                    elif solmethod == 'bicgstab': self.sol0 = sp.sparse.linalg.bicgstab(self.kern, self.bvec)
                    elif solmethod == 'cgs':      self.sol0 = sp.sparse.linalg.cgs(self.kern, self.bvec)
                    elif solmethod == 'gmres':    self.sol0 = sp.sparse.linalg.gmres(self.kern, self.bvec)
                    elif solmethod == 'lgmres':   self.sol0 = sp.sparse.linalg.lgmres(self.kern, self.bvec)
                    elif solmethod == 'qmr':      self.sol0 = sp.sparse.linalg.qmr(self.kern, self.bvec)
                '''
            else:
                print("WARNING: kern is not generated for calculation of phi0.")
            self.phi0 = self.sol0[0]
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = np.zeros(self.si.ndm0r)
        self.funcp.solmethod = solmethod

    def solve_matrix_free(self):
        """Finds the stationary state using matrix free methods like broyden, krylov, etc."""
        if self.funcp.kerntype in {'Pauli', 'pyPauli'}:
            return 0
        if self.funcp.phi0_init is None:
            print("WARNING: The initial guess phi0_init is not specified.")
            return 0
        solmethod = self.funcp.solmethod
        kerntype = self.funcp.kerntype
        phi0_init = self.funcp.phi0_init
        solmethod = solmethod if solmethod != 'n' else 'krylov'
        try:
            self.sol0 = None
            if kerntype == 'Redfield':
                self.phi1fct, self.phi1fct_energy = c_generate_phi1fct(self)
                self.sol0 = optimize.root(c_generate_vec_redfield, phi0_init, args=(self), method=solmethod)
            elif kerntype == '1vN':
                self.phi1fct, self.phi1fct_energy = c_generate_phi1fct(self)
                self.sol0 = optimize.root(c_generate_vec_1vN, phi0_init, args=(self), method=solmethod)
            elif kerntype == 'Lindblad':
                self.tLba = c_generate_tLba(self)
                self.sol0 = optimize.root(c_generate_vec_lindblad, phi0_init, args=(self), method=solmethod)
            elif kerntype == 'py1vN':
                self.phi1fct, self.phi1fct_energy = generate_phi1fct(self)
                self.sol0 = optimize.root(generate_vec_1vN, phi0_init, args=(self), method=solmethod)
            elif kerntype == 'pyLindblad':
                self.tLba = generate_tLba(self)
                self.sol0 = optimize.root(generate_vec_lindblad, phi0_init, args=(self), method=solmethod)
            #
            if not self.sol0 is None:
                self.phi0 = self.sol0.x
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = np.zeros(self.si.ndm0r)
        self.funcp.solmethod = solmethod

    def calc_current(self):
        """Calculates the current and first order density-matrix elements."""
        # First order approaches
        kerntype = self.funcp.kerntype
        if self.phi0 is None:
            print("WARNING: phi0 is not calculated yet for calculation of current.")
            return 0
        if kerntype == 'Pauli':
            self.current, self.energy_current = c_generate_current_pauli(self)
        elif kerntype == 'Redfield':
            self.phi1, self.current, self.energy_current = c_generate_phi1_redfield(self)
        elif kerntype == '1vN':
            self.phi1, self.current, self.energy_current = c_generate_phi1_1vN(self)
        elif kerntype == 'Lindblad':
            self.current, self.energy_current = c_generate_current_lindblad(self)
        elif kerntype == 'pyPauli':
            self.current, self.energy_current = generate_current_pauli(self)
        elif kerntype == 'py1vN':
            self.phi1, self.current, self.energy_current = generate_phi1_1vN(self)
        elif kerntype == 'pyLindblad':
            self.current, self.energy_current = generate_current_lindblad(self)
        self.current, self.energy_current = self.current.real, self.energy_current.real
        self.heat_current = self.energy_current - self.leads.mulst*self.current

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
            if self.funcp.mfreeq:
                self.solve_matrix_free()
            else:
                self.set_kern()
                self.solve_kern()
            if currentq:
                self.calc_current()

class Transport2vN(object):
    """
    Class for solving second von Neuman (2vN) approach for stationary state.
    Most of the attributes are the same as in Transpot class.

    Attributes
    ----------
    kern : array
        Kernel for zeroth order density matrix elements Phi[0].
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

    def __init__(self, qd=None, leads=None, si=None, funcp=None, Ek_grid=None):
        self.qd = qd
        self.leads = leads
        self.si = si
        self.Ek_grid = Ek_grid
        self.Ek_grid_ext = np.zeros(0, dtype=doublenp)
        self.fkp, self.fkm = None, None
        self.hfkp, self.hfkm = None, None
        self.restart()
        self.funcp = funcp
        # Some exceptions
        if self.si is None:
            pass
        elif type(self.si).__name__ != 'StateIndexingDMc':
            raise TypeError('The state indexing class for 2vN approach has to be StateIndexingDMc')

    def restart(self):
        """Restart values of some variables for new calculations."""
        self.kern, self.bvec = None, None
        self.sol0, self.phi0, self.phi1 = None, None, None
        self.current = None #np.zeros(self.si.nleads, dtype=doublenp) if not self.si is None else None
        self.energy_current = None #np.zeros(self.si.nleads, dtype=doublenp) if not self.si is None else None
        self.heat_current = None #np.zeros(self.si.nleads, dtype=doublenp) if not self.si is None else None
        #
        self.niter = -1
        self.phi1k = None
        self.phi1k_delta = None
        self.hphi1k_delta = None
        self.kern1k = None
        #
        self.iters = []

    def solve_kern(self):
        """Finds the stationary state using least squares or by inverting the matrix."""
        solmethod = self.funcp.solmethod
        symq = self.funcp.symq
        if solmethod == 'n': solmethod = 'solve' if symq else 'lsqr'
        if not symq and solmethod != 'lsqr':
            print("WARNING: Using solmethod=lsqr, because the kernel is not symmetric, symq=False.")
            solmethod = 'lsqr'
        try:
            self.sol0 = [None]
            if   solmethod == 'solve': self.sol0 = [np.linalg.solve(self.kern, self.bvec)]
            elif solmethod == 'lsqr':  self.sol0 = np.linalg.lstsq(self.kern, self.bvec)
            self.phi0 = self.sol0[0]
        except Exception as exept:
            self.funcp.print_error(exept)
            self.phi0 = np.zeros(self.si.ndm0)
        self.funcp.solmethod = solmethod

    def iterate(self):
        """Makes one iteration for solution of the 2vN integral equation."""
        kerntype = self.funcp.kerntype
        if kerntype == '2vN':
            self.phi1k_delta, self.hphi1k_delta, self.kern1k = c_iterate_2vN(self)
            self.phi1k = self.phi1k_delta if self.phi1k is None else self.phi1k + self.phi1k_delta
            self.phi1_phi0, self.e_phi1_phi0 = c_get_phi1_phi0_2vN(self)
        elif kerntype == 'py2vN':
            self.phi1k_delta, self.hphi1k_delta, self.kern1k = iterate_2vN(self)
            self.phi1k = self.phi1k_delta if self.phi1k is None else self.phi1k + self.phi1k_delta
            self.phi1_phi0, self.e_phi1_phi0 = get_phi1_phi0_2vN(self)
        self.niter += 1
        self.kern, self.bvec = kern_phi0_2vN(self)
        self.solve_kern()
        self.phi1, self.current, self.energy_current = generate_current_2vN(self)
        self.heat_current = self.energy_current - self.leads.mulst*self.current
        #
        self.iters.append(Iterations2vN(self))


    def save_data(self, dir_name, savekq=False):
        """
        Saves the data for given iteration niter.

        Parameters
        ----------
        dir_name : 'string'
            Directory for the data.
        savekq : bool
            If savekq=True save the energy resolved variables
            kern1k, phi1k, phi1k_delta, hphi1k_delta.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if savekq:
            np.savez_compressed(dir_name+'iter'+str(self.niter),
                                Ek_grid=self.Ek_grid,
                                Ek_grid_ext=self.Ek_grid_ext,
                                kern1k=self.kern1k,
                                phi1k=self.phi1k,
                                phi1k_delta=self.phi1k_delta,
                                hphi1k_delta=self.hphi1k_delta,
                                phi0=self.phi0,
                                phi1=self.phi1,
                                current=self.current,
                                energy_current=self.energy_current,
                                heat_current=self.heat_current,
                                niter=self.niter)
        else:
            np.savez_compressed(dir_name+'iter'+str(self.niter),
                                phi0=self.phi0,
                                phi1=self.phi1,
                                current=self.current,
                                energy_current=self.energy_current,
                                heat_current=self.heat_current,
                                niter=self.niter)

    def get_data(self, dir_name, it):
        """
        Load the data for given iteration it.

        Parameters
        ----------
        dir_name : 'string'
            Directory for the data.
        it : int
            Number of the iteration.
        """
        npzfile = np.load(dir_name+'iter'+str(it)+'.npz')
        try:
            self.Ek_grid = npzfile['Ek_grid']
            self.Ek_grid_ext = npzfile['Ek_grid_ext']
            self.kern1k = npzfile['kern1k']
            self.phi1k = npzfile['phi1k']
            self.phi1k_delta = npzfile['phi1k_delta']
            self.hphi1k_delta = npzfile['hphi1k_delta']
        except:
            self.Ek_grid = None
            self.Ek_grid_ext = None
            self.kern1k = None
            self.phi1k = None
            self.phi1k_delta = None
            self.hphi1k_delta = None
        self.phi0 = npzfile['phi0']
        self.phi1 = npzfile['phi1']
        self.current = npzfile['current']
        self.energy_current = npzfile['energy_current']
        self.heat_current = npzfile['heat_current']
        self.niter = npzfile['niter']

    def solve(self, qdq=True, rotateq=True, masterq=True, saveq=False, savekq=False, restartq=True,
                    niter=None, dir_name='./iterations/', *args, **kwargs):
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
        saveq : bool
            Save the data after each iteration in the directory dir_name.
        savekq : bool
            Also save the energy resolved variables.
        restartq : bool
            Call restart() and erase old values of variables.
            To continue from the last iteration set restarq=False.
        niter : int
            Number of iterations to perform.
        dir_name : 'string'
            Directory for the data.
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
            #
            for it in range(niter):
                #print(self.niter+1)
                self.iterate()
                if saveq: self.save_data(dir_name, savekq)
                #print(self.current)

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
