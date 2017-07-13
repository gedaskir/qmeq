"""Module containing Builder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from .aprclass import Approach
from .indexing import StateIndexingDM
from .indexing import StateIndexingDMc
from .qdot import QuantumDot
from .leadstun import LeadsTunneling

from .various import get_phi0
from .various import get_phi1
from .various import print_state
from .various import print_all_states
from .various import sort_eigenstates
from .various import remove_coherences
from .various import remove_states
from .various import use_all_states

#-----------------------------------------------------------
# Python modules

from .approach.pauli import Approach_pyPauli
from .approach.lindblad import Approach_pyLindblad
from .approach.redfield import Approach_pyRedfield
from .approach.neumann1 import Approach_py1vN
from .approach.neumann2 import Approach_py2vN

# Cython compiled modules

try:
    from .approach.c_pauli import Approach_Pauli
    from .approach.c_lindblad import Approach_Lindblad
    from .approach.c_redfield import Approach_Redfield
    from .approach.c_neumann1 import Approach_1vN
    from .approach.c_neumann2 import Approach_2vN
except:
    print("Cannot import Cython compiled modules for the approaches.")
    Approach_Pauli = Approach_pyPauli
    Approach_Lindblad = Approach_pyLindblad
    Approach_Redfield = Approach_pyRedfield
    Approach_1vN = Approach_py1vN
    Approach_2vN = Approach_py2vN
#-----------------------------------------------------------

class Builder(object):
    """
    Class for building the system for stationary transport calculations.

    Attributes
    ----------
    nsingle : int
        Number of single-particle states.
    hsingle : dict, list, or array
        Dictionary, list, or array corresponding to single-particle hopping (tunneling) Hamiltonian.
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
    tleads : dict, list, or array
        Dictionary, list, or numpy array defining single particle tunneling amplitudes.
        numpy array has to be nleads by nsingle.
    mulst : dict, list, or array
        Dictionary, list, or array containing chemical potentials of the leads.
    tlst : dict, list, or array
        Dictionary, list, or array containing temperatures of the leads.
    dband : float, dict, list, or array
        Float, dictionary, list, or array determining bandwidths of the leads.
    indexing : str
        String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        Note that 'sz' indexing for Fock states is used for 'ssq' indexing, with
        additional specification of eigensates in Fock basis.
    symmetry : str
        String determining that the states will be augmented by the symmetry.
        Possible value is 'spin'.
    kpnt : int
        Number of energy grid points on which 2vN approach equations are solved.
    kerntype : string, Approach class
        String describing what master equation approach to use.
        For Approach class the possible values are 'Pauli', '1vN', 'Redfield', 'Lindblad', \
                                                   'pyPauli', 'py1vN', 'pyRedfield', 'pyLindblad'.
        For Approach2vN class the possible values are '2vN' and 'py2vN'.
        The approaches with 'py' in front are not compiled using cython.
        kerntype can also be an Approach class defining a custom approach.
    symq : bool
        For symq=False keep all equations in the kernel, and the matrix is of size N by N+1.
        For symq=True replace one equation by the normalisation condition,
        and the matrix is square N by N.
    norm_row : int
        If symq=True this row will be replaced by normalisation condition in the kernel matrix.
    solmethod : string
        String specifying the solution method of the equation L(Phi0)=0.
        The possible values are matrix inversion 'solve' and least squares 'lsqr'.
        Method 'solve' works only when symq=True.
        For matrix free methods (used when mfreeq=True) the possible values are
        'krylov', 'broyden', etc.
    itype : int
        Type of integral for first order approach calculations.
        itype=0: the principal parts are evaluated using Fortran integration package QUADPACK \
                 routine dqawc through SciPy.
        itype=1: the principal parts are kept, but approximated by digamma function valid for \
                 large bandwidht D.
        itype=2: the principal parts are neglected.
        itype=3: the principal parts are neglected and infinite bandwidth D is assumed.
    dqawc_limit : int
        For itype=0 dqawc_limit determines the maximum number of subintervals
        in the partition of the given integration interval.
    mfreeq : bool
        If mfreeq=True the matrix free solution method is used for first order methods.
    phi0_init : array
        For mfreeq=True the initial value of zeroth order density matrix elements.
    mtype_qd : float or complex
        Type for the many-body quantum dot Hamiltonian matrix.
    mtype_leads : float or complex
        Type for the many-body tunneling matrix Tba.
    symmetry : str
        String determining if the states will be augmented by a symmetry.
        Possible value is 'spin'.
    herm_hs : bool
        When herm_hs=True the conjugated elements are added automatically to many-body Hamiltonian
        from hsingle. For example, when hsingle={(0,1): val} the element {(1,0): val.conjugate()}
        will also be added.
    herm_c : bool
        When herm_c=True the conjugated elements are added automatically to many-body Hamiltonian
        from coulomb. For example, when coulomb={(0,2,3,1): val} the element
        {(1,3,2,0): val.conjugate()} will also be added. However, we note that for m_less_n=True
        if coulomb={(0,2,1,3): val} the element (3,1,2,0) will not be included.
    m_less_n : bool
        When m_less_n=True the coulomb matrix element (m, n, k, l) has to have m<n.
    qd : QuantumDot
        QuantumDot object.
    leads : LeadsTunneling
        LeadsTunneling object.
    si : StateIndexingDM
        StateIndexingDM object.
    appr : Approach or Approach2vN.
        Approach or Approach2vN object.
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

    def __init__(self, nsingle=0, hsingle={}, coulomb={},
                       nleads=0, tleads={}, mulst={}, tlst={}, dband={},
                       indexing='n', kpnt=None,
                       kerntype='Pauli', symq=True, norm_row=0, solmethod='n',
                       itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                       mtype_qd=complex, mtype_leads=complex,
                       symmetry='n', herm_hs=True, herm_c=False, m_less_n=True):

        if indexing is 'n':
            if symmetry is 'spin' and kerntype not in {'py2vN', '2vN'}:
                indexing = 'ssq'
            else:
                indexing = 'charge'

        if not indexing in {'Lin', 'charge', 'sz', 'ssq'}:
            print("WARNING: Allowed indexing values are: \'Lin\', \'charge\', \'sz\', \'ssq\'. "+
                  "Using default indexing=\'charge\'.")
            indexing = 'charge'

        if not itype in {0,1,2,3}:
            print("WARNING: itype needs to be 0, 1, 2, or 3. Using default itype=0.")
            itype = 0

        if isinstance(kerntype, str):
            if not kerntype in {'Pauli', 'Lindblad', 'Redfield', '1vN', '2vN',
                                'pyPauli', 'pyLindblad', 'pyRedfield', 'py1vN', 'py2vN'}:
                print("WARNING: Allowed kerntype values are: "+
                      "\'Pauli\', \'Lindblad\', \'Redfield\', \'1vN\', \'2vN\', "+
                      "\'pyPauli\', \'pyLindblad\', \'pyRedfield\', \'py1vN\', \'py2vN\'. "+
                      "Using default kerntype=\'Pauli\'.")
                kerntype = 'Pauli'
            self.Approach = globals()['Approach_'+kerntype]
        else:
            if issubclass(kerntype, Approach):
                self.Approach = kerntype
                kerntype = self.Approach.kerntype

        if not indexing in {'Lin', 'charge'} and kerntype in {'py2vN', '2vN'}:
            print("WARNING: For 2vN approach indexing needs to be \'Lin\' or \'charge\'. "+
                  "Using indexing=\'charge\' as a default.")
            indexing = 'charge'

        # Make copies of initialized parameters.
        hsingle = copy.deepcopy(hsingle)
        coulomb = copy.deepcopy(coulomb)
        tleads = copy.deepcopy(tleads)
        mulst = copy.deepcopy(mulst)
        tlst = copy.deepcopy(tlst)
        dband = copy.deepcopy(dband)
        phi0_init = copy.deepcopy(phi0_init)

        self.funcp = FunctionProperties(symq=symq, norm_row=norm_row, solmethod=solmethod,
                                        itype=itype, dqawc_limit=dqawc_limit,
                                        mfreeq=mfreeq, phi0_init=phi0_init,
                                        mtype_qd=mtype_qd, mtype_leads=mtype_leads,
                                        kpnt=kpnt, dband=dband)

        icn = self.Approach.indexing_class_name
        self.si = globals()[icn](nsingle, indexing, symmetry)
        self.qd = QuantumDot(hsingle, coulomb, self.si, herm_hs, herm_c, m_less_n, mtype_qd)
        self.leads = LeadsTunneling(nleads, tleads, self.si, mulst, tlst, dband, mtype_leads)

        self.appr = self.Approach(self)

    def __getattr__(self, item):
        if item in {'solve', 'current', 'energy_current', 'heat_current', 'phi0', 'phi1',
                    'niter', 'iters', 'kern'}:
            return getattr(self.appr, item)
        elif item in {'kpnt', 'symq', 'norm_row', 'solmethod', 'itype', 'dqawc_limit',
                      'mfreeq', 'phi0_init'}:
            return getattr(self.funcp, item)
        elif item in {'tleads', 'mulst', 'tlst', 'dlst', 'Tba'}:
            return getattr(self.leads, item)
        elif item in {'nsingle', 'nleads'}:
            return getattr(self.si, item)
        elif item in {'hsingle', 'coulomb', 'Ea'}:
            return getattr(self.qd, item)
        else:
            #return self.__getattribute__(item)
            return super(Builder, self).__getattribute__(item)

    def __setattr__(self, item, value):
        if item in {'solve', 'current', 'energy_current', 'heat_current', 'phi0', 'phi1',
                    'niter', 'iters', 'kern'}:
            setattr(self.appr, item, value)
        elif item in {'kpnt', 'symq', 'norm_row', 'solmethod', 'itype', 'dqawc_limit',
                      'mfreeq', 'phi0_init'}:
            setattr(self.funcp, item, value)
        elif item in {'tleads', 'mulst', 'tlst', 'dlst', 'Tba'}:
            setattr(self.leads, item, value)
        elif item in {'nsingle', 'nleads'}:
            setattr(self.si, item, value)
        elif item in {'hsingle', 'coulomb', 'Ea'}:
            setattr(self.qd, item, value)
        else:
            super(Builder, self).__setattr__(item, value)

    def change_si(self):
        si = self.si
        icn = self.Approach.indexing_class_name
        if not isinstance(si, globals()[icn]):
            self.si = globals()[icn](si.nsingle, si.indexing, si.symmetry, si.nleads)
            self.qd.si = self.si
            self.leads.si = self.si

    # kerntype
    def get_kerntype(self):
        return self.appr.kerntype
    def set_kerntype(self, value):
        if isinstance(value, str):
            if self.appr.kerntype != value:
                self.Approach = globals()['Approach_'+value]
                self.change_si()
                self.appr = self.Approach(self)
        else:
            if issubclass(value, Approach):
                self.Approach = value
                self.change_si()
                self.appr = self.Approach(self)
    kerntype = property(get_kerntype, set_kerntype)

    # indexing
    def get_indexing(self):
        return self.si.indexing
    def set_indexing(self, value):
        if self.si.indexing != value:
            print('WARNING: Cannot change indexing from \''+self.si.indexing+'\' to \''+value
                  +'\' Consider contructing a new system using \'indexing='+value+'\'')
    indexing = property(get_indexing, set_indexing)

    # dband
    def get_dband(self):
        return self.funcp.dband
    def set_dband(self, value):
        self.funcp.dband = value
        self.leads.change(dlst=dband)
    dband = property(get_dband, set_dband)

    def add(self, hsingle=None, coulomb=None, tleads=None, mulst=None, tlst=None, dlst=None):
        """
        Adds the values to the specified dictionaries and correspondingly redefines
        relevant many-body properties of the system.

        Parameters
        ----------
        hsingle,coulomb,tleads,mulst,tlst,dlst : dict
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
        hsingle,coulomb,tleads,mulst,tlst,dlst : dict
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
        Get the energy integrated current amplitudes corresponding to
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

    def sort_eigenstates(self, srt='n'):
        '''
        Sort many-body states of the system by given order of properties.
        '''
        sort_eigenstates(self, srt=srt)

    def remove_coherences(self, dE):
        '''
        Remove the coherences with energy difference larger than dE.
        '''
        remove_coherences(self, dE)

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
    Class containing miscellaneous variables for Approach and Approach2vN classes.

    Attributes
    ----------
    symq : bool
        For symq=False keep all equations in the kernel, and the matrix is of size N by N+1.
        For symq=True replace one equation by the normalisation condition,
        and the matrix is square N by N.
    norm_row : int
        If symq=True this row will be replaced by normalisation condition in the kernel matrix.
    solmethod : string
        String specifying the solution method of the equation L(Phi0)=0.
        The possible values are matrix inversion 'solve' and least squares 'lsqr'.
        Method 'solve' works only when symq=True.
        For matrix free methods (used when mfreeq=True) the possible values are
        'krylov', 'broyden', etc.
    itype : int
        Type of integral for first order approach calculations.
        itype=0: the principal parts are evaluated using Fortran integration package QUADPACK \
                 routine dqawc through SciPy.
        itype=1: the principal parts are kept, but approximated by digamma function valid for \
                 large bandwidht D.
        itype=2: the principal parts are neglected.
        itype=3: the principal parts are neglected and infinite bandwidth D is assumed.
    dqawc_limit : int
        For itype=0 dqawc_limit determines the maximum number of subintervals
        in the partition of the given integration interval.
    mfreeq : bool
        If mfreeq=True the matrix free solution method is used for first order methods.
    phi0_init : array
        For mfreeq=True the initial value of zeroth order density matrix elements.
    mtype_qd : float or complex
        Type for the many-body quantum dot Hamiltonian matrix.
    mtype_leads : float or complex
        Type for the many-body tunneling matrix Tba.
    kpnt_left, kpnt_right : int
        Number of points Ek_grid is extended to the left and the right for '2vN' approach.
    ht_ker : array
        Kernel used when performing Hilbert transform using FFT.
        It is generated using specfunc.kernel_fredriksen(n).
    emin, emax : float
        Minimal and maximal energy in the updated Ek_grid generated by neumann2py.get_grid_ext(sys).
        Note that emin<=Dmin and emax>=Dmax.
    dmin, dmax : float
        Bandedge Dmin and Dmax values of the lead electrons.
    ext_fct : float
        Multiplication factor used in neumann2py.get_grid_ext(sys), when determining emin and emax.
    suppress_err : bool
        Determines whether to print the warning when the inversion of the kernel failed.
    """

    def __init__(self, kerntype='2vN', symq=True, norm_row=0, solmethod='n',
                       itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                       mtype_qd=float, mtype_leads=complex, kpnt=None, dband=None):
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
        self.kpnt = kpnt
        self.dband = dband
        #
        self.kpnt_left = 0
        self.kpnt_right = 0
        self.ht_ker = None
        #
        self.dmin, self.dmax = 0, 0
        self.emin, self.emax = 0, 0
        self.ext_fct = 1.1
        #
        self.suppress_err = False
        self.suppress_wrn = [False]

    def print_error(self, exept):
        if not self.suppress_err:
            print(str(exept))
            print("WARNING: Could not invert the kernel. "+
                  "All the transport channels may be outside the bandwidth. "+
                  "This warning will not be shown again.")
            self.suppress_err = True

    def print_warning(self, i, message):
        if not self.suppress_wrn[i]:
            print(message)
            self.suppress_wrn[i] = True
