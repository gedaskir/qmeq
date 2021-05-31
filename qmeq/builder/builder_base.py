"""Module containing BuilderBase and BuilderManyBody classes."""

import copy
import numpy as np

from ..wrappers.mytypes import longnp

from ..approach.aprclass import Approach
from ..indexing import StateIndexingDM
from ..indexing import StateIndexingDMc
from ..qdot import QuantumDot
from ..leadstun import LeadsTunneling
from .funcprop import FunctionProperties

from .various import get_phi0
from .various import get_phi1
from .various import print_state
from .various import print_all_states
from .various import sort_eigenstates
from .various import remove_coherences
from .various import remove_states
from .various import use_all_states

from .validation import validate_kerntype
from .validation import validate_itype
from .validation import validate_indexing

# -----------------------------------------------------------
# Python modules

from ..approach.base.pauli import ApproachPauli as ApproachPyPauli
from ..approach.base.lindblad import ApproachLindblad as ApproachPyLindblad
from ..approach.base.redfield import ApproachRedfield as ApproachPyRedfield
from ..approach.base.neumann1 import Approach1vN as ApproachPy1vN
from ..approach.base.neumann2 import Approach2vN as ApproachPy2vN
from ..approach.base.RTD import ApproachPyRTD as ApproachPyRTD

# Cython compiled modules

try:
    from ..approach.base.c_pauli import ApproachPauli
    from ..approach.base.c_lindblad import ApproachLindblad
    from ..approach.base.c_redfield import ApproachRedfield
    from ..approach.base.c_neumann1 import Approach1vN
    from ..approach.base.c_neumann2 import Approach2vN
    from ..approach.base.c_RTD import ApproachRTD
except ImportError as ie:
    print("WARNING: Cannot import Cython compiled modules for the approaches (builder_base.py).")
    ApproachPauli = ApproachPyPauli
    ApproachLindblad = ApproachPyLindblad
    ApproachRedfield = ApproachPyRedfield
    Approach1vN = ApproachPy1vN
    Approach2vN = ApproachPy2vN
    ApproachRTD = ApproachPyRTD
# -----------------------------------------------------------

attribute_map = dict(
    # StateIndexing
    nsingle='si', nleads='si',
    # QuantumDot
    hsingle='qd', coulomb='qd', Ea='qd',
    # LeadsTunneling
    tleads='leads', mulst='leads', tlst='leads', dlst='leads',
    Tba='leads',
    # Approach
    solve='appr', current='appr', energy_current='appr',
    heat_current='appr', phi0='appr', phi1='appr', niter='appr',
    iters='appr', kern='appr', success='appr', make_kern_copy='appr',
    # FunctionProperties
    kpnt='funcp', symq='appr', norm_row='appr', solmethod='appr',
    itype='appr', dqawc_limit='funcp',
    mfreeq='appr', phi0_init='funcp', off_diag_corrections='funcp'
    )


class ModelParameters(object):

    def __init__(self, params):
        for i in params:
            if i not in {'self'}:
                setattr(self, i, copy.deepcopy(params[i]))


class BuilderBase(object):
    """
    Class for building the system for stationary transport calculations.

    For descriptions of all attributes use help(Builder).
    """

    def __init__(self,
                 nsingle=0, hsingle={}, coulomb={},
                 nleads=0, tleads={}, mulst={}, tlst={}, dband={},
                 indexing=None, kpnt=None,
                 kerntype='Pauli', symq=True, norm_row=0, solmethod=None,
                 itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                 mtype_qd=complex, mtype_leads=complex,
                 symmetry=None, herm_hs=True, herm_c=False, m_less_n=True):

        self._init_copy_data(locals())
        self._init_validate_data()
        self._init_set_globals()
        self._init_set_approach_class()
        self._init_create_setup()
        self._init_create_appr()

    def _init_copy_data(self, data):
        self.data = ModelParameters(data)

    def _init_validate_data(self):
        data = self.data
        data.itype = validate_itype(data.itype, data.kerntype)
        data.kerntype = validate_kerntype(data.kerntype)
        data.indexing, data.symmetry = validate_indexing(data.indexing,
                                          data.symmetry,
                                          data.kerntype)

    def _init_set_globals(self):
        self.globals = globals()

    def _init_set_approach_class(self):
        kerntype = self.data.kerntype
        if isinstance(kerntype, str):
            approach_string = kerntype[0].capitalize() + kerntype[1:]
            self.Approach = self.globals['Approach'+approach_string]
        elif issubclass(kerntype, Approach):
            self.Approach = kerntype
            self.kerntype = self.Approach.kerntype

    def _init_create_setup(self):
        data = self.data
        self.funcp = FunctionProperties(symq=data.symq, norm_row=data.norm_row, solmethod=data.solmethod,
                                        itype=data.itype, dqawc_limit=data.dqawc_limit,
                                        mfreeq=data.mfreeq, phi0_init=data.phi0_init,
                                        mtype_qd=data.mtype_qd, mtype_leads=data.mtype_leads,
                                        kpnt=data.kpnt, dband=data.dband)

        icn = self.Approach.indexing_class_name
        self.si = self.globals[icn](data.nsingle, data.indexing, data.symmetry)
        self.qd = QuantumDot(data.hsingle, data.coulomb, self.si,
                             data.herm_hs, data.herm_c, data.m_less_n, data.mtype_qd)
        self.leads = LeadsTunneling(data.nleads, data.tleads, self.si,
                                    data.mulst, data.tlst, data.dband, data.mtype_leads)

    def _init_create_appr(self):
        self.appr = self.Approach(self)

    def __getattr__(self, item):
        sub_class_str = attribute_map.get(item)
        if sub_class_str is None:
            return super(BuilderBase, self).__getattribute__(item)
        else:
            sub_class = getattr(self, attribute_map[item])
            return getattr(sub_class, item)

    def __setattr__(self, item, value):
        sub_class_str = attribute_map.get(item)
        if sub_class_str is None:
            super(BuilderBase, self).__setattr__(item, value)
        else:
            sub_class = getattr(self, attribute_map[item])
            setattr(sub_class, item, value)

    def change_si(self):
        si = self.si
        icn = self.Approach.indexing_class_name
        # noinspection PyTypeHints
        if not isinstance(si, self.globals[icn]):
            self.si = self.globals[icn](si.nsingle, si.indexing, si.symmetry, si.nleads)
            self.qd.si = self.si
            self.leads.si = self.si
        self.si.states_changed = True

    # kerntype
    def get_kerntype(self):
        return self.appr.kerntype

    def set_kerntype(self, value):
        if isinstance(value, str):
            if self.appr.kerntype != value:
                approach_string = value[0].capitalize() + value[1:]
                self.Approach = self.globals['Approach'+approach_string]
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
                  + '\' Consider constructing a new system using \'indexing='+value+'\'')
    indexing = property(get_indexing, set_indexing)

    # dband
    def get_dband(self):
        return self.funcp.dband

    def set_dband(self, value):
        self.funcp.dband = value
        self.leads.change(dlst=value)
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
        """
        Get the reduced density matrix element corresponding to
        many-body states b and bp.
        """
        return get_phi0(self, b, bp)

    def get_phi1(self, l, c, b):
        """
        Get the energy integrated current amplitudes corresponding to
        lead l and many-body states c and b.
        """
        return get_phi1(self, l, c, b)

    def print_state(self, b, eps=0.0, prntq=True, filename=None, separator=''):
        """
        Prints properties of given many-body eigenstate of the quantum dot Hamiltonian
        """
        print_state(self, b, eps, prntq, filename, separator)

    def print_all_states(self, filename, eps=0.0, separator='', mode='w'):
        """
        Prints properties of all many-body eigenstates to a file.
        """
        print_all_states(self, filename, eps, separator, mode)

    def sort_eigenstates(self, srt=None):
        """
        Sort many-body states of the system by given order of properties.
        """
        sort_eigenstates(self, srt=srt)

    def remove_coherences(self, dE):
        """
        Remove the coherences with energy difference larger than dE.
        """
        remove_coherences(self, dE)

    def remove_states(self, dE):
        """
        Remove the states with energy dE larger than the ground state
        for the transport calculations.
        """
        remove_states(self, dE)

    def remove_fock_states(self, lin_state_indices):
        """Remove Fock states."""
        self.si.remove_fock_states(lin_state_indices)
        self.qd._init_hamiltonian()
        self.leads._init_coupling()

    def use_all_states(self):
        """
        Use all states for the transport calculations.
        """
        use_all_states(self)


class BuilderManyBody(BuilderBase):
    """
    Class for building the system for stationary transport calculations,
    using many-body states as an input.

    For missing descriptions of attributes use help(Builder).

    Attributes
    ----------
    Ea : array
        nmany by 1 array containing many-body Hamiltonian eigenvalues.
    Na : array
        nmany by 1 array containing particle numbers of many-body states.
    Tba : array
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix.
    """

    def __init__(self,
                 Ea=None, Na=[0], Tba=None,
                 mulst={}, tlst={}, dband={}, kpnt=None,
                 kerntype='Pauli', symq=True, norm_row=0, solmethod=None,
                 itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                 mtype_qd=complex, mtype_leads=complex,
                 symmetry=None, herm_hs=True, herm_c=False, m_less_n=True):

        nleads = Tba.shape[0] if Tba is not None else 0

        # noinspection PyPep8
        BuilderBase.__init__(self,
            nleads=nleads, mulst=mulst, tlst=tlst, dband=dband, kpnt=kpnt,
            kerntype=kerntype, symq=symq, norm_row=norm_row, solmethod=solmethod,
            itype=itype, dqawc_limit=dqawc_limit, mfreeq=mfreeq, phi0_init=phi0_init,
            mtype_qd=mtype_qd, mtype_leads=mtype_leads,
            symmetry=symmetry, herm_hs=herm_hs, herm_c=herm_c, m_less_n=m_less_n,
            indexing='charge')

        self._init_state_indexing(Na, Ea)

        self.qd.Ea = Ea
        self.leads.Tba = Tba

    def _init_state_indexing(self, Na, Ea):
        Na = np.array(Na, dtype=int)
        nmin, nmax = Na.min(), Na.max()
        ncharge = nmax-nmin+1
        nmany = len(Ea) if Ea is not None else 0

        statesdm = [[] for _ in range(ncharge)]
        for i in range(nmany):
            statesdm[Na[i]].append(i)

        self.Na = Na
        self.si.nmany = nmany
        self.si.ncharge = ncharge
        self.si.shiftlst0 = np.zeros(ncharge+1, dtype=longnp)
        self.si.shiftlst1 = np.zeros(ncharge, dtype=longnp)
        self.si.lenlst = np.zeros(ncharge, dtype=longnp)
        self.si.dictdm = np.zeros(nmany, dtype=longnp)
        self.si.set_statesdm(statesdm)
