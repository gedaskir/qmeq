"""Module containing Builder_elph and Builder_many_body_elph classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy

from .builder_base import Builder_base
from .builder_base import attribute_map
from .builder_base import Builder_many_body
from ..aprclass import Approach
from ..indexing import StateIndexingDM
from ..indexing import StateIndexingDMc
from ..qdot import QuantumDot
from ..leadstun import LeadsTunneling
from ..baths import PhononBaths
from .funcprop import FunctionProperties

from .validation import validate_itype_ph

#-----------------------------------------------------------
# Python modules

from ..approach.elph.pauli import Approach_pyPauli
from ..approach.elph.lindblad import Approach_pyLindblad
from ..approach.elph.neumann1 import Approach_py1vN
from ..approach.elph.redfield import Approach_pyRedfield
from ..approach.base.neumann2 import Approach_py2vN

# Cython compiled modules

try:
    from ..approach.elph.c_pauli import Approach_Pauli
    from ..approach.elph.c_lindblad import Approach_Lindblad
    from ..approach.elph.c_redfield import Approach_Redfield
    from ..approach.elph.c_neumann1 import Approach_1vN
    from ..approach.base.c_neumann2 import Approach_2vN
except:
    print("WARNING: Cannot import Cython compiled modules for the approaches (builder_elph.py).")
    Approach_Pauli = Approach_pyPauli
    Approach_Lindblad = Approach_pyLindblad
    Approach_Redfield = Approach_pyRedfield
    Approach_1vN = Approach_py1vN
    Approach_2vN = Approach_py2vN
#-----------------------------------------------------------

attribute_map_elph = {
    # StateIndexing
    'nbaths':'si',
    # PhononBaths
    'velph':'baths', 'tlst_ph':'baths', 'dlst_ph':'baths',
    'Vbbp':'baths', 'bath_func':'baths',
    # FunctionProperties
    'itype_ph':'funcp', 'eps_elph':'funcp'
    }
attribute_map.update(attribute_map_elph)

class Builder_elph(Builder_base):
    '''
    Class for building the system for stationary transport calculations
    with Electron-Phonon (elph) coupling.

    For missing descriptions of attributes use help(Builder).

    Attributes
    ----------
    nbaths : int
        Number of the phonon baths.
    velph : dict, list, or array
        Dictionary, list, or numpy array defining single particle tunneling amplitudes.
        numpy array has to be nbaths by nsingle by nsingle.
    tlst_ph : dict, list, or array
        Dictionary, list, or array containing temperatures of the phonon baths.
    dband_ph : dict, list, or array
        Dictionary, list, or array determining bandwidths of the phonon baths.
    bath_func : list
        List of length nbaths containing density of states functions for the phonon baths.
    eps_elph : float
        Small parameter which stabilizes the integration for integrand with a Bose function.
    '''

    def __init__(self, nsingle=0, hsingle={}, coulomb={},
                       nleads=0, tleads={}, mulst={}, tlst={}, dband={},
                       nbaths=0, velph={}, tlst_ph={}, dband_ph={},
                       indexing='n', kpnt=None,
                       kerntype='Pauli', symq=True, norm_row=0, solmethod='n',
                       itype=0, itype_ph=0, dqawc_limit=10000,
                       mfreeq=False, phi0_init=None,
                       mtype_qd=complex, mtype_leads=complex,
                       symmetry='n', herm_hs=True, herm_c=False, m_less_n=True,
                       bath_func=None, eps_elph=1.0e-6):

        self._init_copy_data(locals())
        self._init_validate_data()
        self._init_set_globals()
        self._init_set_approach_class()
        self._init_create_setup()
        self._init_create_appr()

    def _init_validate_data(self):
        Builder_base._init_validate_data(self)
        data = self.data
        data.itype_ph = validate_itype_ph(data.itype_ph)

    def _init_set_globals(self):
        self.globals = globals()

    def _init_create_setup(self):
        Builder_base._init_create_setup(self)
        data = self.data

        self.funcp.itype_ph = data.itype_ph
        self.funcp.eps_elph = data.eps_elph

        self.baths = PhononBaths(data.nbaths, data.velph, self.si,
                                 data.tlst_ph, data.dband_ph, data.bath_func)

        self.create_si_elph()

    def change_si(self):
        Builder_base.change_si(self)
        self.create_si_elph()

    def create_si_elph(self):
        si = self.si
        si_elph = StateIndexingDMc(si.nsingle, si.indexing,
                                   si.symmetry, si.nleads)
        si_elph.nbaths = si.nbaths
        self.si_elph = si_elph

    def add(self, hsingle=None, coulomb=None, tleads=None, mulst=None, tlst=None, dlst=None,
                  velph=None, tlst_ph=None, dlst_ph=None):
        Builder_base.add(self, hsingle, coulomb, tleads, mulst, tlst, dlst)
        if not (velph is None and tlst_ph is None and dlst_ph is None):
            self.baths.add(velph, tlst_ph, dlst_ph)

    def change(self, hsingle=None, coulomb=None, tleads=None, mulst=None, tlst=None, dlst=None,
                     velph=None, tlst_ph=None, dlst_ph=None):
        Builder_base.change(self, hsingle, coulomb, tleads, mulst, tlst, dlst)
        if not (velph is None and tlst_ph is None and dlst_ph is None):
            self.baths.change(velph, tlst_ph, dlst_ph)

    def remove_states(self, dE):
        Builder_base.remove_states(self, dE)
        self.si_elph.set_statesdm(si.statesdm)

    def remove_fock_states(self, lin_state_indices):
        Builder_base.remove_fock_states(self, lin_state_indices)
        self.baths._init_coupling()
        self.si_elph.remove_fock_states(lin_state_indices)

class Builder_many_body_elph(Builder_elph, Builder_many_body):
    """
    Class for building the system for stationary transport calculations,
    using many-body states as an input. Also includes Electron-Phonon coupling.

    For missing descriptions of attributes use:
        help(Builder), help(Builder_many_body), help(Builder_elph).

    Attributes
    ----------
    Vbbp : array
        nbaths by nmany by nmany array, which contains many-body electron-phonon coupling matrix.
    """

    def __init__(self, Ea=None, Na=[0], Tba=None, Vbbp=None,
                       mulst={}, tlst={}, dband={}, tlst_ph={}, dband_ph={}, kpnt=None,
                       kerntype='Pauli', symq=True, norm_row=0, solmethod='n',
                       itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                       mtype_qd=complex, mtype_leads=complex,
                       symmetry='n', herm_hs=True, herm_c=False, m_less_n=True,
                       bath_func=None, eps_elph=1.0e-6):

        nleads = Tba.shape[0] if Tba is not None else 0
        nbaths = Vbbp.shape[0] if Vbbp is not None else 0

        Builder_elph.__init__(self,
            nleads=nleads, mulst=mulst, tlst=tlst, dband=dband,
            nbaths=nbaths, tlst_ph=tlst_ph, dband_ph=dband_ph, kpnt=kpnt,
            kerntype=kerntype, symq=symq, norm_row=norm_row, solmethod=solmethod,
            itype=itype, dqawc_limit=dqawc_limit, mfreeq=mfreeq, phi0_init=phi0_init,
            mtype_qd=mtype_qd, mtype_leads=mtype_leads,
            symmetry=symmetry, herm_hs=herm_hs, herm_c=herm_c, m_less_n=m_less_n,
            bath_func=bath_func, eps_elph=eps_elph,
            indexing='charge')

        self._init_state_indexing(Na, Ea)

        self.qd.Ea = Ea
        self.leads.Tba = Tba
        self.baths.Vbbp = Vbbp
