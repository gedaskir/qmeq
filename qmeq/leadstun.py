"""Module for defining tunneling amplitudes from quantum dot to leads."""

import itertools
import numbers

import numpy as np

from .indexing import szrange
from .indexing import ssqrange
from .indexing import sz_to_ind
from .indexing import ssq_to_ind

from .wrappers.mytypes import doublenp


def construct_Tba(leads, tleads, Tba_=None):
    """
    Constructs many-body tunneling amplitude matrix Tba from single particle
    tunneling amplitudes.

    Parameters
    ----------
    leads : LeadsTunneling
        LeadsTunneling object.
    tleads : dict
        Dictionary containing single particle tunneling amplitudes.
        tleads[(lead, state)] = tunneling amplitude.
    Tba_ : None or ndarray
        nbaths by nmany by nmany numpy array containing old values of Tba.
        The values in tleads are added to Tba_.

    Returns
    -------
    Tba : ndarray
        nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
        The returned Tba corresponds to Fock basis.
    """
    si, mtype = leads.si, leads.mtype
    if Tba_ is None:
        Tba = np.zeros((si.nleads, si.nmany, si.nmany), dtype=mtype)
    else:
        Tba = Tba_
    # Iterate over many-body states
    for j1 in range(si.nmany):
        state = si.get_state(j1)
        # Iterate over single particle states
        for j0 in tleads:
            (j3, j2), tamp = j0, tleads[j0]
            # Calculate fermion sign for added/removed electron in a given state
            fsign = np.power(-1, sum(state[0:j2]))
            if state[j2] == 0:
                statep = list(state)
                statep[j2] = 1
                ind = si.get_ind(statep)
                if ind is None:
                    continue
                Tba[j3, ind, j1] += fsign*tamp
            else:
                statep = list(state)
                statep[j2] = 0
                ind = si.get_ind(statep)
                if ind is None:
                    continue
                Tba[j3, ind, j1] += fsign*np.conj(tamp)
    return Tba


def construct_full_pmtr(vecslst, si, mtype=complex):
    """
    From definite charge eigenvectors constructs full eigenvectors
    defined by indexing in si.

    Parameters
    ----------
    vecslst : list of ndarrays
        List of size ncharge containing arrays defining eigenvector matrices for given charge.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of pmtr matrix. For example, float, complex, etc.

    Returns
    -------
    pmtr : ndarray
        nmany by nmany numpy array containing many-body eigenvectors.
    """
    pmtr = np.zeros((si.nmany, si.nmany), dtype=mtype)
    if si.indexing == 'sz':
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                szind = sz_to_ind(sz, charge, si.nsingle)
                # Iterate over many-body states for given charge
                for j1, j2 in itertools.product(range(len(si.szlst[charge][szind])),
                                                range(len(si.szlst[charge][szind]))):
                    (pmtr[si.szlst[charge][szind][j1], si.szlst[charge][szind][j2]]) = (
                        vecslst[charge][szind][j1, j2])
    elif si.indexing == 'ssq':
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                szind = sz_to_ind(sz, charge, si.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, si.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge
                    for j1, j2 in itertools.product(range(len(si.szlst[charge][szind])),
                                                    range(len(si.ssqlst[charge][szind][ssqind]))):
                        (pmtr[si.szlst[charge][szind][j1], si.ssqlst[charge][szind][ssqind][j2]]) = (
                            vecslst[charge][szind][ssqind][j1, j2])
    else:
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over many-body states for given charge
            for j1, j2 in itertools.product(range(len(si.chargelst[charge])),
                                            range(len(si.chargelst[charge]))):
                pmtr[si.chargelst[charge][j1], si.chargelst[charge][j2]] = vecslst[charge][j1, j2]
    return pmtr


def rotate_Tba(Tba0, vecslst, si, indexing=None, mtype=complex):
    """
    Rotates tunneling amplitude matrix Tba0 in Fock basis to Tba,
    which is in eigenstate basis of the quantum dot.

    Parameters
    ----------
    Tba0 : ndarray
        nleads by nmany by nmany numpy array, giving tunneling amplitudes in Fock basis.
    vecslst : list of ndarrays
        List of size ncharge containing arrays defining eigenvector matrices for given charge.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    indexing : string
        Specifies what kind of rotation procedure to use. Default is si.indexing.
    mtype : type
        Defines type of Tba matrix. For example, float, complex, etc.

    Returns
    -------
    Tba : ndarray
        nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
        The returned Tba corresponds to the quantum dot eigenbasis.
    """
    if indexing is None:
        indexingp = si.indexing
    else:
        indexingp = indexing
    Tba = np.zeros((si.nleads, si.nmany, si.nmany), dtype=mtype)
    if indexingp == 'Lin':
        pmtr = construct_full_pmtr(vecslst, si, mtype)
        for l in range(si.nleads):
            # Calculate many-body tunneling matrix Tba=P^(-1).Tba0.P
            # in eigenbasis of Hamiltonian from tunneling matrix Tba0 in Fock basis.
            # pmtr.conj().T denotes the conjugate transpose of pmtr.
            Tba[l] = np.dot(pmtr.conj().T, np.dot(Tba0[l], pmtr))
    elif indexingp == 'sz':
        for l, charge in itertools.product(range(si.nleads), range(si.ncharge-1)):
            szrng = szrange(charge, si.nsingle)
            # Lead labels from 0 to nleads//2 correspond to spin up
            # and nleads//2+1 to nleads-1 correspond to spin down
            if charge >= si.ncharge//2:
                szrng = szrng[0:-1] if l < si.nleads//2 else szrng[1:]
            # s=+1/-1 add/remove spin up/down
            s = +1 if l < si.nleads//2 else -1
            for sz in szrng:
                szind = sz_to_ind(sz, charge, si.nsingle)
                szind2 = sz_to_ind(sz+s, charge+1, si.nsingle)
                if not si.szlst[charge][szind] or not si.szlst[charge+1][szind2]:
                    continue
                i1 = si.szlst[charge][szind][0]
                i2 = si.szlst[charge][szind][-1] + 1
                i3 = si.szlst[charge+1][szind2][0]
                i4 = si.szlst[charge+1][szind2][-1] + 1
                Tba[l, i1:i2][:, i3:i4] = np.dot(vecslst[charge][szind].conj().T,
                                                 np.dot(Tba0[l, i1:i2][:, i3:i4],
                                                        vecslst[charge+1][szind2]))
                Tba[l, i3:i4][:, i1:i2] = Tba[l, i1:i2][:, i3:i4].conj().T
    elif indexingp == 'ssq':
        for l, charge in itertools.product(range(si.nleads), range(si.ncharge-1)):
            szrng = szrange(charge, si.nsingle)
            # Lead labels from 0 to nleads//2 correspond to spin up
            # and nleads//2+1 to nleads-1 correspond to spin down
            if charge >= si.ncharge//2:
                szrng = szrng[0:-1] if l < si.nleads//2 else szrng[1:]
            # s=+1/-1 add/remove spin up/down
            s = +1 if l < si.nleads//2 else -1
            for sz in szrng:
                szind = sz_to_ind(sz, charge, si.nsingle)
                szind2 = sz_to_ind(sz+s, charge+1, si.nsingle)
                i1 = si.szlst[charge][szind][0]
                i2 = si.szlst[charge][szind][-1] + 1
                i3 = si.szlst[charge+1][szind2][0]
                i4 = si.szlst[charge+1][szind2][-1] + 1
                vecslst1 = np.concatenate(vecslst[charge][szind], axis=1)
                vecslst2 = np.concatenate(vecslst[charge+1][szind2], axis=1)
                Tba[l, i1:i2][:, i3:i4] = np.dot(vecslst1.conj().T,
                                                 np.dot(Tba0[l, i1:i2][:, i3:i4],
                                                        vecslst2))
                Tba[l, i3:i4][:, i1:i2] = Tba[l, i1:i2][:, i3:i4].conj().T
    elif indexingp == 'charge':
        for l, charge in itertools.product(range(si.nleads), range(si.ncharge-1)):
            if not si.chargelst[charge] or not si.chargelst[charge+1]:
                continue
            i1 = si.chargelst[charge][0]
            i2 = si.chargelst[charge][-1] + 1
            i3 = si.chargelst[charge+1][0]
            i4 = si.chargelst[charge+1][-1] + 1
            Tba[l, i1:i2][:, i3:i4] = np.dot(vecslst[charge].conj().T,
                                             np.dot(Tba0[l, i1:i2][:, i3:i4],
                                                    vecslst[charge+1]))
            Tba[l, i3:i4][:, i1:i2] = Tba[l, i1:i2][:, i3:i4].conj().T
    return Tba
# ---------------------------------------------------------------------------------------------------


def make_tleads_mtr(tleads, nleads, nsingle, mtype=complex):
    """
    Makes single particle tunneling matrix from a list or a dictionary.

    Parameters
    ----------
    tleads : list or dictionary
        Contains single particle tunneling amplitudes
    nleads : int
        Number of leads.
    nsingle : int
        Number of single particle states.
    mtype : type
        Defines type of tleads matrix. For example, float, complex, etc.

    Returns
    -------
    tleads_mtr : ndarray
        nleads by nsingle numpy array containing single particle tunneling amplitudes.
    """
    tleads_mtr = np.zeros((nleads, nsingle), dtype=mtype)
    for j0 in tleads:
        if isinstance(tleads, list):
            j1, j2, tamp = j0
        elif isinstance(tleads, dict):
            (j1, j2), tamp = j0, tleads[j0]
        else:
            continue
        tleads_mtr[j1, j2] += tamp
    return tleads_mtr


def make_tleads_dict(tleads, si, add_zeros=False):
    """
    Makes single particle tunneling amplitude dictionary.

    Parameters
    ----------
    tleads : list, dict, or ndarray
        Contains single particle tunneling amplitudes.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    add_zeros : bool
        Flag indicating whether to add zeros to dictionary.

    Returns
    -------
    tleads_dict : dictionary
        Dictionary containing tunneling amplitudes.
        tleads[(lead, state)] gives the tunneling amplitude.
    """
    if isinstance(tleads, dict):
        tleads_dict = tleads
    elif isinstance(tleads, list):
        tleads_dict = {}
        for j0 in tleads:
            j1, j2, tamp = j0
            tleads_dict.update({(j1, j2): tamp})
    elif isinstance(tleads, np.ndarray):
        nleads, nsingle = tleads.shape
        tleads_dict = {}
        for j1 in range(nleads):
            for j2 in range(nsingle):
                if tleads[j1, j2] != 0 or add_zeros:
                    tleads_dict.update({(j1, j2): tleads[j1, j2]})
    else:
        return {}
    #
    if si.symmetry is 'spin':
        tleads_dict_spin = dict(tleads_dict)
        for j0 in tleads_dict:
            j1, j2 = j0
            tamp = tleads_dict[j0]
            tleads_dict_spin.update({(j1+si.nleads_sym,
                                      j2+si.nsingle_sym): tamp})
        return tleads_dict_spin
    else:
        return tleads_dict

def make_tleads_array(tleads, si, mtype=complex):
    """Converts a dictionary containing matrix elements to a nleads x nsingle array.

    Parameters
    __________
    tleads : dict
        Contains single particle tunnel matrix elements
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of tleads matrix. For example, float, complex, etc.

    Returns
    -------
    tleads_array: ndarray
        Numpy array representing the single particle tunnel elements
    """
    tleads_array = np.zeros([si.nleads, si.nsingle], dtype=mtype)
    for (lead, state) in tleads:
        tleads_array[lead, state] = tleads[(lead, state)]

    return tleads_array

def make_array(lst_old, lst, si, npar=None, use_symmetry=True):
    """
    Converts dictionary or list of mulst, tlst or dlst to an array.

    Parameters
    ----------
    lst_old,lst : list, dict, or ndarray
        Contains lead parameters.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    npar : int
        Number of entries in the returned array. Default is si.nleads_sym.
    use_symmetry : bool
        Flag indicating whether symmetry specified in si.symmetry should be used.

    Returns
    -------
    lst_arr : ndarray
        Numpy array containing lead parameters.
    """
    #
    if npar is None:
        npar = si.nleads_sym

    if isinstance(lst, dict):
        if lst_old is None:
            lst_arr = np.zeros(npar, dtype=doublenp)
        else:
            lst_arr = lst_old[0:npar]
        for j1 in lst:
            lst_arr[j1] = lst[j1]
    elif isinstance(lst, (list, np.ndarray)):
        lst_arr = np.array(lst, dtype=doublenp)
    elif isinstance(lst, numbers.Number):
        lst_arr = lst*np.ones(npar, dtype=doublenp)
    else:
        lst_arr = np.zeros(npar, dtype=doublenp)
    #
    if si.symmetry is 'spin' and use_symmetry:
        return np.concatenate((lst_arr, lst_arr))
    else:
        return lst_arr


def make_array_dlst(dlst_old, dlst, si, npar=None, use_symmetry=True):
    if npar is None:
        npar = si.nleads_sym
    if dlst_old is None:
        lst_arr = np.zeros((npar, 2), dtype=doublenp)
    else:
        lst_arr = dlst_old[0:npar]
    #
    if isinstance(dlst, numbers.Number):
        lst_arr[:, 0] = -dlst*np.ones(npar, dtype=doublenp)
        lst_arr[:, 1] = +dlst*np.ones(npar, dtype=doublenp)
    elif isinstance(dlst, dict):
        for j1 in dlst:
            if isinstance(dlst[j1], numbers.Number):
                lst_arr[j1] = (-dlst[j1], dlst[j1])
            else:
                lst_arr[j1] = dlst[j1]
    elif isinstance(dlst, (list, np.ndarray)):
        if isinstance(dlst[0], numbers.Number):
            lst_arr[:, 0] = -np.array(dlst, dtype=doublenp)
            lst_arr[:, 1] = +np.array(dlst, dtype=doublenp)
        else:
            lst_arr = np.array(dlst, dtype=doublenp)
    #
    if si.symmetry == 'spin' and use_symmetry:
        return np.concatenate((lst_arr, lst_arr))
    else:
        return lst_arr
# ---------------------------------------------------------------------------------------------------


class LeadsTunneling(object):
    """
    Class for defining tunneling amplitude from leads to the quantum dot.

    Attributes
    ----------
    nleads : int
        Number of the leads.
    tleads : dict, list or ndarray
        Dictionary, list or numpy array defining single particle tunneling amplitudes.
        numpy array has to be nleads by nsingle.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    mulst : dict, list or ndarray
        Dictionary, list or numpy array containing chemical potentials of the leads.
    tlst : dict, list or ndarray
        Dictionary, list or numpy array containing temperatures of the leads.
    dlst : dict, list or ndarray
        Dictionary, list or numpy array containing bandwidths of the leads.
    mtype : type
        Defines type of Tba0 and Tba matrices. For example, float, complex, etc.
    Tba0 : ndarray
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix
        in Fock basis.
    Tba : ndarray
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix,
        which is used in calculations.
    """

    def __init__(self, nleads, tleads, si, mulst, tlst, dlst, mtype=complex):
        """Initialization of the LeadsTunneling class."""
        si.nleads = nleads
        si.nleads_sym = nleads//2 if si.symmetry == 'spin' else nleads
        #
        self.si = si
        self.tleads = make_tleads_dict(tleads, si)
        self.tleads_array = make_tleads_array(self.tleads, si)
        self.mulst = make_array(None, mulst, si)
        self.tlst = make_array(None, tlst, si)
        self.dlst = make_array_dlst(None, dlst, si)
        self.mtype = mtype
        self._init_coupling()

    def _init_coupling(self):
        self.Tba0 = construct_Tba(self, self.tleads)
        self.Tba = np.array(self.Tba0)

    def add(self, tleads=None, mulst=None, tlst=None, dlst=None, updateq=True, lstq=True):
        """
        Adds a value to single particle tunneling amplitudes and correspondingly redefines
        many-body tunneling matrix Tba.

        Parameters
        ----------
        tleads,mulst,tlst,dlst : dict, list, int
            Dictionaries describing what values to add.
            For example, tleads[(lead, state)] = value to add.
        updateq : bool
            Specifies if the values of the single particle amplitudes will be updated.
            The many-body tunneling amplitudes Tba will be updates in either case.
        lstq : bool
            Determines if the values will be added to mulst, tlst, dlst.
        """
        if lstq:
            if mulst is not None:
                self.mulst += make_array(None, mulst, self.si)
            if tlst is not None:
                self.tlst += make_array(None, tlst, self.si)
            if dlst is not None:
                self.dlst += make_array_dlst(None, dlst, self.si)
        if tleads is not None:
            if updateq:
                tleads = make_tleads_dict(tleads, self.si)
                for j0 in tleads:
                    if j0 in self.tleads:
                        self.tleads[j0] += tleads[j0]
                    else:
                        self.tleads.update({j0: tleads[j0]})
                self.tleads_array = make_tleads_array(self.tleads, self.si)
            self.Tba0 = construct_Tba(self, tleads, self.Tba0)

    def change(self, tleads=None, mulst=None, tlst=None, dlst=None):
        """
        Changes the values of the single particle tunneling amplitudes and correspondingly redefines
        many-body tunneling matrix Tba.

        Parameters
        ----------
        tleads,mulst,tlst,dlst : dict, list, int
            Dictionaries describing what values to change.
            For example, tleads[(lead, state)] = the new value.
        """
        if mulst is not None:
            self.mulst[:] = make_array(self.mulst, mulst, self.si)
        if tlst is not None:
            self.tlst[:] = make_array(self.tlst, tlst, self.si)
        if dlst is not None:
            self.dlst[:] = make_array_dlst(self.dlst, dlst, self.si)
        #
        if tleads is not None:
            tleads = make_tleads_dict(tleads, self.si, True)
            # Find the differences from the previous tunneling amplitudes
            tleads_add = {}
            for j0 in tleads:
                if j0 in self.tleads:
                    tleads_diff = tleads[j0]-self.tleads[j0]
                    if tleads_diff != 0:
                        tleads_add.update({j0: tleads_diff})
                        self.tleads[j0] += tleads_diff
                else:
                    tleads_diff = tleads[j0]
                    if tleads_diff != 0:
                        tleads_add.update({j0: tleads_diff})
                        self.tleads.update({j0: tleads_diff})
            # Add the differences
            self.add(tleads_add, updateq=False, lstq=False)

    def rotate(self, vecslst, indexing=None):
        """
        Rotates tunneling amplitude matrix Tba0 in Fock basis to Tba,
        which is in eigenstate basis of the quantum dot.

        Parameters
        ----------
        vecslst : list of ndarrays
            List of size ncharge containing arrays defining eigenvector matrices for given charge.
        indexing : string
            Specifies what kind of rotation procedure to use. Default is si.indexing.
        """
        self.Tba[:] = rotate_Tba(self.Tba0, vecslst, self.si, indexing, self.mtype)

    def use_Tba0(self):
        """
        Sets the Tba matrix for calculation to Tba0 in the Fock basis.
        """
        self.Tba[:] = self.Tba0

    def update_Tba0(self, nleads, tleads, mtype=complex):
        """
        Updates the Tba0 in the Fock basis using new single-particle tunneling amplitudes.

        Parameters
        ----------
        nleads : int
            Number of leads.
        tleads : dict
            The new single-particle tunneling amplitudes. See attribute tleads.
        mtype : type
            Defines type of Tba0 and Tba matrices. For example, float, complex, etc.
        """
        si = self.si
        si.nleads = nleads
        si.nleads_sym = nleads//2 if si.symmetry == 'spin' else nleads
        self.tleads = make_tleads_dict(tleads, si)
        self.tleads_array = make_tleads_array(tleads, self.si)
        self.mtype = mtype
        self.Tba0 = construct_Tba(self, tleads)
