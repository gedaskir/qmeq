"""Module for defining tunneling amplitudes from quantum dot to leads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from .indexing import szrange
from .indexing import ssqrange
from .indexing import sz_to_ind
from .indexing import ssq_to_ind

from .mytypes import doublenp

def construct_Tba(tleads, stateind, mtype=complex, Tba_=None):
    """
    Constructs many-body tunneling amplitude matrix Tba from single particle
    tunneling amplitudes.

    Parameters
    ----------
    tleads : dict
        Dictionary containing single particle tunneling amplitudes.
        tleads[(lead, state)] = tunneling amplitude.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of Tba matrix. For example, float, complex, etc.
    Tba_ : None or array
        nbaths by nmany by nmany numpy array containing old values of Tba.
        The values in tleads are added to Tba\_.

    Returns
    -------
    Tba : array
        nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
        The returned Tba corresponds to Fock basis.
    """
    if Tba_ is None:
        Tba = np.zeros((stateind.nleads, stateind.nmany, stateind.nmany), dtype=mtype)
    else:
        Tba = Tba_
    # Iterate over many-body states
    for j1 in range(stateind.nmany):
        state = stateind.get_state(j1)
        # Iterate over single particle states
        for j0 in tleads:
            (j3, j2), tamp = j0, tleads[j0]
            # Calculate fermion sign for added/removed electron in a given state
            fsign = np.power(-1, sum(state[0:j2]))
            if state[j2] == 0:
                statep = list(state)
                statep[j2] = 1
                ind = stateind.get_ind(statep)
                Tba[j3, ind, j1] += fsign*np.conj(tamp)
            else:
                statep = list(state)
                statep[j2] = 0
                ind = stateind.get_ind(statep)
                Tba[j3, ind, j1] += fsign*tamp
    return Tba

def construct_full_pmtr(vecslst, stateind, mtype=complex):
    """
    From definite charge eigenvectors constructs full eigenvectors
    defined by indexing in stateind.

    Parameters
    ----------
    veclst : list of arrays
        List of size ncharge containing arrays defining eigenvector matrices for given charge.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mtype : type
        Defines type of pmtr matrix. For example, float, complex, etc.

    Returns
    -------
    pmtr : array
        nmany by nmany numpy array containing many-body eigenvectors.
    """
    pmtr = np.zeros((stateind.nmany, stateind.nmany), dtype=mtype)
    if stateind.indexing == 'sz':
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, stateind.nsingle):
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                # Iterate over many-body states for given charge
                for j1, j2 in itertools.product(range(len(stateind.szlst[charge][szind])), range(len(stateind.szlst[charge][szind]))):
                    pmtr[stateind.szlst[charge][szind][j1], stateind.szlst[charge][szind][j2]] = vecslst[charge][szind][j1, j2]
    elif stateind.indexing == 'ssq':
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, stateind.nsingle):
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, stateind.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge
                    for j1, j2 in itertools.product(range(len(stateind.szlst[charge][szind])), range(len(stateind.ssqlst[charge][szind][ssqind]))):
                        pmtr[stateind.szlst[charge][szind][j1], stateind.ssqlst[charge][szind][ssqind][j2]] = vecslst[charge][szind][ssqind][j1, j2]
    else:
        # Iterate over charges
        for charge in range(stateind.ncharge):
            # Iterate over many-body states for given charge
            for j1, j2 in itertools.product(range(len(stateind.chargelst[charge])), range(len(stateind.chargelst[charge]))):
                pmtr[stateind.chargelst[charge][j1], stateind.chargelst[charge][j2]] = vecslst[charge][j1, j2]
    return pmtr

def rotate_Tba(Tba0, vecslst, stateind, indexing='n', mtype=complex):
    """
    Rotates tunneling amplitude matrix Tba0 in Fock basis to Tba,
    which is in eigenstate basis of the quantum dot.

    Parameters
    ----------
    Tba0 : array
        nleads by nmany by nmany numpy array, giving tunneling amplitudes in Fock basis.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    indexing : string
        Specifies what kind of rotation procedure to use. Default is stateind.indexing.
    mtype : type
        Defines type of Tba matrix. For example, float, complex, etc.

    Returns
    -------
    Tba : array
        nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
        The returned Tba corresponds to the quantum dot eigenbasis.
    """
    if indexing == 'n':
        indexingp = stateind.indexing
    else:
        indexingp = indexing
    Tba = np.zeros((stateind.nleads, stateind.nmany, stateind.nmany), dtype=mtype)
    if indexingp == 'Lin':
        pmtr = construct_full_pmtr(vecslst, stateind, mtype)
        for l in range(stateind.nleads):
            # Calculate many-body tunneling matrix Tba=P^(-1).Tba0.P
            # in eigenbasis of Hamiltonian from tunneling matrix Tba0 in Fock basis.
            # pmtr.conj().T denotes the conjugate transpose of pmtr.
            Tba[l] = np.dot(pmtr.conj().T, np.dot(Tba0[l], pmtr))
    elif indexingp == 'sz':
        for l, charge in itertools.product(range(stateind.nleads), range(stateind.ncharge-1)):
            szrng = szrange(charge, stateind.nsingle)
            # Lead labels from 0 to nleads//2 correspond to spin up
            # and nleads//2+1 to nleads-1 correspond to spin down
            if charge >= stateind.ncharge//2:
                szrng = szrng[0:-1] if l < stateind.nleads//2 else szrng[1:]
            # s=+1/-1 add/remove spin up/donw
            s = +1 if l < stateind.nleads//2 else -1
            for sz in szrng:
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                szind2 = sz_to_ind(sz+s, charge+1, stateind.nsingle)
                i1 = stateind.szlst[charge][szind][0]
                i2 = stateind.szlst[charge][szind][-1] + 1
                i3 = stateind.szlst[charge+1][szind2][0]
                i4 = stateind.szlst[charge+1][szind2][-1] + 1
                Tba[l, i1:i2][:, i3:i4] = np.dot(vecslst[charge][szind].conj().T, np.dot(Tba0[l, i1:i2][:, i3:i4], vecslst[charge+1][szind2]))
                Tba[l, i3:i4][:, i1:i2] = Tba[l, i1:i2][:, i3:i4].conj().T
    elif indexingp == 'ssq':
        for l, charge in itertools.product(range(stateind.nleads), range(stateind.ncharge-1)):
            szrng = szrange(charge, stateind.nsingle)
            # Lead labels from 0 to nleads//2 correspond to spin up
            # and nleads//2+1 to nleads-1 correspond to spin down
            if charge >= stateind.ncharge//2:
                szrng = szrng[0:-1] if l < stateind.nleads//2 else szrng[1:]
            # s=+1/-1 add/remove spin up/donw
            s = +1 if l < stateind.nleads//2 else -1
            for sz in szrng:
                szind = sz_to_ind(sz, charge, stateind.nsingle)
                szind2 = sz_to_ind(sz+s, charge+1, stateind.nsingle)
                i1 = stateind.szlst[charge][szind][0]
                i2 = stateind.szlst[charge][szind][-1] + 1
                i3 = stateind.szlst[charge+1][szind2][0]
                i4 = stateind.szlst[charge+1][szind2][-1] + 1
                vecslst1 = np.concatenate(vecslst[charge][szind], axis=1)
                vecslst2 = np.concatenate(vecslst[charge+1][szind2], axis=1)
                Tba[l, i1:i2][:, i3:i4] = np.dot(vecslst1.conj().T, np.dot(Tba0[l, i1:i2][:, i3:i4], vecslst2))
                Tba[l, i3:i4][:, i1:i2] = Tba[l, i1:i2][:, i3:i4].conj().T
    elif indexingp == 'charge':
        for l, charge in itertools.product(range(stateind.nleads), range(stateind.ncharge-1)):
            i1 = stateind.chargelst[charge][0]
            i2 = stateind.chargelst[charge][-1] + 1
            i3 = stateind.chargelst[charge+1][0]
            i4 = stateind.chargelst[charge+1][-1] + 1
            Tba[l, i1:i2][:, i3:i4] = np.dot(vecslst[charge].conj().T, np.dot(Tba0[l, i1:i2][:, i3:i4], vecslst[charge+1]))
            Tba[l, i3:i4][:, i1:i2] = Tba[l, i1:i2][:, i3:i4].conj().T
    return Tba
#---------------------------------------------------------------------------------------------------

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
    tleads_mtr : array
        nleads by nsingle numpy array containing single particle tunneling amplitudes.
    """
    tleads_mtr = np.zeros((nleads, nsingle), dtype=mtype)
    htype = type(tleads).__name__
    for j0 in tleads:
        if htype == 'list':    j1, j2, tamp = j0
        elif htype == 'dict': (j1, j2), tamp = j0, tleads[j0]
        tleads_mtr[j1, j2] += tamp
    return tleads_mtr

def make_tleads_dict(tleads):
    """
    Makes single particle tunneling amplitude dictionary.

    Parameters
    ----------
    tleads : list, dict, or array
        Contains single particle tunneling amplitudes.

    Returns
    -------
    tleads_dict : dictionary
        Dictionary containing tunneling amplitudes.
        tleads[(lead, state)] gives the tunneling amplitude.
    """
    htype = type(tleads).__name__
    if htype == 'list':
        tleads_dict = {}
        for j0 in tleads:
            j1, j2, tamp = j0
            tleads_dict.update({(j1, j2):tamp})
        return tleads_dict
    elif htype == 'ndarray':
        nleads, nsingle = tleads.shape
        tleads_dict = {}
        for j1 in range(nleads):
            for j2 in range(nsingle):
                if tleads[j1, j2] != 0:
                    tleads_dict.update({(j1, j2):tleads[j1, j2]})
        return tleads_dict
    elif htype == 'dict':
        return tleads

def make_array(lst, nleads):
    """
    Converts dictionary or list of mulst, tlst or dlst to an array.

    Parameters
    ----------
    lst : list, dict, or array
        Contains lead parameters.
    nleads : int
        Number of the leads.

    Returns
    -------
    lst_arr : array
        Numpy array containing lead parameters.
    """
    htype = type(lst).__name__
    if htype == 'dict':
        lst_arr = np.zeros(nleads, dtype=doublenp)
        for j1 in lst:
            lst_arr[j1] = lst[j1]
        return lst_arr
    elif htype in {'list', 'ndarray'}:
        return np.array(lst, dtype=doublenp)
    else:
        return np.zeros(nleads, dtype=doublenp)

def make_array_dlst(dlst, nleads):
    htype = type(dlst).__name__
    lst_arr = np.zeros((nleads,2), dtype=doublenp)
    if htype == 'dict':
        for j1 in dlst:
            if type(dlst[j1]).__name__ in {'float', 'int'}:
                lst_arr[j1] = (-dlst[j1], dlst[j1])
            else:
                lst_arr[j1] = dlst[j1]
    elif htype in {'int', 'float'}:
        lst_arr[:,0] = -dlst*np.ones(nleads, dtype=doublenp)
        lst_arr[:,1] = +dlst*np.ones(nleads, dtype=doublenp)
    elif htype in {'list', 'ndarray'}:
        if type(dlst[0]).__name__ in {'float', 'int'}:
            lst_arr[:,0] = -np.array(dlst, dtype=doublenp)
            lst_arr[:,1] = +np.array(dlst, dtype=doublenp)
        else:
            lst_arr = np.array(dlst, dtype=doublenp)
    return lst_arr
#---------------------------------------------------------------------------------------------------

class LeadsTunneling(object):
    """
    Class for defining tunneling amplitude from leads to the quantum dot.

    Attributes
    ----------
    nleads : int
        Number of the leads.
    tleads : dict, list or array
        Dictionary, list or numpy array defining single particle tunneling amplitudes.
        numpy array has to be nleads by nsingle.
    stateind : StateIndexing
        StateIndexing or StateIndexingDM object.
    mulst : dict, list or array
        Dictionary, list or numpy array containing chemical potentials of the leads.
    tlst : dict, list or array
        Dictionary, list or numpy array containing temperatures of the leads.
    dlst : dict, list or array
        Dictionary, list or numpy array containing bandwidths of the leads.
    mtype : type
        Defines type of Tba0 and Tba matrices. For example, float, complex, etc.
    Tba0 : array
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix in Fock basis.
    Tba : array
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix,
        which is used in calculations.
    """

    def __init__(self, nleads, tleads, stateind, mulst, tlst, dlst, mtype=complex):
        """Initialization of the LeadsTunneling class."""
        self.tleads = make_tleads_dict(tleads)
        self.stateind = stateind
        self.stateind.nleads = nleads
        self.mulst = make_array(mulst, nleads)
        self.tlst = make_array(tlst, nleads)
        self.dlst = make_array_dlst(dlst, nleads)
        self.mtype = mtype
        self.Tba0 = construct_Tba(self.tleads, stateind, mtype)
        self.Tba = self.Tba0

    def add(self, tleads=None, mulst=None, tlst=None, dlst=None, updateq=True, lstq=True):
        """
        Adds a value to single particle tunneling amplitudes and correspondingly redefines
        many-body tunneling matrix Tba.

        Parameters
        ----------
        tleads, mulst, tlst, dlst : dict
            Dictionaries describing what values to add.
            For example, tleads[(lead, state)] = value to add.
        updateq : bool
            Specifies if the values of the single particle amplitudes will be updated.
            The many-body tunneling amplitudes Tba will be updates in either case.
        lstq : bool
            Determines if the values will be added to mulst, tlst, dlst.
        """
        if lstq:
            self.mulst = self.mulst if mulst is None else self.mulst + make_array(mulst, self.stateind.nleads)
            self.tlst = self.tlst if tlst is None else self.tlst + make_array(tlst, self.stateind.nleads)
            self.dlst = self.dlst if dlst is None else self.dlst + make_array_dlst(dlst, self.stateind.nleads)
        if not tleads is None:
            tleadsp = tleads if type(tleads).__name__ == 'dict' else make_tleads_dict(tleads)
            self.Tba0 = construct_Tba(tleadsp, self.stateind, self.mtype, self.Tba0)
            if updateq:
                for j0 in tleadsp:
                    try:    self.tleads[j0] += tleadsp[j0]      # if tleads[j0] != 0:
                    except: self.tleads.update({j0:tleads[j0]}) # if tleads[j0] != 0:

    def change(self, tleads=None, mulst=None, tlst=None, dlst=None, updateq=True):
        """
        Changes the values of the single particle tunneling amplitudes and correspondingly redefines
        many-body tunneling matrix Tba.

        Parameters
        ----------
        tleads, mulst, tlst, dlst : dict
            Dictionaries describing what values to change.
            For example, tleads[(lead, state)] = the new value.
        updateq : bool
            Specifies if the values of the single particle amplitudes will be updated.
            The many-body tunneling amplitudes Tba will be updates in either case.
        """
        if not mulst is None:
            if type(mulst).__name__ == 'dict':
                for j0 in mulst:
                    self.mulst[j0] = mulst[j0]
            else:
                self.mulst = make_array(mulst, self.stateind.nleads)
        #
        if not tlst is None:
            if type(tlst).__name__ == 'dict':
                for j0 in tlst:
                    self.tlst[j0] = tlst[j0]
            else:
                self.tlst = make_array(tlst, self.stateind.nleads)
        #
        if not dlst is None:
            if type(dlst).__name__ == 'dict':
                for j0 in dlst:
                    if type(dlst[j0]).__name__ in {'int', 'float'}:
                        self.dlst[j0] = (-dlst[j0], dlst[j0])
                    else:
                        self.dlst[j0] = dlst[j0]
            else:
                self.dlst = make_array_dlst(dlst, self.stateind.nleads)
        #
        if not tleads is None:
            tleadsp = tleads if type(tleads).__name__ == 'dict' else make_tleads_dict(tleads)
            # Find the differences from the previous tunneling amplitudes
            tleads_add = {}
            for j0 in tleadsp:
                try:
                    tleads_diff = tleadsp[j0]-self.tleads[j0]
                    if tleads_diff != 0:
                        tleads_add.update({j0:tleads_diff})
                        if updateq: self.tleads[j0] += tleads_diff
                except:
                    tleads_diff = tleadsp[j0]
                    if tleads_diff != 0:
                        tleads_add.update({j0:tleads_diff})
                        if updateq: self.tleads.update({j0:tleads_diff})
            # Add the differences
            self.add(tleads_add, updateq=False, lstq=False)

    def rotate(self, vecslst, indexing='n'):
        """
        Rotates tunneling amplitude matrix Tba0 in Fock basis to Tba,
        which is in eigenstate basis of the quantum dot.

        Parameters
        ----------
        veclst : list of arrays
            List of size ncharge containing arrays defining eigenvector matrices for given charge.
        indexing : string
            Specifies what kind of rotation procedure to use. Default is stateind.indexing.
        """
        self.Tba = rotate_Tba(self.Tba0, vecslst, self.stateind, indexing, self.mtype)

    def use_Tba0(self):
        """
        Sets the Tba matrix for calculation to Tba0 in the Fock basis.
        """
        self.Tba = self.Tba0

    def update_Tba0(self, nleads, tleads, mtype=complex):
        """
        Updates the Tba0 in the Fock basis using new single-particle tunneling amplitudes.

        Parameters
        ----------
        tleads : array
            The new single-particle tunneling amplitudes. See attribute tleads.
        """
        self.stateind.nleads = nleads
        self.tleads = make_tleads_dict(tleads)
        self.mtype = mtype
        self.Tba0 = construct_Tba(tleads, self.stateind, mtype)
