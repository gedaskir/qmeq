"""Module for constructing many-body quantum dot Hamiltonian and diagonalising it."""

import numpy as np

from .indexing import empty_szlst
from .indexing import empty_ssqlst
from .indexing import szrange
from .indexing import ssqrange
from .indexing import sz_to_ind
from .indexing import ssq_to_ind


def construct_ham_coulomb(qd, coulomb, statelst, ham_=None):
    """
    Constructs many-body Hamiltonian for given Coulomb matrix elements.

    Parameters
    ----------
    qd : QuantumDot
        QuantumDot object.
    coulomb : dict
        Dictionary containing coulomb matrix elements. The dictionary is of the format
        coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels
        and U is the strength of the coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    statelst : list
        List of indices of states under consideration.
    ham_ : None or ndarray
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in coulomb are added to ham_coulomb.

    Returns
    -------
    ham_coulomb : ndarray
        Matrix representation of many-body Coulomb Hamiltonian in Fock basis for states statelst.
    """
    si, mtype = qd.si, qd.mtype
    herm_c, m_less_n = qd.herm_c, qd.m_less_n
    nstates = len(statelst)
    if ham_ is not None:
        ham_coulomb = ham_
    else:
        ham_coulomb = np.zeros((nstates, nstates), dtype=mtype)
    # Iterate over many-body states
    for j1 in range(nstates):
        state = si.get_state(statelst[j1])
        # Iterate over Coulomb matrix elements
        for j2 in coulomb:
            # The suggested convention is: k>l, n>m
            (m, n, k, l), U = j2, coulomb[j2]
            if state[k] == 1 and state[l] == 1 and k != l:
                # Calculate fermion sign due to two removed electrons in a given state
                # Note that if k is larger than l additional sign appears for flipping k with l
                fsign = np.power(-1, sum(state[0:k])+sum(state[0:l])) * (-1 if k > l else +1)
                statep = list(state)
                statep[k] = 0
                statep[l] = 0
                if statep[m] == 0 and statep[n] == 0 and m != n:
                    # Calculate fermion sign due to two added electrons in a given state
                    # Note that if n is smaller than m additional sign appears for flipping m with n
                    fsign = fsign*np.power(-1, sum(statep[0:n])+sum(statep[0:m])) * (+1 if n > m else -1)
                    statep[m] = 1
                    statep[n] = 1
                    state_index = si.get_ind(statep)
                    if state_index is None:
                        continue
                    ind = statelst.index(state_index)
                    ham_coulomb[ind, j1] += U*fsign
                    if herm_c and (m != l or n != k) and not (m_less_n and l > k):
                        ham_coulomb[j1, ind] += U.conjugate()*fsign
    return ham_coulomb


def construct_ham_hopping(qd, hsingle, statelst, ham_=None):
    """
    Constructs many-body Hamiltonian for given single-particle hamiltonian.

    Parameters
    ----------
    qd : QuantumDot
        QuantumDot object.
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
        The dictionary is of the format hsingle[(i, j)] = hij, where i, j are the state labels
        and hij is the matrix element of the single particle Hamiltonian.
    statelst : list
        List of indices of states under consideration.
    ham_ : None or ndarray
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in hsingle are added to ham.

    Returns
    -------
    ham_coulomb : ndarray
        Matrix representation of many-body Hamiltonian in Fock basis for states statelst.
    """
    si, mtype, herm_hs = qd.si, qd.mtype, qd.herm_hs
    nstates = len(statelst)
    if ham_ is not None:
        ham_hopping = ham_
    else:
        ham_hopping = np.zeros((nstates, nstates), dtype=mtype)
    # Iterate over many-body states
    for j1 in range(nstates):
        state = si.get_state(statelst[j1])
        # Iterate over single particle Hamiltonian
        for j0 in hsingle:
            (j2, j3), hop = j0, hsingle[j0]
            if j2 == j3:
                if state[j2] == 1:
                    ham_hopping[j1, j1] += hop
            # Remove particle from j2 single particle state, add particle in j3 single particle state
            elif state[j2] == 1 and state[j3] == 0:
                # Calculate fermion sign for added/removed electrons in a given state
                # Note that if j3 is larger than j2 additional sign appears for flipping j3 with j2
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:j3])) * (+1 if j2 > j3 else -1)
                statep = list(state)
                statep[j2] = 0
                statep[j3] = 1
                state_index = si.get_ind(statep)
                if state_index is None:
                    continue
                ind = statelst.index(state_index)
                ham_hopping[ind, j1] += hop*fsign
                if herm_hs:
                    ham_hopping[j1, ind] += hop.conjugate()*fsign
    return ham_hopping


def construct_manybody_eigenstates(qd, hsingle, coulomb, statelst, ham_=None):
    """
    Calculates eigenstates of many-body Hamiltonian (described by hsingle and coulomb).

    Parameters
    ----------
    qd : QuantumDot
        QuantumDot object.
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
    coulomb : dict
        Dictionary containing coulomb matrix elements.
    statelst : list
        List of indices of states under consideration.
    ham_ : None or ndarray
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.

    Returns
    -------
    ham_vals : ndarray
        len(statelst) by 1 array containing eigenvalues of Hamiltonian.
    ham_vecs : ndarray
        len(statelst) by len(statelst) array containing eigenvector matrix of Hamiltonian.
        Columns of ham_vecs correspond to particular eigenvectors.
    """
    if ham_ is not None:
        ham = ham_
    else:
        ham_coulomb = construct_ham_coulomb(qd, coulomb, statelst)
        ham_hopping = construct_ham_hopping(qd, hsingle, statelst)
        ham = ham_coulomb + ham_hopping
    ham_vals, ham_vecs = np.linalg.eigh(ham)
    return ham_vals, ham_vecs


def construct_Ea_manybody(valslst, si):
    """
    Makes a single array corresponding to a given indexing and containing eigenvalues
    of Hamiltonian.

    Parameters
    ----------
    valslst : list of ndarrays
        List containing eigenvalues. List entry valslst[charge] is an array of definite charge
        eigenvalues.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    Ea : ndarray
        nmany by 1 array containing eigenvalues of the Hamiltonian.
    """
    Ea = np.zeros(si.nmany, dtype=float)
    if si.indexing == 'sz':
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                # Iterate over many-body states for given charge and sz
                szind = sz_to_ind(sz, charge, si.nsingle)
                for ind in range(len(si.szlst[charge][szind])):
                    # The mapping of many-body states is according to szlst
                    Ea[si.szlst[charge][szind][ind]] = valslst[charge][szind][ind]
    elif si.indexing == 'ssq':
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over spin projection sz
            for sz in szrange(charge, si.nsingle):
                szind = sz_to_ind(sz, charge, si.nsingle)
                # Iterate over total spin ssq
                for ssq in ssqrange(charge, sz, si.nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    # Iterate over many-body states for given charge, sz, and ssq
                    for ind in range(len(si.ssqlst[charge][szind][ssqind])):
                        # The mapping of many-body states is according to ssqlst
                        Ea[si.ssqlst[charge][szind][ssqind][ind]] = valslst[charge][szind][ssqind][ind]
    else:
        # Iterate over charges
        for charge in range(si.ncharge):
            # Iterate over many-body states for given charge
            for ind in range(len(si.chargelst[charge])):
                # The mapping of many-body states is according to chargelst
                Ea[si.chargelst[charge][ind]] = valslst[charge][ind]
    return Ea


# ---------------------------------------------------------------------------------------------------
# manyhamssq.py
def operator_sm(charge, sz, si):
    """
    Construct the operator :math:`S^{-}`, which acts on the states with given :math:`S^{z}`.

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    sz : int
        Spin projection.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    sm : ndarray
        Matrix (numpy array) describing the specific :math:`S^{-}` operator.
        The operator is expressed in the Fock basis.
    """
    # The minimum possible value of sz for given number of
    # single particle states nsingle and number of particle charge
    szmin = -min(charge, si.nsingle-charge)
    # Acting with S^{-} on the states with smallest possible sz gives 0
    if sz-2 < szmin:
        return 0
    states = si.get_lst(charge, sz)
    statesm = si.get_lst(charge, sz-2)
    sm = np.zeros((len(statesm), len(states)), dtype=int)
    # States up to index dind have spin up
    # and the states from dind to nsingle have spin down
    dind = int(si.nsingle/2)
    for j1 in range(len(states)):
        state = si.get_state(states[j1])
        for j2 in range(int(si.nsingle/2)):
            if state[j2] == 1 and state[dind+j2] == 0:
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:dind+j2])+1)
                state2 = list(state)
                # Flip the spin in single particle state from up to down
                state2[j2] = 0
                state2[dind+j2] = 1
                ind = statesm.index(si.get_ind(state2))
                sm[ind, j1] = 2*fsign
    return sm


def operator_sp(charge, sz, si):
    """
    Construct the operator :math:`S^{+}`, which acts on the states with given sz.

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    sz : int
        Spin projection.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    sp : ndarray
        Matrix (numpy array) describing the specific :math:`S^{+}` operator.
        The operator is expressed in the Fock basis.
    """
    # The maximum possible value of sz for given number of
    # single particle states nsingle and number of particle charge
    szmax = min(charge, si.nsingle-charge)
    # Acting with S^{+} on the states with largest possible sz gives 0
    if sz+2 > szmax:
        return 0
    states = si.get_lst(charge, sz)
    statesp = si.get_lst(charge, sz+2)
    sp = np.zeros((len(statesp), len(states)), dtype=int)
    # States up to index dind have spin up
    # and the states from dind to nsingle have spin down
    dind = int(si.nsingle/2)
    for j1 in range(len(states)):
        state = si.get_state(states[j1])
        for j2 in range(int(si.nsingle/2)):
            if state[j2] == 0 and state[dind+j2] == 1:
                fsign = np.power(-1, sum(state[0:j2])+sum(state[0:dind+j2]))
                state2 = list(state)
                # Flip the spin in single particle state from down to up
                state2[j2] = 1
                state2[dind+j2] = 0
                ind = statesp.index(si.get_ind(state2))
                sp[ind, j1] = 2*fsign
    return sp


def operator_ssquare(charge, sz, si):
    """
    Construct the operator :math:`S^{2}`, which acts on the states with given sz.
    :math:`S^{2} = (S^{+}S^{-}+S^{-}S^{+})/2+S^{z}S^{z}`

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    sz : int
        Spin projection.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    ssquare : ndarray
        Matrix (numpy array) describing the specific :math:`S^{+}` operator.
        The operator is expressed in the Fock basis.
    """
    szmin = -min(charge, si.nsingle-charge)
    szmax = min(charge, si.nsingle-charge)
    sp_sm = np.dot(operator_sp(charge, sz-2, si), operator_sm(charge, sz, si)) if sz-2 >= szmin else 0
    sm_sp = np.dot(operator_sm(charge, sz+2, si), operator_sp(charge, sz, si)) if sz+2 <= szmax else 0
    sz_square = np.eye(len(si.get_lst(charge, sz)), dtype=int)*sz**2
    ssquare = (sp_sm+sm_sp)//2+sz_square
    return ssquare


def ssquare_eigenstates(charge, sz, si, prnt=False):
    """
    Find the eigenstates of the operator :math:`S^{2}` for given sz.

    Parameters
    ----------
    charge : int
        States with given charge (number of electrons) are considered.
    sz : int
        Value of sz.
        sz odd corresponds to fractional spin. For example, 1->1/2, 3->3/2, etc.
        sz even corresponds to integer spin. For example, 0->0, 2->1, etc.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    prnt : bool
        If true, then print the eigenstates.

    Returns
    -------
    ssq_eigvec : list of ndarrays
        List of numpy arrays containing the eigenstates of :math:`S^{2}`.
        ssq_eigvec[ssqind] gives the eigenstate matrix for :math:`S^{2}` =ssq.
    """
    if charge % 2 != sz % 2:
        print("WARNING: charge and sz need to have the same parity. Return 0.")
        return 0
    ssq_eigvec = [0 for _ in ssqrange(charge, sz, si.nsingle)]
    ssquare = operator_ssquare(charge, sz, si)
    eigval, eigvec = np.linalg.eigh(ssquare)
    eigval = np.rint(-1 + np.sqrt(1+eigval))
    for ssq in ssqrange(charge, sz, si.nsingle):
        v = np.nonzero(eigval == ssq)
        ssqind = ssq_to_ind(ssq, sz)
        ssq_eigvec[ssqind] = eigvec[:][:, v[0][0]:v[0][-1]+1]
        if prnt:
            print('-'*30)
            print('charge =', charge, ', sz =', sz, ', ssq =', ssq)
            print(ssq_eigvec[ssqind])
    return ssq_eigvec


def ssquare_all_szlow(si, prnt=False):
    """
    Find the eigenstates of the operator :math:`S^{2}` for sz=0 or 1.

    Parameters
    ----------
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
    prnt : bool
        If true, then print the eigenstates.

    Returns
    -------
    ssq_eigvec : list of ndarrays
        List of numpy arrays containing the eigenstates of :math:`S^{2}` for lowest abs(sz) value
        possible.
        ssq_eigvec[charge][ssq] gives the eigenstate matrix for given charge and :math:`S^{2}` =ssq.
    """
    ssq_eigvec_all_szlow = [[] for _ in range(si.ncharge)]
    for charge in range(si.ncharge):
        ssq_eigvec_all_szlow[charge] = ssquare_eigenstates(charge, -(charge % 2), si)
    if prnt:
        for charge in range(si.ncharge):
            for ssq in ssqrange(charge, -(charge % 2), si.nsingle):
                print('-'*30)
                print('charge =', charge, ', sz =', -(charge % 2), ', ssq =', ssq)
                print(ssq_eigvec_all_szlow[charge][ssq//2])
    return ssq_eigvec_all_szlow


def construct_manybody_eigenstates_ssq(qd, charge, sz, ssq, hsingle, coulomb, ham_=None):
    """
    Find the many-body eigenstates of the Hamiltonian, which are also the eigenstates of sz
    and :math:`S^{2}` for given charge, sz and :math:`S^{2}`.

    Parameters
    ----------
    qd : QuantumDot
        QuantumDot object.
    charge : int
        Charge (number of particles) of the states to consider.
    sz : int
        sz value of the states to consider.
    ssq : int
        :math:`S^{2}` value of the states to consider.
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
        The dictionary is of the format hsingle[(i, j)] = hij, where i, j are the state labels
        and hij is the matrix element of the single particle Hamiltonian.
    coulomb : dict
        Dictionary containing coulomb matrix elements. The dictionary is of the format
        coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels
        and U is the strength of the coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    ham_ : None or ndarray
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in hsingle and coulomb are added to ham.

    Returns
    -------
    eigvalp : ndarray
        Eigenvalues of the Hamiltonian for given charge, sz, :math:`S^{2}`.
    eigvecssq : ndarray
        Corresponding eigenvectors of the Hamiltonian.
    """
    si = qd.si
    statelst = si.get_lst(charge=charge, sz=sz)
    if ham_ is not None:
        ham = ham_
    else:
        ham = (construct_ham_coulomb(qd, coulomb, statelst)
               + construct_ham_hopping(qd, hsingle, statelst))
    ssq_eigvec = ssquare_eigenstates(charge, sz, si)
    ssqind = ssq_to_ind(ssq, sz)
    # Write the Hamiltonian in the S^{2} basis for given value of S^{2}=ssq
    hamp = np.dot(ssq_eigvec[ssqind].T, np.dot(ham,  ssq_eigvec[ssqind]))
    eigvalp, eigvecp = np.linalg.eigh(hamp)
    eigvecssq = np.dot(ssq_eigvec[ssqind], eigvecp)
    return eigvalp, eigvecssq


def construct_manybody_eigenstates_ssq_all(qd, charge, hsingle, coulomb, ham_=None):
    """
    Find the many-body eigenstates of the Hamiltonian for given charge.
    The eigenstates are also the eigenstates of sz and :math:`S^{2}`.

    Parameters
    ----------
    qd : QuantumDot
        QuantumDot object.
    charge : int
        Charge (number of particles) of the states to consider.
    hsingle : dict
        Dictionary corresponding to single-particle hopping (tunneling) Hamiltonian.
        The dictionary is of the format hsingle[(i, j)] = hij, where i, j are the state labels
        and hij is the matrix element of the single particle Hamiltonian.
    coulomb : dict
        Dictionary containing coulomb matrix elements. The dictionary is of the format
        coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels
        and U is the strength of the coulomb interaction between these states.
        Note that only the matrix elements k>l, n>m have to be specified.
    ham_ : None or ndarray
        nmany by nmany numpy array containing old values of the many-body Hamiltonian.
        The values in hsingle and coulomb are added to ham.

    Returns
    -------
    valslst : list of ndarrays
        Eigenvalues of the Hamiltonian for given charge.
        valslst[szind][ssqind] are the eigenvalues for given sz and :math:`S^{2}`.
    vecslst : list of ndarrays
        Eigenvectors corresponding to valslst.
    """
    si = qd.si
    szlow = -(charge % 2)
    nsingle = si.nsingle
    # valslst = [[None] for k in ssqrange(charge, szlow, nsingle)]
    valslst = [[[None] for _ in ssqrange(charge, j, nsingle)] for j in szrange(charge, nsingle)]
    vecslst = [[[None] for _ in ssqrange(charge, j, nsingle)] for j in szrange(charge, nsingle)]
    #
    statelst = si.get_lst(charge=charge, sz=szlow)
    if ham_ is not None:
        ham = ham_
    else:
        ham = (construct_ham_coulomb(qd, coulomb, statelst)
               + construct_ham_hopping(qd, hsingle, statelst))
    # NOTE: Can be an argument of the function instead
    ssq_eigvec = ssquare_eigenstates(charge, szlow, si)
    #
    # Find eigenstates corresponding to sz=1 or 0
    szind = sz_to_ind(szlow, charge, nsingle)
    for ssq in ssqrange(charge, szlow, nsingle):
        ssqind = ssq_to_ind(ssq, szlow)
        hamp = np.dot(ssq_eigvec[ssqind].T, np.dot(ham,  ssq_eigvec[ssqind]))
        eigvalp, eigvecp = np.linalg.eigh(hamp)
        eigvecssq = np.dot(ssq_eigvec[ssqind], eigvecp)
        valslst[szind][ssqind] = eigvalp
        vecslst[szind][ssqind] = eigvecssq
    # Construct the eigenstates with larger (positive) sz by acting with S^{+}
    szmax = min(charge, si.nsingle-charge)
    for sz in range(szlow, szmax-1, 2):
        szind = sz_to_ind(sz, charge, nsingle)
        szind2 = sz_to_ind(sz+2, charge, nsingle)
        for ssq in ssqrange(charge, sz+2, nsingle):
            ssqind = ssq_to_ind(ssq, sz)
            ssqind2 = ssq_to_ind(ssq, sz+2)
            sp = operator_sp(charge, sz, si)
            fct = np.sqrt(ssq*(ssq+2) - sz*(sz+2))
            vecslst[szind2][ssqind2] = np.dot(sp, vecslst[szind][ssqind])/fct
            valslst[szind2][ssqind2] = valslst[szind][ssqind]
    # Construct the eigenstates with smaller (negative) sz by acting with S^{-}
    for sz in range(szlow, -szmax+1, -2):
        szind = sz_to_ind(sz, charge, nsingle)
        szind2 = sz_to_ind(sz-2, charge, nsingle)
        for ssq in ssqrange(charge, sz-2, nsingle):
            ssqind = ssq_to_ind(ssq, sz)
            ssqind2 = ssq_to_ind(ssq, sz-2)
            sm = operator_sm(charge, sz, si)
            fct = np.sqrt(ssq*(ssq+2) - sz*(sz-2))
            vecslst[szind2][ssqind2] = np.dot(sm, vecslst[szind][ssqind])/fct
            valslst[szind2][ssqind2] = valslst[szind][ssqind]
    return valslst, vecslst
# ---------------------------------------------------------------------------------------------------


def make_hsingle_mtr(hsingle, nsingle, mtype=float):
    """
    Makes single particle Hamiltonian matrix from a list or a dictionary.

    Parameters
    ----------
    hsingle : list or dictionary
        Contains single particle tunneling amplitudes
    nsingle : int
        Number of single particle states.
    mtype : type
        Defines type of tleads matrix. For example, float, complex, etc.

    Returns
    -------
    hsingle_mtr : ndarray
        hsingle by hsingle numpy array containing single particle Hamiltonian.
    """
    hsingle_mtr = np.zeros((nsingle, nsingle), dtype=mtype)
    for j0 in hsingle:
        if isinstance(hsingle, list):
            j1, j2, hop = j0
        elif isinstance(hsingle, dict):
            (j1, j2), hop = j0, hsingle[j0]
        else:
            continue
        hsingle_mtr[j1, j2] += hop
        if j1 != j2:
            hsingle_mtr[j2, j1] += hop.conjugate()
    return hsingle_mtr


def make_hsingle_dict(qd, hsingle, add_zeros=False):
    """
    Makes single particle Hamiltonian dictionary.

    Parameters
    ----------
    hsingle : list, dict, or ndarray
        Contains single particle Hamiltonian.
    qd : QuantumDot
        QuantumDot object.
    add_zeros : bool
        Flag indicating whether to add zeros to dictionary.

    Returns
    -------
    hsingle_dict : dictionary
        Dictionary containing single particle Hamiltonian.
        hsingle[(state1, state2)] gives the matrix element.
    """
    si = qd.si
    #
    if isinstance(hsingle, dict):
        hsingle_dict = hsingle
    elif isinstance(hsingle, list):
        hsingle_dict = {}
        for j0 in hsingle:
            j1, j2, hop = j0
            hsingle_dict.update({(j1, j2): hop})
    elif isinstance(hsingle, np.ndarray):
        nsingle = hsingle.shape[0]
        hsingle_dict = {}
        herm_hs = qd.herm_hs
        for j1 in range(nsingle):
            for j2 in range(j1 if herm_hs else 0, nsingle):
                if hsingle[j1, j2] != 0 or add_zeros:
                    hsingle_dict.update({(j1, j2): hsingle[j1, j2]})
    else:
        return {}
    #
    if si.symmetry == 'spin':
        hsingle_dict_spin = dict(hsingle_dict)
        for j0 in hsingle_dict:
            j1, j2 = j0
            hop = hsingle_dict[j0]
            hsingle_dict_spin.update({(j1+si.nsingle_sym,
                                       j2+si.nsingle_sym): hop})
        return hsingle_dict_spin
    else:
        return hsingle_dict


def make_coulomb_dict(qd, coulomb):
    """
    Makes Coulomb matrix element dictionary.

    Parameters
    ----------
    coulomb : list, dict, or ndarray
        Contains coulomb matrix elements.
    qd : QuantumDot
        QuantumDot object.

    Returns
    -------
    coulomb_dict : dictionary
        Dictionary containing coulomb matrix element.
        coulomb[(state1, state2, state3, state4)] gives the coulomb matrix element U.
    """
    si = qd.si
    #
    if isinstance(coulomb, dict):
        coulomb_dict = coulomb
    elif isinstance(coulomb, (list, np.ndarray)):
        coulomb_dict = {}
        for j0 in coulomb:
            m, n, k, l, U = j0
            if U != 0:
                coulomb_dict.update({(m, n, k, l): U})
    else:
        return {}
    #
    if si.symmetry == 'spin':
        coulomb_dict_spin = dict(coulomb_dict)
        nss = si.nsingle_sym
        herm_c, m_less_n = qd.herm_c, qd.m_less_n
        for j0 in coulomb_dict:
            m, n, k, l = j0
            U = coulomb_dict[j0]
            # up, down, down, up
            if not herm_c or (l, k+nss, n+nss, m) not in coulomb_dict_spin.keys():
                coulomb_dict_spin.update({(m, n+nss, k+nss, l): U})
            # down, up, up, down
            if m_less_n:
                if not herm_c or (k, l+nss, m+nss, n) not in coulomb_dict_spin.keys():
                    coulomb_dict_spin.update({(n, m+nss, l+nss, k): U})
            else:
                coulomb_dict_spin.update({(m+nss, n, k, l+nss): U})
            # down, down, down, down
            coulomb_dict_spin.update({(m+nss, n+nss, k+nss, l+nss): U})
        return coulomb_dict_spin
    else:
        return coulomb_dict
# ---------------------------------------------------------------------------------------------------


class QuantumDot(object):
    """
    Class for constructing and diagonalising many-body Hamiltonian describing
    the quantum dot.

    Attributes
    ----------
    hsingle : dict, list, or ndarray
        Dictionary, list, or array corresponding to single-particle hopping (tunneling) Hamiltonian.
        On input list or array gets converted to dictionary.
    coulomb : dict, list, or ndarray
        Dictionary, list, or array or containing coulomb matrix elements.
        For dictionary:    coulomb[(m, n, k, l)] = U, where m, n, k, l are the state labels.
        For list or array: coulomb[i] is list of the format [m, n, k, l, U].
        U is the strength of the coulomb interaction between the states (m, n, k, l).
        Note that only the matrix elements k>l, n>m have to be specified.
        On input list or array gets converted to dictionary.
    si : StateIndexing
        StateIndexing or StateIndexingDM object.
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
    mtype : type
        Defines the type of the Hamiltonian matrices. For example, float, complex, etc.
    ham_ssq_q : bool
        If ham_ssq_q is False then for 'ssq' indexing only Hamiltonian with lowest abs(sz) value
        is stored.
    valslst : list of ndarrays
        List containing ncharge entries, where valslst[charge] is an array having Hamiltonian
        eigenvalues for definite charge.
    vecslst : list of ndarrays
        List containing ncharge entries, where vecslst[charge] is an array corresponding to
        Hamiltonian eigenvectors for definite charge.
    hamlst : list of ndarrays
        List containing ncharge entries, where hamlst[charge] is an array corresponding to many-body
        Hamiltonian for definite charge.
    Ea : ndarray
        nmany by 1 array containing many-body Hamiltonian eigenvalues.
    """

    def __init__(self, hsingle, coulomb, si,
                 herm_hs=True, herm_c=False, m_less_n=True,
                 mtype=float, ham_ssq_q=False):
        """Initialization of the QuantumDot class."""
        self.si = si
        self.mtype = mtype
        self.ham_ssq_q = ham_ssq_q
        self.herm_hs = herm_hs
        self.herm_c = herm_c
        self.m_less_n = m_less_n
        self.hsingle = make_hsingle_dict(self, hsingle)
        self.coulomb = make_coulomb_dict(self, coulomb)
        self._init_hamiltonian()

    def _init_hamiltonian(self):
        si = self.si
        if si.indexing == 'sz':
            self.valslst = empty_szlst(si.nsingle, True)
            self.vecslst = empty_szlst(si.nsingle, True)
            self.hamlst = empty_szlst(si.nsingle, True)
        elif si.indexing == 'ssq':
            self.valslst = empty_ssqlst(si.nsingle, True)
            self.vecslst = empty_ssqlst(si.nsingle, True)
            self.hamlst = empty_szlst(si.nsingle, True)
        else:
            self.valslst = [None]*si.ncharge
            self.vecslst = [None]*si.ncharge
            self.hamlst = [None]*si.ncharge
        self.add(self.hsingle, self.coulomb, False)
        self.Ea = np.zeros(si.nmany, dtype=float)
        self.Ea_ext = None

    def add(self, hsingle=None, coulomb=None, updateq=True):
        """
        Adds a value to single particle Hamiltonian and Coulomb matrix elements
        and correspondingly redefines many-body Hamiltonian.

        Parameters
        ----------
        hsingle : dict
            Dictionary describing what single-particle Hamiltonian values to add.
            For example, hsingle[(state1, state2)] = value to add.
        coulomb : dict
            Dictionary describing what Coulomb matrix element values to add.
            For example, coulomb[(state1, state2, state3, state4)] = value to add.
        updateq : bool
            Specifies if the values of the single particle Hamiltonian
            and the Coulomb matrix elements will be updated.
            The many-body Hamiltonian will be updates in either case.
        """
        if updateq:
            hsingle = {} if hsingle is None else make_hsingle_dict(self, hsingle)
            coulomb = {} if coulomb is None else make_coulomb_dict(self, coulomb)
            for j0 in hsingle:
                if j0 in self.hsingle:
                    self.hsingle[j0] += hsingle[j0]
                else:
                    self.hsingle.update({j0: hsingle[j0]})
            for j0 in coulomb:
                if j0 in self.coulomb:
                    self.coulomb[j0] += coulomb[j0]
                else:
                    self.hsingle.update({j0: coulomb[j0]})
        #
        if self.si.indexing == 'sz':
            for charge in range(self.si.ncharge):
                for sz in szrange(charge, self.si.nsingle):
                    szlst = self.si.get_lst(charge, sz)
                    szind = sz_to_ind(sz, charge, self.si.nsingle)
                    self.hamlst[charge][szind] = construct_ham_coulomb(self, coulomb, szlst,
                                                                       self.hamlst[charge][szind])
                    self.hamlst[charge][szind] = construct_ham_hopping(self, hsingle, szlst,
                                                                       self.hamlst[charge][szind])
        elif self.si.indexing == 'ssq':
            for charge in range(self.si.ncharge):
                szlow = -(charge % 2)
                szrng = szrange(charge, self.si.nsingle) if self.ham_ssq_q else [szlow]
                for sz in szrng:
                    szlst = self.si.get_lst(charge, sz)
                    szind = sz_to_ind(sz, charge, self.si.nsingle)
                    self.hamlst[charge][szind] = construct_ham_coulomb(self, coulomb, szlst,
                                                                       self.hamlst[charge][szind])
                    self.hamlst[charge][szind] = construct_ham_hopping(self, hsingle, szlst,
                                                                       self.hamlst[charge][szind])
        else:
            for charge in range(self.si.ncharge):
                chargelst = self.si.get_lst(charge)
                self.hamlst[charge] = construct_ham_coulomb(self, coulomb, chargelst,
                                                            self.hamlst[charge])
                self.hamlst[charge] = construct_ham_hopping(self, hsingle, chargelst,
                                                            self.hamlst[charge])

    def change(self, hsingle=None, coulomb=None, updateq=True):
        """
        Changes the values of the single particle Hamiltonian and Coulomb matrix elements
        and correspondingly redefines many-body Hamiltonian.

        Parameters
        ----------
        hsingle : dict
            Dictionary describing what single-particle Hamiltonian values to add.
            For example, hsingle[(state1, state2)] = value to add.
        coulomb : dict
            Dictionary describing what Coulomb matrix element values to add.
            For example, coulomb[(state1, state2, state3, state4)] = value to add.
        updateq : bool
            Specifies if the values of the single particle Hamiltonian
            and the Coulomb matrix elements will be updated.
            The many-body Hamiltonian will be updates in either case.
        """
        hsingle = {} if hsingle is None else make_hsingle_dict(self, hsingle, True)
        coulomb = {} if coulomb is None else make_coulomb_dict(self, coulomb)
        #
        hsingle_add = {}
        coulomb_add = {}
        # Find the differences from the previous single-particle Hamiltonian
        # and from the previous Coulomb matrix elements
        for j0 in hsingle:
            if j0 in self.hsingle:
                hsingle_diff = hsingle[j0]-self.hsingle[j0]
                if hsingle_diff != 0:
                    hsingle_add.update({j0: hsingle_diff})
                    if updateq:
                        self.hsingle[j0] += hsingle_diff
            else:
                hsingle_diff = hsingle[j0]
                if hsingle_diff != 0:
                    hsingle_add.update({j0: hsingle_diff})
                    if updateq:
                        self.hsingle.update({j0: hsingle_diff})
        for j0 in coulomb:
            if j0 in self.coulomb:
                coulomb_diff = coulomb[j0]-self.coulomb[j0]
                if coulomb_diff != 0:
                    coulomb_add.update({j0: coulomb_diff})
                    if updateq:
                        self.coulomb[j0] += coulomb_diff
            else:
                coulomb_diff = coulomb[j0]
                if coulomb_diff != 0:
                    coulomb_add.update({j0: coulomb_diff})
                    if updateq:
                        self.coulomb.update({j0: coulomb_diff})
        # Then add the differences
        self.add(hsingle_add, coulomb_add, False)

    def diagonalise_charge(self, charge=None, sz=None, ssq=None):
        """Diagonalises Hamiltonian for given charge.
           Additionally, spin projection 'sz' and total spin 'ssq' can be specified
           if those symmetries exist for considered model.

        Parameters
        ----------
        charge : int
            Charge under consideration.
        sz : int
            Spin projection.
        ssq : int
            Total spin.
        """
        if charge is None:
            return None
        elif sz is None:
            chargelst = self.si.get_lst(charge)
            (self.valslst[charge],
             self.vecslst[charge]) = (
                construct_manybody_eigenstates(self, self.hsingle, self.coulomb,
                                               chargelst, self.hamlst[charge]))
            return self.valslst[charge], self.vecslst[charge]
        elif ssq is None and (self.si.indexing == 'sz' or self.si.indexing == 'ssq'):
            szlst = self.si.get_lst(charge, sz)
            szind = sz_to_ind(sz, charge, self.si.nsingle)
            (self.valslst[charge][szind],
             self.vecslst[charge][szind]) = (
                construct_manybody_eigenstates(self, self.hsingle, self.coulomb,
                                               szlst, self.hamlst[charge][szind]))
            return self.valslst[charge][szind], self.vecslst[charge][szind]
        elif self.si.indexing == 'ssq':
            szind = sz_to_ind(sz, charge, self.si.nsingle)
            ssqind = ssq_to_ind(ssq, sz)
            (self.valslst[charge][szind][ssqind],
             self.vecslst[charge][szind][ssqind]) = (
                construct_manybody_eigenstates_ssq(self, charge, sz, ssq,
                                                   self.hsingle, self.coulomb,
                                                   self.hamlst[charge][szind]))
            return self.valslst[charge][szind][ssqind], self.vecslst[charge][szind][ssqind]
        else:
            print("WARNING: No indexing by 'sz' or 'ssq'. Return None.")
            return None

    def diagonalise(self):
        """Diagonalises Hamiltonians for all charge states."""
        if self.si.indexing == 'sz':
            for charge in range(self.si.ncharge):
                for sz in szrange(charge, self.si.nsingle):
                    self.diagonalise_charge(charge, sz)
        elif self.si.indexing == 'ssq':
            for charge in range(self.si.ncharge):
                szlow = -(charge % 2)
                szind = sz_to_ind(szlow, charge, self.si.nsingle)
                (self.valslst[charge], self.vecslst[charge]) = (
                    construct_manybody_eigenstates_ssq_all(self, charge, self.hsingle, self.coulomb,
                                                           self.hamlst[charge][szind]))
        else:
            for charge in range(self.si.ncharge):
                self.diagonalise_charge(charge)
        self.set_Ea()

    def set_Ea(self):
        """Sets the many-body eigenstates using construct_Ea_manybody()."""
        self.Ea[:] = construct_Ea_manybody(self.valslst, self.si)
