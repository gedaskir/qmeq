"""Module for indexing many-body states using Lin tables."""

import itertools

import numpy as np
try:
    from scipy.special import factorial
except ImportError:
    # For backwards compatibility with older versions of SciPy
    from scipy.misc import factorial

from .wrappers.mytypes import boolnp
from .wrappers.mytypes import longnp


def binarylist_to_integer(lst):
    """
    Convert a binary number in a form of a list to an integer.

    Parameters
    ----------
    lst : list
        List containing digits of a binary number.

    Returns
    -------
    int
        A decimal integer.
    """
    return int(''.join(map(str, lst)), 2)


def integer_to_binarylist(num, binlen=0, strq=False):
    """
    Convert an integer number to a binary number in a form of a list.

    Parameters
    ----------
    num : int
        Integer number.
    binlen : int
        Length of a list containing digits of a binary number.
    strq : bool
        If true then returns a string instead of a list.

    Returns
    -------
    list
        A list containing digits of a binary number.
    """
    value = format(num, '0'+str(binlen)+'b')
    rez = value if strq else list(map(int, str(value)))
    return rez


def construct_chargelst(nsingle):
    """
    Makes list of lists containing Lin indices of the states for given charge.

    Parameters
    ----------
    nsingle : int
        Number of single particle states.

    Returns
    -------
    chargelst : list of lists
        chargelst[charge] gives a list of state indices for given charge,
        chargelst[charge][ind] gives state index.
    """
    nmany = np.power(2, nsingle)
    chargelst = [[] for _ in range(nsingle+1)]
    # Iterate over many-body states
    for j1 in range(nmany):
        state = integer_to_binarylist(j1, nsingle)
        chargelst[sum(state)].append(j1)
    return chargelst


def sz_to_ind(sz, charge, nsingle):
    """
    Converts :math:`S_{z}` to a list index.

    Parameters
    ----------
    sz : int
        Value :math:`S_{z}` of a spin projection in the z direction.
    charge : int
        Value of the charge.
    nsingle : int
        Number of single particle states.

    Returns
    -------
    int
        Value of the list index corresponding to sz.
    """
    szmax = min(charge, nsingle-charge)
    return int((szmax+sz)/2)


def szrange(charge, nsingle):
    """
    Make a list giving :math:`S_{z}` values for given charge.

    Parameters
    ----------
    charge : int
        Value of the charge.
    nsingle : int
        Number of single particle states.

    Returns
    -------
    list
        List containing :math:`S_{z}` values for given charge.
    """
    szmax = min(charge, nsingle-charge)
    return list(range(-szmax, szmax+1, +2))
    # return range(szmax, -szmax-1, -2)


def empty_szlst(nsingle, noneq=False):
    """
    Make an empty list of lists corresponding to different charges and :math:`S_{z}` values.

    Parameters
    ----------
    nsingle : int
        Number of single particle states.
    noneq : bool
        If True the list contains None objects.
        If False the list contains empty lists [].

    Returns
    -------
    list of lists
        Contains None objects or [].
    """
    ncharge = nsingle+1
    if noneq:
        return [[None for _ in szrange(i, nsingle)] for i in range(ncharge)]
    else:
        return [[[] for _ in szrange(i, nsingle)] for i in range(ncharge)]


def construct_szlst(nsingle):
    """
    Makes list of lists of lists containing Lin indices of the states for given charge
    and spin :math:`S_{z}`.

    Parameters
    ----------
    nsingle : int
        Number of single particle states.

    Returns
    -------
    szlst : list of lists of lists
        szlst[charge] gives a list of lists of state indices for given charge,
        szlst[charge][sz] gives a list corresponding to charge and sz,
        szlst[charge][sz][ind] gives state index.
    """
    nmany = np.power(2, nsingle)
    szlst = empty_szlst(nsingle)
    # Iterate over many-body states
    for j1 in range(nmany):
        state = integer_to_binarylist(j1, nsingle)
        charge = sum(state)
        sz = sum(state[0:int(nsingle/2)])-sum(state[int(nsingle/2):nsingle])
        szind = sz_to_ind(sz, charge, nsingle)
        szlst[charge][szind].append(j1)
    return szlst


def ssq_to_ind(ssq, sz):
    """
    Convert the value of :math:`S^{2}` for a given :math:`S_{z}` to an index.

    Parameters
    ----------
    ssq : int
        Value of :math:`S^{2}`.
    sz : int
        Value of :math:`S_{z}`.

    Returns
    -------
    int
        An index corresponding to given :math:`S^{2}` and :math:`S_{z}`.
    """
    return int((ssq-abs(sz))/2)


def ssqrange(charge, sz, nsingle):
    """
    Make a list giving all possible :math:`S^{2}` values for given charge and :math:`S_{z}`.

    Parameters
    ----------
    charge : int
        Value of the charge.
    sz : int
        Value of sz.
    nsingle : int
        Number of single particle states.

    Returns
    -------
    list
        List of all possible :math:`S^{2}` values for given charge and :math:`S_{z}`.
    """
    szmax = min(charge, nsingle-charge)
    return list(range(abs(sz), szmax+1, +2))


def empty_ssqlst(nsingle, noneq=False):
    """
    Make an empty list of lists of lists corresponding to
    different charge, :math:`S_{z}`, :math:`S^{2}` values.

    Parameters
    ----------
    nsingle : int
        Number of single particle states.
    noneq : bool
        If True the list contains None objects.
        If False the list contains empty lists [].

    Returns
    -------
    list of lists of lists
        Contains None objects or [].
    """
    ncharge = nsingle+1
    if noneq:
        return [[[None for _ in ssqrange(i, j, nsingle)] for j in szrange(i, nsingle)] for i in range(ncharge)]
    else:
        return [[[[] for _ in ssqrange(i, j, nsingle)] for j in szrange(i, nsingle)] for i in range(ncharge)]


def construct_ssqlst(szlst, nsingle):
    """
    Makes list of lists of lists of lists containing Lin indices of the states
    for given charge, spin projection :math:`S_{z}`, and :math:`S^{2}` value.

    Parameters
    ----------
    szlst : list of lists
        List containing the possible values of :math:`S_{z}` for given charge.
        szlst[charge][szind] is an integer corresponding to :math:`S_{z}`.
    nsingle : int
        Number of single particle states.

    Returns
    -------
    sqqlst : list of lists of lists of lists
        ssqlst[charge] gives a list of lists of lists of state indices for given charge,
        ssqlst[charge][sz] gives a list of lists corresponding to charge and :math:`S_{z}`,
        ssqlst[charge][sz][ssq] gives a list corresponding to charge, :math:`S_{z}`, and :math:`S^{2}`.
        ssqlst[charge][sz][ssq][ind] gives a state index.
    """
    ncharge = nsingle+1
    ssqlst = empty_ssqlst(nsingle)
    ssqcount = [[0 for _ in ssqrange(i, i % 2, nsingle)] for i in range(ncharge)]
    # Find the number of multiplets for given charge
    for j1 in range(ncharge):
        ssqr = ssqrange(j1, j1 % 2, nsingle)
        lssqr = len(ssqr)
        for j2 in range(lssqr):
            ssqind = ssq_to_ind(ssqr[lssqr-j2-1], 0)
            ssqcount[j1][ssqind] = len(szlst[j1][j2]) - (len(szlst[j1][j2-1]) if j2 > 0 else 0)
    # For Lin index charge, sz, and ssq are assigned
    ind = 0
    for charge in range(ncharge):
        for sz in szrange(charge, nsingle):
            szind = sz_to_ind(sz, charge, nsingle)
            for ssq in ssqrange(charge, sz, nsingle):
                ssqind = ssq_to_ind(ssq, sz)
                ssqind2 = ssq_to_ind(ssq, 0)
                for j1 in range(ssqcount[charge][ssqind2]):
                    ssqlst[charge][szind][ssqind].append(ind)
                    ind += 1
    return ssqlst


def flatten(lst):
    """
    Flattens one level of a list.

    Parameters
    ----------
    lst : list

    Returns
    -------
    list
        List flattened by one level.
    """
    return list(itertools.chain.from_iterable(lst))


def enum_chargelst(chargelst_lin):
    """
    Make a list of integers from 0 to 2^nsingle having nesting of chargelst_lin.

    Parameters
    ----------
    chargelst_lin : list of lists
        Constructed by construct_chargelst(nsingle).

    Returns
    -------
    chargelst : list of lists
    """
    ncharge = len(chargelst_lin)
    chargelst = [[] for _ in range(ncharge)]
    counter1 = 0
    for j1 in range(ncharge):
        counter2 = counter1 + len(chargelst_lin[j1])
        chargelst[j1] = list(range(counter1, counter2))
        counter1 = counter2
    return chargelst


def enum_szlst(szlst_lin):
    """
    Make a list of integers from 0 to 2^nsingle having nesting of szlst_lin.

    Parameters
    ----------
    szlst_lin : list of lists of lists
        Constructed by construct_szlst(nsingle).

    Returns
    -------
    szlst : list of lists of lists
    """
    ncharge = len(szlst_lin)
    nsingle = ncharge-1
    szlst = empty_szlst(nsingle)
    # [[[] for j in szrange(i, nsingle)] for i in range(ncharge)]
    counter1 = 0
    for j1 in range(ncharge):
        for j2 in range(len(szlst_lin[j1])):
            counter2 = counter1 + len(szlst_lin[j1][j2])
            szlst[j1][j2] = list(range(counter1, counter2))
            counter1 = counter2
    return szlst


def make_inverse_map(lst):
    """
    Make a map from Lin index to a state index.

    Parameters
    ----------
    lst : list
        List containing Lin indices.

    Returns
    -------
    rez : list
        List containing a map from Lin index to a state index.
    """
    rez = [0]*len(lst)
    for j1 in range(len(lst)):
        rez[lst[j1]] = j1
    return rez


def make_quantum_numbers(si):
    """
    Make dictionaries between the state indices and
    quantum numbers corresponding to the states.

    Parameters
    ----------
    si : StateIndexing
        StateIndexing or StateIndexingDM object.

    Returns
    -------
    qn_ind : dict
        Dictionary from quantum number to a state.
    ind_qn : dict
        Dictionary from state index to quantum numbers of the state.
    """
    ncharge = si.ncharge
    nsingle = si.nsingle
    qn_ind = {}
    ind_qn = {}
    if si.indexing == 'ssq':
        for charge in range(ncharge):
            for sz in szrange(charge, nsingle):
                szind = sz_to_ind(sz, charge, nsingle)
                for ssq in ssqrange(charge, sz, nsingle):
                    ssqind = ssq_to_ind(ssq, sz)
                    for alpha in range(len(si.ssqlst[charge][szind][ssqind])):
                        ind = si.ssqlst[charge][szind][ssqind][alpha]
                        qn_ind.update({(charge, sz, ssq, alpha): ind})
                        ind_qn.update({ind: (charge, sz, ssq, alpha)})
    elif si.indexing == 'sz':
        for charge in range(ncharge):
            for sz in szrange(charge, nsingle):
                szind = sz_to_ind(sz, charge, nsingle)
                for alpha in range(len(si.szlst[charge][szind])):
                    ind = si.szlst[charge][szind][alpha]
                    qn_ind.update({(charge, sz, alpha): ind})
                    ind_qn.update({ind: (charge, sz, alpha)})
    else:
        for charge in range(ncharge):
            for alpha in range(len(si.chargelst[charge])):
                ind = si.chargelst[charge][alpha]
                qn_ind.update({(charge, alpha): ind})
                ind_qn.update({ind: (charge, alpha)})
    return qn_ind, ind_qn


class StateIndexing(object):
    """
    Class for indexing many-body states.

    Attributes
    ----------
    i, j : list
        List of integers containing indexing.
    indexing : str
        String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        Note that 'sz' indexing for Fock states is used for 'ssq' indexing, with
        additional specification of eigenstates in Fock basis.
    symmetry : str
            String determining if the states will be augmented by a symmetry.
            Possible value is 'spin'.
    ncharge : int
        Number of charge states.
    nmany : int
        Number of many-body states.
    nsingle : int
        Number of single-particle states.
    nleads : int
        Number of leads. Set to zero by default.
    chargelst_lin : list of lists
        chargelst_lin[charge] gives a list of state Lin indices for given charge,
        chargelst_lin[charge][ind] gives state Lin index
    szlst_lin : list of lists
        szlst_lin[charge] gives a list of lists of state indices for given charge,
        szlst_lin[charge][sz] gives a list corresponding to charge and sz,
        szlst_lin[charge][sz][ind] gives state Lin index
    chargelst : list of lists
        Indices of states for different charges for chosen indexing.
    szlst : list of lists
        Indices of states for different :math:`S_{z}` values for chosen indexing.
    ssqlst : list of lists of lists
        Indices of states for different :math:`S_{z}` and :math:`S^{2}` values.
    qn_ind : dictionary
        Maps quantum number to state index
    ind_qn : dictionary
        Maps state index to quantum numbers
    states_order : list
        Order of states used by many-body state printing functions in qmeq.various
        print_state() and print_all_states()
    """

    def __init__(self, nsingle, indexing='Lin', symmetry=None, nleads=0, nbaths=0):
        """
        Initialization of the StateIndexing class

        Parameters
        ----------
        nsingle : int
            Number of single-particle states.
        indexing : str
            String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        symmetry : str
            String determining that the states will be augmented by the symmetry.
            Possible value is 'spin'.
        """
        #
        self.nsingle_sym = nsingle//2 if symmetry == 'spin' else nsingle
        self.nsingle = nsingle
        self.indexing = indexing
        self.symmetry = symmetry
        self.ncharge = nsingle+1
        self.nmany = 2**nsingle
        self.nleads = nleads
        self.nleads_sym = nleads//2 if symmetry == 'spin' else nleads
        self.nbaths = nbaths
        #
        self.szlst_lin = None
        self.szlst = None
        self.ssqlst = None
        if indexing == 'charge':
            self.chargelst_lin = construct_chargelst(nsingle)
            self.chargelst = enum_chargelst(self.chargelst_lin)
            self.i = flatten(self.chargelst_lin)
        elif (indexing == 'sz' or indexing == 'ssq') and self.nsingle % 2 == 0:
            self.chargelst_lin = construct_chargelst(nsingle)
            self.chargelst = enum_chargelst(self.chargelst_lin)
            self.szlst_lin = construct_szlst(self.nsingle)
            self.szlst = enum_szlst(self.szlst_lin)
            self.i = flatten(flatten(self.szlst_lin))
            if indexing == 'ssq':
                self.ssqlst = construct_ssqlst(self.szlst, nsingle)
        elif (indexing == 'sz' or indexing == 'ssq') and self.nsingle % 2 != 0:
            print("WARNING: For 'sz' or 'ssq' indexing, nsingle has to be even. \
                   Using 'Lin' indexing.")
            self.indexing = 'Lin'
            self.chargelst = construct_chargelst(nsingle)
            self.i = list(range(self.nmany))
        elif indexing == 'Lin':
            self.chargelst = construct_chargelst(nsingle)
            self.i = list(range(self.nmany))
        else:
            print("WARNING: The indexing has to be 'Lin', 'charge', or 'sz'. \
                   Using 'Lin' indexing.")
            self.indexing = 'Lin'
            self.chargelst = construct_chargelst(nsingle)
            self.i = list(range(self.nmany))
        self.j = make_inverse_map(self.i)
        # Note that these quantum numbers to state and state to quantum numbers dictionaries
        # are necessary only for ssq indexing
        self.qn_ind, self.ind_qn = make_quantum_numbers(self)
        self.states_order = list(range(self.nmany))
        self.removed_fock_states = None
        self.nmany_ = self.nmany

        self.states_changed = True

    def get_state(self, ind, linq=False, strq=False):
        """
        Returns a list containing digits of a binary number corresponding to a state index.

        Parameters
        ----------
        ind : int
            Index of the state.
        linq : bool
            For linq=True uses Lin indexing, for linq=False uses specified indexing.
        strq : bool
            If true then returns a string instead of a list.

        Returns
        -------
        list
            List containing digits of a binary number corresponding to a state index.
        """
        if linq:
            return integer_to_binarylist(ind, self.nsingle, strq)
        else:
            return integer_to_binarylist(self.i[ind], self.nsingle, strq)

    def get_ind(self, state, linq=False):
        """
        Returns an index for given state

        Parameters
        ----------
        state : list
           List containing digits of a binary number.
        linq : bool
            For linq=True uses Lin indexing, for linq=False uses specified indexing

        Returns
        -------
        int
            Index of the state.
        """
        if linq:
            return binarylist_to_integer(state)
        else:
            return self.j[binarylist_to_integer(state)]

    def get_lst(self, charge=None, sz=None, ssq=None):
        """
        Gives a list of state Lin indices corresponding to charge or charge and sz.

        Parameters
        ----------
        charge : int
            Value of charge
        sz : int
            Value of sz
        ssq : int
            Value of total angular momentum ssq

        Returns
        -------
        list
            List of indices corresponding to given charge, chargelst[charge],
            or given charge and sz, szlst[charge][szind],
            or given charge, sz, and, ssq, szlst[charge][szind][ssqind].
        """
        if charge is None and sz is None:
            return None
        elif sz is None:
            return self.chargelst[charge]
        elif self.szlst is not None and ssq is None:
            szind = sz_to_ind(sz, charge, self.nsingle)
            return self.szlst[charge][szind]
        elif self.szlst is not None and self.ssqlst is not None:
            szind = sz_to_ind(sz, charge, self.nsingle)
            ssqind = ssq_to_ind(ssq, sz)
            return self.ssqlst[charge][szind][ssqind]
        else:
            print("WARNING: No indexing by 'sz' or 'ssq'. Returning charge list.")
            return self.chargelst[charge]

    def remove_fock_states(self, lin_state_indices):
        """
        Remove Fock states.

        lin_state_indices : list
            Lin state indices of Fock states to remove.
        """
        indexing = self.indexing
        if indexing == 'ssq':
            print("WARNING: For 'ssq' indexing removal of Fock states is not supported.")
            return

        if self.removed_fock_states is None:
            self.removed_fock_states = np.zeros(self.nmany, dtype=boolnp)

        for ind in lin_state_indices:
            self.removed_fock_states[ind] = True
        self.nmany = self.nmany_ - sum(self.removed_fock_states)

        chargelst_lin = self.chargelst if self.indexing == 'Lin' else self.chargelst_lin
        ncharge = self.ncharge
        for charge in range(ncharge):
            charge_states = []
            for j1 in chargelst_lin[charge]:
                if not self.removed_fock_states[j1]:
                    charge_states.append(j1)
                else:
                    self.j[j1] = None
            chargelst_lin[charge] = charge_states

        if self.indexing == 'charge' or self.indexing == 'Lin':
            self.i = flatten(chargelst_lin)
            self.chargelst = enum_chargelst(chargelst_lin)
        elif self.indexing == 'sz':
            nsingle = self.nsingle
            for charge in range(ncharge):
                charge_states = []
                for sz in szrange(charge, nsingle):
                    szind = sz_to_ind(sz, charge, nsingle)
                    sz_states = []
                    for j1 in self.szlst_lin[charge][szind]:
                        if not self.removed_fock_states[j1]:
                            sz_states.append(j1)
                        else:
                            self.j[j1] = None
                    charge_states.append(sz_states)
                self.szlst_lin[charge] = charge_states

            self.chargelst = enum_chargelst(chargelst_lin)
            self.szlst = enum_szlst(self.szlst_lin)
            self.i = flatten(flatten(self.szlst_lin))

        for j1 in range(len(self.i)):
            self.j[self.i[j1]] = j1

        self.qn_ind, self.ind_qn = make_quantum_numbers(self)

        self.states_changed = True


class StateIndexingPauli(StateIndexing):
    """
    Class for indexing diagonal density matrix elements (Pauli master equation).
    Derived from StateIndexing.

    Attributes
    ----------
    statesdm : list
        List of states considered in calculations involving density matrix.
    shiftlst0 : numpy array
        Array necessary for labeling states for Pauli master equation. See method get_ind_dm0().
    dictdm : numpy array
        Array giving enumeration of considered many-body states in statesdm.
    mapdm0 : list
        List showing which states are mapped to each other due to symmetries.
        For example mapdm0[1]=1, mapdm0[2]=1, shows that 1 and 2 are equivalent.
        Also mapdm0[ind]=-1 shows that the state is excluded.
    booldm0 : list
        List giving states which will be used. Other states will be mapped to states,
        which have booldm0[ind]=True.
        From example for mapdm0 we have booldm0[1]=True, booldm0[2]=False.
    """

    def __init__(self, nsingle, indexing='Lin', symmetry=None, nleads=0):
        """
        Initialization of the StateIndexingDM class

        Parameters
        ----------
        nsingle : int
            Number of single-particle states.
        indexing : str
            String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        symmetry : str
            String determining that the states will be augmented by the symmetry.
        """
        StateIndexing.__init__(self, nsingle, indexing, symmetry, nleads)
        self.shiftlst0 = np.zeros(self.ncharge+1, dtype=longnp)
        self.dictdm = np.zeros(self.nmany, dtype=longnp)
        self.statesdm = None
        self.mapdm0, self.booldm0 = None, None
        self.set_statesdm(self.chargelst)

    def set_statesdm(self, statesdm):
        """
        Sets the considered many-body states for Pauli master equation calculations.

        Parameters
        ----------
        statesdm: list
            List containing indices of many-body state under consideration.
        """
        self.statesdm = statesdm
        self.statesdm.append([])
        self.npauli_ = 0
        for j1 in range(self.ncharge):
            self.npauli_ += len(statesdm[j1])
        self.set_dictdm()

        self.states_changed = True

    def set_dictdm(self):
        """
        Makes dictdm, shiftlst0 for Pauli master equation indexing.
        """
        for j1 in range(self.ncharge):
            self.shiftlst0[j1+1] = self.shiftlst0[j1] + len(self.statesdm[j1])
            counter = 0
            for j2 in self.statesdm[j1]:
                self.dictdm[j2] = counter
                counter += 1
        self.set_mapdm()

    def set_mapdm(self):
        """
        Reduce the number of diagonal matrix elements by using symmetries.
        """
        # noinspection PyShadowingNames
        def add_elem(counter, b, bp, charge, dictq=True):
            bbp = self.get_ind_dm0(b, bp, charge, maptype=0)
            self.mapdm0[bbp] = counter
            self.booldm0[bbp] = dictq
            # if dictq:
            #     self.inddm0.update({counter:(b, bp)})
            return counter+1
        #
        self.mapdm0 = np.ones(self.npauli_, dtype=longnp)*(-1)
        self.booldm0 = np.zeros(self.npauli_, dtype=boolnp)
        # self.inddm0 = {}
        counter = 0
        # Diagonal density matrix elements
        for charge in range(self.ncharge):
            for b in self.statesdm[charge]:
                if self.indexing == 'ssq':
                    (b_ch,  b_sz,  b_ssq,  b_alpha) = self.ind_qn[b]
                    if b_ssq == -b_sz:
                        add_elem(counter, b, b, charge)
                        for sz in range(b_sz+2, b_ssq+1, 2):
                            b1 = self.qn_ind[(b_ch,  sz, b_ssq,  b_alpha)]
                            add_elem(counter, b1, b1, charge, dictq=False)
                        counter = counter+1
                else:
                    counter = add_elem(counter, b, b, charge)
        self.npauli = counter

    def get_ind_dm0(self, b, bp, charge, maptype=1):
        """
        Get the state index for Pauli master equation.

        Parameters
        ----------
        b,bp : int
            Indices of many-body states within the same charge state.
            Note that bp is just a dummy variable for Pauli master equation.
        charge : int
            Charge of b and bp.
        maptype : int
            Determines what kind of mapping to use when returning the index.

        Returns
        -------
        int
            Index of the zeroth order density matrix element.
        """
        if maptype == 0:
            return self.dictdm[b] + self.shiftlst0[charge]
        elif maptype == 1:
            return self.mapdm0[self.dictdm[b] + self.shiftlst0[charge]]
        elif maptype == 2:
            return self.booldm0[self.dictdm[b] + self.shiftlst0[charge]]

    def remove_fock_states(self, lin_state_indices):
        StateIndexing.remove_fock_states(self, lin_state_indices)
        self.set_statesdm(self.chargelst)


class StateIndexingDM(StateIndexing):
    """
    Class for indexing density matrix elements.
    Separates the off-diagonal density matrix elements into real and imaginary parts.
    Derived from StateIndexing.

    Attributes
    ----------
    ndm0 : int
        Number of density matrix elements to zeroth order corresponding to statesdm,
        when symmetries are used.
    ndm0r : int
        Number of unique density matrix elements to zeroth order corresponding to statesdm,
        when symmetries are used.
        The real and imaginary parts of the off-diagonal matrix element are considered
        as separate entities.
    ndm0_ : int
        Number of density matrix elements to zeroth order corresponding to statesdm,
        when no symmetries are used.
    ndm0_tot : int
        Total number of density matrix elements to zeroth order for given nsingle,
        when no symmetries are used.
    ndm1, ndm1_ : int, int
        Number of density matrix elements to first order corresponding to statesdm,
        when no symmetries are used.
    ndm1_tot : int
        Total number of density matrix elements to first order for given nsingle,
        when no symmetries are used.
    npauli_ : int
        Number of diagonal density matrix elements to zeroth order corresponding to statesdm,
        when no symmetries are used.
    npauli : int
        Number of diagonal density matrix elements to zeroth order corresponding to statesdm,
        when symmetries are used.
    statesdm : list
        List of states considered in calculations involving density matrix.
    shiftlst0 : numpy array
        Array necessary for labeling zeroth order density matrix elements. See method get_ind_dm0().
    shiftlst1 : numpy array
        Array necessary for labeling first order density matrix elements. See method get_ind_dm1().
    lenlst : numpy array
        Array containing number of many-body states for each charge. lenlst[charge]=number of states.
    dictdm : numpy array
        Array giving enumeration of considered many-body states in statesdm.
    inddm0 : dict
        Dictionary between index of density matrix element and pair of many-body states (a,b).
    mapdm0 : list
        List showing which density matrix elements are mapped to each other due to symmetries.
        For example mapdm0[1]=1, mapdm0[2]=1, shows that 1 and 2 are equivalent.
        Also mapdm0[ind]=-1 shows that the state is excluded.
    booldm0 : list
        List giving density matrix elements which will be used.
        Other elements will be mapped to elements, which have booldm0[ind]=True.
        From example for mapdm0 we have booldm0[1]=True, booldm0[2]=False.
    conjdm0 : list
        List showing, which density matrix elements are complex conjugate and are not unique.
    """

    def __init__(self, nsingle, indexing='Lin', symmetry=None, nleads=0):
        """
        Initialization of the StateIndexingDM class

        Parameters
        ----------
        nsingle : int
            Number of single-particle states.
        indexing : str
            String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        symmetry : str
            String determining that the states will be augmented by the symmetry.
        """
        StateIndexing.__init__(self, nsingle, indexing, symmetry, nleads)
        self.ndm0_tot = int(factorial(2*self.nsingle)/factorial(self.nsingle)**2)
        self.ndm1_tot = int(self.nsingle/(self.nsingle+1)*self.ndm0_tot)
        #
        self.shiftlst0 = np.zeros(self.ncharge+1, dtype=longnp)
        self.shiftlst1 = np.zeros(self.ncharge, dtype=longnp)
        self.lenlst = np.zeros(self.ncharge, dtype=longnp)
        self.dictdm = np.zeros(self.nmany, dtype=longnp)
        #
        self.statesdm, self.mapdm0, self.mapdm1 = None, None, None
        self.set_statesdm(self.chargelst)

    def set_statesdm(self, statesdm):
        """
        Sets the considered many-body states for density matrix calculations.

        Parameters
        ----------
        statesdm: list
            List containing indices of many-body state under consideration.
        """
        self.statesdm = list(statesdm)
        self.statesdm.append([])
        self.ndm0_, self.ndm1_, self.npauli_ = 0, 0, 0
        for j1 in range(self.ncharge):
            self.npauli_ += len(statesdm[j1])
            self.ndm0_ += len(statesdm[j1])**2
            if j1 < self.ncharge-1:
                self.ndm1_ += len(statesdm[j1])*len(statesdm[j1+1])
        self.set_dictdm()

        self.states_changed = True

    def set_dictdm(self):
        """
        Makes dictdm, shiftlst0, and shiftlst1 necessary for density-matrix element indexing.
        """
        for j1 in range(self.ncharge):
            self.lenlst[j1] = len(self.statesdm[j1])
            self.shiftlst0[j1+1] = self.shiftlst0[j1] + len(self.statesdm[j1])**2
            if j1 < self.ncharge-1:
                self.shiftlst1[j1+1] = (self.shiftlst1[j1]
                                        + len(self.statesdm[j1])*len(self.statesdm[j1+1]))
            counter = 0
            for j2 in self.statesdm[j1]:
                self.dictdm[j2] = counter
                counter += 1
        self.set_mapdm()

    def set_mapdm(self):
        """
        Reduce the number of diagonal matrix elements by using symmetries.
        """
        # noinspection PyShadowingNames
        def add_elem(counter, b, bp, charge, dictq=True, conjq=True):
            bbp = self.get_ind_dm0(b, bp, charge, maptype=0)
            self.mapdm0[bbp] = counter
            self.booldm0[bbp] = dictq
            self.conjdm0[bbp] = conjq
            if dictq:
                self.inddm0.update({counter: (b, bp)})
            return counter+1
        #
        self.mapdm0 = np.ones(self.ndm0_, dtype=longnp)*(-1)
        self.booldm0 = np.zeros(self.ndm0_, dtype=boolnp)
        self.conjdm0 = np.zeros(self.ndm0_, dtype=boolnp)
        self.inddm0 = {}
        counter = 0
        # Diagonal density matrix elements
        for charge in range(self.ncharge):
            for b in self.statesdm[charge]:
                if self.indexing == 'ssq':
                    (b_ch,  b_sz,  b_ssq,  b_alpha) = self.ind_qn[b]
                    if b_ssq == -b_sz:
                        add_elem(counter, b, b, charge)
                        for sz in range(b_sz+2, b_ssq+1, 2):
                            b1 = self.qn_ind[(b_ch,  sz, b_ssq,  b_alpha)]
                            add_elem(counter, b1, b1, charge, dictq=False)
                        counter = counter+1
                else:
                    counter = add_elem(counter, b, b, charge)
        self.npauli = counter
        # Off-diagonal density matrix elements
        for charge in range(self.ncharge):
            for b, bp in itertools.combinations(self.statesdm[charge], 2):
                if self.indexing == 'sz':
                    if self.ind_qn[b][1] == self.ind_qn[bp][1]:
                        add_elem(counter, bp, b, charge, dictq=False, conjq=False)
                        counter = add_elem(counter, b, bp, charge)
                elif self.indexing == 'ssq':
                    (b_ch,  b_sz,  b_ssq,  b_alpha) = self.ind_qn[b]
                    (bp_ch, bp_sz, bp_ssq, bp_alpha) = self.ind_qn[bp]
                    if (b_sz, b_ssq) == (bp_sz, bp_ssq):
                        if b_ssq == -b_sz:
                            add_elem(counter, bp, b, charge, dictq=False, conjq=False)
                            add_elem(counter, b, bp, charge)
                            for sz in range(b_sz+2, b_ssq+1, 2):
                                b1 = self.qn_ind[(b_ch,  sz, b_ssq,  b_alpha)]
                                b1p = self.qn_ind[(bp_ch, sz, bp_ssq, bp_alpha)]
                                add_elem(counter, b1p, b1, charge, dictq=False, conjq=False)
                                add_elem(counter, b1, b1p, charge, dictq=False)
                            counter = counter+1
                else:
                    add_elem(counter, bp, b, charge, dictq=False, conjq=False)
                    counter = add_elem(counter, b, bp, charge)
        self.ndm0 = counter
        self.ndm0r = self.npauli+2*(self.ndm0-self.npauli)
        self.ndm1 = self.ndm1_

    def get_ind_dm0(self, b, bp, charge, maptype=1):
        """
        Get the index of zeroth order density matrix element.

        Parameters
        ----------
        b,bp : int
            Indices of many-body states within the same charge state.
        charge : int
            Charge of b and bp.
        maptype : int
            Determines what kind of mapping to use when returning the index.

        Returns
        -------
        int
            Index of the zeroth order density matrix element.
        """
        # l = len(self.statesdm[charge])
        # i = self.dictdm[b]
        # j = self.dictdm[bp]
        # shiftas = self.shiftlst0[charge]
        # print('index is', l*i + j + shiftas)
        if maptype == 0:
            return (self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp]
                    + self.shiftlst0[charge])
        elif maptype == 1:
            return (self.mapdm0[self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp]
                    + self.shiftlst0[charge]])
        elif maptype == 2:
            return (self.booldm0[self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp]
                    + self.shiftlst0[charge]])
        elif maptype == 3:
            return (self.conjdm0[self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp]
                    + self.shiftlst0[charge]])

    def get_ind_dm1(self, c, b, bcharge):
        """
        Get the index of first order density matrix element.

        Parameters
        ----------
        c,b : int
            Indices of many-body states differing by single charge, charge(c)>charge(b).
        bcharge : int
            Charge of many-body state b.

        Returns
        -------
        int
            Index of the first order density matrix element.
        """
        # l = len(self.statesdm[charge])
        # i = self.dictdm[c]
        # j = self.dictdm[b]
        # shiftas = self.shiftlst1[charge]
        # print('index is', l*i + j + shiftas)
        #       b
        #     -----
        #   c |   |
        #     -----
        return self.lenlst[bcharge]*self.dictdm[c] + self.dictdm[b] + self.shiftlst1[bcharge]

    def remove_fock_states(self, lin_state_indices):
        StateIndexing.remove_fock_states(self, lin_state_indices)
        self.set_statesdm(self.chargelst)


class StateIndexingDMc(StateIndexing):
    """
    Class for indexing density matrix elements.
    Does not separates the off-diagonal density matrix elements into real and imaginary parts,
    and treats Phi[0]_{b,bp} and Phi[0]_{bp,b} as separate entities.
    Derived from StateIndexing.

    Attributes
    ----------
    Same as in StateIndexingDM.
    """

    def __init__(self, nsingle, indexing='Lin', symmetry=None, nleads=0):
        """
        Initialization of the StateIndexingDMc class

        Parameters
        ----------
        nsingle : int
            Number of single-particle states.
        indexing : str
            String determining type of the indexing. Possible values are 'Lin', 'charge', 'sz', 'ssq'.
        symmetry : str
            String determining that the states will be augmented by the symmetry.
        """
        StateIndexing.__init__(self, nsingle, indexing, symmetry, nleads)
        self.ndm0_tot = int(factorial(2*self.nsingle)/factorial(self.nsingle)**2)
        self.ndm1_tot = int(self.nsingle/(self.nsingle+1)*self.ndm0_tot)
        #
        self.shiftlst0 = np.zeros(self.ncharge+1, dtype=longnp)
        self.shiftlst1 = np.zeros(self.ncharge, dtype=longnp)
        self.lenlst = np.zeros(self.ncharge, dtype=longnp)
        self.dictdm = np.zeros(self.nmany, dtype=longnp)
        #
        self.statesdm, self.mapdm0, self.mapdm1 = None, None, None
        self.set_statesdm(self.chargelst)

    def set_statesdm(self, statesdm):
        """
        Sets the considered many-body states for density matrix calculations.

        Parameters
        ----------
        statesdm: list
            List containing indices of many-body state under consideration.
        """
        self.statesdm = statesdm
        self.statesdm.append([])
        self.ndm0_, self.ndm1_, self.npauli_ = 0, 0, 0
        for j1 in range(self.ncharge):
            self.npauli_ += len(statesdm[j1])
            self.ndm0_ += len(statesdm[j1])**2
            if j1 < self.ncharge-1:
                self.ndm1_ += len(statesdm[j1])*len(statesdm[j1+1])
        self.set_dictdm()

        self.states_changed = True

    def set_dictdm(self):
        """
        Makes dictdm, shiftlst0, and shiftlst1 necessary for density-matrix element indexing.
        """
        for j1 in range(self.ncharge):
            self.lenlst[j1] = len(self.statesdm[j1])
            self.shiftlst0[j1+1] = self.shiftlst0[j1] + len(self.statesdm[j1])**2
            if j1 < self.ncharge-1:
                self.shiftlst1[j1+1] = (self.shiftlst1[j1]
                                        + len(self.statesdm[j1])*len(self.statesdm[j1+1]))
            counter = 0
            for j2 in self.statesdm[j1]:
                self.dictdm[j2] = counter
                counter += 1
        self.set_mapdm()

    def set_mapdm(self):
        """
        Reduce the number of diagonal matrix elements by using symmetries.
        """
        # noinspection PyShadowingNames
        def add_elem(counter, b, bp, charge, dictq=True):
            bbp = self.get_ind_dm0(b, bp, charge, maptype=0)
            self.mapdm0[bbp] = counter
            self.booldm0[bbp] = dictq
            if dictq:
                self.inddm0.update({counter: (b, bp)})
            return counter+1
        #
        self.mapdm0 = np.ones(self.ndm0_, dtype=longnp)*(-1)
        self.booldm0 = np.zeros(self.ndm0_, dtype=boolnp)
        self.conjdm0 = None
        self.inddm0 = {}
        counter = 0
        # Diagonal density matrix elements
        for charge in range(self.ncharge):
            for b in self.statesdm[charge]:
                if self.indexing == 'ssq':
                    (b_ch,  b_sz,  b_ssq,  b_alpha) = self.ind_qn[b]
                    if b_ssq == -b_sz:
                        add_elem(counter, b, b, charge)
                        for sz in range(b_sz+2, b_ssq+1, 2):
                            b1 = self.qn_ind[(b_ch,  sz, b_ssq,  b_alpha)]
                            add_elem(counter, b1, b1, charge, dictq=False)
                        counter = counter+1
                else:
                    counter = add_elem(counter, b, b, charge)
        self.npauli = counter
        # self.npauli = 0
        # Off-diagonal density matrix elements
        for charge in range(self.ncharge):
            # for b, bp in itertools.product(self.statesdm[charge], self.statesdm[charge]):
            for b, bp in itertools.permutations(self.statesdm[charge], 2):
                if self.indexing == 'sz':
                    if self.ind_qn[b][1] == self.ind_qn[bp][1]:
                        counter = add_elem(counter, b, bp, charge)
                elif self.indexing == 'ssq':
                    (b_ch,  b_sz,  b_ssq,  b_alpha) = self.ind_qn[b]
                    (bp_ch, bp_sz, bp_ssq, bp_alpha) = self.ind_qn[bp]
                    if (b_sz, b_ssq) == (bp_sz, bp_ssq):
                        if b_ssq == -b_sz:
                            add_elem(counter, b, bp, charge)
                            for sz in range(b_sz+2, b_ssq+1, 2):
                                b1 = self.qn_ind[(b_ch,  sz, b_ssq,  b_alpha)]
                                b1p = self.qn_ind[(bp_ch, sz, bp_ssq, bp_alpha)]
                                add_elem(counter, b1, b1p, charge, dictq=False)
                            counter = counter+1
                else:
                    counter = add_elem(counter, b, bp, charge)
        self.ndm0 = counter
        self.ndm0r = self.npauli+2*(self.ndm0-self.npauli)
        self.ndm1 = self.ndm1_

    def get_ind_dm0(self, b, bp, charge, maptype=1):
        """
        Get the index of zeroth order density matrix element.

        Parameters
        ----------
        b,bp : int
            Indices of many-body states within the same charge state.
        charge : int
            Charge of b and bp.
        maptype : int
            Determines what kind of mapping to use when returning the index.

        Returns
        -------
        int
            Index of the zeroth order density matrix element.
        """
        # l = len(self.statesdm[charge])
        # i = self.dictdm[b]
        # j = self.dictdm[bp]
        # shiftas = self.shiftlst0[charge]
        # print('index is', l*i + j + shiftas)
        if maptype == 0:
            return self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp] + self.shiftlst0[charge]
        elif maptype == 1:
            return (self.mapdm0[self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp]
                    + self.shiftlst0[charge]])
        elif maptype == 2:
            return (self.booldm0[self.lenlst[charge]*self.dictdm[b] + self.dictdm[bp]
                    + self.shiftlst0[charge]])

    def get_ind_dm1(self, c, b, bcharge):
        """
        Get the index of first order density matrix element.

        Parameters
        ----------
        c,b : int
            Indices of many-body states differing by single charge, charge(c)>charge(b).
        bcharge : int
            Charge of many-body state b.

        Returns
        -------
        int
            Index of the first order density matrix element.
        """
        # l = len(self.statesdm[charge])
        # i = self.dictdm[c]
        # j = self.dictdm[b]
        # shiftas = self.shiftlst1[charge]
        # print('index is', l*i + j + shiftas)
        #       b
        #     -----
        #   c |   |
        #     -----
        return self.lenlst[bcharge]*self.dictdm[c] + self.dictdm[b] + self.shiftlst1[bcharge]

    def remove_fock_states(self, lin_state_indices):
        StateIndexing.remove_fock_states(self, lin_state_indices)
        self.set_statesdm(self.chargelst)
