"""Module containing Builder class."""

from .builder_base import BuilderBase
from .builder_base import BuilderManyBody
from .builder_elph import BuilderElPh
from .builder_elph import BuilderManyBodyElPh


# noinspection PyUnresolvedReferences
class Builder(BuilderBase):
    """
    Class for building the system for stationary transport calculations.

    This is a wrapper class which determines what kind of Builder object to use.
    The available options are:
    | Builder_base
    | Builder_many_body
    | Builder_elph
    | Builder_many_body_elph

    Use the factory methods to construct the relevant objects:
    | Builder.base(...)
    | Builder.many_body(...)
    | Builder.elph(...)
    | Builder.many_body_elph(...)

    For further documentation of particular builder (for example, Builder_elph)
    and it's attributes use: help(Builder_elph)

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
        additional specification of eigenstates in Fock basis.
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
                 large bandwidth D.
        itype=2: the principal parts are neglected.
        itype=3: the principal parts are neglected and infinite bandwidth D is assumed.
    dqawc_limit : int
        For itype=0 dqawc_limit determines the maximum number of sub-intervals
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

    @classmethod
    def base(cls, *args, **kwargs):
        return BuilderBase(*args, **kwargs)

    @classmethod
    def many_body(cls, *args, **kwargs):
        return BuilderManyBody(*args, **kwargs)

    @classmethod
    def elph(cls, *args, **kwargs):
        return BuilderElPh(*args, **kwargs)

    @classmethod
    def many_body_elph(cls, *args, **kwargs):
        return BuilderManyBodyElPh(*args, **kwargs)
