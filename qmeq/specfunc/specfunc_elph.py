"""Module containing various special functions."""

from scipy import pi
from scipy import exp
from scipy.integrate import quad


# -----------------------------------------------------------------------
class Func(object):
    """
    Sample class for functions which can be used to derive
    density of states functions.
    """
    def eval(self, x):
        return 1.


class FuncBose(Func):
    """Bose function."""
    def eval(self, x):
        return 1/(exp(x)-1)


class FuncPauliElPh(object):
    """
    Class for function used when generating Pauli master equation kernel
    due to electron-phonon coupling.
    """

    def __init__(self, tlst, dlst, bath_func, eps):
        self.tlst, self.dlst = tlst, dlst
        self.bath_func = bath_func
        self.bath_func_q = False if bath_func is None else True
        self.eps = eps
        #
        self.bose, self.dos = FuncBose(), Func()
        self.val = 0

    def eval(self, Ebbp, l):
        T, omm, omp = self.tlst[l], self.dlst[l, 0], self.dlst[l, 1]
        # alpha, Rm, Rp = Ebbp/T, omm/T, omp/T
        alpha = max(abs(Ebbp/T), self.eps) * (1 if Ebbp >= 0 else -1)
        Rm, Rp = max(omm/T, 0.9*self.eps), omp/T
        if self.bath_func_q:
            self.dos = self.bath_func[l]
        if Rm < alpha < Rp:
            # Absorption
            self.val = 2*pi*self.bose.eval(alpha)*self.dos.eval(T*alpha)
        elif Rm < -alpha < Rp:
            # Emission
            self.val = 2*pi*(1+self.bose.eval(-alpha))*self.dos.eval(-T*alpha)
        else:
            self.val = 0


class Func1vNElPh(object):
    """
    Class for function used when generating 1vN or Redfield approach kernel
    due to electron-phonon coupling.
    """

    def __init__(self, tlst, dlst, itype, limit, bath_func, eps):
        self.tlst, self.dlst = tlst, dlst
        self.itype, self.limit = itype, limit
        self.bath_func = bath_func
        self.bath_func_q = False if bath_func is None else True
        self.eps = eps
        #
        self.bose, self.dos = FuncBose(), Func()
        self.val0, self.val1 = 0., 0.

    def iplus(self, x):
        return +self.dos.eval(self.T*x)*self.bose.eval(x)

    def iminus(self, x):
        return -self.dos.eval(self.T*x)*(1.+self.bose.eval(x))

    def eval(self, Ebbp, l):
        T, omm, omp = self.tlst[l], self.dlst[l, 0], self.dlst[l, 1]
        # alpha, Rm, Rp = Ebbp/T, omm/T, omp/T
        alpha = max(abs(Ebbp/T), self.eps) * (1 if Ebbp >= 0 else -1)
        Rm, Rp = max(omm/T, 0.9*self.eps), omp/T
        self.T = T
        if self.bath_func_q:
            self.dos = self.bath_func[l]
        if self.itype == 0:
            self.val0, err = quad(self.iplus, Rm, Rp,
                                  weight='cauchy', wvar=alpha, epsabs=1.0e-6, epsrel=1.0e-6, limit=self.limit)
            self.val0 = self.val0 + (-1.0j*pi*self.iplus(alpha) if Rm < alpha < Rp else 0)
            self.val1, err = quad(self.iminus, Rm, Rp,
                                  weight='cauchy', wvar=alpha, epsabs=1.0e-6, epsrel=1.0e-6, limit=self.limit)
            self.val1 = self.val1 + (-1.0j*pi*self.iminus(alpha) if Rm < alpha < Rp else 0)
        elif self.itype == 2:
            self.val0 = -1.0j*pi*self.iplus(alpha) if Rm < alpha < Rp else 0
            self.val1 = -1.0j*pi*self.iminus(alpha) if Rm < alpha < Rp else 0
# -----------------------------------------------------------------------
