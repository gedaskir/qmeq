import numpy as np
from numpy.linalg import norm
from scipy import exp
from scipy.integrate import quad
from qmeq.specfunc import *
import qmeq

try:
    from qmeq.specfuncc import *
except:
    print("Cannot import Cython compiled module qmeq.specfuncc.")
    c_fermi_func = fermi_func
    c_func_pauli = func_pauli
    c_func_1vN = func_1vN

EPS = 1e-14
EPS2 = 1e-11

def test_fermi_func():
    for f in [fermi_func, c_fermi_func]:
        assert f(0) == 0.5
        assert norm(f(2)+f(-2) - 1.0) < EPS

def test_func_pauli():
    for f in [func_pauli, c_func_pauli]:
        Ecb, mu, T, Dm, Dp, itype = 0, 0, 1, -5, 5, 0
        assert norm(f(Ecb, mu, T, Dm, Dp, itype) - np.pi) < EPS
        fm = f(-2, mu, T, Dm, Dp, itype)
        fp = f(+2, mu, T, Dm, Dp, itype)
        assert norm(fm+fp - 2*np.pi*np.ones(2)) < EPS
        #
        for itype in {0, 2, 4}:
            Ecb = 4.99
            assert norm( f(Ecb, mu, T, Dm, Dp, itype) - [0.04247219960243555, 6.240713107577151] ) < EPS
            Ecb = 5.01
            assert f(Ecb, mu, T, Dm, Dp, itype).tolist() == [0., 0.]
        for itype in {1, 3}:
            assert norm( f(Ecb, mu, T, Dm, Dp, itype) - [0.04163676679420959, 6.241548540385377] ) < EPS

def test_func_1vN():

    def test_rez(Ecb, mu, T, Dm, Dp):
        def f0(x): return 1/(exp((x-mu)/T)+1)
        def f1(x): return -(1-1/(exp((x-mu)/T)+1))
        def f2(x): return x/(exp((x-mu)/T)+1)
        def f3(x): return -x*(1-1/(exp((x-mu)/T)+1))
        cur0, err = quad(f0, Dm, Dp, weight='cauchy', wvar=Ecb, epsabs=1.0e-6, epsrel=1.0e-6, limit=10000)
        cur1, err = quad(f1, Dm, Dp, weight='cauchy', wvar=Ecb, epsabs=1.0e-6, epsrel=1.0e-6, limit=10000)
        en0, err = quad(f2, Dm, Dp, weight='cauchy', wvar=Ecb, epsabs=1.0e-6, epsrel=1.0e-6, limit=10000)
        en1, err = quad(f3, Dm, Dp, weight='cauchy', wvar=Ecb, epsabs=1.0e-6, epsrel=1.0e-6, limit=10000)
        return np.array([cur0, cur1, en0, en1])

    for f in [func_1vN, c_func_1vN]:
        Ecb, mu, T, Dm, Dp = 0.5, 0.3, 1.0, -10., 15.
        rr0 = test_rez(Ecb, mu, T, Dm, Dp)
        rr1 = np.array([-2.4197253517177657, -2.8251904598259303, 8.790137324141117, -16.412595229912966])
        ri = np.array([-1.4142382069390027, 1.7273544466507904, -0.7071191034695014, 0.8636772233253952])
        #
        r0 = f(Ecb, mu, T, Dm, Dp, 0, 10000)
        r1 = f(Ecb, mu, T, Dm, Dp, 1, 10000)
        r2 = f(Ecb, mu, T, Dm, Dp, 2, 10000)
        assert norm(r0 - rr0-1j*ri) < EPS2
        assert norm(r1 - rr1-1j*ri) < EPS2
        assert norm(r2 - 1j*ri) < EPS2
        for itype in {0, 2}:
            r0 = f(Dp+0.1, mu, T, Dm, Dp, itype, 10000)
            assert norm(r0.imag - np.zeros(4)) < EPS
            r0 = f(Dm-0.1, mu, T, Dm, Dp, itype, 10000)
            assert norm(r0.imag - np.zeros(4)) < EPS
