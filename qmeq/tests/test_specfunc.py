import numpy as np
from numpy.linalg import norm
from numpy import exp
from scipy.integrate import quad
from qmeq.specfunc import *

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

def test_diff_fermi():
    for f in [diff_fermi, c_diff_fermi]:
        assert abs(f(0.55) - -0.23200764322418327) < EPS

def test_digamma():
    for f in [digamma, c_digamma]:
        temp = f(1 + 5j) - (1.612784844615746 + 1.4707963267949677j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(.01 + 5j) - (1.6125868048556047 + 1.668809372590705j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS

def test_polygamma():
    for f in [polygamma, c_polygamma]:
        temp = f(1 + 5j, 0) - (1.612784844615746 + 1.4707963267949677j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(.01 + 5j, 0) - (1.6125868048556047 + 1.668809372590705j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(1 + 5j, 1) - (0.01999999999955169 - 0.1986556763597955j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(1 + 5j, 2) - (0.03918887182635747 + 0.007999999997183265j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(.01 + 5j, 1) - (-0.01960787277389306 - 0.19873490840041347j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(.01 + 5j, 2) - (0.03923642415486224 - 0.007846344695290304j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS

def test_phi():
    for f in [phi, c_phi]:
        assert (f(10, 10000, 10000) - 6.925720809890757) < EPS

def test_delta_phi():
    for f in [delta_phi, c_delta_phi]:
        assert (f(10, -5, 1000, 1000) - 0.7599463446332866) < EPS

def test_diff_phi():
    for f in [diff_phi, c_diff_phi]:
        assert abs(c_diff_phi(-.10, 1000) - 42.54546520657971) < EPS

def test_diff2_phi():
    for f in [diff2_phi, c_diff2_phi]:
        assert abs(f(-.10, 1000) - -423.8096493945452) < EPS

def test_bose():
    for f in [bose, c_bose]:
        assert abs(f(0.2) - 4.516655566126994) < EPS

def test_BW_Ozaki():
    for f in [BW_Ozaki, c_BW_Ozaki]:
        b_and_R = f(100)
        assert np.allclose(b_and_R[:, 0], [0.007451490725924492, 0.02166498443734055, 0.03391482302880566,
                                           0.04542385072204902, 0.06366191745692201, 0.1061032953943358,
                                           0.3183098861837907])
        assert np.allclose(b_and_R[:, 1], [42.39056017683527, 4.545407954380445, 1.543756675702128, 1.0202523534327939,
                                           1.0000228395795256, 1.0000000000674338, 1.0000000000000013])
def test_X_integral():
    BW = 2e5
    b_and_R = BW_Ozaki(BW)
    for f in [integralX, c_integralX]:
        print(f.__name__)
        temp = f(1, 1, 10, 0, -15, 10, 20, -4, 2, BW, b_and_R, False) -(-0.08253203115377988+0.007904048913053564j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(1, 1, 15, 0, 15, 10, 20, -4, 2, BW, b_and_R, False) - (-0.0684892140474174+0.03968851698180709j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(-1, -1, 0, 100, 5, 10, 20, 0, 0, BW, b_and_R, False) - (0.06109101805561376-0.03673158282336319j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS

        assert abs(f(1, 1, 50, 1, 67, 10, 10, 0, 0, 10*BW, b_and_R, False) - -0.010434716600846468) < EPS
        assert abs(f(-1, 1, 50, 1, 67,10, 10, 0, 0, 10*BW, b_and_R, False) - 0.010434716600846468) < EPS
        assert abs(f(1, 1, 50, 1, 50, 10, 10, 0, 0, 10*BW,b_and_R, False) - -0.014316911535272273) < EPS
        assert abs(f(-1, 1, 50, 1, 50, 10, 10, 0, 9, 10*BW, b_and_R, False) - 0.022032067555540023) < EPS
        assert abs(f(1, 1, 50, 0, 50, 10, 10, 0, 0, 10*BW, b_and_R, False) - -0.01471930410595051) < EPS
        assert abs(f(-1, 1, 50, 0, 50, 10, 10, 0, 0, 10*BW, b_and_R, False) - 0.01471930410595051) < EPS
        assert abs(f(1, 1, 500, 0, 670, 10, 10, 0, 0, 10*BW, b_and_R, False) - -9.387941233923406e-05) < EPS
        assert abs(f(-1, 1, 500, 0, 670, 10, 10, 50, -6, 10*BW, b_and_R, False) - 0.00042552097225860233) < EPS

def test_D_integrals():
    BW = 2e5
    b_and_R = BW_Ozaki(BW)
    for f in [integralD, c_integralD]:
        temp = f(1, 1, 10, 0, -15, 10, 20, -4, 2, BW, b_and_R, False) -(-0.5720195468017749-0.0661761480679941j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(1, 1, 15, 0, 15, 10, 20, -4, 2, BW, b_and_R, False) - (-0.34799468862372424+0.32194353953897303j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS
        temp = f(-1, -1, 0, 100, 5, 10, 20, 0, 0, BW, b_and_R, False) - (0.5431481709725874-0.28340078060897006j)
        assert abs(temp.real) < EPS
        assert abs(temp.imag) < EPS

        assert abs(f(1, 1, 50, 1, 67, 10, 10, 0, 0, 10*BW, b_and_R, False) - -0.022669451594648456) < EPS
        assert abs(f(-1, 1, 50, 1, 67,10, 10, 0, 0, 10*BW, b_and_R, False) - -0.038816547604131724) < EPS
        assert abs(f(1, 1, 50, 1, 50, 10, 10, 0, 0, 10*BW, b_and_R, False) - -0.03833737405481647) < EPS
        assert abs(f(-1, 1, 50, 1, 50, 10, 10, 0, 9, 10*BW, b_and_R, False) - -0.030799301800540173) < EPS
        assert abs(f(1, 1, 50, 0, 50, 10, 10, 0, 0, 10*BW, b_and_R, False) - -0.03881982276789282) < EPS
        assert abs(f(-1, 1, 50, 0, 50, 10, 10, 0, 0, 10*BW, b_and_R, False) - -0.03477669776185972) < EPS
        assert abs(f(1, 1, 500, 0, 670, 10, 10, 0, 0, 10*BW, b_and_R, False) - -9.406521280339516e-05) < EPS
        assert abs(f(-1, 1, 500, 0, 670, 10, 10, 50, -6, 10*BW, b_and_R, False) - -0.005465272683842759) < EPS