"""Module containing various special functions, cython implementation."""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

from ..wrappers.mytypes import doublenp
from ..wrappers.mytypes import complexnp

cimport numpy as np
cimport cython

cdef double_t pi = 3.14159265358979323846
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport tanh
from libc.math cimport atan
from libc.math cimport floor
from libc.math cimport sqrt
from libc.math cimport tan
from libc.math cimport fabs

#Factors needed to evaluate the digamma function
cdef double[8] Digamma_factors
Digamma_factors[0] =   0.0
Digamma_factors[1] = -.083333333333333333333333333333333333333333333333333
Digamma_factors[2] =  .008333333333333333333333333333333333333333333333333
Digamma_factors[3] = -.003968253968253968253968253968253968253968253968254
Digamma_factors[4] =  .0041666666666666666666666666666666666666666666666667
Digamma_factors[5] = -.0075757575757575757575757575757575757575757575757576
Digamma_factors[6] =  .021092796092796092796092796092796092796092796092796
Digamma_factors[7] = -.083333333333333333333333333333333333333333333333333

cdef double[ 6 ][ 5 ] C
cdef double[5] SGN = [ -1, 1, -1, 1, -1]
cdef double[6] FCT = [0, 1, 1, 2, 6, 24]

#Factors needed to evaluate the polygamma function.
C[ 0 ][ 0 ] = -Digamma_factors[1]
C[ 1 ][ 0 ] = -Digamma_factors[2]
C[ 2 ][ 0 ] = -Digamma_factors[3]
C[ 3 ][ 0 ] = -Digamma_factors[4]
C[ 4 ][ 0 ] = -Digamma_factors[5]
C[ 5 ][ 0 ] = -Digamma_factors[6]

C[ 0 ][ 1 ] = 1.66666666666666667e-1
C[ 1 ][ 1 ] = -3.33333333333333333e-2
C[ 2 ][ 1 ] = 2.38095238095238095e-2
C[ 3 ][ 1 ] = -3.33333333333333333e-2
C[ 4 ][ 1 ] = 7.57575757575757576e-2
C[ 5 ][ 1 ] = -2.53113553113553114e-1

C[ 0 ][ 2 ] = 5.00000000000000000e-1
C[ 1 ][ 2 ] = -1.66666666666666667e-1
C[ 2 ][ 2 ] = 1.66666666666666667e-1
C[ 3 ][ 2 ] = -3.00000000000000000e-1
C[ 4 ][ 2 ] = 8.33333333333333333e-1
C[ 5 ][ 2 ] = -3.29047619047619048e+0

C[ 0 ][ 3 ] = 2.00000000000000000e+0
C[ 1 ][ 3 ] = -1.00000000000000000e+0
C[ 2 ][ 3 ] = 1.33333333333333333e+0
C[ 3 ][ 3 ] = -3.00000000000000000e+0
C[ 4 ][ 3 ] = 1.00000000000000000e+1
C[ 5 ][ 3 ] = -4.60666666666666667e+1

C[ 0 ][ 4 ] = 10
C[ 1 ][ 4 ] = -7
C[ 2 ][ 4 ] = 12
C[ 3 ][ 4 ] = -33
C[ 4 ][ 4 ] = 130
C[ 5 ][ 4 ] = -691

cdef double_t TMIN = 1e-5


cdef double_t cabs(complex_t z) nogil:
    return sqrt(z.real*z.real + z.imag*z.imag)


@cython.cdivision(True)
cdef complex_t clog(complex_t z) nogil:
    """Calculates log(z) where z is a complex number."""
    cdef double_t x = z.real
    cdef double_t y = z.imag
    cdef complex_t ret = 0.5*log(x*x + y*y) + 1j*(atan(y/x))
    return ret


@cython.cdivision(True)
cdef double_t fermi_func(double_t x) nogil:
    """Fermi function."""
    if x > 709.77: # To avoid overflow in the denimonator.
        return 0.0
    return 1.0/(exp(x)+1.0)


@cython.cdivision(True)
cdef double_t diff_fermi(double_t x, double_t sign=1) nogil:
    """Calculates the derivative of the Fermi function."""
    if x > 354.885:
        return 0.0
    cdef double_t e = exp(sign*x)
    return -sign*e / ((e + 1.0)**2)


@cython.cdivision(True)
cdef double_t bose(double_t x, double_t sign=1) nogil:
    """Calculates the Bose distribution function at x."""
    if x > 709.77: #To avoid overflow in the denominator.
        return 0.0
    cdef double_t e = exp( sign * x )
    return 1.0 / (e - 1.0)


@cython.cdivision(True)
cdef double_t phi(double_t x, double_t Dp, double_t Dm, double_t sign=1) nogil:
    """ Calculates the phi function in Leijnse & Wegevijs 2008. Assumes Dm
        and Dp are rescaled with temperature."""
    cdef complex_t Z = 0.5 + x/(2*pi)*1j
    cdef double_t ret = sign*(-digamma(Z).real + log(0.5*(fabs(Dp)+fabs(Dm))/(2.0*pi)))
    return ret


@cython.cdivision(True)
cdef double_t diff_phi(double_t x, double_t sign=1) nogil:
    """Calculates the derivative of the phi function."""
    cdef complex_t Z = 0.5 + x/(2.0*pi)*1j
    cdef double_t ret = sign* 1.0/(2.0*pi)*polygamma(Z, 1).imag
    return ret


@cython.cdivision(True)
cdef double_t diff2_phi(double_t x, double_t sign=1) nogil:
    """Calculates the second derivative of the phi function."""
    cdef complex_t Z = 0.5 + x/(2.0*pi)*1j
    cdef double_t ret = sign* 1.0/(2.0*pi)**2*polygamma(Z, 2).real
    return ret


cdef double_t delta_phi(double_t x1, double_t x2, double_t Dp, double_t Dm, double_t sign=1) nogil:
    """Calculates difference of phi functions with different arguments."""
    return phi( x1, Dp, Dm, sign=sign ) - phi( x2, Dp, Dm, sign=sign )


@cython.boundscheck(False)
@cython.cdivision(True)
cdef int_t func_pauli(double_t Ecb, double_t mu, double_t T,
                      double_t Dm, double_t Dp, int_t itype,
                      double_t [:] rez):
    cdef double_t alpha, Rm, Rp, cur0, cur1
    alpha = (Ecb-mu)/T
    Rm, Rp = (Dm-mu)/T, (Dp-mu)/T
    if itype == 1 or itype == 3 or (Rm < alpha < Rp):
        cur0 = fermi_func(alpha)
        cur1 = 1.0-cur0
        rez[0] = 2*pi*cur0
        rez[1] = 2*pi*cur1
    else:
        rez[0], rez[1] = 0.0, 0.0
    return 0


@cython.boundscheck(False)
@cython.cdivision(True)
cdef int_t func_1vN(double_t Ecb, double_t mu, double_t T,
                    double_t Dm, double_t Dp,
                    int_t itype, int_t limit,
                    complex_t [:] rez):
    cdef double_t alpha, Rm, Rp, err
    cdef complex_t cur0, cur1, en0, en1, const0, const1
    # -------------------------
    if itype == 0:
        alpha, Rm, Rp = (Ecb-mu)/T, (Dm-mu)/T, (Dp-mu)/T
        cur0, err = quad(fermi_func, Rm, Rp,
                         weight='cauchy', wvar=alpha,
                         epsabs=1.0e-6, epsrel=1.0e-6, limit=limit)
        cur0 = cur0 + (-1.0j*pi*fermi_func(alpha) if Rm < alpha < Rp else 0.0j)
        cur1 = cur0 + log(abs((Rm-alpha)/(Rp-alpha)))
        cur1 = cur1 + (1.0j*pi if Rm < alpha < Rp else 0.0j)
        #
        const0 = T*((-Rm if Rm < -40.0 else log(1+exp(-Rm)))
                   -(-Rp if Rp < -40.0 else log(1+exp(-Rp))))
        const1 = const0 + Dm-Dp
        #
        en0 = const0 + Ecb*cur0
        en1 = const1 + Ecb*cur1
    elif itype == 1:
        alpha, Rm, Rp = (Ecb-mu)/T, Dm/T, Dp/T
        cur0 = digamma(0.5+1.0j*alpha/(2*pi)).real - log(abs(Rm)/(2*pi))
        cur0 = cur0 - 1.0j*pi*fermi_func(alpha)
        cur1 = cur0 + log(abs(Rm/Rp))
        cur1 = cur1 + 1.0j*pi
        #
        en0 = -T*Rm + Ecb*cur0
        en1 = -T*Rp + Ecb*cur1
    elif itype == 2:
        alpha, Rm, Rp = (Ecb-mu)/T, (Dm-mu)/T, (Dp-mu)/T
        cur0 = -1.0j*pi*fermi_func(alpha) if Rm < alpha < Rp else 0.0j
        cur1 = cur0 + (1.0j*pi if Rm < alpha < Rp else 0.0j)
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    elif itype == 3:
        alpha = (Ecb-mu)/T
        cur0 = -1.0j*pi*fermi_func(alpha)
        cur1 = cur0 + 1.0j*pi
        en0 = Ecb*cur0
        en1 = Ecb*cur1
    # -------------------------
    rez[0], rez[1], rez[2], rez[3] = cur0, cur1, en0, en1
    return 0


@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex_t digamma(complex_t z) nogil:
    """Returns the complex valued digamma function with argument Z. This
    can also be calculated using polygamma(z, 0), but this way is faster."""
    cdef double_t a, b, b_new, r, s, x, y
    cdef complex_t t, ret
    cdef int_t n, v, i
    x, y = z.real, z.imag

    if x <= 0.0:
        return polygamma(z, 0)
    else:
        ret = 0.0+0.0j

    if x<8:
        n = 8 - <int>(x)
        for v in range(0, n):
            ret -= 1/(z + v)
        z += n

    t = 1/z
    ret -= t/2
    t = t/z
    x, y = t.real, t.imag
    r = x+x
    s = x**2 + y**2

    a = Digamma_factors[7]
    b = Digamma_factors[6]
    for i in range(2, 8):
        b_new = Digamma_factors[7-i] - s*a
        a = b + r*a
        b = b_new

    return ret + t*a + b + clog(z)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef complex_t polygamma( complex_t U, long_t K ) nogil:
    """Calculates the Kth derivative of the digamma function with argument U."""
    cdef complex_t hh1, hh2, hh3, V, H, R, P
    cdef double X, A, B, T, Y
    cdef long_t i, K1

    X = U.real
    A = fabs( X )
    H = 0 + 0j
    K1 = K + 1

    if K < 0 or K > 4:
        return H

    if X < 0:
       U = - U
    V = U

    if A < 15:
        H = 1/(V**float(K1))
        for i in range(15-<int>A):
            V = V + 1.0
            H = H + 1/(V**float(K1))
        V = V + 1.0

    R = 1/(V**2)
    P = R*C[5][K]
    H = H*FCT[ K + 1 ]

    for i in reversed(range(1, 6)):
        hh1 = P + C[ i-1 ][ K ]
        P =  R * hh1

    hh1 = 1/(V**float(K1))
    hh2 = P + FCT[ K ]
    hh3 = V*hh2 + 0.5*FCT[K+1]
    H = H + hh1*hh3
    H = H*SGN[ K ]

    if K == 0:
        H =  H + clog( V )

    if X < 0 :
        V =  V * pi
        X = V.real
        Y = V.imag
        A = sin( X )
        B = cos( X )
        T = tanh( X )
        P = B - 1j*A*T / (A + 1j*A*T)

        if K == 0 :
            H = H + 1/ U
            H = H + P*pi
        elif K == 1 :
            H =  H + 1/( U**2)
            H = H + (P**2 + 1.0)*9.869604401089358
        elif K == 2 :
            H = H + 1/(U**3)
            H = H + (P**2 + 1.0)*62.01255336059963

    return H


@cython.cdivision(True)
cdef complex_t integralD(double_t p1, double_t eta1, double_t E1, double_t E2, double_t E3, double_t T1,
            double_t T2, double_t mu1, double_t mu2, double_t D, double_t[:,:] b_and_R, bint ImGamma) nogil:
    cdef complex_t ret
    cdef double_t lambda1, lambda2, lambda3
    if fabs(T2-T1) < TMIN and not ImGamma:
        lambda1 = (E1 - mu1) / T1
        lambda2 = (E2 - mu1 - eta1 * mu2) / T1
        lambda3 = (E3 - mu1) /T1
        ret = D_integral_equal_T(p1, 1, lambda3, lambda2,  lambda1, D/2/T1, D/2/T1)
        return ret*pi/T1
    else:
        ret = D_integral(1, p1, -E1, -E2, -E3, T1, T2, mu1, eta1*mu2, D/2, D/2, b_and_R)
        return -1j*ret


@cython.cdivision(True)
@cython.boundscheck(False)
cdef complex_t integralX(double_t p1, double_t eta1, double_t E1, double_t E2, double_t E3, double_t T1,
            double_t T2, double_t mu1, double_t mu2, double_t D, double_t[:,:] b_and_R, bint ImGamma) nogil:
    cdef complex_t ret
    cdef double_t lambda1, lambda2, lambda3
    if fabs(T2-T1) < TMIN and not ImGamma:
        lambda1 = (E1 - mu1) / T1
        lambda2 = (E2 - mu1 - eta1 * mu2) / T1
        lambda3 = (E3 - eta1*mu2) /T1
        ret = X_integral_equal_T(p1, 1, lambda3, lambda2,  lambda1, D/2/T1, D/2/T1)
        return ret*pi/T1
    else:
        ret = X_integral(1, p1, -E1, -E2, -E3, T1, T2, mu1, eta1*mu2, D/2, D/2, b_and_R)
        return -1j*ret


@cython.cdivision(True)
cdef double_t D_integral_equal_T(double_t p1, double_t p2, double_t E1, double_t deltaE, double_t E2,
                                                                      double_t Dp, double_t Dm) nogil:
    cdef double_t E_MIN = 1e-10
    cdef double_t  ret = 0.0

    if fabs(E1 - E2) < E_MIN:
        if fabs( deltaE ) < E_MIN:
            ret = ( p1*p2*( -diff2_phi(E1) + diff_fermi(E1)*phi(-E1,Dp,Dm) - diff_phi(-E1)*fermi_func(E1)
                    - 0.5*diff_phi(E1)) + 0.5*p2*diff_phi(E1) )
        else:
            ret = ( p1*p2*( -fermi_func(E1)*diff_phi(deltaE - E1) - bose(deltaE)*diff_phi(deltaE - E1)
                + diff_fermi(E1)*phi(deltaE - E1,Dp,Dm) - (bose(deltaE) + 0.5)*diff_phi(E1)) +
                0.5*p2*diff_phi(E1) )
    else:
        if fabs(deltaE) < E_MIN:
            ret = ( 1.0/(E1 - E2)*(p1*p2*( -(diff_phi(E1) - diff_phi(E2)) + fermi_func(E1)*phi(-E1,Dp,Dm)
                    -fermi_func(E2)*phi(-E2,Dp,Dm) - 0.5*delta_phi(E1, E2, Dp, Dm)) + 0.5*p2*delta_phi(E1, E2, Dp, Dm)) )

        else:
            ret = ( 1.0/(E1 - E2)*(p1*p2*((fermi_func(E1) + bose(deltaE))*phi(deltaE - E1, Dp, Dm) -
                    ( fermi_func(E2) + bose(deltaE))*phi(deltaE - E2, Dp, Dm)
                    - (bose(deltaE) + 0.5)*delta_phi(E1, E2, Dp, Dm)) + 0.5*p2*delta_phi(E1, E2, Dp, Dm)) )
    return ret


@cython.cdivision(True)
cdef double_t X_integral_equal_T(double_t p1, double_t p2, double_t E1, double_t deltaE, double_t E2,
                                                                     double_t Dp, double_t Dm) nogil:
    cdef double_t E_MIN = 1e-10
    cdef double_t ret = 0.0

    if fabs( deltaE - E1 - E2 ) < E_MIN:
        if fabs( deltaE ) < E_MIN:
            ret = ( diff2_phi(E1) + fermi_func(E1)*diff_phi(-E1) + diff_phi(E1)*fermi_func(-E1) )
        else:
            ret = ( (fermi_func(E1) + bose( E1 + E2))*diff_phi(E2) + diff_phi(E1)*(fermi_func(E2)
                + bose(E1 + E2)) )
    else:
        if fabs( deltaE ) < E_MIN:
            ret = ( -1.0/(E1 + E2)*(fermi_func(E1)*delta_phi(-E1, E2, Dp, Dm) + fermi_func(E2)*
                    delta_phi(-E2, E1, Dp, Dm) - (diff_phi( E1) + diff_phi(E2))) )
        else:
            ret = ( 1.0/(deltaE - E1 - E2)*((fermi_func(E1) + bose(deltaE))*delta_phi(deltaE - E1, E2, Dp, Dm)
                    + (fermi_func(E2) + bose(deltaE))*delta_phi(deltaE - E2, E1, Dp, Dm)) )

    return p1*p2*ret


@cython.cdivision(True)
@cython.boundscheck(False)
cdef complex_t D_integral(double_t p1, double_t p2, double_t z1, double_t z2, double_t z3, double_t T1, double_t T2,
                    double_t mu1, double_t mu2, double_t Dp, double_t Dm, double_t[:,:] b_and_R) nogil:
    cdef double_t BW, BWT, val, pi2T, x
    cdef complex_t temp_f1, temp_f2, ret, A, B, C
    cdef long_t i

    E_MIN = 1e-10
    BW = (fabs(Dp) + fabs(Dm))*0.5
    BWT = log(BW/(2*pi*T1))
    pi2T =  2*pi*T2

    temp_f1 = 0+0j
    for i in range(b_and_R.shape[0]):

        val = b_and_R[i, 0]
        if 1/val > BW/T2:
            continue
        x = T1/val
        A = BWT+0j - digamma(0.5+x/(pi2T) -(z2+mu1+mu2)/(pi2T)*1j)
        B = x - (z3+mu1)*1j
        C = x - (z1+mu1)*1j
        temp_f1 +=  A * 1.0/B * 1.0/C * b_and_R[i, 1]

    temp_f1 *= 0 - 8*pi*T1*1j

    if( fabs(z3/T1-z1/T1)>E_MIN ):
        temp_f2 = ( digamma(0.5 - (z3+mu1)/(2*pi*T1)*1j ) - digamma(0.5 - (z1+mu1)/( 2*pi*T1 )*1j )) / (z3/T1-z1/T1)
        temp_f2 = temp_f2*(-2*pi*1j)/T1
    else:
        temp_f2 = polygamma(0.5 -(z3+mu1)/(2*pi*T1)*1j , 1 ) * (-1.0/T1)
    ret = 0.25*( p1*p2*temp_f1 - p1*temp_f2)
    return ret


@cython.cdivision(True)
@cython.boundscheck(False)
cdef complex_t X_integral(double_t p1, double_t p2, double_t z1, double_t z2, double_t z3, double_t T1, double_t T2,
                    double_t mu1, double_t mu2, double_t Dp, double_t Dm, double_t[:,:] b_and_R) nogil:
    cdef double_t pi2T, x, val
    cdef complex_t ret, A, B, C, D
    cdef long_t i

    BW = (fabs(Dp) + fabs(Dm))*0.5
    pi2T = 2*pi*T2
    ret = 0.0 + 0.0j

    for i in range(b_and_R.shape[0]):

        val = b_and_R[i, 0]
        if 1 /val > BW/ T2:
            continue
        x = T1 / val
        A = digamma( 0.5 + x/(pi2T) - (z2+mu1+mu2)/(pi2T)*1j )
        B = digamma( 0.5 - (z3+mu2)/(pi2T)*1j )
        C = x + (-z2+z3-mu1)*1j
        D = x - (z1+mu1)*1j
        ret +=  (A - B) / C * 1.0/D * b_and_R[i, 1]

    ret *= - 8*pi*T1*1j
    ret *= p1*p2*0.25
    return ret


cdef double_t[:,:] BW_Ozaki(double_t BW):
    cdef double_t p, q
    cdef double_t[:,:] b_and_R

    p = 1.27324094/2.54647909
    q = (0.52325856 - BW)/2.54647909

    cdef long_t N =  abs(int( p/2 - sqrt(p*p/4 - q) ) )
    b_and_R = Ozaki(N * 2 + 2)
    return b_and_R


@cython.cdivision(True)
cdef double_t[:,:] Ozaki(long_t N):
    cdef long_t n, i, count
    cdef double_t t, r, val
    cdef double_t[:] B, b
    cdef double_t[:,:] b_and_R, v

    B = np.zeros(N-1, dtype=doublenp)

    for n in range(1, N + 0):
        t = 1 / float(2 * sqrt((2.0 * float(n) - 1.0) * (2.0 * float(n) + 1.0)))
        B[n-1] = t

    b, v = eigh_tridiagonal(np.zeros(N), B)
    b_and_R = np.zeros([N//2, 2], dtype=doublenp)

    count = 0
    for i in range(N):
        val = b[i]
        if val > 0:
            r = v[0, i] ** 2 / (4 * val ** 2)
            b_and_R[count, 0] = val
            b_and_R[count, 1] = r
            count += 1
    return b_and_R


cdef void diag_matrix_multiply(double_t[:] D, double_t[:,:] M):
    """ Calculates D*M ehere D is the diagonal entries of a
    diagonal matrix. M is a matrix. M gets modified.
    """
    cdef long_t r,c
    cdef double_t k

    for r in range(M.shape[0]):
        k = D[r]
        for c in range(M.shape[1]):
            M[r, c] = k*M[r, c]



def c_fermi_func(double_t x):
    return fermi_func(x)


def c_func_pauli(double_t Ecb, double_t mu, double_t T,
                 double_t Dm, double_t Dp, int_t itype):
    rez = np.zeros(2, dtype=doublenp)
    func_pauli(Ecb, mu, T, Dm, Dp, itype, rez)
    return rez


def c_func_1vN(double_t Ecb, double_t mu, double_t T,
               double_t Dm, double_t Dp,
               int_t itype, int_t limit):
    rez = np.zeros(4, dtype=complexnp)
    func_1vN(Ecb, mu, T, Dm, Dp, itype, limit, rez)
    return rez

def c_diff_fermi(double_t x, double_t sign=1):
    return diff_fermi(x, sign=sign)


def c_bose(double_t x, double_t sign=1):
    return bose(x, sign=sign)


def c_phi(double_t x, double_t Dp, double_t Dm, double_t sign=1):
    return phi(x, Dp, Dm, sign=sign)


def c_diff_phi(double_t x, double_t sign=1):
    return diff_phi(x, sign=sign)


def c_diff2_phi(double_t x, double_t sign=1):
    return diff2_phi(x, sign=sign)


def c_delta_phi(double_t x1, double_t x2, double_t Dp, double_t Dm, double_t sign=1):
    return delta_phi(x1, x2, Dp, Dm, sign=sign)


def c_polygamma(complex_t U, int_t K):
    return polygamma(U, K)


def c_digamma(complex_t z):
    return digamma(z)


def c_integralD(double p1, double eta1, double z1, double z2, double z3, double T1, double T2,
                    double mu1, double mu2, double D, double_t[:,:] b_and_R, bint ImGamma):
    return integralD(p1, eta1, z1, z2, z3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)


def c_integralX(double p1, double eta1, double z1, double z2, double z3, double T1, double T2,
                    double mu1, double mu2, double D, double_t[:,:] b_and_R, bint ImGamma):
    return integralX(p1, eta1, z1, z2, z3, T1, T2, mu1, mu2, D, b_and_R, ImGamma)


def c_Ozaki(int_t N):
    return Ozaki(N)


def c_BW_Ozaki(double_t BW):
    return BW_Ozaki(BW)


def c_fermi_func(double_t x):
    return fermi_func(x)


def c_func_pauli(double_t Ecb, double_t mu, double_t T,
                 double_t Dm, double_t Dp, int_t itype):
    rez = np.zeros(2, dtype=doublenp)
    func_pauli(Ecb, mu, T, Dm, Dp, itype, rez)
    return rez


def c_func_1vN(double_t Ecb, double_t mu, double_t T,
               double_t Dm, double_t Dp,
               int_t itype, int_t limit):
    rez = np.zeros(4, dtype=complexnp)
    func_1vN(Ecb, mu, T, Dm, Dp, itype, limit, rez)
    return rez
