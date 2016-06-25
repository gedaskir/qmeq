import numpy as np
cimport numpy as np

ctypedef np.uint8_t boolnp
#ctypedef bint boolnp
ctypedef np.int_t intnp
ctypedef np.long_t longnp
ctypedef np.double_t doublenp
#ctypedef double doublenp
ctypedef np.complex128_t complexnp
#ctypedef complex complexnp

cdef doublenp fermi_func(doublenp)
cdef doublenp func_pauli(doublenp, doublenp, doublenp)
cdef complexnp func_1vN(doublenp, doublenp, doublenp, doublenp, intnp, intnp)
