import numpy as np
cimport numpy as np

from ..c_aprclass cimport ApproachElPh

from ...wrappers.c_mytypes cimport bool_t
from ...wrappers.c_mytypes cimport int_t
from ...wrappers.c_mytypes cimport long_t
from ...wrappers.c_mytypes cimport double_t
from ...wrappers.c_mytypes cimport complex_t


cdef class ApproachPauli(ApproachElPh):
    pass
