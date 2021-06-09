import numpy as np
cimport numpy as np

from ..c_aprclass cimport Approach

from ...wrappers.c_mytypes cimport bool_t
from ...wrappers.c_mytypes cimport int_t
from ...wrappers.c_mytypes cimport long_t
from ...wrappers.c_mytypes cimport double_t
from ...wrappers.c_mytypes cimport complex_t


cdef class ApproachRedfield(Approach):
    pass
