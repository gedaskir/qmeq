# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from .aprclass import Approach as ApproachPy
from .aprclass import ApproachElPh as ApproachElPhPy

from ..mytypes import doublenp

# Cython imports

cimport numpy as np
cimport cython


cdef class Approach:

    kerntype = 'not defined'
    dtype = doublenp
    indexing_class_name = 'StateIndexingDM'

    def __init__(self, builder):
        ApproachPy.__init__(self, builder)

    cpdef generate_fct(self):
        pass

    cpdef generate_kern(self):
        pass

    cpdef generate_current(self):
        pass

    cpdef generate_vec(self, phi0):
        pass

    def get_kern_size(self):
        return ApproachPy.get_kern_size(self)

    def restart(self):
        ApproachPy.restart(self)

    def set_phi0_init(self):
        return ApproachPy.set_phi0_init(self)

    def prepare_kern(self):
        ApproachPy.prepare_kern(self)

    def solve_kern(self):
        ApproachPy.solve_kern(self)

    def solve_matrix_free(self):
        ApproachPy.solve_matrix_free(self)

    @cython.boundscheck(False)
    def generate_norm_vec(self, length):
        si, symq, norm_row = (self.si, self.funcp.symq, self.funcp.norm_row)

        self.bvec_ext = np.zeros(length+1, dtype=self.dtype)
        self.bvec_ext[-1] = 1

        self.bvec = self.bvec_ext[0:-1]
        self.bvec[norm_row] = 1 if symq else 0

        self.norm_vec = np.zeros(length, dtype=self.dtype)
        cdef np.ndarray[double_t, ndim=1] norm_vec = self.norm_vec

        cdef np.ndarray[long_t, ndim=1] lenlst = si.lenlst
        cdef np.ndarray[long_t, ndim=1] dictdm = si.dictdm
        cdef np.ndarray[long_t, ndim=1] shiftlst0 = si.shiftlst0
        cdef np.ndarray[long_t, ndim=1] mapdm0 = si.mapdm0

        cdef int_t charge, b, bb
        for charge in range(si.ncharge):
            for b in si.statesdm[charge]:
                bb = mapdm0[lenlst[charge]*dictdm[b] + dictdm[b] + shiftlst0[charge]]
                norm_vec[bb] += 1

    def rotate(self):
        ApproachPy.rotate(self)

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True, *args, **kwargs):
        ApproachPy.solve(self, qdq, rotateq, masterq, currentq, args, kwargs)


cdef class ApproachElPh(Approach):

    def __init__(self, builder):
        ApproachElPhPy.__init__(self, builder)

    def restart(self):
        ApproachElPhPy.restart(self)

    def rotate(self):
        ApproachElPhPy.rotate(self)
