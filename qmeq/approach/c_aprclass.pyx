from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .aprclass import Approach as ApproachPy
from .aprclass import ApproachElPh as ApproachElPhPy


cdef class Approach:

    kerntype = 'not defined'
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

    def restart(self):
        ApproachPy.restart(self)

    def set_phi0_init(self):
        return ApproachPy.set_phi0_init(self)

    def solve_kern(self):
        ApproachPy.solve_kern(self)

    def solve_matrix_free(self):
        ApproachPy.solve_matrix_free(self)

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
