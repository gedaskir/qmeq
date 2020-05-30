# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# Cython imports

cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free

cimport scipy.linalg.cython_lapack


cdef class LapackSolver:

    cdef void solve(self):
        pass

    cdef void determine_pointer_a(self, a):
        cdef double [:, :] a_matrix = a
        self.a_double = &a_matrix[0, 0]

    cdef void determine_pointer_b(self, b):
        cdef double [:] b_vector
        cdef double [:, :] b_matrix

        if len(b.shape) == 1:
            b_vector = b
            self.b_double = &b_vector[0]
        else:
            b_matrix = b
            self.b_double = &b_matrix[0, 0]


#------------------------------------------
# XGESV
#------------------------------------------
cdef class LapackSolverXGESV(LapackSolver):

    def __init__(self, a, b):
        self.n = a.shape[0]
        self.nrhs = 1 if len(b.shape) == 1 else b.shape[1]
        self.lda = self.n
        self.ldb = max(self.n, self.nrhs)

        self.ipiv = <int*> malloc(self.n * sizeof(int))
        if not self.ipiv:
            raise MemoryError()

    def __dealloc__(self):
        free(self.ipiv)


cdef class LapackSolverDGESV(LapackSolverXGESV):

    def __init__(self, a, b):
        LapackSolverXGESV.__init__(self, a, b)
        self.determine_pointer_a(a)
        self.determine_pointer_b(b)

    cdef void solve(self):
        scipy.linalg.cython_lapack.dgesv(
            &self.n,
            &self.nrhs,
             self.a_double,
            &self.lda,
             self.ipiv,
             self.b_double,
            &self.ldb,
            &self.info)


#------------------------------------------
# XGELSD
#------------------------------------------
cdef class LapackSolverXGELSD(LapackSolver):

    def __init__(self, a, b):
        self.m = a.shape[0]
        self.n = a.shape[1]
        self.nrhs = 1 if len(b.shape) == 1 else b.shape[1]
        self.lda = self.m
        self.ldb = max(self.m, self.n)
        self.rcond = -1

        self.s = <double*> malloc(min(self.m, self.n) * sizeof(double))
        if not self.s:
            raise MemoryError()

        self.determine_work_size()

        self.iwork = <int*> malloc(self.liwork * sizeof(int))
        if not self.iwork:
            raise MemoryError()

    cdef void determine_work_size(self):
        pass

    def __dealloc__(self):
        free(self.s)
        free(self.iwork)


cdef class LapackSolverDGELSD(LapackSolverXGELSD):

    def __init__(self, a, b):
        LapackSolverXGELSD.__init__(self, a, b)
        self.determine_pointer_a(a)
        self.determine_pointer_b(b)

        self.work = <double*> malloc(self.lwork * sizeof(double))
        if not self.work:
            raise MemoryError()

    def __dealloc__(self):
        free(self.work)

    cdef void determine_work_size(self):
        self.lwork = -1
        cdef double lwork

        scipy.linalg.cython_lapack.dgelsd(
            &self.m,
            &self.n,
            &self.nrhs,
             self.a_double,
            &self.lda,
             self.b_double,
            &self.ldb,
             self.s,
            &self.rcond,
            &self.rank,
            &lwork,
            &self.lwork,
            &self.liwork,
            &self.info)

        self.lwork = int(lwork)

    cdef void solve(self):
        scipy.linalg.cython_lapack.dgelsd(
            &self.m,
            &self.n,
            &self.nrhs,
             self.a_double,
            &self.lda,
             self.b_double,
            &self.ldb,
             self.s,
            &self.rcond,
            &self.rank,
             self.work,
            &self.lwork,
             self.iwork,
            &self.info)
