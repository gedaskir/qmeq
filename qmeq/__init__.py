"""
QmeQ: Quantum master equation for Quantum dot transport calculations
====================================================================

QmeQ is an open-source Python package for transport calculations through
quantum  dot devices. The so-called Anderson-type models are used to describe
the quantum dot device, where quantum dots are coupled to the leads by
tunneling. QmeQ can calculate the stationary state particle and energy currents
using various approximate density matrix approaches. As for now we have
implemented the following first-order methods

* Pauli (classical) master equation
* Lindblad approach
* Redfield approach
* First order von Neumann (1vN) approach

which can describe the effect of Coulomb blockade. QmeQ also has one
second-order method

* Second order von Neumann (2vN) approach

which can additionally address cotunneling, pair tunneling, and
broadening effects.

Physics disclaimer
------------------

All the methods in QmeQ are approximate so depending on parameter regime they
can fail, and a good knowledge of the method is required whether to trust the
result or not. For example, Redfield, 1vN, and 2vN approaches can violate
positivity of the reduced density matrix and lead to currents flowing against
the bias. We still think it is important to have a package where a user can
duplicate existing calculations, check applicability of different methods, or
simply discover new kind of physics using different approximate master equations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .aprclass import Approach
from .aprclass import Approach2vN
from .builder import Builder
from .builder import FunctionProperties
from .indexing import StateIndexing
from .indexing import StateIndexingPauli
from .indexing import StateIndexingDM
from .indexing import StateIndexingDMc
from .leadstun import LeadsTunneling
from .qdot import QuantumDot

__version__ = '1.0'
