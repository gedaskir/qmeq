QmeQ: Quantum master equation for Quantum dot transport calculations
====================================================================

QmeQ is an open-source Python package for calculations of transport through
quantum  dot devices. The so-called Anderson-type models are used to describe
the quantum dot device, where quantum dots are coupled to the leads by
tunneling. QmeQ can calculate the stationary state **particle** and
**energy currents** using various approximate density matrix approaches. As for
now we have implemented the following first-order methods

-  Pauli (classical) master equation
-  Lindblad approach
-  Redfield approach
-  First order von Neumann (1vN) approach

which can describe the effect of Coulomb blockade. QmeQ also has two
second-order methods

-  Second order von Neumann (2vN) approach
-  Real Time Diagrammatic approach

which can additionally address cotunneling, pair tunneling, and
broadening effects.
