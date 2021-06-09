# QmeQ Changelog

## [Unreleased]

## [1.1] - 2021-06-04

### Added

- First-order approaches to describe electron-phonon coupling inside a quantum dot
  * Pauli (classical)
  * Lindblad
  * Redfield
  * First order von Neumann (1vN)

- Approaches to describe tunneling from metallic leads
  * Second order Real Time Diagramatic (RTD) approach

- Added BuilderManyBody class for dealing with many-body state input
- Support for Fock state removal when calculating quantum dot eigenstates

### Changed

- Refactored Approach classes:
  * Introduced separate Cython class
  * Introduced KernelHandler class for more convenient dealing with master equation matrix elements

### Fixed

- Add to a coulomb matrix element correctly when before it was not defined/used

### Removed

- Python 2.7 support

## [1.0] - 2017-07-13

### Added

- Quantum dot eigenstate calculations

- Approaches to describe tunneling from metallic leads
  * Pauli (classical)
  * Lindblad
  * Redfield
  * First order von Neumann (1vN)
  * Second order von Neumann (2vN)

[unreleased]: https://github.com/gedaskir/qmeq/compare/1.1...HEAD
[1.1]: https://github.com/gedaskir/qmeq/releases/tag/1.1
[1.0]: https://github.com/gedaskir/qmeq/releases/tag/1.0
