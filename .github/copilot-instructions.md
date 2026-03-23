# Project Instructions
- This library generates force field parameters from QM calculations using DMFF as backend
- DMFF uses JAX for automatic differentiation; preserve differentiability in all changes
- Force field XML format must be OpenMM-compatible (Hamiltonian class)
- Python style: type hints, docstrings in NumPy format
- Test framework: pytest; all new functions must have unit tests
- Dependencies: JAX, DMFF, OpenMM, RDKit
