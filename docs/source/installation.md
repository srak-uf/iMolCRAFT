# Installation

## Requirements

iMolCRAFT needs Python 3.11 or later. The scientific dependencies (OpenMM, Psi4, AmberTools, GROMACS, DMFF, JAX, ...) cannot be installed from PyPI and come from the conda environment files in the repository:

| File | Environment name | Use |
|------|------------------|-----|
| `env.yml` | `imc_cpu` | CPU (Mac and Linux) |
| `env_cuda.yml` | `imc_cuda` | NVIDIA GPU (Linux, CUDA 12 build of JAX) |

GROMACS is part of both environments; it is needed by `GMXCalculator`. LAMMPS and Gaussian 16 are not included (see {doc}`exporter` for LAMMPS).

## Procedure

```bash
git clone https://github.com/srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f env.yml        # or env_cuda.yml
conda activate imc_cpu             # or imc_cuda
pip install .
```

## Building the documentation

```bash
pip install '.[docs]'
cd docs
make html
```

## Testing

```bash
pip install '.[test]'
pytest tests/test_io -m "not g16"
```

Tests are grouped by directory under `tests/` and tagged with pytest markers: `g16` (needs Gaussian 16), `qm` (needs Psi4), `gmx` (needs GROMACS), `tp`, `dihedral` and `distance`. Use `-m "not g16"` unless Gaussian 16 is available; the CI runs each `tests/test_*` directory that way.

## Troubleshooting/issues

Contact us at [issues](https://github.com/srak-uf/iMolCRAFT/issues) or by [email](<mailto:sasaki.r.1788@m.isct.ac.jp>) when you have trouble.
