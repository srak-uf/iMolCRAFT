# About iMolCRAFT

**iMolCRAFT** (**I**onic **Mol**ecular **CR**ystal **A**utomatic **F**orcefield **T**ool) is a Python package for automated force field development and optimization, with a specialization in liquid elecrtolytes and molecular crystal electrolytes. It is designed to streamline the entire workflow from crystal structure analysis to force field parameterization and validation for molecular dynamics simulations.

iMolCRAFT leverages [DMFF](https://github.com/deepmodeling/DMFF) (Differentiable Molecular Force Field) as its computational backend, enabling automatic differentiation-based parameter optimization from highly accurate calculations and experimental data.

## Key Features

The package automates and accelerates the force field development process through:

- **Crystal structure analysis**: Extract molecular structures and information from crystal structures
- **Automated force field generation**: Assign force field parameters and calculate point charges
- **High-accuracy informed optimization**: Optimize parameters by combining high-accuracy calculations (QM, DFT, or other computational methods) with experimental data (density, RDF, thermodynamic properties, etc.)
- **Differentiable architecture**: Leverage JAX and automatic differentiation for efficient gradient-based optimization, enabling end-to-end parameter tuning
- **Diverse system support**: Specialized for electrolytic solutions and molecular crystal electrolytes, while supporting organic molecules, polymers, and other molecular systems
- **Multi-platform compatibility**: Seamlessly export to and validate with LAMMPS, GROMACS, and OpenMM

## Specialization

iMolCRAFT is particularly optimized for **liquid electrolytes** and **molecular crystal electrolytes**, where accurate representation of electrostatic interactions and ion dynamics is critical. The framework enables researchers to systematically develop transferable force fields for these complex systems by bridging QM accuracy and classical MD efficiency.

# Installation
```
git clone git@github.com:srak-uf/iMolCRAFT.git
cd iMolCRAFT
conda env create -f env.yml # this script can be used in linux and mac
conda activate imc_cpu
pip install .

# make document (Optional)
conda install -c conda-forge sphinx myst-parser sphinx_rtd_theme myst-nb
cd docs
make html

# test
conda install -c conda-forge pytest
```
You can optimize force field parameters by using [DMFF](https://github.com/srak-uf/DMFF). This DMFF repository is the modified version.    
The sample codes for the optimization are located in [the example/opt](https://github.com/srak-uf/imolcryff/tree/main/examples/opt) directory.
