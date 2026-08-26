# Crafter: Crystal Systems for MD Simulations

## Overview

The **Crafter** is a versatile feature in iMolCRAFT that automates the preparation of molecular systems for molecular dynamics (MD) simulations. This section focuses on the **Crystal System** workflow, where Crafter takes a crystallographic information file (CIF), performs quantum mechanical (QM) geometry optimization and charge calculations, and generates MD input files for GROMACS or LAMMPS with proper force field parameters from GAFF or other force fields.

## Theoretical Foundation

### Crystal Structure Handling

Crystal systems in iMolCRAFT are:
1. **Loaded** from CIF (Crystallographic Information File) format
2. **Repeated** to create supercells for accurate MD simulations
3. **Optimized** using QM methods to refine atomic positions
4. **Charged** using Restrained Electrostatic Potential (RESP) or other methods
5. **Parameterized** with GAFF or custom force fields
6. **Exported** to MD engine formats (GROMACS/LAMMPS)

## Key Components

### Crafter Class

The `Crafter` class is the unified interface for building MD systems from various structure types. It handles both **crystal** and **liquid** systems with a consistent workflow.

```python
from imolcraft.crafter import Crafter

# Initialize with configuration file
craft = Crafter("craft_params.yml")
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `from_yaml(filename)` | Load parameters from a YAML configuration file |
| `prep(do_opt=True, do_charge=True)` | Prepare the system with structure optimization and charge calculation |
| `build()` | Generate force field parameters and OpenMM system |
| `save_crafter(filename)` | Save the Crafter object to a pickle file for later use |
| `load_crafter(filename)` | Load a previously saved Crafter object from a pickle file |

**Key Attributes:**

- `mol_info`: Dictionary containing molecular information for each unique species
- `structure`: Dictionary with structure type (`crystal` or `liquid`) and configuration
- `params_geoopt`: Geometry optimization parameters (basis set, QM method, software)
- `params_charge`: Charge calculation parameters (method, basis set)
- `params_ff`: Force field parameters (type, ion parameters, charge scaling)

### Configuration via `craft_params.yml`

The YAML file controls all aspects of system preparation:

```yaml
geoopt:
  basis: 6-311+g(2d,p)        # Basis set for QM optimization
  method: wb97xd               # DFT method
  software: psi4               # QM software (psi4 or gaussian)

charge:
  type: resp                   # Charge calculation method
  basis: 6-31g(d)             # Basis set for charge fitting
  method: hf                   # HF or DFT
  software: psi4              # QM software

forcefield:
  fftype: gaff-2.11           # Force field type
  iontype: amber/ions/ionsff99_tip3p.xml  # Ion parameters (the default; or Gmanr / Madrid / Wu-Wick / SMM for Li+)
  charge_scale_ion: 0.8       # Charge scaling for ionic interactions
  charge_scale_neutral: 1.0   # Charge scaling for neutral molecules

structure:
  type: crystal               # System type: "crystal" or "liquid"
  cif: structure.cif          # Path to CIF file
  repeat: [2, 2, 2]           # Supercell repetition (a, b, c directions)
```

### Exporter Function

The `exporter()` function converts the prepared system to MD engine formats:

```python
from imolcraft.io import exporter

exporter(
    pdb="supercell_bonds.pdb",        # Input PDB file with bonds
    system="system.xml",               # OpenMM system XML file
    filename="crystal",                # Output filename (without extension)
    format="gmx"                       # "gmx" for GROMACS, "lmp" for LAMMPS
)
```

**Output Files:**

- **GROMACS** (`format="gmx"`): 
  - `crystal.top`: Topology file
  - `crystal.gro`: Structure file (GROMACS format)

- **LAMMPS** (`format="lmp"`, *experimental*):
  - `crystal.data`: Data file

```{warning}
The LAMMPS exporter is experimental and not considered stable; verify its output before use.
LAMMPS is not installed by `env.yml`. See {doc}`exporter` for details.
```


## Key Parameters Reference

### Geometry Optimization (`geoopt`)

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `basis` | str | `6-311+g(2d,p)` | Gaussian basis set |
| `method` | str | `wb97xd` | DFT method (B3LYP, PBE, ωB97X-D, etc.) |
| `software` | str | `psi4` or `g16` | QM software to use |

### Charge Calculation (`charge`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `type` | str | `resp`, `am1bcc` | Charge fitting method |
| `basis` | str | `6-31g(d)` | Basis set for charge fitting |
| `method` | str | `hf` | Method for charge calculation |
| `software` | str | `psi4`, `g16` | QM software |

### Force Field (`forcefield`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `fftype` | str | `gaff-2.11`, `gaff-2.1`, `gaff-1.81`, `gaff-1.8`, `gaff-1.4` | Force field type |
| `iontype` | str | `Gmanr`, `Madrid`, `Wu-Wick`, `SMM`, or a path to an XML file. Default `amber/ions/ionsff99_tip3p.xml` | Ion force field parameters. The four names each supply Li<sup>+</sup> alone; every other ion falls back to the default Amber library. A path is taken as it stands if it exists, otherwise as relative to the `ffxml` directory of openmmforcefields |
| `charge_scale_ion` | float | 0.0-1.0 | Scale factor for ionic charges |
| `charge_scale_neutral` | float | 0.0-1.0 | Scale factor for neutral charges |

### Structure Definition (`structure`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `type` | str | `crystal`, `liquid` | System type |
| `cif` | str | File path | Path to CIF file (for crystal) |
| `repeat` | list | `[2, 2, 2]` | Supercell repetition factors (a, b, c) |
