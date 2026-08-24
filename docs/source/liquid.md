# Crafter: Liquid Systems for MD Simulations

## Overview

The **Crafter** is a versatile feature in iMolCRAFT that automates the preparation of molecular systems for molecular dynamics (MD) simulations. This section focuses on the **Liquid System** workflow, where Crafter takes multiple molecular structures (as PDB files), performs charge calculations and QM geometry optimization (optionally), builds a liquid box with specified density and cell size, and generates MD input files for GROMACS or LAMMPS with proper force field parameters from GAFF or other force fields.

## Theoretical Foundation

### Liquid Structure Handling

Liquid systems in iMolCRAFT are:
1. **Loaded** from individual PDB files for each unique molecular species
2. **Repeated** to create the desired number of molecules at specified density
3. **Packed** into a simulation box with specified dimensions (cell size or density)
4. **Optimized** using QM methods to refine atomic positions (optional)
5. **Charged** using Restrained Electrostatic Potential (RESP) or other methods
6. **Parameterized** with GAFF or custom force fields
7. **Exported** to MD engine formats (GROMACS/LAMMPS)

The key difference from crystal systems is that liquid systems require careful packing of multiple molecular species into a simulation box while maintaining target density and avoiding steric clashes.

## Key Components

### Crafter Class

The `Crafter` class is the unified interface for building MD systems from various structure types. It handles both **crystal** and **liquid** systems with a consistent workflow.

```python
from imolcraft.crafter import Crafter

# Initialize with configuration file
craft = Crafter("crafter.yml")
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `from_yaml(filename)` | Load parameters from a YAML configuration file |
| `prep(do_opt=True, do_charge=True)` | Prepare the system with optional geometry optimization and charge calculation |
| `build()` | Generate force field parameters and OpenMM system |
| `save_crafter(filename)` | Save the Crafter object to a pickle file for later use |
| `load_crafter(filename)` | Load a previously saved Crafter object from a pickle file |

**Key Attributes:**

- `mol_info`: Dictionary containing molecular information for each unique species
- `structure`: Dictionary with structure type (`crystal` or `liquid`) and configuration
- `params_geoopt`: Geometry optimization parameters (basis set, QM method, software)
- `params_charge`: Charge calculation parameters (method, basis set)
- `params_ff`: Force field parameters (type, ion parameters, charge scaling)

### Configuration via `crafter.yml`

The YAML file controls all aspects of liquid system preparation:

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
  iontype: amber/ions/ionsff99_tip3p.xml  # Ion parameters (if needed)
  charge_scale_ion: 0.8       # Charge scaling for ionic interactions
  charge_scale_neutral: 1.0   # Charge scaling for neutral molecules

structure:
  type: liquid                # System type: "crystal" or "liquid"
  molecules:                  # List of PDB files for each molecular species
    - Li.pdb
    - BF4.pdb
    - SL.pdb
  nmols:                      # Number or ratio of each molecule in the system
    - 1
    - 1
    - 10
  density_kgm3: 1304          # Target liquid density (kg/m³)
  cell_A: [40, 40, 40]        # Simulation box dimensions (Ångströms)
  fixed_property: "density"   # Which property is fixed during packing
  priority_property: "cell"   # Which property takes priority if conflict exists
```


### Exporter Function

The `exporter()` function converts the prepared system to MD engine formats:

```python
from imolcraft.io import exporter

exporter(
    pdb="supercell_bonds.pdb",        # Input PDB file with bonds
    system="system.xml",               # OpenMM system XML file
    filename="liquid",                 # Output filename (without extension)
    format="gmx"                       # "gmx" for GROMACS, "lmp" for LAMMPS
)
```

**Output Files:**

- **GROMACS** (`format="gmx"`): 
  - `liquid.top`: Topology file
  - `liquid.gro`: Structure file (GROMACS format)
  - `posre.itp`: Position restraint file

- **LAMMPS** (`format="lmp"`, *experimental*):
  - `liquid.data`: Data file

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
| `fftype` | str | `gaff-2.11`, `gaff-2.1`, `gaff-1.81`, `gaff-1.8`, `gaff-1.4`   | Force field type |
| `iontype` | str | Path to XML | Ion force field parameters |
| `charge_scale_ion` | float | 0.0-1.0 | Scale factor for ionic charges |
| `charge_scale_neutral` | float | 0.0-1.0 | Scale factor for neutral charges |

### Structure Definition (`structure`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `type` | str | `crystal`, `liquid` | System type |
| `molecules` | list | File paths | PDB files for each molecular species |
| `nmols` | list | Integers | Number or ratio of each molecule type |
| `density_kgm3` | float | > 0 | Target liquid density in kg/m³ |
| `cell_A` | list | `[x, y, z]` | Simulation box dimensions in Ångströms |
| `fixed_property` | str | `nmols`, `density`, `volume` | Which property is fixed during packing |
| `priority_property` | str | `cell`, `density` | Which property takes priority if conflict exists |

#### Kewword of `fixed_property` and `priority_property` 
| fixed_property | priority_property | Process |
|--------   |---------|---------|
| `nmols`   | `cell` | `nmols` of molecules are packed in `cell_A`. |
| `nmols`   | `density` | `cell_A` is determined by `nmols` and `density_kgm3`|
| `cell`    | `density` | `nmol` is adjusted to approach the `density_kgm3` while maintaining the ratio. |
| `density` | `cell` | `nmol` is adjusted while maintaining the ratio, and the value of `cell_A` is also adjusted so that it equals `density_kgm3`.|

## Usage Example: Liquid System to MD Simulation

### Step 1: Prepare Configuration File

Create a `crafter.yml` with your liquid system specification:

```yaml
geoopt:
  basis: 6-311+g(2d,p)
  method: wb97xd
  software: psi4

charge:
  type: resp
  basis: 6-31g(d)
  method: hf
  software: psi4

forcefield:
  fftype: gaff-2.11
  iontype: amber/ions/ionsff99_tip3p.xml
  charge_scale_ion: 0.8
  charge_scale_neutral: 1.0

structure:
  type: liquid
  molecules:
    - Li.pdb         # Lithium ion
    - BF4.pdb        # Tetrafluoroborate anion
    - SL.pdb         # Solvent molecule
  nmols:
    - 1              # 1 Li+ ion
    - 1              # 1 BF4- ion
    - 10             # 10 solvent molecules
  density_kgm3: 1304 # Ionic liquid density
  cell_A: [40, 40, 40]
  fixed_property: "density"
  priority_property: "cell"
```

### Step 2: Prepare Component Molecules

For each molecular species, create an optimized PDB file:

```bash
# Li.pdb, BF4.pdb, SL.pdb should be:
# - Individual molecular structures
# - Already optimized (or will be if do_opt=True)
# - In proper PDB format with correct connectivity
```

### Step 3: Initialize Crafter and Prepare System

```python
from imolcraft.crafter import Crafter

# Initialize with configuration
craft = Crafter("crafter.yml")

# Prepare the system:
# - Load individual molecules from PDB files
# - For liquid systems, do_opt=False is common (molecules often pre-optimized)
# - Calculate partial charges using RESP
craft.prep(do_opt=False, do_charge=True)
```

**What happens during `prep()`:**
1. **Molecule loading**: PDB files are read for each species
2. **RDKit molecule generation**: Structures are converted for bond/connectivity detection
3. **Charge calculation**: RESP charges calculated at provided geometry
4. **Density-based packing**: Molecules are packed into a box achieving target density

### Step 4: Build Force Field

```python
# Generate force field parameters and OpenMM system
craft.build()

# This produces:
# - system.xml: OpenMM system file with all force field parameters
# - supercell_bonds.pdb: Packed liquid box with all molecules and bonds
# - Force field parameters automatically generated from GAFF
```

### Step 5: Export to MD Engines

```python
from imolcraft.io import exporter

# Export liquid box for MD simulation

# Option 1: GROMACS format
exporter(
    pdb="supercell_bonds.pdb",
    system="system.xml",
    filename="liquid_md",
    format="gmx"
)

# Option 2: LAMMPS format (experimental, see the Exporter page)
exporter(
    pdb="supercell_bonds.pdb",
    system="system.xml",
    filename="liquid_md",
    format="lmp"
)
```

## Saving and Loading Crafter State

### Saving the Crafter Object

After preparing your liquid system, save the Crafter object to avoid re-running charge calculations:

```python
# Save after preparation and building
craft.save_crafter(filename="liquid_system.pkl")
```

The `save_crafter()` method:
- Saves the complete Crafter object with all internal state
- Preserves all charge calculations
- Stores molecular information for each species
- Allows reuse across multiple MD export formats

### Loading the Crafter Object

Resume work from a saved Crafter object in a new session:

```python
from imolcraft.crafter import Crafter
from imolcraft.io import exporter

# Initialize empty Crafter
craft = Crafter()

# Load from saved pickle file
craft.load_crafter(filename="liquid_system.pkl")

# Immediately export to different MD engine format
exporter(
    pdb="supercell_bonds.pdb",
    system="system.xml",
    filename="liquid_md",
    format="gmx"
)
```
