# Crafter: Liquid Systems for MD Simulations

## Overview

The **Crafter** automates the preparation of molecular systems for molecular dynamics (MD) simulations. This page covers the **liquid** workflow: Crafter takes one PDB file per molecular species, runs a QM geometry optimization (optional) and a charge fit for each species, packs the molecules into a box with packmol at the requested density or cell size, assigns GAFF parameters and writes an OpenMM system. The result can be exported to GROMACS (or, experimentally, LAMMPS).

## Workflow

1. **Load** one PDB file per unique species.
2. **Optimize** each species in vacuum at the QM level (`prep(do_opt=True)`, optional). The optimized geometry is used only for the charge fit; packmol places the molecules as they are in the PDB files.
3. **Charge** each species by RESP (or AM1-BCC on the Gaussian/antechamber path) and scale the charges (`prep(do_charge=True)`).
4. **Pack** the molecules into the box and **parameterize** with GAFF (`build()`).
5. **Export** to an MD engine format with `imolcraft.io.exporter`.

Unlike a crystal, a liquid needs the number of molecules, the box size and the density to be made consistent; `fixed_property` and `priority_property` decide which of them is derived from the others.

## Key Components

### Crafter Class

```python
from imolcraft.crafter import Crafter

craft = Crafter("crafter.yml")
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `from_yaml(filename)` | Load the settings and the structure from a YAML file |
| `prep(do_opt=True, do_charge=True)` | Optional QM geometry optimization and charge fit for every species |
| `build()` | Pack the box, generate the GAFF force field and the OpenMM system. Needs the charges from `prep(do_charge=True)` |
| `get_ffxml()` | Write only the `gaffxml_{i}.xml` files, without building the system |
| `save_crafter(filename="crafter.pkl")` | Save the Crafter object to a pickle file |
| `load_crafter(filename)` | Load a previously saved Crafter object |

**Key Attributes:**

- `mol_info`: dictionary keyed by `MOL_0`, `MOL_1`, ... (in the order of `structure.molecules`) with the structures, charges and settings of each species
- `structure`: the `structure` section of the YAML file
- `params_geoopt`, `params_charge`, `params_ff`: the `geoopt`, `charge` and `forcefield` sections

### Configuration via `crafter.yml`

```yaml
geoopt:
  basis: 6-311+g(2d,p)        # Basis set for the QM optimization
  method: wb97xd               # DFT method (translated to wb97x-d for Psi4)
  software: psi4               # psi4 or g16

charge:
  type: resp                   # resp (psi4 or g16) or am1bcc (g16 path only)
  software: psi4               # psi4 or g16
  # params:                    # optional; overrides the RESP level of theory
  #   method: hf
  #   basis: 6-31g(d)

forcefield:
  fftype: gaff-2.11            # GAFF version
  iontype: amber/ions/ionsff99_tip3p.xml  # Ion parameters (the default; or Gmanr / Madrid / Wu-Wick / SMM for Li+)
  charge_scale_ion: 0.8        # Charge scaling for ions

structure:
  type: liquid                 # "crystal" or "liquid"
  molecules:                   # One PDB file per species
    - Li.pdb
    - BF4.pdb
    - SL.pdb
  nmols: [1, 1, 10]            # Number (or ratio) of each species
  density_kgm3: 1304           # Target density (kg/m³)
  cell_A: [40, 40, 40]         # Box lengths (Å)
  fixed_property: density      # See the table below
  priority_property: cell
```

See {doc}`crystal` for the meaning of the `geoopt`, `charge` and `forcefield` keys; they are the same for both structure types.

### Output Files

| File | Written by | Content |
|------|-----------|---------|
| `MOL_{i}/` | `prep()` | QM inputs and outputs of species `i` (`MOL_{i}.xyz`, `MOL_{i}.sdf`, `MOL_{i}_resp.mol2`, optimization logs) |
| `gaffxml_{i}.xml` | `build()` | OpenMM force field of species `i` |
| `supercell.pdb` | `build()` | packmol output, without bonds |
| `supercell_bonds.pdb` | `build()` | Packed box with CONECT records |
| `system.xml` | `build()` | Serialized OpenMM `System` |
| `crafter.pkl`, `MOL_{i}/mol_info.pkl` | `save_crafter()` | Crafter state, stamped with the iMolCRAFT version |

### Exporter Function

```python
from imolcraft.io import exporter

exporter(
    pdb="supercell_bonds.pdb",   # PDB file with bonds
    system="system.xml",         # OpenMM system XML file
    filename="liquid",           # Output name without extension
    fmt="gmx",                   # "gmx" for GROMACS, "lmp" for LAMMPS
)
```

- **GROMACS** (`fmt="gmx"`): `liquid.top` and `liquid.gro`
- **LAMMPS** (`fmt="lmp"`, *experimental*): `liquid.data`

```{warning}
The LAMMPS exporter is experimental and not considered stable; verify its output before use.
LAMMPS is not installed by `env.yml`. See {doc}`exporter` for details.
```

## Key Parameters Reference

### Structure Definition (`structure`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `type` | str | `crystal`, `liquid` | System type |
| `molecules` | list | File paths | One PDB file per species. Relative paths are also searched next to the YAML file |
| `nmols` | list | Integers | Number of each species, or their ratio when the number is derived |
| `density_kgm3` | float | > 0 | Target density in kg/m³ |
| `cell_A` | list | `[x, y, z]` | Box lengths in Å |
| `fixed_property` | str | `nmols`, `cell`, `density` | Which property is kept fixed. Default `cell` |
| `priority_property` | str | `cell`, `density` | Which property is honoured when the others are derived. Default `density` |

#### Combinations of `fixed_property` and `priority_property`

| fixed_property | priority_property | Process |
|--------|---------|---------|
| `nmols`   | `cell`    | `nmols` molecules are packed into `cell_A`. `density_kgm3` is not used. |
| `nmols`   | `density` | A cubic box is derived from `nmols` and `density_kgm3`. |
| `cell`    | `density` | `nmols` is scaled, keeping the ratio, to approach `density_kgm3` in `cell_A`. |
| `density` | `cell`    | `nmols` is scaled as above, then the box is made cubic and resized so that the density equals `density_kgm3`. |

Only these four combinations are implemented. With the defaults (`cell` / `density`) `density_kgm3` is required.

## Usage Example: Liquid System to MD Simulation

### Step 1: Prepare the configuration file and the molecules

Write a `crafter.yml` as above and put one PDB file per species next to it (`Li.pdb`, `BF4.pdb`, `SL.pdb`). Each file holds a single molecule with reasonable 3D coordinates; packmol uses these coordinates as they are.

### Step 2: Prepare the species

```python
from imolcraft.crafter import Crafter

craft = Crafter("crafter.yml")

# The geometry optimization only affects the charge fit; it is often skipped
# for liquids when the PDB geometries are already reasonable.
craft.prep(do_opt=False, do_charge=True)
```

What happens during `prep()`:

1. The PDB files are read and converted to RDKit molecules, and the net charge of every species is assigned.
2. With `do_opt=True`, each species is optimized at the `geoopt` level.
3. With `do_charge=True`, RESP charges are fitted and scaled by `charge_scale_ion`.

### Step 3: Build the box and the force field

```python
craft.build()
# -> gaffxml_{i}.xml, supercell.pdb, supercell_bonds.pdb, system.xml
```

### Step 4: Export to an MD engine

```python
from imolcraft.io import exporter

exporter(pdb="supercell_bonds.pdb", system="system.xml", filename="liquid_md", fmt="gmx")
# LAMMPS (experimental, see the Exporter page)
exporter(pdb="supercell_bonds.pdb", system="system.xml", filename="liquid_md", fmt="lmp")
```

## Saving and Loading Crafter State

Save the Crafter object after `prep()` to avoid re-running the QM calculations:

```python
craft.save_crafter("liquid_system.pkl")
```

`save_crafter()` pickles the whole object (charges, settings and `mol_info` of every species) and also writes `MOL_{i}/mol_info.pkl` for each species. Both are stamped with the iMolCRAFT version that wrote them.

Resume in a new session:

```python
from imolcraft.crafter import Crafter

craft = Crafter()
craft.load_crafter("liquid_system.pkl")
craft.build()   # e.g. rebuild with another box after editing craft.structure
```
