# Crafter: Crystal Systems for MD Simulations

## Overview

The **Crafter** automates the preparation of molecular systems for molecular dynamics (MD) simulations. This page covers the **crystal** workflow: Crafter reads a crystal structure (CIF or any other format ASE can read), splits it into molecules, runs a QM geometry optimization and a charge fit for each unique species, assigns GAFF parameters and writes an OpenMM system together with a supercell PDB. The result can be exported to GROMACS (or, experimentally, LAMMPS).

## Workflow

1. **Load** the crystal structure and split it into molecules. Symmetry-equivalent copies of the same species share one entry in `mol_info`.
2. **Optimize** each species in vacuum at the QM level (`prep(do_opt=True)`). The optimized geometry is used only for the charge fit; the supercell is built from the crystal coordinates as given.
3. **Charge** each species by RESP (or AM1-BCC on the Gaussian/antechamber path) and scale the charges (`prep(do_charge=True)`).
4. **Parameterize** with GAFF and **repeat** the unit cell into a supercell (`build()`).
5. **Export** to an MD engine format with `imolcraft.io.exporter`.

## Key Components

### Crafter Class

```python
from imolcraft.crafter import Crafter

craft = Crafter("craft_params.yml")   # same as Crafter(); craft.from_yaml("craft_params.yml")
craft.prep(do_opt=True, do_charge=True)
craft.build()
craft.save_crafter("crafter.pkl")
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `from_yaml(filename)` | Load the settings and the structure from a YAML file |
| `prep(do_opt=True, do_charge=True)` | QM geometry optimization and charge fit for every species |
| `build()` | Generate the GAFF force field, the supercell and the OpenMM system. Needs the charges from `prep(do_charge=True)` |
| `get_ffxml()` | Write only the `gaffxml_{i}.xml` files, without building the system |
| `save_crafter(filename="crafter.pkl")` | Save the Crafter object to a pickle file |
| `load_crafter(filename)` | Load a previously saved Crafter object |

**Key Attributes:**

- `mol_info`: dictionary keyed by `MOL_0`, `MOL_1`, ... with the structures, charges and settings of each unique species
- `structure`: the `structure` section of the YAML file
- `params_geoopt`, `params_charge`, `params_ff`: the `geoopt`, `charge` and `forcefield` sections

### Configuration via `craft_params.yml`

```yaml
geoopt:
  basis: 6-311+g(2d,p)        # Basis set for the QM optimization
  method: wb97xd               # DFT method (translated to wb97x-d for Psi4)
  software: psi4               # psi4 or g16

charge:
  type: resp                   # resp (psi4 or g16) or am1bcc (g16 path only)
  software: psi4               # psi4 or g16
  # params:                    # optional; overrides the RESP level of theory
  #   method: hf               # defaults: hf / 6-31G* (Psi4), hf / 6-31g(d) (Gaussian)
  #   basis: 6-31g(d)

forcefield:
  fftype: gaff-2.11            # GAFF version
  iontype: amber/ions/ionsff99_tip3p.xml  # Ion parameters (the default; or Gmanr / Madrid / Wu-Wick / SMM for Li+)
  charge_scale_ion: 0.8        # Charge scaling for ions

structure:
  type: crystal                # "crystal" or "liquid"
  cif: structure.cif           # Crystal structure file
  repeat: [2, 2, 2]            # Supercell repetition (a, b, c)
```

Relative paths in `structure` are resolved against the directory of the YAML file when they are not found as given. A top-level section other than `geoopt`, `charge`, `forcefield` and `structure` raises an error.

### Output Files

| File | Written by | Content |
|------|-----------|---------|
| `MOL_{i}/` | `prep()` | QM inputs and outputs of species `i` (`MOL_{i}.xyz`, `MOL_{i}.sdf`, `MOL_{i}_resp.mol2`, optimization logs) |
| `gaffxml_{i}.xml` | `build()` | OpenMM force field of species `i` |
| `supercell_bonds.pdb` | `build()` | Supercell with CONECT records |
| `system.xml` | `build()` | Serialized OpenMM `System` |
| `crafter.pkl`, `MOL_{i}/mol_info.pkl` | `save_crafter()` | Crafter state, stamped with the iMolCRAFT version |

### Exporter Function

```python
from imolcraft.io import exporter

exporter(
    pdb="supercell_bonds.pdb",   # PDB file with bonds
    system="system.xml",         # OpenMM system XML file
    filename="crystal",          # Output name without extension
    fmt="gmx",                   # "gmx" for GROMACS, "lmp" for LAMMPS
)
```

- **GROMACS** (`fmt="gmx"`): `crystal.top` and `crystal.gro`
- **LAMMPS** (`fmt="lmp"`, *experimental*): `crystal.data`

```{warning}
The LAMMPS exporter is experimental and not considered stable; verify its output before use.
LAMMPS is not installed by `env.yml`. See {doc}`exporter` for details.
```

## Key Parameters Reference

### Geometry Optimization (`geoopt`)

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `basis` | str | `6-311+g(2d,p)` | Basis set |
| `method` | str | `wb97xd` | DFT method |
| `software` | str | `psi4` or `g16` | QM software. Psi4 optimizes with geomeTRIC; with `g16` any further key is passed to the ASE Gaussian calculator (default `opt: maxcycle=256`) |

A conformer within 0.2 Å RMSD of an already optimized copy of the same species is not optimized again.

### Charge Calculation (`charge`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `type` | str | `resp`, `am1bcc` | Charge model. `am1bcc` runs antechamber/sqm and is not available with `software: psi4` |
| `software` | str | `psi4`, `g16` | QM software for the electrostatic potential |
| `params` | dict | `{method: hf, basis: 6-31g(d)}` | Optional overrides of the RESP level of theory. With `g16` also `ioplist` and `pop` |

```{note}
`method` and `basis` written directly under `charge:` (as in older examples) are ignored. Put them under `charge.params`.
```

### Force Field (`forcefield`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `fftype` | str | `gaff-2.11`, `gaff-2.1`, `gaff-1.81`, `gaff-1.8`, `gaff-1.4` | GAFF version. Other force-field families are not supported |
| `iontype` | str | `Gmanr`, `Madrid`, `Wu-Wick`, `SMM`, or a path to an XML file. Default `amber/ions/ionsff99_tip3p.xml` | Ion force field parameters. The four names each supply Li<sup>+</sup> alone; every other ion falls back to the default Amber library. A path is taken as it stands if it exists, otherwise as relative to the `ffxml` directory of openmmforcefields |
| `charge_scale_ion` | float | 0.0-1.0 | Scale factor for the charges of ions |
| `charge_scale_neutral` | float | 0.0-1.0 | Scale factor for the charges of neutral molecules. Ignored when `charge_scale_ion` is also given |
| `fsa_assign` | dict | default `{FSA: {S: s6, N: n, O: o, F: f}}` | GAFF atom types forced on the FSA anion |

The net charge of each species is assigned automatically (monatomic ions, quaternary nitrogen, imidazolium rings; any remaining charge is shared out). PF<sub>6</sub><sup>-</sup> is recognized from its SMILES and receives a dedicated template instead of GAFF.

### Structure Definition (`structure`)

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `type` | str | `crystal`, `liquid` | System type |
| `cif` | str | File path | Crystal structure file (CIF, or any format ASE can read) |
| `repeat` | list | `[2, 2, 2]` | Supercell repetition factors (a, b, c). Default `[1, 1, 1]` |
