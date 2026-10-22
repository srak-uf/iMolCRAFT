# Exporter: Generating MD Input Files from Prepared Systems

## Overview

The **Exporter** turns a prepared system (a PDB file with bond information and a serialized OpenMM `System`) into the input files of an MD engine. GROMACS is the supported target; LAMMPS output is experimental. The `system.xml` and `supercell_bonds.pdb` written by {doc}`Crafter.build() <crystal>` are exactly the inputs it expects.

```{warning}
**The LAMMPS exporter (`fmt="lmp"`) is experimental and is not considered stable.**
Its output has received far less validation than the GROMACS exporter, and the generated
data file may be incomplete or incorrect for some systems. Always verify the resulting
topology, force field parameters and box definition before running a production simulation.
The GROMACS exporter (`fmt="gmx"`) is the recommended and supported path.

LAMMPS itself is **not** included in `env.yml`. The `lammps` conda-forge package cannot be
installed alongside `ambertools>=25` because the two require incompatible `libnetcdf`
versions. iMolCRAFT never invokes the `lmp` binary -- it only writes input files -- so this
does not affect any iMolCRAFT functionality. If you want to run the exported input, install
LAMMPS separately (for example in its own conda environment).
```

## `exporter(pdb, system, filename, fmt)`

| Argument | Description |
|----------|-------------|
| `pdb` | PDB file with CONECT records (the topology) |
| `system` | OpenMM `System` serialized to XML |
| `filename` | Output name without extension |
| `fmt` | `"gmx"` (GROMACS) or `"lmp"` (LAMMPS). Anything else raises `ValueError` |

The argument is called `fmt`, not `format`; `format=` raises a `TypeError`.

| `fmt` | Files written | Notes |
|-------|---------------|-------|
| `gmx` | `{filename}.top`, `{filename}.gro` | Written through ParmEd; existing files are overwritten. OpenMM virtual sites are not translated to `[ virtual_sites2 ]` |
| `lmp` | `{filename}.data` | Written through OpenFF Interchange (sets `INTERCHANGE_EXPERIMENTAL=1` in the process). One molecule is built per residue name, so residues with the same name must be the same molecule |

## Example

### 1. Generating `system.xml` from a PDB file and force field XML files

`Crafter.build()` already writes `system.xml`. To make one by hand:

```python
from openmm import XmlSerializer
from openmm.app import PME, ForceField, PDBFile
from openmm.unit import nanometer

pdb = PDBFile("supercell_bonds.pdb")
forcefield = ForceField("gaffxml_0.xml")
system = forcefield.createSystem(
    pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.5 * nanometer, constraints=None
)

with open("system.xml", "w") as output:
    output.write(XmlSerializer.serialize(system))
```

### 2. Exporting to MD engine formats

```python
from imolcraft.io import exporter

exporter("supercell_bonds.pdb", "system.xml", "liquid", fmt="gmx")   # liquid.top, liquid.gro
exporter("supercell_bonds.pdb", "system.xml", "liquid", fmt="lmp")   # liquid.data (experimental)
```

To run the MD itself, use `imolcraft.calculator.MDCalculator`, which drives OpenMM or GROMACS from the same `.ffxml` and `.pdb` and calls this exporter internally for GROMACS (see {doc}`thermodyn_perturb`).
