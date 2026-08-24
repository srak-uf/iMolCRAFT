# Exporter: Generating MD Input Files from Prepared Systems

## Overview

The **Exporter** is a versatile feature in iMolCRAFT that automates the generation of MD input files from prepared molecular systems (pdb files with bond information and force field xml files). It supports GAFF-type force fields and can export to both GROMACS and LAMMPS formats.

```{warning}
**The LAMMPS exporter (`format="lmp"`) is experimental and is not considered stable.**
Its output has received far less validation than the GROMACS exporter, and the generated
data file may be incomplete or incorrect for some systems. Always verify the resulting
topology, force field parameters and box definition before running a production simulation.
The GROMACS exporter (`format="gmx"`) is the recommended and supported path.

LAMMPS itself is **not** included in `env.yml`. The `lammps` conda-forge package cannot be
installed alongside `ambertools>=25` because the two require incompatible `libnetcdf`
versions. iMolCRAFT never invokes the `lmp` binary -- it only writes input files -- so this
does not affect any iMolCRAFT functionality. If you want to run the exported input, install
LAMMPS separately (for example in its own conda environment).
```


## Example
### 1. Generating the system.xml file from a pdb file and a force field xml file.
```python
from openmm.app import *
from openmm import *
from openmm.unit import *
from imolcraft.io import exporter

pdb = PDBFile('supercell_bonds.pdb')
forcefield = ForceField('gaffxml_0.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=0.5*nanometer, constraints=None)

with open("system.xml", "w") as output:
    output.write(XmlSerializer.serialize(system))

```

### 2. Exporting to MD Engine Formats
```python
exporter("supercell_bonds.pdb", "system.xml", "liquid", format="gmx")
exporter("supercell_bonds.pdb", "system.xml", "liquid", format="lmp")
```
- **GROMACS** (`format="gmx"`): 
  - `liquid.top`: Topology file
  - `liquid.gro`: Structure file (GROMACS format)

- **LAMMPS** (`format="lmp"`, *experimental -- see the warning above*):
  - `liquid.data`: Data file
