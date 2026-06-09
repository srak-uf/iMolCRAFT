# Exporter: Generating MD Input Files from Prepared Systems

## Overview

The **Exporter** is a versatile feature in iMolCRAFT that automates the generation of MD input files from prepared molecular systems (pdb files with bond information and force field xml files). It supports GAFF-type force fields and can export to both GROMACS and LAMMPS formats. 


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

- **LAMMPS** (`format="lmp"`):
  - `liquid.data`: Data file
