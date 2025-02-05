#!/usr/bin/env python
import os, json
import sys
import numpy as np
import shutil
from openmm import app
from openmm.app.forcefield import ForceField
from openmm.app.pdbfile import PDBFile
from parmed.openmm import load_topology

# results.jsonを読み込む
with open("results.json", "r") as f:
    results = json.load(f)

loss = results["loss"][1:]
epoch = results["epoch"][1:]

for i in np.array(loss).argsort()[:4]:
    print(epoch[i], loss[i])


for i, j in enumerate(np.array(loss).argsort()[:4]):
    if os.path.isfile(f"xmlfiles/epoch-{epoch[j]}.xml"):
        shutil.copy(f"xmlfiles/epoch-{epoch[j]}.xml", f"{i+1}_epoch-{epoch[j]}.xml")
    else:
        # shutil.copy(f"xmlfiles/epoch-{epoch[j]-1}.xml", f"{i+1}_epoch-{epoch[j]}.xml")
        AssertionError(f"xml file: xmlfiles/epoch-{epoch[j]}.xml not found")
    forcefield = ForceField(f"{i+1}_epoch-{epoch[j]}.xml")
    pdb = PDBFile("merged_supercell_bonds.pdb")
    sys = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME)
    parm_top = load_topology(topology=pdb.topology, system=sys, xyz=pdb.positions)
    parm_top.save(f"{i+1}_epoch-{epoch[j]}.top", overwrite=True)
    parm_top.save(f"{i+1}_epoch-{epoch[j]}.gro", overwrite=True)
