#!/usr/bin/env python
from pymatgen.core import Structure
from ase.io import read
from ase.visualize import view
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
from pymatgen.io.ase import AseAtomsAdaptor
import subprocess
# import nglview

from IPython import get_ipython
from shutil import which
import os

from sevenn.sevennet_calculator import SevenNetCalculator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="CIF file to be processed", type=str)
parser.add_argument("-r", "--repl", help="Repetition of the unit cell", type=int, nargs=3, default=[1, 1, 1])
args = parser.parse_args()


vesta_path = ["/Applications/VESTA.app/Contents/MacOS/VESTA"] # get_ipython().getoutput('which vesta')
vesta_path = vesta_path[0]


filename = args.filename
struc = Structure.from_file(filename)

subprocess.Popen([vesta_path, '-open',  filename])


repl = args.repl
struc = struc * repl
struc.to(f"{filename}_{repl[0]}x{repl[1]}x{repl[2]}.cif")


subprocess.Popen([vesta_path, '-open',  f"{filename}_{repl[0]}x{repl[1]}x{repl[2]}.cif"])




enum_cmd = which("enum.x") or which("multienum.x")
# prefer makestr.x at present
makestr_cmd = which("makestr.x") or which("makeStr.x") or which("makeStr.py")
print(enum_cmd, makestr_cmd)


en = EnumerateStructureTransformation()
cadi = en.apply_transformation(struc, 10000)



filename = f"{filename}_{repl[0]}x{repl[1]}x{repl[2]}.cif"
filename_woext = os.path.splitext(filename)[0]
filename_woext = os.path.basename(filename_woext)
os.makedirs(f"{filename_woext}_candidate", exist_ok=True)


aseatoms_list = []
for i, st_i in enumerate(cadi):
    st_i['structure'].to(f"{filename_woext}_candidate/{i}.cif")
    aseatoms_list.append(AseAtomsAdaptor.get_atoms(st_i['structure']))



sevenet_0_cal = SevenNetCalculator("7net-0", device='cpu')  



def _get_stats(aseatoms):
    aseatoms.calc = sevenet_0_cal
    energy = float(aseatoms.get_potential_energy())  
    return {"num_sites": len(aseatoms), "energy": energy, "atoms": aseatoms}    

from joblib import Parallel, delayed

print("calculating energies......")
all_structures = Parallel(n_jobs=-1)(delayed(_get_stats)(aseatoms) for aseatoms in aseatoms_list)


def sort_func(struct):
    return (
        struct["energy"] / struct["num_sites"]
    )



all_structures = sorted(all_structures, key=sort_func)



for i in range(len(all_structures)):
    all_structures[i]["atoms"].write(f"{filename_woext}_candidate/opt_{i}.cif")


