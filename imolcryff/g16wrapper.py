from ase.io import read, write
from ase.calculators.gaussian import Gaussian
import os, subprocess


default_params_opt = {
    "method": "wb97xd",
    "basis": "6-311++g(d,p)",
    "opt": "maxcycle=256",
    # "mem": "92GB",
    # "nprocshared": 40,
}

default_params_charge = {
    "method": "hf",
    "basis": "6-31g(d)",
    # "mem": "92GB",
    # "nprocshared": 40,
    "ioplist": ["6/33=2", "6/42=6"], 
    "pop": "mk"
}

default_params_dihedral = {
    "method": "wb97xd",
    "basis": "6-311++g(d,p)",
    "opt": "modredundant",
    # "mem": "92GB",
    # "nprocshared": 40,
}

class G16Calculator:
    def __init__(self, atoms, charge, output_dir, label, params=None):
        self.atoms = atoms
        self.params = params
        self.charge = charge
        self.output_dir = output_dir
        self.label = label
        self.g16 = None

    def input_g16(self):
        output_xyz = os.path.join(self.output_dir, f"{self.label}.xyz")
        output_label = os.path.join(self.output_dir, f"{self.label}")
        self.atoms.pbc = False
        self.atoms.cell = None
        write(output_xyz, self.atoms, format="xyz")
        self.g16 = Gaussian(
            label=output_label,
            charge=self.charge,
            **self.params,
        )
        self.g16.write_input(self.atoms, system_changes=0)
    
    def get_resp(self, output_mol2, g16logfile):
        g16logfile = os.path.abspath(g16logfile)
        cmd_antech = (
                    f"antechamber -i {g16logfile} -fi gout "
                    f"-o {output_mol2} -fo mol2 -at sybyl -c resp -nc {self.charge} -pf y -dr no")
        print(cmd_antech)
        output = subprocess.getoutput(cmd_antech)
        print(output)
    




def input_g16(atoms, params, charge, output_dir, label):
    output_xyz = os.path.join(output_dir, f"{label}.xyz")
    output_label = os.path.join(output_dir, f"{label}")
    atoms.pbc = False
    atoms.cell = None
    write(output_xyz, atoms, format="xyz")
    g16 = Gaussian(
        label=output_label,
        charge = charge,
        **params,
    )
    g16.write_input(atoms, system_changes=0)
    return g16

def molinfo_setg16opt(mol_info, params_opt=None):
    if params_opt is None:
        params_opt = default_params_opt
    for key in mol_info.keys():
        output_dir = key
        mol_info[key]["g16opt"] = []
        mol_info[key]["g16optlog"] = []
        for i in range(len(mol_info[key]["aseatoms_list"])):
            atoms = mol_info[key]["aseatoms_list"][i]
            charge = mol_info[key]["charge"]
            label = f"{key}_{i}"
            g16 = input_g16(atoms, params_opt, charge, output_dir, label)
            mol_info[key]["g16opt"].append(g16)

def molinfo_setg16charge(mol_info, params_charge=None):
    if params_charge is None:
        params_charge = default_params_charge
    else:
        if "algo" in params_charge:
            if params_charge["algo"] == "resp":
                del params_charge['algo']
                params_charge["ioplist"] = ["6/33=2", "6/42=6"]
                params_charge["pop"] = "mk"
            else:
                # AssertionError
                assert False, f"Invalid algo {params_charge['algo']}"

    for key in mol_info.keys():
        if "geoopt_energy" in mol_info[key]:
            energy_list = mol_info[key]["geoopt_energy"]
            # energy_listが全てNoneの場合はエラーを出力し次の分子へ
            if all(x is None for x in energy_list):
                print(f"{key} has no geoopt results")
                atoms = mol_info[key]["aseatoms_list"][0]
            else:
                energy_list = [1e10 if x is None else x for x in energy_list]
                min_idx = mol_info[key]["geoopt_energy"].index(min(energy_list))
                # stable_atoms = mol_info[key]["g16opt"][min_idx]
                # optlog = mol_info[key]["g16opt"][min_idx].label + ".log"
                optlog = mol_info[key]["g16optlog"][min_idx]
                atoms = read(optlog)
        else:
            atoms = mol_info[key]["aseatoms_list"][0]
        charge = mol_info[key]["charge"]
        label = f"{key}_charge"
        g16 = input_g16(atoms, params_charge, charge, key, label)
        mol_info[key]["g16charge"] = g16


def molinfo_setg16dihedral(mol_info, params_dihedral=None):
    if params_dihedral is None:
        params_dihedral = default_params_dihedral

    for key in mol_info.keys():
        energy_list = mol_info[key]["geoopt_energy"]
        # energy_listが全てNoneの場合はエラーを出力し次の分子へ
        if all(x is None for x in energy_list):
            print(f"{key} has no geoopt results")
            continue
        elif "rotatable_dihedral" not in mol_info[key].keys():
            print(f"{key} get rotatable dihedral")
            print("mol_info.mol_info.get_rotatable_dihedral()")
            continue
        else:
            # energy_listでもしNoneの場合は1e10に置き換える
            mol_info[key]["g16dihedral"] = []
            energy_list = [1e10 if x is None else x for x in energy_list]
            min_idx = mol_info[key]["geoopt_energy"].index(min(energy_list))
            stable_atoms = mol_info[key]["g16opt"][min_idx]
            optlog = mol_info[key]["g16opt"][min_idx].label + ".log"
            atoms = read(optlog)
            charge = mol_info[key]["charge"]
            for di in range(len(mol_info[key]["rotatable_dihedral"])):
                label = f"{key}_dihed_{di}"
                d0 = mol_info[key]["rotatable_dihedral"][di][0] + 1
                d1 = mol_info[key]["rotatable_dihedral"][di][1] + 1
                d2 = mol_info[key]["rotatable_dihedral"][di][2] + 1
                d3 = mol_info[key]["rotatable_dihedral"][di][3] + 1
                params_dihedral["addsec"] = f"D {d0} {d1} {d2} {d3} S 35 10.0"
                g16 = input_g16(atoms, params_dihedral, charge, key, label)
                with open(g16.label+".com") as f:
                    lines = f.readlines()
                del lines[-3]     # to cope with bug of ase
                with open(g16.label+".com",mode="w") as f:
                    f.writelines(lines)
                mol_info[key]["g16dihedral"].append(g16)

def parse_g16scan(file):
    with open(file) as f:
        lines = f.readlines()
        parserd_lines = []
        scanned_energy = []
        for line in lines:
            if "Optimization completed." in line:
                scanned_energy.append(parserd_lines[-1])
                parserd_lines.append(line)
            elif "SCF Done:" in line:
                parserd_lines.append(float(line.split()[4]))
    return scanned_energy
