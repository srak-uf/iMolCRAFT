from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDepictor
from ase.io import read, write
from ase import Atoms
from ase.calculators.gaussian import Gaussian
import os, copy, pickle
import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit import Quantity
from openff import toolkit
from openmm.app import *
from openmm import *
from .asemol import asemol_wrapper, cast_molecules
from .dihedral import DihedCalculator
from .charge import ChargeCalculator

MOLINFO_KEYS = {
    "aseatoms_list": list,
    "aseatoms_geoopt": list,
    "aseatoms_stable": type(None),
    "ChargeCalc": type(None),
    "DihedCalc": type(None),
    "directory": str,
    "metadata": dict,
    "Molecule_OFF": type(None),
    "Natoms": type(None),
    "Nmols": type(None),
    "netcharge": type(None),
    "networkX": list,
    "partial_charges": list,
    "rdkit": dict, 
    "SMILES": str, 
}

class Crafter:
    def __init__(self):
        self.mol_info = {}

    def _initialize_molinfo(self, key: str):
        """
        Initialize mol_info[key] with predefined keys.
        """
        self.mol_info[key] = {k: v() if callable(v) else v for k, v in MOLINFO_KEYS.items()}

    def from_crystal(self, filename: str, assign_totalcharge=True):
        """
        Read cif, xyz, POSCAR, ... file and append to mol_info dictionary
        filename: Structure file name (It should be ASE compatible)
        """
        atoms = read(filename)
        asemol_wrap = asemol_wrapper(atoms)
        self.atoms_unwrap = asemol_wrap.unwrap_molecules()
        self.molatoms, self.molecule_list, self.networkX \
            = asemol_wrap.get_ase_molecules(out_nX=True)

        for i, mol_idx in enumerate(self.molecule_list):
            al = [self.molatoms[i] for i in mol_idx]
            nX = [self.networkX[i] for i in mol_idx]
            self.append_fromAtomsList(al, 
                                      key=f"MOL_{i}", 
                                      Nmols=len(mol_idx), 
                                      networkX=nX)
        
            if assign_totalcharge:
                self._assign_totalcharge()
            else:
                print(f"Warning: MOL_{i} has no charge information")
                print(f"Please define the total charge manually")

    def append_fromAtomsList(self, atomslist: list, key: str, Nmols: int = None, networkX: list = []):
        """
        Append List of ase.Atoms to mol_info dictionary
        
        Parameters:
        ----------
        atomslist: List of ase.Atoms
        key: key of mol_info dictionary
        Nmols: Number of molecules
        networkX: List of NetworkX graph
        """
        self._initialize_molinfo(key)
        self.mol_info[key]["aseatoms_list"] = atomslist
        self.mol_info[key]["aseatoms_geoopt"] = [None for _ in range(len(atomslist))]
        self.mol_info[key]["Natoms"] = len(atomslist[0])
        self.mol_info[key]["Nmols"] = len(atomslist) if Nmols is None else Nmols
        if networkX != []:
            assert len(atomslist) == len(networkX), "The length of atomslist and networkX should be the same"
            self.mol_info[key]["networkX"] = networkX
        os.makedirs(key, exist_ok=True)
        self.mol_info[key]["directory"] = key
        write(f"{key}/{key}.xyz", atomslist[0], format="xyz")

    def get_rdkitmol(self, keys=None, il_assing=True):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            natoms = self.mol_info[key]["Natoms"]
            directory = self.mol_info[key]["directory"]

            if natoms > 1:
                mol = Chem.MolFromXYZFile(f"{directory}/{key}.xyz")
                mol = Chem.Mol(mol)
                nc = self.mol_info[key]["netcharge"]
                try:
                    mol_tmp = copy.deepcopy(mol)
                    rdDetermineBonds.DetermineBonds(mol_tmp,charge=nc)
                    mol = mol_tmp
                except:
                    from rdkit.Chem import Draw
                    mol_tmp = copy.deepcopy(mol)
                    rdDetermineBonds.DetermineConnectivity(mol_tmp, charge=nc)
                    mol = copy.deepcopy(mol_tmp)
                    print("Warning: The bonds may be assigned incorrectly. Please check carefully.")
                    print("Warning: This may give wrong SMILES")
                    Draw.MolToImage(mol) # Necessary
                mol2d = copy.deepcopy(mol)
                rdDepictor.Compute2DCoords(mol2d)
                self.mol_info[key]["rdkit"]["mol"] = mol
                self.mol_info[key]["rdkit"]["mol2d"] = mol2d
            else:
                symbol = self.mol_info[key]["symbol"]
                nc = self.mol_info[key]["netcharge"]
                if nc > 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}+{abs(nc)}]")
                elif nc < 0:
                    mol = Chem.MolFromSmiles(f"[{symbol}-{abs(nc)}]")
                else:
                    mol = Chem.MolFromSmiles(f"[{symbol}]")
                self.mol_info[key]["rdkit"]["mol"] = mol
                self.mol_info[key]["rdkit"]["mol2d"] = mol

            nc = self.mol_info[key]["netcharge"]
            self._il_assign(mol, mol2d, int(nc))

    def get_smiles(self, keys=None):
        if keys is None:
            keys = self.mol_info.keys()
        for key in keys:
            if self.mol_info[key]["rdkit"] is None:
                self.get_rdkitmol(key)
            mol = self.mol_info[key]["rdkit"]["mol"]
            self.mol_info[key]["SMILES"] = Chem.MolToSmiles(mol)

    def get_sdf(self, keys=None):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if self.mol_info[key]["rdkit"] is None:
                self.get_rdkitmol(key)

            mol = self.mol_info[key]["rdkit"]["mol"]
            output_dir = f"{key}"
            self.mol_info[key]["directory"] = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f"{key}.sdf")
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()

    def get_optstructure(self, keys=None, do_calc=True, rmsd=0.2):
        geoopt_params = {
                "method": "wb97xd",
                "basis": "6-311+g(2d,p)",
                "opt": "maxcycle=256",
        }
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            for i, atoms in enumerate(self.mol_info[key]["aseatoms_list"]):
                atoms_tmp = atoms.copy()
                atoms_tmp.pbc = False
                atoms_tmp.cell = None
                if self.mol_info[key]["netcharge"] is None:
                    self._assign_totalcharge()
                nc = self.mol_info[key]["netcharge"]
                calc_geoopt = Gaussian(label=f'{key}_{i}', charge=nc, **geoopt_params)
                calc_geoopt.directory = self.mol_info[key]["directory"]
                atoms_tmp.calc = calc_geoopt
                if do_calc:
                    # geoopt is skipped if the similar conformation was already calculated
                    rmsd_skip = False
                    for j in range(0, i):
                        gj = self.mol_info[key]["networkX"][j]
                        gi = self.mol_info[key]["networkX"][i]
                        rmsd_ji, _ = cast_molecules(gj, gi)
                        print(rmsd_ji, i, j)
                        if rmsd_ji < rmsd:
                            rmsd_skip = True
                            break
                                
                    if rmsd_skip:
                        self.mol_info[key]["aseatoms_geoopt"][i] \
                                  = self.mol_info[key]["aseatoms_geoopt"][j]
                        print(f"Skip geometry optimization of {key}_{i}: RMSD = {rmsd_ji} A < {rmsd}")
                        rmsd_skip = False
                    else:
                        self.mol_info[key]["aseatoms_geoopt"][i] = atoms_tmp
                        print(f"Calculating geometry optimization of {key}_{i}")
                        _ = self.mol_info[key]["aseatoms_geoopt"][i].get_potential_energy()
                        print(f"Finished geometry optimization of {key}_{i}")

    def get_partial_charges(self, charge_type="resp", keys=None, ff_params=None):
        """
        Get partial charges by RESP or AM1-BCC
        charge_type: "resp" or "am1bcc"
        keys: list of keys to get partial charges
        """
        if keys is None:
            keys = self.mol_info.keys()
        
        if charge_type not in ["resp", "am1bcc"]:
            assert False, "charge_type should be resp or am1bcc"

        for key in keys:
            if self.mol_info[key]["aseatoms_stable"] != None:
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]
            
            netcharge = 0
            label = key
            self.mol_info[key]["charge"] = ChargeCalculator(
                atoms,
                charge_type,
                netcharge,
                label,
                directory=self.mol_info[key]["directory"]
            )

            self.mol_info[key]["partial_charges"] = \
                self.mol_info[key]["charge"].get_partialcharges()
        
        if ff_params is not None:
            self._adjust_charges(ff_params)
    
    def get_dihedral_qm(self, keys=None, do_calc=True):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            if isinstance(self.mol_info[key]["aseatoms_stable"], Atoms):
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]

            if self.mol_info[key]["DihedCalc"] is None:
                self.mol_info[key]["DihedCalc"] = DihedCalculator(
                    atoms=atoms,
                    rdkitmol=self.mol_info[key]["rdkit"]["mol"],
                    label=key,
                    directory = self.mol_info[key]["directory"],
                )
            
            self.mol_info[key]["DihedCalc"].get_dihedral_qm(do_calc=do_calc)
    
    def get_dihedral_ff(self, keys=None, do_calc=True):
        if keys is None:
            keys = self.mol_info.keys()
        for key in keys:
            self.mol_info[key]["DihedCalc"].get_dihedral_ff(do_calc=do_calc)

    def _adjust_charges(self, ff_params=None):
        for key in self.mol_info.keys():
            net_charge = self.mol_info[key]["charge"]
            charges = np.array([ np.float64(ee) for ee in self.mol_info[key]["charges"]])

            if ff_params is not None:
                if "charge_scale_ion" in ff_params.keys():
                    charge_scale_ion = ff_params["charge_scale_ion"]
                    if not np.isclose(net_charge, 0.0):
                        charges *= charge_scale_ion
                        net_charge *= charge_scale_ion
                elif "charge_scale_neutral" in ff_params.keys():
                    charge_scale_neutral = ff_params["charge_scale_neutral"]
                    if np.isclose(net_charge, 0.0):
                        charges *= charge_scale_neutral
                        net_charge *= charge_scale_neutral

            total_charge = np.sum(charges)

            charge_deficit = total_charge - net_charge
            print(charge_deficit)

            if not np.isclose(charge_deficit, 0.0):
                print(f"Net charge of {key} is {net_charge} and total charge is {total_charge}")
                charges = charges - charge_deficit / len(charges)
                total_charge = np.sum(charges)
                charge_deficit = total_charge - net_charge
                charges[0] -= charge_deficit
            
            self.mol_info[key]["charges"] = charges


    def get_molecule_off(self):
        molecules_off = []
        for mol in self.mol_info.keys():
            dirname = self.mol_info[mol]["directory"]
            sdffile = os.path.join(dirname, f"{mol}.sdf")

            if not os.path.exists(sdffile):
                self.get_sdf()
            
            molecule_off = Molecule.from_file(sdffile)
            molecule_off.partial_charges = Quantity(self.mol_info[mol]["charges"], toolkit.unit.elementary_charge)
            molecule_off.name = mol
            # molecule_mm.total_charge = Quantity(self.mol_info[mol]["charge"], unit.elementary_charge)  ## why ??
            for meta_key in self.mol_info[mol]["metadata"].keys():
                for meta_ind in self.mol_info[mol]["metadata"][meta_key]:
                    molecule_off.atoms[meta_ind].metadata[meta_key] = True
            if "mol2file" in self.mol_info[mol]:
                molecule_off.mol2file = os.path.join(self.mol_info[mol]["directory"], self.mol_info[mol]["mol2file"])
            molecules_off.append(molecule_off)
        return molecules_off
        
    def save_crafter(self, filename="crafter.pkl"):
        with open(filename, mode='wb') as f:
            pickle.dump(self,f)
        
        for key in self.mol_info.keys():
            directory = self.mol_info[key]["directory"]
            with open(f"{directory}/mol_info.pkl", mode='wb') as f:
                mol_info = copy.deepcopy(self.mol_info[key])
                mol_info.pop("directory")
                pickle.dump(mol_info, f)
    
    def load_crafter(self, filename):
        with open(filename, mode='rb') as f:
            crafter = pickle.load(f)
        self.mol_info = crafter.mol_info

    def _assign_totalcharge(self):
        num_charge_none = 0
        total_charge = 0
        for key in self.mol_info.keys():
            totalnum_elec = self.mol_info[key]["aseatoms_list"][0].get_atomic_numbers().sum()
            openshell_flag = totalnum_elec % 2 == 1
            self.mol_info[key]["symbol"] = str(self.mol_info[key]["aseatoms_list"][0].symbols)
            if self.mol_info[key]["symbol"] in ["Li", "Na", "K", "Rb", "Cs", "Mg"]:
                self.mol_info[key]["netcharge"] = 1
            elif self.mol_info[key]["symbol"] in ["F", "Cl", "Br", "I"]:
                self.mol_info[key]["netcharge"] = -1
            elif self.mol_info[key]["symbol"] in ["Mg", "Ca", "Sr", "Ba"]:
                self.mol_info[key]["netcharge"] = 2
            elif openshell_flag:
                self.mol_info[key]["netcharge"] = None
            else:
                self.mol_info[key]["netcharge"] = 0
        for key in self.mol_info.keys():
            if self.mol_info[key]["netcharge"] is None:
                num_charge_none += 1 * self.mol_info[key]["Nmols"]
            else:
                total_charge += self.mol_info[key]["netcharge"] * self.mol_info[key]["Nmols"]
        if num_charge_none > 0 and total_charge % num_charge_none == 0:
            charge_per_none = int(-total_charge / num_charge_none)
            for key in self.mol_info.keys():
                if self.mol_info[key]["netcharge"] is None:
                    self.mol_info[key]["netcharge"] = charge_per_none

    def _il_assign(self, mol, mol2d, nc):
        fsalike_Nindex = []
        fsalike_Sindex = []
        fsalike_Oindex = []
        if nc < 0:
            atoms = mol.GetAtoms()
            atoms2d = mol2d.GetAtoms()

            # Check FSA-like N, S, and O atoms
            for i in range(len(atoms)):
                if atoms[i].GetTotalValence() == 3 and atoms[i].GetSymbol() == "N":
                    for bond in atoms[i].GetBonds():
                        if bond.GetEndAtom().GetSymbol() == "S":
                            s_index = bond.GetEndAtomIdx()
                            if atoms[s_index].GetTotalValence() == 6:
                                fsalike_Nindex.append(i)
                                fsalike_Sindex.append(s_index)
                        elif bond.GetBeginAtom().GetSymbol() == "S":
                            s_index = bond.GetBeginAtomIdx()
                            if atoms[s_index].GetTotalValence() == 6:
                                fsalike_Nindex.append(i)
                                fsalike_Sindex.append(s_index)

            fsalike_Nindex = list(set(fsalike_Nindex))
            fsalike_Sindex = list(set(fsalike_Sindex))

            # Sに結合しているO原子のindexをfsalike_Oindexに追加
            for i in fsalike_Sindex:
                for bond in atoms[i].GetBonds():
                    if bond.GetEndAtom().GetSymbol() == "O":
                        fsalike_Oindex.append(bond.GetEndAtomIdx())
                    elif bond.GetBeginAtom().GetSymbol() == "O":
                        fsalike_Oindex.append(bond.GetBeginAtomIdx())
            fsalike_Oindex = list(set(fsalike_Oindex))

            for i in fsalike_Nindex:
                atoms[i].SetFormalCharge(-1)
                atoms2d[i].SetFormalCharge(-1)
                for b_idx, bond in enumerate(atoms[i].GetBonds()):
                    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        ## 3d rdkitmol
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                        ## 2d rdkitmol
                        bond2d = atoms2d[i].GetBonds()[b_idx]
                        bond2d.SetBondType(Chem.rdchem.BondType.SINGLE)

            for i in fsalike_Sindex:
                for b_idx, bond in enumerate(atoms[i].GetBonds()):
                    bonded_element = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE \
                        and "O" in bonded_element:
                        oxygen_idx = bond.GetBeginAtomIdx() if bonded_element[0] == "O" else bond.GetEndAtomIdx()
                        ## 3d rdkitmol
                        bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
                        atoms[oxygen_idx].SetFormalCharge(0)

                        ## 2d rdkitmol
                        bond2d = atoms2d[i].GetBonds()[b_idx]
                        bond2d.SetBondType(Chem.rdchem.BondType.DOUBLE)
                        atoms2d[oxygen_idx].SetFormalCharge(0)

            # self.mol_info[key]["metadata"]["FSA_N"] = fsalike_Nindex
            # self.mol_info[key]["metadata"]["FSA_S"] = fsalike_Sindex
            # self.mol_info[key]["metadata"]["FSA_O"] = fsalike_Oindex


