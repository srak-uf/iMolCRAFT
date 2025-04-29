from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDepictor
from ase.io import read, write
from ase import Atoms
from ase.calculators.gaussian import Gaussian
from ase.optimize import BFGS
from ase.calculators.psi4 import Psi4
import os, copy, pickle, yaml, tempfile
import numpy as np
from openff.toolkit.topology import Molecule
from openff.toolkit import Quantity
from openff import toolkit
from openmm.app import *
from openmm import *
from .asemol import asemol_wrapper, cast_molecules, pdb2packmol
from ..calculator.dihedral import DihedCalculator
from ..calculator.charge import ChargeCalculator, Psi4ChargeCalculator
from .gaffil_generators import GAFFilTemplateGenerator
from .ffxml import gafftemplate2xml
from .asemol import merge_asemols, aseatoms2pdb

MOLINFO_KEYS = {
    "aseatoms_list": list,
    "aseatoms_geoopt": list,
    "aseatoms_stable": type(None),
    "ChargeCalc": type(None),
    "DihedCalc": type(None),
    "directory": str,
    "metadata": dict,
    "molecule_OFF": type(None),
    "natoms": type(None),
    "nmols": type(None),
    "netcharge": type(None),
    "networkX": list,
    "partial_charges": list,
    "rdkit": dict, 
    "SMILES": str, 
}

class Crafter:
    """
    Crafter is a class designed to handle molecular information and perform various operations 
    such as geometry optimization, charge calculation, dihedral angle analysis, and force field 
    generation. It integrates multiple tools and libraries like ASE, OpenFF, Psi4, and RDKit 
    to streamline molecular modeling workflows.

    Parameters:
    ----------
    yml : str, optional
        Path to a YAML file containing parameters for geometry optimization, 
        charge calculation, force field generation, and molecular structure.
        If provided, the parameters and structure will be loaded automatically.

    Attributes:
    ----------
    mol_info : dict
        A dictionary to store molecular information for each molecule.
    params_geoopt : dict or None
        Parameters for geometry optimization.
    params_charge : dict or None
        Parameters for charge calculation.
    params_ff : dict or None
        Parameters for force field generation.
    structure : dict or None
        Information about the molecular structure (e.g., crystal or liquid).
    """
    def __init__(self, yml=None):
        """
        Initialize the Crafter object.

        Parameters:
        ----------
        yml : str, optional
            Path to a YAML file containing parameters for geometry optimization, 
            charge calculation, force field generation, and molecular structure.
            If provided, the parameters and structure will be loaded automatically.
        """
        self.mol_info = {}
        self.params_geoopt = None
        self.params_charge = None
        self.params_ff = None
        self.structure = None
        if yml is not None:
            self.from_yaml(yml)

    def from_yaml(self, filename):
        data = self._parser_yaml(filename)
        self.params_geoopt = data["geoopt"]
        self.params_charge = data["charge"]
        self.params_ff = data["forcefield"]
        self.structure = data["structure"]

    def prep(self, do_opt=True, do_charge=True):
        # initial structure generation
        if self.structure["type"] == "crystal":
            self.from_crystal(self.structure["cif"])
        elif self.structure["type"] == "liquid":
            for i, atoms in enumerate(self.structure["molecules"]):
                atoms = read(atoms)
                self.append_fromAtomsList([atoms],
                                          f"MOL_{i}",
                                          Nmols=int(self.structure["nmols"][i]))
            self._assign_totalcharge()
        self.get_rdkitmol()
        self.get_molecule_off()
        if do_opt == True:
            self.get_optstructure(**self.params_geoopt)
        if do_charge == True:
            self.get_partial_charges(params_ff=self.params_ff, **self.params_charge)
            self.get_molecule_off()
        
    def build(self):
        self.get_molecule_off()
        molecules = []
        
        for key in self.mol_info.keys():
            molecule = self.mol_info[key]["molecule_OFF"]
            molecule.mol2file = self.mol_info[key]["ChargeCalc"].mol2file
            molecules.append(molecule)
        
        if self.params_ff["fftype"].split("-")[0] == "gaff":
            fftemplate = GAFFilTemplateGenerator(molecules=molecules,
                                                 forcefield=self.params_ff["fftype"],
                                                 il_assign=self.params_ff.get("fsa_assign", None))
        else:
            assert False, "Unknown forcefield type. Please check the forcefield type."
        
        # output xml files
        ffxmlfiles = gafftemplate2xml(molecules, fftemplate, ion_ffxml=self.params_ff.get("iontype", "amber/ions/ionsff99_tip3p.xml"))

        # create structure
        ## crystal structure
        if self.structure["type"] == "crystal":
            system_atoms = merge_asemols(self.molatoms)
            system_atoms = system_atoms.repeat(self.structure.get("repeat", [1,1,1]))
            with tempfile.NamedTemporaryFile() as temp_pdb:
                temp_pdb_name = temp_pdb.name
                aseatoms2pdb(temp_pdb_name, system_atoms)
                pdb = PDBFile(temp_pdb_name)
                pdb_b = asemol_wrapper(system_atoms)
                _ = pdb_b.get_bonds()
                bonds_list = [ [bond[0], bond[1]] for bond in pdb_b.bonds]
                atomlist_openmm = [a for a in pdb.topology.atoms()]
                for bond in bonds_list:
                    a1 = atomlist_openmm[bond[0]]
                    a2 = atomlist_openmm[bond[1]]
                    pdb.topology.addBond(a1, a2)
        ## liquid structure
        elif self.structure["type"] == "liquid":
            molstructures = self.structure["molecules"]
            b, a, m   = pdb2packmol(molstructures, 
                        self.structure.get("nmols"),
                        desired_density=self.structure.get("density_kgm3", None),
                        cell=self.structure.get("cell_A", None),
                        outfile="supercell.pdb")
            pdb = PDBFile("supercell.pdb")
            atomlist_openmm = [a for a in pdb.topology.atoms()]
            for bond in b:
                a1 = atomlist_openmm[bond[0]]
                a2 = atomlist_openmm[bond[1]]
                pdb.topology.addBond(a1, a2)

        # Save the supecell with bonds
        pdb.writeFile(pdb.topology, pdb.positions, open("supercell_bonds.pdb", "w"))
        pdb = PDBFile("supercell_bonds.pdb")

        # create system
        forcefield = ForceField(*ffxmlfiles)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME)

        with open('system.xml', 'w') as output:
            output.write(XmlSerializer.serialize(system))

    def _initialize_molinfo(self, key: str):
        """
        Initialize mol_info[key] with predefined keys.
        """
        self.mol_info[key] = {k: v() if callable(v) else v for k, v in MOLINFO_KEYS.items()}

    def _parser_yaml(self, filename):
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)

        if set(data.keys()) <= set(["geoopt", "charge", "forcefield", "structure"]):
            pass
        else:
            assert False, "Unknown keys in the yaml file. Please check the file."

        return data

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

    def get_rdkitmol(self, keys=None, il_assign=True):
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
            il_dict = self._il_assign(self.mol_info[key]["rdkit"]["mol"],
                                      self.mol_info[key]["rdkit"]["mol2d"],
                                      int(nc))
            self.mol_info[key]["metadata"].update(il_dict)

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

    # GeoOptimizerとして別ファイルに移す案もあり
    def get_optstructure(self, keys=None, do_calc=True, rmsd=0.2, **kwargs):
        geoopt_params_g16 = {
                "method": "wb97xd",
                "basis": "6-311+g(2d,p)",
                "opt": "maxcycle=256",
        }
        geoopt_params_psi4 = {
                "method": "wb97x-d",
                "basis": "6-311+g(2d,p)",
        }

        if kwargs.get("software") == "psi4":
            geoopt_params_psi4.update(kwargs)
            method = geoopt_params_psi4.get("method")
            if method == "wb97xd":
                geoopt_params_psi4["method"] = "wb97x-d"
            geoopt_params_psi4.pop("software", None)
            psi4_flag = True
            g16_flag = False
        elif kwargs.get("software") == "g16":
            geoopt_params_g16.update(kwargs)
            geoopt_params_g16.pop("software", None)
            psi4_flag = False
            g16_flag = True
        else:
            software = kwargs.get("software")
            assert False, f"Unknown software: {software}. Please check the software name."

        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            trajectory_psi4 = []
            for i, atoms in enumerate(self.mol_info[key]["aseatoms_list"]):
                atoms_tmp = atoms.copy()
                atoms_tmp.pbc = False
                atoms_tmp.cell = None
                if self.mol_info[key]["netcharge"] is None:
                    self._assign_totalcharge()
                nc = self.mol_info[key]["netcharge"]

                if g16_flag == True:
                    calc_geoopt = Gaussian(label=f'{key}_{i}', charge=nc, **geoopt_params_g16)
                    calc_geoopt.directory = self.mol_info[key]["directory"]
                    atoms_tmp.calc = calc_geoopt
                elif psi4_flag == True:
                    psi4traj = os.path.join(self.mol_info[key]["directory"], f'{key}_{i}_opt.traj')
                    psi4log = os.path.join(self.mol_info[key]["directory"], f'{key}_{i}_opt.log')
                    calc_geoopt = Psi4(atoms = atoms_tmp,
                                       num_threads = "max", 
                                       memory = "16GB", 
                                       charge=nc, 
                                       **geoopt_params_psi4)
                    opt = BFGS(atoms_tmp, trajectory=psi4traj, logfile=psi4log)
                    trajectory_psi4.append(psi4traj)
                
                if do_calc:
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
                                  = read(trajectory_psi4[j],index=-1) # self.mol_info[key]["aseatoms_geoopt"][j]
                        print(f"Skip geometry optimization of {key}_{i}: RMSD = {rmsd_ji} A < {rmsd}")
                        rmsd_skip = False
                    else:
                        self.mol_info[key]["aseatoms_geoopt"][i] = atoms_tmp
                        print(f"Calculating geometry optimization of {key}_{i}")
                        if g16_flag == True:
                            _ = self.mol_info[key]["aseatoms_geoopt"][i].get_potential_energy()
                        elif psi4_flag == True:
                            opt.run(fmax=0.01)
                            self.mol_info[key]["aseatoms_geoopt"][i] = read(psi4traj, index=-1)
                        print(f"Finished geometry optimization of {key}_{i}")
                        
            minidx = np.array([a.get_potential_energy() for a \
                                in self.mol_info[key]["aseatoms_geoopt"]]).argmin()
            self.mol_info[key]["aseatoms_stable"] = self.mol_info[key]["aseatoms_geoopt"][minidx]

    def get_molecule_off(self, keys=None, **kwargs):
        if keys is None:
            keys = self.mol_info.keys()

        for key in keys:
            dirname = self.mol_info[key]["directory"]
            sdffile = os.path.join(dirname, f"{key}.sdf")
            if not os.path.exists(sdffile):
                self.get_sdf()
            
            molecule_off = Molecule.from_file(sdffile)
            if list(self.mol_info[key]["partial_charges"]) != []:
                molecule_off.partial_charges = Quantity(self.mol_info[key]["partial_charges"],
                                                        toolkit.unit.elementary_charge)

            molecule_off.name = key
            for meta_key in self.mol_info[key]["metadata"].keys():
                for meta_ind in self.mol_info[key]["metadata"][meta_key]:
                    molecule_off.atoms[meta_ind].metadata[meta_key] = True
            
            for attr_key, attr_value in kwargs.items():
                setattr(molecule_off, attr_key, attr_value)

            self.mol_info[key]["molecule_OFF"] = molecule_off

    def get_partial_charges(self, keys=None, params_ff=None, **kwargs):
        """
        Get partial charges by RESP or AM1-BCC
        charge_type: "resp" or "am1bcc"
        keys: list of keys to get partial charges
        """
        if keys is None:
            keys = self.mol_info.keys()

        charge_type = kwargs.get("type", "resp")
        software = kwargs.get("software", False)
        if software == "psi4":
            psi4_flag = True
        else:
            psi4_flag = False
        kwargs.pop("type", None)
        kwargs.pop("software", None)
        
        if charge_type not in ["resp", "am1bcc"]:
            assert False, "charge_type should be resp or am1bcc"

        for key in keys:
            if self.mol_info[key]["aseatoms_stable"] != None:
                atoms = self.mol_info[key]["aseatoms_stable"]
            else:
                atoms = self.mol_info[key]["aseatoms_list"][0]
            
            netcharge = self.mol_info[key]["netcharge"]
            label = key

            if psi4_flag == False:
                self.mol_info[key]["ChargeCalc"] = ChargeCalculator(
                    atoms,
                    charge_type,
                    netcharge,
                    label,
                    directory=self.mol_info[key]["directory"],
                    params=kwargs
                )
            elif psi4_flag == True:
                if self.mol_info[key]["molecule_OFF"] == None:
                    self.get_molecule_off(keys=[key])
                molecule = self.mol_info[key]["molecule_OFF"]
                molecule.conformers[0].magnitude[:] = atoms.positions
                self.mol_info[key]["ChargeCalc"] = Psi4ChargeCalculator(
                    molecule,
                    charge_type,
                    netcharge,
                    label,
                    directory=self.mol_info[key]["directory"],
                    params=kwargs
                )

            self.mol_info[key]["ChargeCalc"].get_partialcharges()
            self.mol_info[key]["partial_charges"] = self.mol_info[key]["ChargeCalc"].partial_charges
            self.get_molecule_off(keys=[key],
                                  **{"mol2file": self.mol_info[key]["ChargeCalc"].mol2file})
        
        if params_ff is not None:
            self._adjust_charges(params_ff)
    
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
            net_charge = self.mol_info[key]["netcharge"]
            charges = np.array([ np.float64(ee) for ee in self.mol_info[key]["partial_charges"]])

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

            if not np.isclose(charge_deficit, 0.0):
                print(f"Net charge of {key} is {net_charge} and total charge is {total_charge}")
                charges = charges - charge_deficit / len(charges)
                total_charge = np.sum(charges)
                charge_deficit = total_charge - net_charge
                charges[0] -= charge_deficit
            
            self.mol_info[key]["partial_charges"] = charges

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
        il_dict = {}
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

            il_dict["FSA_N"] = fsalike_Nindex
            il_dict["FSA_S"] = fsalike_Sindex
            il_dict["FSA_O"] = fsalike_Oindex
        return il_dict

