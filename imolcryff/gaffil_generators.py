from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule
from collections import OrderedDict
import subprocess
import os
from .molinfo import read_mol2, write_mol2


class GAFFilTemplateGenerator(GAFFTemplateGenerator):
    INSTALLED_FORCEFIELDS = [
        "gaff-1.4",
        "gaff-1.8",
        "gaff-1.81",
        "gaff-2.1",
        "gaff-2.11",
    ]

    def __init__(self, molecules, il_assign=None, forcefield_files=None, cache=None, **kwargs):
        super().__init__(molecules, forcefield_files=None, cache=None, **kwargs)
        for molecule in molecules:
            for mm in self._molecules.items():
                if mm[1].to_smiles() == molecule.to_smiles():
                    mm[1].mol2file = molecule.mol2file

        if il_assign == None:
            self.il_assign =  {"FSA": {"S": "s6", "N": "n2", "O": "o", "F": "f"}}
        else:
            self.il_assign = il_assign

    def generate_residue_template(self, molecule, residue_atoms=None):
        import numpy as np
        from openff.units import unit, Quantity

        self._generate_unique_atom_names(molecule)
        smiles = molecule.to_smiles()
        mol2file = self.run_antech(molecule)
        mol2_dict = read_mol2(mol2file) 
        for index, atominfo in enumerate(mol2_dict["@<TRIPOS>ATOM"]):
            molecule.atoms[index].gaff_type = atominfo[5]
        
        frcmod_filename = self.run_parmchk(mol2file)
        from io import StringIO
        from inspect import (  # use introspection to support multiple parmed versions
            signature,
        )
        leaprc = StringIO(f"parm = loadamberparams {frcmod_filename}")

        import parmed
        params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)
        kwargs = {}
        if "remediate_residues" in signature(parmed.openmm.OpenMMParameterSet.from_parameterset).parameters:
            kwargs["remediate_residues"] = False
        params = parmed.openmm.OpenMMParameterSet.from_parameterset(params, **kwargs)
        ffxml = StringIO()
        kwargs = {}

        for atom_type in params.atom_types.copy().keys():
            if atom_type not in self._gaff_atom_types_observed:
                self._gaff_atom_types_observed.add(atom_type)
            # if we have seen the atom type, delete it from the OG params,
            # not the copy!
            else:
                del params.atom_types[atom_type]

        params.write(ffxml, **kwargs)
        ffxml_contents = ffxml.getvalue()

        # Create the residue template
        from lxml import etree
        root = etree.fromstring(ffxml_contents)
        # Create residue definitions
        residues = etree.SubElement(root, "Residues")
        residue = etree.SubElement(residues, "Residue", name=smiles)
        for atom in molecule.atoms:
            charge_string = str(atom.partial_charge.m_as(unit.elementary_charge))
            atom = etree.SubElement(
                residue,
                "Atom",
                name=atom.name,
                type=atom.gaff_type,
                charge=charge_string,
            )

        # If residue_atoms == None, add all atoms to the residues
        if not residue_atoms:
            residue_atoms = [atom for atom in molecule.atoms]
        for bond in molecule.bonds:
            if (bond.atom1 in residue_atoms) and (bond.atom2 in residue_atoms):
                bond = etree.SubElement(
                    residue,
                    "Bond",
                    atomName1=bond.atom1.name,
                    atomName2=bond.atom2.name,
                )
            elif (bond.atom1 in residue_atoms) and (bond.atom2 not in residue_atoms):
                bond = etree.SubElement(residue, "ExternalBond", atomName=bond.atom1.name)
            elif (bond.atom1 not in residue_atoms) and (bond.atom2 in residue_atoms):
                bond = etree.SubElement(residue, "ExternalBond", atomName=bond.atom2.name)
        # Render XML into string and append to parameters
        ffxml_contents = etree.tostring(root, pretty_print=True, encoding="unicode")

        return ffxml_contents

    def run_antech(self, molecule):
        gaff_ver = self._gaff_major_version
        if len(molecule.atoms) == 1:
            mol2file = molecule.mol2file
            return f"{mol2file}"
        else:
            mol2file = molecule.mol2file
            cmd = f"antechamber -i {mol2file} -fi mol2 -o {mol2file}.gaff -fo mol2 -at {gaff_ver} -c dc -dr no"
            output = subprocess.getoutput(cmd)
            gaffmol2 = read_mol2(f"{mol2file}.gaff")
            chgmol2 = read_mol2(f"{mol2file}")
            # charges 
            for i in range(len(gaffmol2["@<TRIPOS>ATOM"])):
                gaffmol2["@<TRIPOS>ATOM"][i][8] = chgmol2["@<TRIPOS>ATOM"][i][8]

            # Modify atom types
            for i, atom in enumerate(molecule.atoms):
                if "FSA_S" in atom.metadata and atom.metadata["FSA_S"] == True:
                    gaffmol2["@<TRIPOS>ATOM"][i][5] = self.il_assign["FSA"]["S"]
                elif "FSA_N" in atom.metadata and atom.metadata["FSA_N"] == True:
                    gaffmol2["@<TRIPOS>ATOM"][i][5] = self.il_assign["FSA"]["N"]
                elif "FSA_O" in atom.metadata and atom.metadata["FSA_O"] == True:
                    gaffmol2["@<TRIPOS>ATOM"][i][5] = self.il_assign["FSA"]["O"]

            # Modify bond types
            for i, bond in enumerate(molecule.bonds):
                atom1_idx = bond.atom1_index + 1
                atom2_idx = bond.atom2_index + 1
                gaffmol2["@<TRIPOS>BOND"][i] = [i+1, atom1_idx, atom2_idx, bond.bond_order]

            write_mol2(f"{mol2file}.gaff", gaffmol2)
            return f"{mol2file}.gaff"

    def run_parmchk(self, mol2file):
        import shutil
        frcmod_filename = "molecule.frcmod"
        shutil.copy(self.gaff_dat_filename, "gaff.dat")
        cmd = f"parmchk2 -i {mol2file} -f mol2 -p gaff.dat -o {frcmod_filename} -s {self._gaff_major_version} -a Y"
        output = subprocess.getoutput(cmd)
        return frcmod_filename


def mod_il(mol2file):
    pass
