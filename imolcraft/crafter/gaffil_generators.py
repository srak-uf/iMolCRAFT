import shutil
import subprocess
from inspect import signature
from io import StringIO
from typing import Any, Dict, List, Optional

import parmed
from lxml import etree
from openff.toolkit.topology import Molecule
from openff.units import unit
from openmmforcefields.generators import GAFFTemplateGenerator

from ..io.mol2 import read_mol2, write_mol2

#: Default GAFF atom types assigned to the FSA anion.
DEFAULT_IL_ASSIGN = {"FSA": {"S": "s6", "N": "n", "O": "o", "F": "f"}}

#: (metadata flag, il_assign element key) pairs checked in ``run_antech``.
#: The order matters: the first matching flag wins, as in the original
#: if/elif chain.
_FSA_METADATA_KEYS = (("FSA_S", "S"), ("FSA_N", "N"), ("FSA_O", "O"))


class GAFFilTemplateGenerator(GAFFTemplateGenerator):
    INSTALLED_FORCEFIELDS = [
        "gaff-1.4",
        "gaff-1.8",
        "gaff-1.81",
        "gaff-2.1",
        "gaff-2.11",
    ]

    def __init__(
        self,
        molecules: List[Molecule],
        il_assign: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs
    ) -> None:
        super().__init__(molecules, **kwargs)
        # ``super().__init__`` stores deep copies of the molecules, so the
        # ``mol2file`` attribute attached by the caller has to be propagated
        # to the stored copies by matching them on SMILES.  The cache maps a
        # molecular formula to a list of ``(Molecule, matching_template)``
        # pairs, so both levels have to be unwrapped here.
        for molecule in molecules:
            for stored_entries in self._molecules.values():
                for stored, _matching_template in stored_entries:
                    if stored.to_smiles() == molecule.to_smiles():
                        stored.mol2file = molecule.mol2file

        self.il_assign = DEFAULT_IL_ASSIGN if il_assign is None else il_assign

    def generate_residue_template(
        self, molecule: Any, residue_atoms: Optional[List[Any]] = None
    ) -> str:
        self._generate_unique_atom_names(molecule)
        smiles = molecule.to_smiles()
        mol2file = self.run_antech(molecule)
        mol2_dict = read_mol2(mol2file)

        for index, atominfo in enumerate(mol2_dict["@<TRIPOS>ATOM"]):
            molecule.atoms[index].gaff_type = atominfo[5]

        frcmod_filename = self.run_parmchk(mol2file)
        params = self._load_openmm_parameters(frcmod_filename)

        ffxml = StringIO()
        params.write(ffxml)
        root = etree.fromstring(ffxml.getvalue())

        self._append_residue(root, molecule, smiles, residue_atoms)

        return etree.tostring(root, pretty_print=True, encoding="unicode")

    @staticmethod
    def _load_openmm_parameters(frcmod_filename: str) -> Any:
        """Load an frcmod file through parmed and convert it to OpenMM form."""
        leaprc = StringIO(f"parm = loadamberparams {frcmod_filename}")
        params = parmed.amber.AmberParameterSet.from_leaprc(leaprc)

        kwargs = {}
        # use introspection to support multiple parmed versions
        from_parameterset = parmed.openmm.OpenMMParameterSet.from_parameterset
        if "remediate_residues" in signature(from_parameterset).parameters:
            kwargs["remediate_residues"] = False
        return from_parameterset(params, **kwargs)

    @staticmethod
    def _append_residue(
        root: Any,
        molecule: Any,
        smiles: str,
        residue_atoms: Optional[List[Any]],
    ) -> None:
        """Append a ``<Residues>`` block describing ``molecule`` to ``root``."""
        residues = etree.SubElement(root, "Residues")
        residue = etree.SubElement(residues, "Residue", name=smiles)
        for atom in molecule.atoms:
            etree.SubElement(
                residue,
                "Atom",
                name=atom.name,
                type=atom.gaff_type,
                charge=str(atom.partial_charge.m_as(unit.elementary_charge)),
            )

        # If residue_atoms == None, add all atoms to the residues
        if not residue_atoms:
            residue_atoms = list(molecule.atoms)

        for bond in molecule.bonds:
            in1 = bond.atom1 in residue_atoms
            in2 = bond.atom2 in residue_atoms
            if in1 and in2:
                etree.SubElement(
                    residue,
                    "Bond",
                    atomName1=bond.atom1.name,
                    atomName2=bond.atom2.name,
                )
            elif in1:
                etree.SubElement(residue, "ExternalBond", atomName=bond.atom1.name)
            elif in2:
                etree.SubElement(residue, "ExternalBond", atomName=bond.atom2.name)

    def run_antech(self, molecule: Any) -> str:
        mol2file = molecule.mol2file
        if len(molecule.atoms) == 1:
            return f"{mol2file}"

        cmd = (
            f"antechamber -i {mol2file} -fi mol2 -o {mol2file}.gaff "
            f"-fo mol2 -at {self._gaff_major_version} -c dc -dr no"
        )
        _ = subprocess.getoutput(cmd)

        gaffmol2 = read_mol2(f"{mol2file}.gaff")
        chgmol2 = read_mol2(f"{mol2file}")
        gaff_atoms = gaffmol2["@<TRIPOS>ATOM"]

        # antechamber recomputes the charges, so restore the original ones
        for i, atominfo in enumerate(chgmol2["@<TRIPOS>ATOM"]):
            gaff_atoms[i][8] = atominfo[8]

        # Modify atom types
        fsa_types = self.il_assign["FSA"]
        for i, atom in enumerate(molecule.atoms):
            for meta_key, elem_key in _FSA_METADATA_KEYS:
                if atom.metadata.get(meta_key) is True:
                    gaff_atoms[i][5] = fsa_types[elem_key]
                    break

        # Modify bond types
        for i, bond in enumerate(molecule.bonds):
            gaffmol2["@<TRIPOS>BOND"][i] = [
                i + 1,
                bond.atom1_index + 1,
                bond.atom2_index + 1,
                bond.bond_order,
            ]

        write_mol2(f"{mol2file}.gaff", gaffmol2)
        return f"{mol2file}.gaff"

    def run_parmchk(self, mol2file: str) -> str:
        frcmod_filename = "molecule.frcmod"
        shutil.copy(self.gaff_dat_filename, "gaff.dat")
        cmd = (
            f"parmchk2 -i {mol2file} -f mol2 -p gaff.dat -o {frcmod_filename} "
            f"-s {self._gaff_major_version} -a Y"
        )
        _ = subprocess.getoutput(cmd)
        return frcmod_filename
