from imolcryff.crafter.gaffil_generators import GAFFilTemplateGenerator
from imolcryff.crafter.ffxml import gafftemplate2xml
from imolcryff.io.rdkit import _il_assign
from openff.toolkit.topology import Molecule
from rdkit import Chem
import os
import pytest


@pytest.mark.parametrize(
    "smiles, resp_mol2",
    [
        (
            "[O]=[S](=[O])([F])[N-][S](=[O])(=[O])[F]",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "fsa_resp.mol2"
            )
        ),
        (
            "[F][P-](F)(F)(F)(F)F",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "pf6_resp.mol2"
            )
        )
    ])
def test_GAFF2assignment(smiles, resp_mol2):
    molecule = Molecule.from_smiles(smiles)
    rdkitmol = Chem.MolFromSmiles(smiles)
    metadata = _il_assign(rdkitmol, -1)
    for meta_key in metadata.keys():
        for meta_ind in metadata[meta_key]:
            molecule.atoms[meta_ind].metadata[meta_key] = True
    molecule.mol2file = resp_mol2
    if not os.path.isfile(molecule.mol2file):
        raise FileNotFoundError(f"{molecule.mol2file} not found!")
    gen = GAFFilTemplateGenerator([molecule],
                                  forcefield="gaff-2.11")
    _ = gafftemplate2xml([molecule], gen)
