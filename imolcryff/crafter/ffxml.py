import imolcryff
from dmff import Hamiltonian
import openmmforcefields
from openff.toolkit import Molecule
import shutil
import os
import numpy as np

def get_element_fromtype(ptype, ff):
    for at in ff.ffinfo['AtomTypes']:
        if at["class"] == ptype:
            return at["element"]

def gafftemplate2xml(mmm, fftemplate_gen, ion_ffxml=None):
    """
    Generate GAFF XML files for a list of molecules.
    Parameters
    ----------
    mmm : list
        List of molecules to generate XML files.
    fftemplate_gen : openmmforcefields.GAFFTemplateGenerator
        GAFFTemplateGenerator object.
    ion_ffxml : str, optional
        Path to the ion XML file. If None, it will use the default ion XML file.
    
    Returns
    -------
    list
        List of paths to the generated XML files.
    """
    def write_PF6(molecule_off, ffxml):
        def molecule2aseatoms(molecule):
            from ase import Atoms
            from ase.data import chemical_symbols
            positions = molecule.conformers[0].magnitude
            symbols = [chemical_symbols[atom.atomic_number] for atom in molecule.atoms]
            atoms = Atoms(symbols=symbols, positions=positions)
            return atoms
        
        def get_element_angles(atoms, elem1, elem2, elem3, rcut=2.0):
            """
            Get the angles of a given element in the molecule.
            elem1-elem2-elem3
            """
            elem1_indices = [i for i, atom in enumerate(atoms) if atom.symbol == elem1]
            elem2_indices = [i for i, atom in enumerate(atoms) if atom.symbol == elem2]
            elem3_indices = [i for i, atom in enumerate(atoms) if atom.symbol == elem3]

            angles = []
            angles_idx = []
            for i in elem1_indices:
                for j in elem2_indices:
                    for k in elem3_indices:
                        if i != j and j != k and i < k:
                            # Calculate the distance between the atoms
                            d1 = atoms.get_distance(i, j)
                            d2 = atoms.get_distance(j, k)
                            if d1 < rcut and d2 < rcut:
                                angles.append(atoms.get_angle(i, j, k, mic=True))
                                angles_idx.append((i, j, k))
            return np.array(angles), np.array(angles_idx)
        
        path = imolcryff.__path__[0]
        pf6xml = os.path.join(path, "..", "data/PF6_gaff.xml")
        ff = Hamiltonian(pf6xml)
        atoms = molecule2aseatoms(molecule_off)
        angles, angles_idx = get_element_angles(atoms, "F", "P", "F")
        angles_180_idx = np.where(np.abs(angles - 180) < 1e-2)[0]
        P_idx = [ i for i, atom in enumerate(molecule_off.atoms) \
                                            if atom.atomic_number == 15 ]
        F_idx = [ i for i, atom in enumerate(molecule_off.atoms) \
                                            if atom.atomic_number ==  9 ]
        # P charge
        ff.ffinfo["Residues"][0]["particles"][0]["charge"] = molecule_off.partial_charges[P_idx[0]].magnitude
        # F charge
        for i, charge in enumerate(molecule_off.partial_charges[F_idx].magnitude):
            ff.ffinfo["Residues"][0]["particles"][i+1]["charge"] = charge

        angles_180 = angles_idx[angles_180_idx]
        for i, ag in enumerate(angles_180):
            ii = F_idx.index(ag[0])
            kk = F_idx.index(ag[2])
            ff.generators["HarmonicAngleForce"].angle_keys[i] = \
                                (f"f{ii+1}_pf6",
                                ff.generators["HarmonicAngleForce"].angle_keys[i][1],
                                f"f{kk+1}_pf6")
                                
        angles_90 = np.delete(angles_idx, angles_180_idx, axis=0)
        for i, ag in enumerate(angles_90,3):
            ii = F_idx.index(ag[0])
            kk = F_idx.index(ag[2])
            ff.generators["HarmonicAngleForce"].angle_keys[i] = \
                                (f"f{ii+1}_pf6",
                                 ff.generators["HarmonicAngleForce"].angle_keys[i][1],
                                 f"f{kk+1}_pf6")
        ff.renderXML(ffxml)

    ffxmlfiles = []
    for i in range(len(mmm)):
        if mmm[i] == Molecule.from_smiles("F[P-](F)(F)(F)(F)F"):
            write_PF6(mmm[i], f"gaffxml_{i}.xml")
        elif mmm[i].n_atoms > 1 or (mmm[i].total_charge == 0 and mmm[i].n_atoms == 1):
            ffxml = fftemplate_gen.generate_residue_template(mmm[i])
            with open(f"gaffxml_{i}.xml", "w") as f:
                f.write(ffxml)
        else: # single-atom ion
            ion_ffxml = os.path.join(os.path.dirname(openmmforcefields.__file__),
                                    "ffxml",
                                     ion_ffxml)
            ion_ffxml_base = os.path.basename(ion_ffxml)
            shutil.copy(ion_ffxml,  f"./gaffxml_{i}.xml")
            ff = Hamiltonian(f"./gaffxml_{i}.xml") 
            symbol = mmm[i].atoms[0].symbol
            for res in ff.ffinfo["Residues"]:
                if len(res["particles"]) == 1:
                    ptype = res["particles"][0]["type"]
                    elem = get_element_fromtype(ptype, ff)
                    if elem == symbol:
                        res["particles"][0]["charge"] = float(mmm[i].partial_charges[0].magnitude)
                        target_ptype = ptype
                        target_elem  = elem
            
            for ii, at in enumerate(ff.ffinfo["AtomTypes"]):
                if at["class"] == target_ptype:
                    ff.ffinfo["AtomTypes"] = [ff.ffinfo["AtomTypes"][ii]]
            for ii, res in enumerate(ff.ffinfo["Residues"]):
                if res['particles'][0]["type"] == target_ptype:
                    ff.ffinfo["Residues"]  = [ff.ffinfo["Residues"][ii]]
            for ii, nb in enumerate(ff.ffinfo["Forces"]['NonbondedForce']["node"]):
                if nb["name"] == "UseAttributeFromResidue":
                    atrib_def = nb
                    break
            for ii, nb in enumerate(ff.ffinfo["Forces"]['NonbondedForce']["node"]):
                if nb["name"] == "Atom" and nb["attrib"]["type"] == target_ptype:
                    target_param = ff.ffinfo["Forces"]['NonbondedForce']["node"][ii]
                    ff.ffinfo["Forces"]['NonbondedForce']["node"] = [atrib_def]
                    ff.ffinfo["Forces"]['NonbondedForce']["node"].append(target_param)
            
            ff.generators["NonbondedForce"].atom_keys = [target_ptype]
            ff.renderXML(f"./gaffxml_{i}.xml")
        
        ffxmlfiles.append(f"./gaffxml_{i}.xml")
    
    return ffxmlfiles
