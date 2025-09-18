from openff.toolkit.topology import Molecule
from ase import Atoms


def read_mol2(filename):
    """
    Read a mol2 file and return a dictionary with the contents.
    The dictionary will contain the keys "@<TRIPOS>MOLECULE", "@<TRIPOS>ATOM",
    "@<TRIPOS>BOND", and "@<TRIPOS>SUBSTRUCTURE".
    Each key will contain a list of lists, where each inner list represents a line
    in the corresponding section of the mol2 file.
    The first element of each inner list is the line number, and the rest are the
    corresponding values.

    Parameters
    ----------
    filename : str
        The name of the mol2 file.

    Returns
    -------
    dict
        A dictionary containing the contents of the mol2 file.
    """
    with open(filename) as f:
        lines = f.readlines()
        mol2_dict = {}
        for i in range(len(lines)):
            if lines[i].startswith("@<TRIPOS>"):
                key_name = lines[i].strip()
                mol2_dict[key_name] = []
            elif lines[i].strip() != "":
                mol2_dict[key_name].append(lines[i].strip().split())
    return mol2_dict


def write_mol2(filename, input):
    """
    Write a mol2 file from a dictionary or a molecule object.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    input : dict or openff.toolkit.topology.Molecule
        The input data to write. If a dictionary, it should contain the keys
        "@<TRIPOS>MOLECULE", "@<TRIPOS>ATOM", "@<TRIPOS>BOND", and
        "@<TRIPOS>SUBSTRUCTURE".
        If a molecule object, it will be written in the mol2 format.
    """
    if isinstance(input, dict):
        dict_flag = True
        offmol_flag = False
    elif isinstance(input, Molecule):
        offmol_flag = True
        dict_flag = False
    else:
        raise ValueError("Input must be a dictionary or a molecule object.")

    with open(filename, mode="w") as f:
        if dict_flag:
            # Handle mol2_dict writing
            for key in input:
                f.write(key + "\n")
                for i in range(len(input[key])):
                    outline = [str(x) for x in input[key][i]]
                    f.write(" ".join(outline) + "\n")
        elif offmol_flag:
            # Handle molecule writing
            f.write("@<TRIPOS>MOLECULE\n")
            f.write("MOL\n")
            f.write(f"{len(input.atoms)} {len(input.bonds)} 0 0 0\n")
            f.write("SMALL\n")
            f.write("CHARGES\n")
            f.write("\n")
            f.write("\n")
            f.write("@<TRIPOS>ATOM\n")
            for i, atom in enumerate(input.atoms):
                xx = input.conformers[0].magnitude[i, 0]
                yy = input.conformers[0].magnitude[i, 1]
                zz = input.conformers[0].magnitude[i, 2]
                charge = input.partial_charges[i].magnitude
                f.write(
                    f"{i+1} {atom.symbol} {xx} {yy} {zz} {atom.symbol}  1  "
                    f"MOL {charge}\n"
                )
            f.write("@<TRIPOS>BOND\n")
            for i, bond in enumerate(input.bonds):
                f.write(
                    f"{i+1} {bond.atom1_index+1} "
                    f"{bond.atom2_index+1} {bond.bond_order}\n"
                )
            f.write("@<TRIPOS>SUBSTRUCTURE\n")
            f.write("1 MOL 1 TEMP              0 ****  ****    0 ROOT\n")
        else:
            raise ValueError("Either mol2_dict or molecule must be provided.")


def mol2_to_aseatoms(mol2file):
    mol2_dict = read_mol2(mol2file)
    symbols = [atom[1] for atom in mol2_dict["@<TRIPOS>ATOM"]]
    positions = [list(map(float, atom[2:5])) for atom in mol2_dict["@<TRIPOS>ATOM"]]
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms
