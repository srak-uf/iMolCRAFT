def read_mol2(filename):
    with open(filename) as f:
        l = f.readlines()
        mol2_dict = {}
        for i in range(len(l)):
            if l[i].startswith("@<TRIPOS>"):
                key_name = l[i].strip()
                mol2_dict[key_name] = []
            elif l[i].strip() != "":
                mol2_dict[key_name].append(l[i].strip().split())
    return mol2_dict

def write_mol2(filename, mol2_dict):
    with open(filename, mode="w") as f:
        for key in mol2_dict:
            f.write(key + "\n")
            for i in range(len(mol2_dict[key])):
                outline = [ str(x) for x in mol2_dict[key][i] ]
                f.write(" ".join(outline) + "\n")

def write_mol2_off(filename, molecule):
    with open(filename, mode="w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{molecule.name}\n")
        f.write(f"{len(molecule.atoms)} {len(molecule.bonds)} 0 0 0\n")
        f.write("SMALL\n")
        f.write("CHARGES\n")
        f.write("\n")
        f.write("\n")
        f.write("@<TRIPOS>ATOM\n")
        for i, atom in enumerate(molecule.atoms):
            xx = molecule.conformers[0].magnitude[i, 0]
            yy = molecule.conformers[0].magnitude[i, 1]
            zz = molecule.conformers[0].magnitude[i, 2]
            charge = molecule.partial_charges[i].magnitude
            f.write(f"{i+1} {atom.symbol} {xx} {yy} {zz} {atom.symbol}  1  {molecule.name} {charge}\n")
        f.write("@<TRIPOS>BOND\n")
        for i, bond in enumerate(molecule.bonds):
            f.write(f"{i+1} {bond.atom1_index+1} {bond.atom2_index+1} {bond.bond_order}\n")
        f.write("@<TRIPOS>SUBSTRUCTURE\n")
        f.write(f"1 {molecule.name} 1 TEMP              0 ****  ****    0 ROOT\n")
