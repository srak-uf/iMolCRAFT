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
                