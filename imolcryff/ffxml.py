from dmff import Hamiltonian
import openmmforcefields
import shutil
import os

def getelement_fromtype(ptype, ff):
    for at in ff.ffinfo['AtomTypes']:
        if at["class"] == ptype:
            return at["element"]

def gafftemplate2xml(mmm, gaff, ion_ffxml=None):
    for i in range(len(mmm)):
        if mmm[i].n_atoms > 1 or (mmm[i].total_charge == 0 and mmm[i].n_atoms == 1):
            ffxml = gaff.generate_residue_template(mmm[i])
            with open(f"gaffxml_{i}.xml", "w") as f:
                f.write(ffxml)
        else: # single-atom ion
            ion_ffxml = os.path.join(os.path.dirname(openmmforcefields.__file__), \
                                    "ffxml", \
                                     ion_ffxml)
            ion_ffxml_base = os.path.basename(ion_ffxml)
            shutil.copy(ion_ffxml,  f"./gaffxml_{i}.xml")
            ff = Hamiltonian(f"./gaffxml_{i}.xml") 
            symbol = mmm[i].atoms[0].symbol
            for res in ff.ffinfo["Residues"]:
                if len(res["particles"]) == 1:
                    ptype = res["particles"][0]["type"]
                    elem = getelement_fromtype(ptype, ff)
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
