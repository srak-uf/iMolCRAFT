#!/usr/bin/env python
from dmff import Hamiltonian
import os
import numpy as np
import mdtraj as md
from openmm import app
import jax.numpy as jnp
import dmff
from dmff.operators.templatetype import TemplateATypeOperator

def merge_xml(ffxml_list, outxml):
    ff = Hamiltonian(*ffxml_list)
    del_idx = []
    attribfromres_flag = False
    # ffinfo_nb = ff.ffinfo["Forces"]["NonbondedForce"]["node"]
    for i, f in enumerate(ff.ffinfo["Forces"]["NonbondedForce"]["node"]):
        if "name" in f and "attrib" in f:
            if f["name"] == "UseAttributeFromResidue" and f["attrib"]["name"] == "charge" and attribfromres_flag == False:
                attribfromres_flag = True
            elif f["name"] == "UseAttributeFromResidue" and f["attrib"]["name"] == "charge" and attribfromres_flag == True:
                del_idx.append(i)

    n_del = 0
    for i in del_idx:
        ff.ffinfo["Forces"]["NonbondedForce"]["node"].pop(i-n_del)
        n_del += 1
    
    os.makedirs("xmlfiles", exist_ok=True)
    ff.renderXML(os.path.join("xmlfiles",outxml))

def neutralize(params, num_elems):
    net_q = jnp.dot(params['NonbondedForce']['charges'], num_elems)
    params['NonbondedForce']['charges'] = params['NonbondedForce']['charges'] - net_q / num_elems.sum()
    return params

def get_charges_types(topdata: app.Topology, ff, gen_dmfftop=False):
    template = TemplateATypeOperator(ff.ffinfo)
    topdata = dmff.api.DMFFTopology(from_top=topdata)
    topdata = template(topdata)
    charges = [a.meta["charge"] for a in topdata.atoms()]
    types = [a.meta['type'] for a in topdata.atoms()]
    if gen_dmfftop:
        return charges, types, topdata
    else:
        return charges, types

def update_ffinfo_from_rescharges(ff, rescharges):
    for i_res in range(len(ff.ffinfo['Residues'])):
        for key in rescharges[i_res]:
            for key2 in rescharges[i_res][key]:
                res_idx = rescharges[i_res][key][key2]["index"]
                for i in res_idx:
                    ff.ffinfo['Residues'][i_res]["particles"][i]["charge"] = rescharges[i_res][key][key2]["value"]
    return ff


def update_ffinfo_from_params(ff, params):
    idx = 0
    for i_res in range(len(ff.ffinfo['Residues'])):
        for i,_ in enumerate(ff.ffinfo['Residues'][i_res]["particles"]):
            ff.ffinfo['Residues'][i_res]["particles"][i]["charge"] = params["NonbondedForce"]["charges"][idx]
            idx += 1
    return ff

def get_chgparams_from_rescharges(params, rescharges):
    natoms = np.array([len(rescharges[i_res][key][key2]["index"]) for i_res in range(len(rescharges)) for key in rescharges[i_res] for key2 in rescharges[i_res][key]]).sum()
    params["NonbondedForce"]["charges"] = jnp.zeros(natoms)
    ishift = 0
    for res in rescharges:
        natoms = 0
        for t in res:
            for key in res[t]:
                for idx in res[t][key]["index"]:
                    params["NonbondedForce"]["charges"] = params["NonbondedForce"]["charges"].at[idx+ishift].set(res[t][key]["value"])
                    natoms += 1
        ishift += natoms
    return params

def update_rescharges_from_params(rescharges, params):
    ishift = 0
    for res in rescharges:
        natoms = 0
        for t in res:
            for key in res[t]:
                charge_forave = []
                for idx in res[t][key]["index"]:
                    natoms += 1
                    charge_forave.append(params["NonbondedForce"]["charges"][idx+ishift])
                res[t][key]["value"] = np.mean(charge_forave)
                # res[t][key]["value"] = params["NonbondedForce"]["charges"][idx+ishift]
        ishift += natoms
    return rescharges


def get_rescharges_from_residues(ff, ratio=None):
    residues = ff.ffinfo['Residues']
    rescharges = []
    
    if ratio is not None:
        assert len(residues) == len(ratio), "len(residues) != len(ratio)"
        num_elems = []

    for i_res in range(len(residues)):
        residue = residues[i_res]
        chargedict = {}
        for i, p in enumerate(residue["particles"]):
            t = p["type"]
            c = p["charge"]
            if ratio is not None:
                num_elems.append(ratio[i_res])

            if t not in chargedict:
                chargedict[t] = {}
                
            if len(chargedict[t]) == 0:
                chargedict[t][f"{t}_0"] = {"value": c, "index":[i]}
            elif len(chargedict[t]) > 0:
                sameTYPEflag = False
                for c_idx, key in enumerate(chargedict[t]):
                    if np.isclose(float(chargedict[t][key]["value"]), c, atol=1e-3):
                        chargedict[t][key]["index"].append(i)
                        sameTYPEflag = True
                if not sameTYPEflag:
                    chargedict[t][f"{t}_{c_idx+1}"] = {"value": c, "index": [i]}
        rescharges.append(chargedict)

    if ratio is not None:
        return rescharges, jnp.array(num_elems)
    else:
        return rescharges

