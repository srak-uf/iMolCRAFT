from imolcraft.trainer import dmff_utils
import os
import pytest
import numpy as np
import jax.numpy as jnp


def test_parser_dmffyaml():
    d = dmff_utils.parser_dmffyaml(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "dmff.yml"
        )
    )
    assert set(d.keys()) == set(['sampling', 'targets'])
    assert set(d["targets"].keys()) == set([
        'density_gcm3', 'La_A', 'Lc_A', 'rdf', 'adf'
    ])
    assert set(d["sampling"].keys()) == set([
        'init_structure', 'ensemble', 'dt_fs', 'rcut_nm', 'temperature_K',
        'pressure_bar', 'anneal_steps', 'anneal_Tmax', 'anneal_totalsteps',
        'relax_steps', 'prod_steps', 'nstxout', 'neff', 'anneal_totaltime'
    ])


def test_get_target_gt():
    params = {
        "density_gcm3": {"gt": 1.0, "weight": 2.0},
        "La_A": {"gt": 2.0, "weight": 3.0},
        "Lb_A": {"gt": 3.0, "weight": 4.0},
        "Lc_A": {"gt": 4.0, "weight": 5.0},
    }
    gt = dmff_utils.get_target_gt(params)
    assert gt["density_gcm3"]["gt"] == 1.0
    assert gt["La_A"]["weight"] == 3.0


def test_neutralize():
    ffparams = {"NonbondedForce": {"charges": jnp.array([1.0, -1.0])}}
    natoms_list = jnp.array([1, 1])
    result = dmff_utils.neutralize(ffparams, natoms_list, 0.0)
    assert np.isclose(jnp.dot(result["NonbondedForce"]["charges"], natoms_list), 0.0)
    result = dmff_utils.neutralize(ffparams, natoms_list, 0.1)
    assert np.isclose(jnp.dot(result["NonbondedForce"]["charges"], natoms_list), 0.1)

    ffparams = {"NonbondedForce": {"charges": jnp.array([1.0, -1.0, 0.5, -0.5, 0.0])}}
    natoms_list = jnp.array([1, 1, 2, 2, 1])
    nc = 0.4
    target_lists = [[0, 1], [2, 3]]
    target_charges = [0.2, 0.2]
    result = dmff_utils.neutralize(
        ffparams, natoms_list, nc, target_lists, target_charges
    )
    assert np.isclose(
        result["NonbondedForce"]["charges"][0] + result["NonbondedForce"]["charges"][1],
        0.2
    )
    assert np.isclose(
        result["NonbondedForce"]["charges"][2] +
        result["NonbondedForce"]["charges"][3],
        0.1
    )
    assert np.isclose(jnp.dot(result["NonbondedForce"]["charges"], natoms_list), 0.4)


def test_update_ffinfo_from_params_and_rescharges():
    class DummyFF:
        def __init__(self):
            self.ffinfo = {
                "Residues": [
                    {"particles": [{"charge": 0.0}, {"charge": 0.0}], "vsites": []}
                ]
            }
    ff = DummyFF()
    params = {"NonbondedForce": {"charges": jnp.array([1.0, 2.0])}}
    ff2 = dmff_utils.update_ffinfo_from_params(ff, params)
    assert ff2.ffinfo["Residues"][0]["particles"][0]["charge"] == 1.0
    assert ff2.ffinfo["Residues"][0]["particles"][1]["charge"] == 2.0


def test_vsiteinfo_to_params():
    class DummyFF:
        def __init__(self):
            self.ffinfo = {
                "Residues": [
                    {"vsites": [{"weight1": 1.0, "weight2": 2.0}]}
                ]
            }
    ff = DummyFF()
    params = {}
    params2 = dmff_utils.vsiteinfo_to_params(ff, params)
    assert np.allclose(params2["VsiteForce"]["weight"], jnp.array([1.0, 2.0]))
