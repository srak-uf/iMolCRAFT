from imolcraft.trainer import dmff_utils
import os
import shutil
import pytest
import numpy as np
import jax.numpy as jnp


def test_parser_dmffyaml(tmp_path):
    # parser_dmffyaml writes *_parsed.yaml next to the input yaml, so copy the whole
    # input set into tmp_path first to keep tests/data/ clean
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for name in ("dmff.yml", "LiBH4.cif"):
        shutil.copy(os.path.join(data_dir, name), tmp_path / name)

    d = dmff_utils.parser_dmffyaml(str(tmp_path / "dmff.yml"))
    assert (tmp_path / "dmff_parsed.yaml").is_file()  # the output goes to tmp_path
    assert set(d.keys()) == set(['sampling', 'targets', 'validation'])
    assert set(d["targets"].keys()) == set([
        'density_gcm3', 'La_A', 'Lc_A', 'rdf', 'adf'
    ])
    assert set(d["sampling"].keys()) == set([
        'init_structure', 'ensemble', 'dt_fs', 'rcut_nm', 'temperature_K',
        'pressure_bar', 'anneal_T', 'anneal_steps', 'anneal_interval',
        'relax_steps', 'prod_steps', 'nstxout', 'neff',
        'dispcorr', 'nonbondedmethod'
    ])
    assert set(d["validation"].keys()) == set([
        'dself_Li', 'dself_B', 'rho', 'rdf_Li_O'
    ])
    assert d["validation"]["dself_B"]["fit_range"] == [0.1, 0.6]


def test_check_validation_accepts_the_targets_of_the_loss():
    """A property usable in loss is usable in validation too"""
    params = {
        "rho": {"property": "density_gcm3"},
        "a": {"property": "La_A"},
        "rdf_LiO": {"property": "rdf", "elem1": "Li", "elem2": "O",
                    "rcut12_A": 8.0, "gt": "rdf_Li_O.txt"},
    }
    assert dmff_utils._check_validation(params) is params


def test_check_validation_rejects_an_unknown_metric():
    with pytest.raises(ValueError):
        dmff_utils._check_validation({
            "rdf_LiO": {"property": "rdf", "elem1": "Li", "elem2": "O",
                        "rcut12_A": 8.0, "gt": "rdf.txt", "metric": "hoge"}
        })


def test_check_validation_requires_a_reference_for_a_distribution():
    """A distribution cannot be scored without a reference, so gt is required"""
    with pytest.raises(KeyError):
        dmff_utils._check_validation({
            "rdf_LiO": {"property": "rdf", "elem1": "Li", "elem2": "O",
                        "rcut12_A": 8.0}
        })


def test_parser_dmffyaml_rejects_dself_as_a_target(tmp_path):
    """dself_cm2s cannot be handled perturbatively, so it cannot go in targets"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for name in ("dmff.yml", "LiBH4.cif"):
        shutil.copy(os.path.join(data_dir, name), tmp_path / name)

    yml = tmp_path / "dmff.yml"
    text = yml.read_text().replace(
        "targets:\n", "targets:\n    dself_cm2s:\n        gt: 1.0e-6\n"
        "        weight: 1.0\n", 1
    )
    yml.write_text(text)

    with pytest.raises(ValueError, match="validation"):
        dmff_utils.parser_dmffyaml(str(yml))


def test_parser_dmffyaml_rejects_a_dispersion_correction_with_ljpme(tmp_path):
    """LJPME adds the long-range dispersion itself, so a correction double counts"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for name in ("dmff.yml", "LiBH4.cif"):
        shutil.copy(os.path.join(data_dir, name), tmp_path / name)

    yml = tmp_path / "dmff.yml"
    yml.write_text(yml.read_text().replace(
        "    neff: 30\n", "    neff: 30\n    nonbondedmethod: LJPME\n"
        "    dispcorr: True\n", 1
    ))

    with pytest.raises(ValueError, match="dispcorr"):
        dmff_utils.parser_dmffyaml(str(yml))


def test_parser_dmffyaml_allows_a_dispersion_correction_with_pme(tmp_path):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for name in ("dmff.yml", "LiBH4.cif"):
        shutil.copy(os.path.join(data_dir, name), tmp_path / name)

    yml = tmp_path / "dmff.yml"
    yml.write_text(yml.read_text().replace(
        "    neff: 30\n", "    neff: 30\n    dispcorr: True\n", 1
    ))

    assert dmff_utils.parser_dmffyaml(str(yml))["sampling"]["dispcorr"] is True


def test_validation_only_properties_are_absent_from_the_targets():
    for name in dmff_utils.VALIDATION_ONLY_PROPERTIES:
        assert name not in dmff_utils.REQUIRED_TARGET_KEYS


def test_check_validation_defaults_to_an_empty_section():
    """Even without a validation block the caller receives an empty dict"""
    assert dmff_utils._check_validation(None) == {}


def test_check_validation_rejects_an_unknown_property():
    with pytest.raises(ValueError):
        dmff_utils._check_validation({"x": {"property": "viscosity", "select": "all"}})


def test_check_validation_requires_the_property_key():
    with pytest.raises(KeyError):
        dmff_utils._check_validation({"x": {"select": "all"}})


def test_check_validation_requires_the_keys_of_the_property():
    # dself_cm2s requires select
    with pytest.raises(KeyError):
        dmff_utils._check_validation({"x": {"property": "dself_cm2s"}})


def test_get_validation_gt_keeps_only_the_blocks_with_a_reference():
    params = {
        "a": {"property": "dself_cm2s", "select": "all", "gt": 1.0e-6},
        "b": {"property": "dself_cm2s", "select": "element Li"},
        # A distribution gt is a filename and only the distance is recorded, so skip
        "c": {"property": "rdf", "elem1": "Li", "elem2": "O",
              "rcut12_A": 8.0, "gt": "rdf.txt", "metric": "wrightfactor"},
    }
    assert dmff_utils.get_validation_gt(params) == {"a": 1.0e-6}


def test_get_validation_pred_rejects_an_unknown_property():
    with pytest.raises(ValueError):
        dmff_utils.get_validation_pred(
            "nonexistent.xtc", "nonexistent.pdb", {"x": {"property": "viscosity"}}
        )


def test_check_validation_rejects_an_unknown_scalar_metric():
    with pytest.raises(ValueError):
        dmff_utils._check_validation(
            {"x": {"property": "density_gcm3", "gt": 1.0, "metric": "wrightfactor"}}
        )


def test_check_validation_rejects_a_scalar_metric_without_a_reference():
    """Giving only how to measure the deviation, with no target, is a mistake"""
    with pytest.raises(KeyError, match="gt"):
        dmff_utils._check_validation(
            {"x": {"property": "density_gcm3", "metric": "relerr"}}
        )


def test_score_scalar_is_none_without_a_reference():
    assert dmff_utils._score_scalar(1.5, {"property": "density_gcm3"}) is None


def test_score_scalar_is_the_signed_relative_error_by_default():
    block = {"property": "density_gcm3", "gt": 2.0}
    assert dmff_utils._score_scalar(1.8, block) == pytest.approx(-0.1)


def test_score_scalar_follows_the_metric_of_the_block():
    block = {"property": "density_gcm3", "gt": 2.0, "metric": "sqrelerr"}
    assert dmff_utils._score_scalar(1.8, block) == pytest.approx(0.01)
    block = {"property": "density_gcm3", "gt": 2.0, "metric": "diff"}
    assert dmff_utils._score_scalar(1.8, block) == pytest.approx(-0.2)


def test_score_distribution_rejects_a_reference_of_another_length(tmp_path):
    """A reference with a mismatched bin count fails instead of silently comparing"""
    gt_file = tmp_path / "rdf.txt"
    np.savetxt(gt_file, np.column_stack([np.arange(5.0), np.ones(5)]))
    block = {"gt": str(gt_file), "metric": "wrightfactor"}

    with pytest.raises(ValueError, match="bins"):
        dmff_utils._score_distribution(np.ones(4), block)


def test_score_distribution_is_zero_for_a_perfect_match(tmp_path):
    gt_file = tmp_path / "rdf.txt"
    gt = np.array([0.0, 1.0, 2.0, 1.0])
    np.savetxt(gt_file, np.column_stack([np.arange(4.0), gt]))
    block = {"gt": str(gt_file), "metric": "wrightfactor"}

    score, curve = dmff_utils._score_distribution(gt, block)
    assert np.isclose(score, 0.0)
    # The curve has three columns: x, pred, gt
    assert curve.shape == (4, 3)


def test_plot_validation_curves_does_nothing_without_curves(tmp_path):
    dmff_utils.plot_validation_curves({}, {}, label=str(tmp_path / "curves"))
    assert not (tmp_path / "curves.png").exists()


def test_plot_validation_does_nothing_without_history(tmp_path):
    # Plotting from an empty history does not crash
    dmff_utils.plot_validation([], label=str(tmp_path / "validation"))
    assert not (tmp_path / "validation.png").exists()


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
    ffparams = {"NonbondedForce": {"charge": jnp.array([1.0, -1.0])}}
    natoms_list = jnp.array([1, 1])
    result = dmff_utils.neutralize(ffparams, natoms_list, 0.0)
    assert np.isclose(jnp.dot(result["NonbondedForce"]["charge"], natoms_list), 0.0)
    result = dmff_utils.neutralize(ffparams, natoms_list, 0.1)
    assert np.isclose(jnp.dot(result["NonbondedForce"]["charge"], natoms_list), 0.1)

    ffparams = {"NonbondedForce": {"charge": jnp.array([1.0, -1.0, 0.5, -0.5, 0.0])}}
    natoms_list = jnp.array([1, 1, 2, 2, 1])
    nc = 0.4
    target_lists = [[0, 1], [2, 3]]
    target_charges = [0.2, 0.2]
    result = dmff_utils.neutralize(
        ffparams, natoms_list, nc, target_lists, target_charges
    )
    assert np.isclose(
        result["NonbondedForce"]["charge"][0] + result["NonbondedForce"]["charge"][1],
        0.2
    )
    assert np.isclose(
        result["NonbondedForce"]["charge"][2] +
        result["NonbondedForce"]["charge"][3],
        0.1
    )
    assert np.isclose(jnp.dot(result["NonbondedForce"]["charge"], natoms_list), 0.4)


def test_update_ffinfo_from_params_and_rescharges():
    class DummyFF:
        def __init__(self):
            self.ffinfo = {
                "Residues": [
                    {"particles": [{"charge": 0.0}, {"charge": 0.0}], "vsites": []}
                ]
            }
    ff = DummyFF()
    params = {"NonbondedForce": {"charge": jnp.array([1.0, 2.0])}}
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
    assert np.allclose(params2["VirtualSite"]["weight"], jnp.array([1.0, 2.0]))
