import os

import numpy as np
import pytest

from imolcraft.analyzer.analyzer import (
    _join_on_center,
    _zero_first_bin,
    calc_cellpar_frame,
    calc_density,
    calc_density_frame,
    calc_rdf,
    calc_rdf_frame,
)

DATA = os.path.join(os.path.dirname(__file__), "..")
PDB = os.path.join(DATA, "vs_supercell_bonds.pdb")
XTC = os.path.join(DATA, "xtcfiles", "sample_0.xtc")


@pytest.fixture
def universe():
    MDAnalysis = pytest.importorskip("MDAnalysis")
    return MDAnalysis.Universe(PDB, XTC)


@pytest.mark.parametrize("first", [0.0, 0.3, 1.0, 5.0, 477.46, np.nan])
def test_zero_first_bin_always_zeroes(first):
    """r = 0 の RDF は値によらず常に 0 になる"""
    g = np.array([first, 1.0, 2.0])
    out = _zero_first_bin(g)
    assert out[0] == 0.0
    assert out is g  # in-place
    assert np.array_equal(out[1:], [1.0, 2.0])  # 他のビンは触らない


def test_calc_rdf_first_bin_is_zero(universe):
    r, g = calc_rdf(universe, "C", "H", rmax=6.0, dr=0.05)
    assert r.shape == g.shape
    assert g[0] == 0.0


def test_calc_rdf_frame_returns_independent_rows(universe):
    """フレームごとの RDF が同じ配列を共有していない"""
    frames = calc_rdf_frame(universe, "C", "H", rmax=6.0, dr=0.05)
    assert frames.shape[0] == len(universe.trajectory)
    assert np.all(frames[:, 0] == 0.0)
    assert not all(np.array_equal(frames[0], f) for f in frames[1:])


def test_join_on_center_matches_shared_atom():
    pairs_12 = np.array([[0, 10], [1, 11], [2, 10]])
    pairs_23 = np.array([[10, 20], [11, 21], [10, 22]])
    assert _join_on_center(pairs_12, pairs_23) == [
        [0, 10, 20],
        [0, 10, 22],
        [1, 11, 21],
        [2, 10, 20],
        [2, 10, 22],
    ]


def test_join_on_center_without_match():
    assert _join_on_center(np.array([[0, 1]]), np.array([[2, 3]])) == []


def test_calc_density(universe):
    frames = calc_density_frame(universe)
    assert frames.shape == (len(universe.trajectory),)
    assert np.all(frames > 0)
    assert np.isclose(calc_density(universe), np.mean(frames))


def test_calc_cellpar_frame_shapes(universe):
    n_frames = len(universe.trajectory)
    assert calc_cellpar_frame(universe).shape == (n_frames, 6)
    for target in ["La_A", "Lb_A", "Lc_A", "alpha_deg", "beta_deg", "gamma_deg"]:
        assert calc_cellpar_frame(universe, target).shape == (n_frames,)


def test_calc_cellpar_frame_rejects_unknown_target(universe):
    with pytest.raises(ValueError, match="target must be"):
        calc_cellpar_frame(universe, "hoge")


def test_package_exports_only_public_api():
    """imolcraft.analyzer が import したモジュールを露出しないこと"""
    import imolcraft.analyzer as pkg

    exported = {n for n in vars(pkg) if not n.startswith("_")}
    assert exported == {
        "analyzer",  # サブモジュール自身
        "CELLPAR_INDICES",
        "calc_adf",
        "calc_adf_frame",
        "calc_cellpar_frame",
        "calc_density",
        "calc_density_frame",
        "calc_rdf",
        "calc_rdf_frame",
    }
