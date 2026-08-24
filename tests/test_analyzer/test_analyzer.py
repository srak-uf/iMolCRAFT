import os

import numpy as np
import pytest

from imolcraft.analyzer.analyzer import (
    _fit_msd_line,
    _join_on_center,
    _zero_first_bin,
    calc_cellpar_frame,
    calc_density,
    calc_density_frame,
    calc_dself,
    calc_msd,
    calc_rdf,
    calc_rdf_frame,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PDB = os.path.join(DATA, "vs_supercell_bonds.pdb")
XTC = os.path.join(DATA, "sample_0.xtc")


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
        "calc_dself",
        "calc_msd",
        "calc_rdf",
        "calc_rdf_frame",
    }


def test_fit_msd_line_recovers_a_straight_line():
    """直線を与えれば窓の取り方によらず傾きと切片がそのまま返る"""
    t = np.arange(100) * 0.5
    msd = 3.0 * t + 7.0
    assert np.allclose(_fit_msd_line(t, msd, (0.0, 0.5)), (3.0, 7.0))
    assert np.allclose(_fit_msd_line(t, msd, (0.25, 1.0)), (3.0, 7.0))


def test_fit_msd_line_uses_only_the_requested_window():
    """窓の外がどれだけ暴れても傾きに効かない"""
    t = np.arange(100) * 0.5
    msd = 3.0 * t
    msd[50:] = 1e6  # 後半をノイズで潰す
    assert np.isclose(_fit_msd_line(t, msd, (0.0, 0.5))[0], 3.0)


@pytest.mark.parametrize("fit_range", [(0.5, 0.5), (0.6, 0.2), (-0.1, 0.5), (0.0, 1.5)])
def test_fit_msd_line_rejects_invalid_range(fit_range):
    with pytest.raises(ValueError, match="fit_range must satisfy"):
        _fit_msd_line(np.arange(10.0), np.arange(10.0), fit_range)


def test_fit_msd_line_rejects_too_few_points():
    """点が 2 つ未満しか入らない窓は直線を引けない"""
    with pytest.raises(ValueError, match="at least 2"):
        _fit_msd_line(np.arange(10.0), np.arange(10.0), (0.0, 0.1))


def test_calc_msd_axis_and_zero_lag(universe):
    lagtime, msd = calc_msd(universe, select="element C")
    n_frames = len(universe.trajectory)
    assert lagtime.shape == msd.shape == (n_frames,)
    assert lagtime[0] == 0.0
    assert np.isclose(msd[0], 0.0)  # ラグ 0 の変位は 0
    assert np.allclose(np.diff(lagtime), universe.trajectory.dt)


def test_calc_msd_step_stretches_the_time_axis(universe):
    """step でフレームを間引くと 1 点あたりの時間間隔が step 倍になる"""
    lagtime, _ = calc_msd(universe, select="element C", step=2)
    assert np.allclose(np.diff(lagtime), universe.trajectory.dt * 2)


def test_calc_msd_applies_nojump_once(universe):
    calc_msd(universe, select="element C")
    calc_msd(universe, select="element C")  # 2 回目でも例外にならない
    assert len(universe.trajectory.transformations) == 1


def test_calc_msd_without_nojump_leaves_trajectory_untouched(universe):
    calc_msd(universe, select="element C", nojump=False)
    assert not universe.trajectory.transformations


def test_calc_msd_rejects_unknown_type(universe):
    with pytest.raises(ValueError, match="msd_type must be"):
        calc_msd(universe, select="element C", msd_type="hoge")


def test_calc_dself_matches_the_einstein_relation(universe):
    """calc_dself が MSD の傾き / (2d) に単位換算を掛けた値と一致する"""
    lagtime, msd = calc_msd(universe, select="element C")
    slope, _ = _fit_msd_line(lagtime, msd, (0.0, 0.5))
    expected = slope / 6 * 1e-4  # xyz なので 2d = 6, A^2/ps -> cm^2/s
    assert np.isclose(calc_dself(universe, select="element C"), expected)


@pytest.mark.parametrize("msd_type,dof", [("xyz", 6), ("xy", 4), ("z", 2)])
def test_calc_dself_scales_with_dimensionality(universe, msd_type, dof):
    lagtime, msd = calc_msd(universe, select="element C", msd_type=msd_type)
    slope, _ = _fit_msd_line(lagtime, msd, (0.0, 0.5))
    got = calc_dself(universe, select="element C", msd_type=msd_type)
    assert np.isclose(got, slope / dof * 1e-4)
