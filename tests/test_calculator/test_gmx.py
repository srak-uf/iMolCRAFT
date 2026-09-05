"""Unit tests for GMXCalculator: mdp generation, command lines and a GROMACS smoke run"""
import importlib.resources
import os
import shutil

import pytest

from imolcraft.calculator import GMXCalculator, gmx, gmx_sample, md

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PDB = os.path.join(DATA, "supercell_bonds.pdb")
#: vsite_average2.xml with its two virtual sites stripped: the same molecule
#: without extra particles, which the GROMACS exporter can write.
FFXML = os.path.join(DATA, "supercell_bonds_novsite.xml")
VSITE_FFXML = os.path.join(DATA, "vsite_average2.xml")

TEMPLATES = ("min", "nvt", "isonpt", "anisonpt_xyz", "trinpt_xyz_xy_yz_zx")


def _floats(value):
    return [float(x) for x in value.split()]


# -- A. no GROMACS binary needed ---------------------------------------------


@pytest.mark.parametrize("name", TEMPLATES)
def test_shipped_templates_are_readable_as_package_data(name):
    """A.2: the five tutorial mdp files ship with the package"""
    resource = importlib.resources.files("imolcraft.data") / "mdp" / f"{name}.mdp"
    options = gmx._read_mdp(resource)
    assert options["pbc"] == "xyz"
    assert options["coulombtype"] == "PME"


@pytest.mark.parametrize("name", TEMPLATES)
def test_mdp_parser_round_trips_the_templates(name, tmp_path):
    """A.1: read -> format -> read keeps keys, order and values; comments vanish"""
    resource = importlib.resources.files("imolcraft.data") / "mdp" / f"{name}.mdp"
    first = gmx._read_mdp(resource)
    path = tmp_path / f"{name}.mdp"
    path.write_text(gmx._format_mdp(first))
    second = gmx._read_mdp(path)

    assert list(first) == list(second)
    assert first == second
    assert all(";" not in value for value in first.values())
    # The inline comment of the template did not leak into the value
    assert first["integrator"] in ("steep", "md")


def test_mdp_parser_folds_underscores_onto_hyphens(tmp_path):
    path = tmp_path / "x.mdp"
    path.write_text("ref_t = 300 ; kelvin\n\ncompressed-x-grps =\n")
    assert gmx._read_mdp(path) == {"ref-t": "300", "compressed-x-grps": ""}


@pytest.mark.parametrize("dt_fs, expected", [(2.0, 0.002), (1.0, 0.001), (0.5, 0.0005)])
def test_timestep_is_converted_from_fs_to_ps(dt_fs, expected):
    """A.3: dt_fs -> dt [ps]"""
    calc = GMXCalculator("start.pdb", dt_fs=dt_fs)
    assert float(calc.mdp_options("prod")["dt"]) == pytest.approx(expected, rel=1e-12)
    assert "dt" not in calc.mdp_options("min")


@pytest.mark.parametrize("rcut, rlist", [(1.2, 1.4), (1.0, 1.2)])
def test_cutoff_is_written_in_nm_without_conversion(rcut, rlist):
    options = GMXCalculator("start.pdb", rcut_nm=rcut).mdp_options("prod")
    assert float(options["rvdw"]) == pytest.approx(rcut, rel=1e-12)
    assert float(options["rcoulomb"]) == pytest.approx(rcut, rel=1e-12)
    assert float(options["rlist"]) == pytest.approx(rlist, rel=1e-12)


def test_temperature_sets_ref_t_and_gen_temp():
    options = GMXCalculator("start.pdb", temperature_K=233.15).mdp_options("relax")
    assert float(options["ref-t"]) == pytest.approx(233.15)
    assert float(options["gen-temp"]) == pytest.approx(233.15)


def test_annealing_is_mapped_onto_gromacs_native_annealing():
    """A.3: cumulative step counts become annealing-time in ps"""
    calc = GMXCalculator(
        "start.pdb", anneal_T=[300, 400, 300], anneal_steps=[50000, 30000], dt_fs=2.0
    )
    options = calc.mdp_options("anneal")
    assert options["annealing"] == "single"
    assert options["annealing-npoints"] == "3"
    assert _floats(options["annealing-temp"]) == pytest.approx([300, 400, 300])
    assert _floats(options["annealing-time"]) == pytest.approx([0, 100, 160], rel=1e-9)
    assert options["nsteps"] == "80000"
    assert options["gen-vel"] == "yes"
    assert float(options["gen-temp"]) == pytest.approx(300.0)
    assert options["pcoupl"] == "no"
    # The relaxation then inherits the velocities instead of drawing new ones
    relax = calc.mdp_options("relax")
    assert relax["gen-vel"] == "no" and relax["continuation"] == "yes"


def test_annealing_options_are_empty_without_a_schedule():
    assert gmx._annealing_options([], 2.0) == {}
    options = GMXCalculator("start.pdb").mdp_options("anneal")
    assert "annealing" not in options and options["nsteps"] == "0"


@pytest.mark.parametrize("ensemble", md.VALID_ENSEMBLES)
def test_every_md_ensemble_is_accepted(ensemble):
    """The ensemble set is md.VALID_ENSEMBLES, nve included"""
    calc = GMXCalculator("start.pdb", ensemble=ensemble)
    assert calc.mdp_options("prod")["integrator"] == "md"


def test_gmx_pcoupl_covers_exactly_the_md_ensembles():
    assert set(gmx.GMX_PCOUPL) == set(md.VALID_ENSEMBLES)


@pytest.mark.parametrize("ensemble", ["nve", "nvt"])
def test_fixed_volume_ensembles_carry_no_barostat_keys(ensemble):
    """A.4: nvt.mdp's ref-p = 5.0 must not leak into a run without a barostat"""
    options = GMXCalculator("start.pdb", ensemble=ensemble).mdp_options("prod")
    assert options["pcoupl"] == "no"
    for key in ("ref-p", "pcoupltype", "compressibility", "tau-p"):
        assert key not in options
    assert options["tcoupl"] == ("no" if ensemble == "nve" else "nose-hoover")


def test_isonpt_pressure_options():
    options = GMXCalculator("start.pdb", ensemble="isonpt").mdp_options("prod")
    assert options["pcoupl"] == "Parrinello-Rahman"
    assert options["pcoupltype"] == "isotropic"
    assert options["ref-p"] == "1.0"
    assert float(options["compressibility"]) == pytest.approx(4.5e-5, rel=1e-12)
    assert float(options["tau-p"]) == pytest.approx(5.0)
    assert float(options["tau-t"]) == pytest.approx(1.0)


@pytest.mark.parametrize("ensemble, c_mask", [
    ("anisonpt", [1, 1, 1, 0, 0, 0]),
    ("trinpt", [1, 1, 1, 1, 1, 1]),
])
def test_anisotropic_pressure_vectors(ensemble, c_mask):
    c = 4.5e-5
    options = GMXCalculator(
        "start.pdb", ensemble=ensemble, pressure_bar=1.0, compressibility_bar=c
    ).mdp_options("prod")
    assert options["pcoupltype"] == "anisotropic"
    assert _floats(options["ref-p"]) == pytest.approx([1, 1, 1, 0, 0, 0], rel=1e-12)
    assert _floats(options["compressibility"]) == pytest.approx(
        [c * m for m in c_mask], rel=1e-12
    )


def test_pressure_and_coupling_settings_reach_the_mdp():
    options = GMXCalculator(
        "start.pdb", ensemble="isonpt", pressure_bar=50.0, tau_p_ps=2.0,
        tau_t_ps=0.5, tcoupl="v-rescale", pcoupl="C-rescale",
    ).mdp_options("relax")
    assert float(options["ref-p"]) == pytest.approx(50.0)
    assert options["tcoupl"] == "v-rescale" and options["pcoupl"] == "C-rescale"
    assert float(options["tau-p"]) == pytest.approx(2.0)
    assert float(options["tau-t"]) == pytest.approx(0.5)


@pytest.mark.parametrize("flag, expected", [(True, "h-bonds"), (False, "none")])
def test_hbond_constraint_maps_onto_constraints(flag, expected):
    """A.5"""
    for stage in gmx.GMX_STAGES:
        options = GMXCalculator("start.pdb", useHbondConstraint=flag).mdp_options(stage)
        assert options["constraints"] == expected


@pytest.mark.parametrize("flag, expected", [(True, "EnerPres"), (False, "no")])
def test_dispcorr_maps_onto_DispCorr(flag, expected):
    for stage in gmx.GMX_STAGES:
        assert GMXCalculator("start.pdb", dispcorr=flag).mdp_options(stage)["DispCorr"] == expected


@pytest.mark.parametrize("method, vdwtype", [("PME", "Cut-off"), ("LJPME", "PME")])
def test_nonbondedmethod_maps_onto_vdwtype(method, vdwtype):
    for stage in gmx.GMX_STAGES:
        options = GMXCalculator("start.pdb", nonbondedmethod=method).mdp_options(stage)
        assert options["vdwtype"] == vdwtype
        assert options["coulombtype"] == "PME"


def test_stages_differ_as_intended():
    """A.6: min is steepest descent, prod continues without new velocities, xtc only"""
    calc = GMXCalculator("start.pdb", min_steps=123, emtol=10.0, relax_steps=456,
                         prod_steps=789, nstxout=25)
    options = {stage: calc.mdp_options(stage) for stage in gmx.GMX_STAGES}
    assert all(isinstance(o, dict) for o in options.values())

    assert options["min"]["integrator"] == "steep"
    assert options["min"]["nsteps"] == "123"
    assert float(options["min"]["emtol"]) == pytest.approx(10.0)
    for stage in ("anneal", "relax", "prod"):
        assert options[stage]["integrator"] == "md"

    assert options["relax"]["nsteps"] == "456"
    assert options["relax"]["gen-vel"] == "yes"       # no annealing before it
    assert options["relax"]["continuation"] == "no"

    prod = options["prod"]
    assert prod["nsteps"] == "789"
    assert prod["continuation"] == "yes"
    assert prod["gen-vel"] == "no"
    assert prod["nstxout"] == "0" and prod["nstvout"] == "0"
    assert prod["nstxout-compressed"] == "25"
    assert prod["nstenergy"] == "25" and prod["nstlog"] == "25"


def test_mdp_extra_wins_over_everything():
    """A.7: raw mdp entries are applied last, key spelling folded"""
    calc = GMXCalculator(
        "start.pdb", rcut_nm=1.2,
        mdp_extra={"rvdw": 0.9, "nstlist": 20, "fourier_spacing": "0.1"},
    )
    options = calc.mdp_options("prod")
    assert options["rvdw"] == "0.9"
    assert options["nstlist"] == "20"
    assert options["fourier-spacing"] == "0.1"
    assert options["rcoulomb"] == "1.2"


def test_mdp_templates_override_the_shipped_ones(tmp_path):
    template = tmp_path / "my_prod.mdp"
    template.write_text("integrator = md\nnstlist = 42\nfoo-bar = baz\n")
    calc = GMXCalculator("start.pdb", mdp_templates={"prod": str(template)})
    options = calc.mdp_options("prod")
    assert options["nstlist"] == "42" and options["foo-bar"] == "baz"
    # Other stages still come from the package
    assert "foo-bar" not in calc.mdp_options("relax")


def test_mdp_templates_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="unknown stages"):
        GMXCalculator("start.pdb", mdp_templates={"production": "x.mdp"})


def test_mdp_options_rejects_an_unknown_stage():
    with pytest.raises(ValueError, match="stage must be one of"):
        GMXCalculator("start.pdb").mdp_options("equil")


def test_calculator_records_every_setting_including_the_defaults():
    """A.8: to_dict writes omitted settings with their defaults"""
    calc = GMXCalculator("start.pdb", temperature_K=350.0, prod_steps=1000)
    record = calc.to_dict()
    assert record["init_structure"] == "start.pdb"
    assert record["temperature_K"] == 350.0
    assert record["prod_steps"] == 1000
    assert set(record) == {"init_structure"} | set(GMXCalculator.SETTINGS)
    assert record["gmx_bin"] == "gmx"


def test_calculator_round_trips_through_a_record():
    calc = GMXCalculator(
        "start.pdb", temperature_K=350.0, ensemble="trinpt", pressure_bar=10.0,
        anneal_T=[300, 400, 300], anneal_steps=[10, 20], md_log="none",
        mpi_command=["mpirun", "-np", "4"], mdp_extra={"nstlist": 20},
    )
    assert GMXCalculator.from_dict(calc.to_dict()).to_dict() == calc.to_dict()


def test_calculator_rejects_an_unknown_setting():
    with pytest.raises(TypeError, match="unknown GROMACS MD settings: temperature"):
        GMXCalculator("start.pdb", temperature=350.0)


def test_calculator_rejects_an_unknown_ensemble():
    with pytest.raises(ValueError, match="Invalid ensemble"):
        GMXCalculator("start.pdb", ensemble="npt")


def test_calculator_rejects_an_unknown_nonbondedmethod():
    with pytest.raises(ValueError, match="Invalid nonbonded method"):
        GMXCalculator("start.pdb", nonbondedmethod="cutoff")


def test_calculator_rejects_an_unknown_md_log_mode():
    with pytest.raises(ValueError, match="md_log"):
        GMXCalculator("start.pdb", md_log="quiet")


def test_calculator_expands_the_annealing_schedule_on_construction():
    calc = GMXCalculator("start.pdb", anneal_T=[300, 400], anneal_steps=[50])
    assert calc.anneal_legs == [(300.0, 400.0, 50)]
    with pytest.raises(ValueError, match="one entry fewer"):
        GMXCalculator("start.pdb", anneal_T=[300, 400, 300], anneal_steps=[50])


def test_settings_are_a_superset_of_the_openmm_calculator():
    """A.9: every MDCalculator setting is accepted under the same name"""
    assert set(md.MDCalculator.SETTINGS) <= set(GMXCalculator.SETTINGS)
    for name, default in md.MDCalculator.SETTINGS.items():
        assert GMXCalculator.SETTINGS[name] == default


def test_a_sampling_block_for_the_openmm_calculator_is_accepted_as_is():
    calc = GMXCalculator("start.pdb", **md.MDCalculator.SETTINGS)
    assert calc.rigidWater is False and calc.anneal_interval == 100


def test_module_reuses_the_openmm_constants_and_exports_little():
    assert gmx.VALID_ENSEMBLES is md.VALID_ENSEMBLES
    assert gmx.MD_LOG_MODES is md.MD_LOG_MODES
    assert gmx.NONBONDED_METHODS is md.NONBONDED_METHODS
    assert set(gmx.__all__) == {"GMXCalculator", "gmx_sample", "GMX_STAGES", "GMX_PCOUPL"}


def test_grompp_command():
    """A.10"""
    calc = GMXCalculator("start.pdb", maxwarn=2, gmx_bin="gmx_d")
    argv = calc.grompp_command("a.mdp", "a.gro", "a.top", "w/a.tpr")
    assert isinstance(argv, list) and all(isinstance(x, str) for x in argv)
    assert argv[:2] == ["gmx_d", "grompp"]
    for flag, value in (("-f", "a.mdp"), ("-c", "a.gro"), ("-p", "a.top"),
                        ("-o", "w/a.tpr"), ("-maxwarn", "2"), ("-po", "w/a_mdout.mdp")):
        assert argv[argv.index(flag) + 1] == value
    assert "-t" not in argv
    with_cpt = calc.grompp_command("a.mdp", "a.gro", "a.top", "a.tpr", checkpoint="p.cpt")
    assert with_cpt[with_cpt.index("-t") + 1] == "p.cpt"
    # mpi_command does not apply to grompp
    calc = GMXCalculator("start.pdb", mpi_command=["mpirun", "-np", "4"])
    assert calc.grompp_command("a.mdp", "a.gro", "a.top", "a.tpr")[0] == "gmx"


def test_mdrun_command_defaults():
    argv = GMXCalculator("start.pdb").mdrun_command("gmxfiles/s_prod")
    assert isinstance(argv, list) and all(isinstance(x, str) for x in argv)
    assert argv[:4] == ["gmx", "mdrun", "-deffnm", "gmxfiles/s_prod"]
    assert argv[argv.index("-nb") + 1] == "cpu"
    assert "-ntmpi" not in argv and "-ntomp" not in argv


def test_mdrun_command_with_mpi_threads_and_gpu():
    calc = GMXCalculator(
        "start.pdb", mpi_command=["mpirun", "-np", "4"], gmx_bin="gmx_mpi",
        ntmpi=2, ntomp=8, device="CUDA",
    )
    argv = calc.mdrun_command("s_prod")
    assert argv[:3] == ["mpirun", "-np", "4"]
    assert argv[3:5] == ["gmx_mpi", "mdrun"]
    assert argv[argv.index("-nb") + 1] == "gpu"
    assert argv[argv.index("-ntmpi") + 1] == "2"
    assert argv[argv.index("-ntomp") + 1] == "8"


@pytest.mark.parametrize("device, nb", [("CPU", "cpu"), ("cpu", "cpu"),
                                        ("CUDA", "gpu"), ("OpenCL", "gpu"), ("HIP", "gpu")])
def test_mdrun_device_maps_onto_nb(device, nb):
    argv = GMXCalculator("start.pdb", device=device).mdrun_command("x")
    assert argv[argv.index("-nb") + 1] == nb


def test_write_mdp_writes_a_readable_file(tmp_path):
    """A.11"""
    calc = GMXCalculator("start.pdb", ensemble="isonpt", prod_steps=77)
    path = calc.write_mdp("prod", str(tmp_path / "sub" / "prod.mdp"))
    assert os.path.exists(path)
    assert gmx._read_mdp(path) == calc.mdp_options("prod")


def test_run_fails_early_without_the_binary(tmp_path, monkeypatch):
    """A.12: the PATH check comes first, before anything is exported or written"""
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(gmx, "exporter", lambda *a, **k: calls.append(a))
    calc = GMXCalculator("start.pdb", gmx_bin="definitely-not-a-gmx", md_log="none")
    with pytest.raises(FileNotFoundError, match="imc_cpu|GMXRC"):
        calc.run("ff.xml", "s_0.xtc")
    assert calls == []
    assert not (tmp_path / "gmxfiles").exists()


def test_run_wraps_a_failing_command_with_the_stage(tmp_path, monkeypatch):
    """A failing grompp/mdrun surfaces as RuntimeError naming the stage"""
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gmx.shutil, "which", lambda name: "/fake/gmx")
    monkeypatch.setattr(
        GMXCalculator, "_export_inputs", lambda self, ffxml, tmpdir: ("s.top", "s.gro")
    )

    def fail(argv, logstream, env=None):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(gmx, "_run_command", fail)
    calc = GMXCalculator("start.pdb", md_log="none", workdir="w")
    with pytest.raises(RuntimeError, match="min stage"):
        calc.run("ff.xml", "s_0.xtc")
    assert (tmp_path / "w" / "s_0_min.mdp").exists()


def test_run_command_uses_a_list_and_no_shell(monkeypatch):
    import subprocess

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(gmx.subprocess, "run", fake_run)
    gmx._run_command(["gmx", "mdrun"], None)
    assert seen["argv"] == ["gmx", "mdrun"]
    assert seen["check"] is True
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.STDOUT
    assert "shell" not in seen


def test_build_system_matches_the_openmm_calculator():
    """The System GROMACS gets is the one MDCalculator would integrate (minus constraints)"""
    from openmm import NonbondedForce, unit

    calc = GMXCalculator(PDB, rcut_nm=1.0, nonbondedmethod="LJPME", dispcorr=True)
    topology, positions, system = calc._build_system(FFXML)

    reference = md.MDCalculator(
        PDB, rcut_nm=1.0, nonbondedmethod="LJPME", dispcorr=True,
        useHbondConstraint=False,
    )
    simulation, _, ref_positions = reference._build_simulation(FFXML, lambda m: None)
    ref_system = simulation.system

    assert topology.getNumAtoms() == 10 and len(positions) == 10
    assert system.getNumParticles() == ref_system.getNumParticles()
    assert system.getNumConstraints() == 0
    assert [type(f).__name__ for f in system.getForces()] == [
        type(f).__name__ for f in ref_system.getForces()
    ]
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    ref_nb = next(f for f in ref_system.getForces() if isinstance(f, NonbondedForce))
    assert nb.getNonbondedMethod() == ref_nb.getNonbondedMethod() == NonbondedForce.LJPME
    assert nb.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(1.0)
    assert nb.getUseDispersionCorrection() is True
    for i in range(nb.getNumParticles()):
        assert nb.getParticleParameters(i) == ref_nb.getParticleParameters(i)


def test_export_inputs_writes_top_and_gro_from_an_ffxml(tmp_path):
    """The .top / .gro come out of the ffxml through exporter(fmt='gmx')"""
    calc = GMXCalculator(PDB)
    top, gro = calc._export_inputs(FFXML, str(tmp_path))
    assert os.path.dirname(top) == str(tmp_path) and os.path.dirname(gro) == str(tmp_path)
    assert os.path.getsize(top) > 0
    # second line of a .gro is the atom count
    assert int(gro_lines := open(gro).read().splitlines()[1]) == 10, gro_lines
    top_text = open(top).read()
    assert "[ bonds ]" in top_text and "[ molecules ]" in top_text
    # The intermediate system.xml / system.pdb live in the same directory
    assert (tmp_path / "system.xml").exists() and (tmp_path / "system.pdb").exists()


def test_run_refuses_a_force_field_with_virtual_sites(tmp_path, monkeypatch):
    """Virtual sites are not translated to [ virtual_sites2 ], so a vsite ffxml is refused before any GROMACS call"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gmx.shutil, "which", lambda name: "/fake/gmx")
    calls = []
    monkeypatch.setattr(gmx, "_run_command", lambda *a, **k: calls.append(a))
    calc = GMXCalculator(PDB, md_log="none")
    with pytest.raises(ValueError, match="virtual sites are not supported yet"):
        calc.run(VSITE_FFXML, "s_0.xtc")
    assert calls == []


# -- B. smoke test with the real gmx -----------------------------------------


@pytest.mark.gmx
@pytest.mark.skipif(shutil.which("gmx") is None, reason="gmx binary not on PATH")
def test_smoke_run_with_gromacs(tmp_path, monkeypatch):
    """A 10-atom system through min -> relax -> prod; the xtc reads back with the pdb"""
    mdtraj = pytest.importorskip("mdtraj")
    monkeypatch.chdir(tmp_path)
    calc = GMXCalculator(
        PDB, ensemble="nvt", dt_fs=1.0, min_steps=50, relax_steps=100,
        prod_steps=200, nstxout=20, md_log="none",
        useHbondConstraint=True, dispcorr=False,
    )
    xtc = calc.run(FFXML, "smoke_0.xtc")

    assert xtc == os.path.join("xtcfiles", "smoke_0.xtc")
    assert os.path.getsize(xtc) > 0
    traj = mdtraj.load(xtc, top=PDB)
    assert traj.n_atoms == 10
    assert traj.n_frames in (200 // 20, 200 // 20 + 1)

    for stage in ("min", "relax", "prod"):
        assert (tmp_path / "gmxfiles" / f"smoke_0_{stage}.mdp").exists()
        assert (tmp_path / "gmxfiles" / f"smoke_0_{stage}.tpr").exists()
    assert not (tmp_path / "gmxfiles" / "smoke_0_anneal.tpr").exists()
    # The .top / .gro inputs lived in a temporary directory, not in workdir
    assert not any(p.suffix == ".top" for p in (tmp_path / "gmxfiles").iterdir())
    assert not any(p.suffix in (".top", ".gro", ".mdp") for p in tmp_path.iterdir())


@pytest.mark.gmx
@pytest.mark.skipif(shutil.which("gmx") is None, reason="gmx binary not on PATH")
def test_smoke_gmx_sample_with_annealing_and_npt(tmp_path, monkeypatch):
    """Annealing -> isonpt relaxation -> production through the wrapper"""
    mdtraj = pytest.importorskip("mdtraj")
    monkeypatch.chdir(tmp_path)
    xtc = gmx_sample(
        PDB, FFXML, "anneal_0.xtc", ensemble="isonpt", pcoupl="C-rescale",
        dt_fs=1.0, min_steps=50, relax_steps=100, prod_steps=100, nstxout=20,
        anneal_T=[300, 400, 300], anneal_steps=[50, 50], md_log="file",
    )
    traj = mdtraj.load(xtc, top=PDB)
    assert traj.n_atoms == 10
    assert traj.n_frames in (5, 6)
    assert (tmp_path / "gmxfiles" / "anneal_0_anneal.tpr").exists()
    assert (tmp_path / "mdlogs" / "anneal_0.log").stat().st_size > 0
