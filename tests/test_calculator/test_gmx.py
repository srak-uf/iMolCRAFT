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


@pytest.fixture(autouse=True)
def _no_gmx_environment(monkeypatch):
    """Every test starts without the IMOLCRAFT_GMX_* variables, whatever the shell has"""
    for variable in gmx.GMX_ENV.values():
        monkeypatch.delenv(variable, raising=False)


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


@pytest.mark.parametrize("rcut", [1.2, 1.0, 1.5])
def test_cutoff_is_written_in_nm_without_conversion(rcut):
    for stage in gmx.GMX_STAGES:
        options = GMXCalculator("start.pdb", rcut_nm=rcut).mdp_options(stage)
        assert float(options["rvdw"]) == pytest.approx(rcut, rel=1e-12)
        assert float(options["rcoulomb"]) == pytest.approx(rcut, rel=1e-12)


def test_pair_list_is_left_to_gromacs():
    """Neither rlist nor verlet-buffer-tolerance is written; the template's are dropped"""
    for stage in gmx.GMX_STAGES:
        options = GMXCalculator("start.pdb").mdp_options(stage)
        assert "rlist" not in options
        assert "verlet-buffer-tolerance" not in options
        assert "nstlist" in options
    # ...unless the user pins the pair list explicitly through mdp_extra
    pinned = GMXCalculator(
        "start.pdb", mdp_extra={"verlet-buffer-tolerance": -1, "rlist": 1.4}
    ).mdp_options("prod")
    assert pinned["verlet-buffer-tolerance"] == "-1"
    assert pinned["rlist"] == "1.4"


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
        assert options["coulomb-modifier"] == "None"


def test_ljpme_writes_the_openmm_scheme_explicitly():
    """Geometric mesh + LB pairs, potential shift; a plain cutoff keeps the template's None"""
    ljpme = GMXCalculator("start.pdb", nonbondedmethod="LJPME").mdp_options("prod")
    assert ljpme["lj-pme-comb-rule"] == "Geometric"
    assert ljpme["vdw-modifier"] == "Potential-Shift"
    assert "ewald-rtol-lj" in ljpme

    pme = GMXCalculator("start.pdb", nonbondedmethod="PME").mdp_options("prod")
    assert "lj-pme-comb-rule" not in pme
    assert pme["vdw-modifier"] == "None"
    assert "ewald-rtol-lj" not in pme


def test_ewald_options_reproduce_openmm_alpha_and_grid():
    """ewald-rtol / ewald-rtol-lj / fourierspacing are OpenMM's alpha and mesh in GROMACS terms"""
    import math

    tol = 5e-4                       # OpenMM NonbondedForce default
    x = math.sqrt(-math.log(2 * tol))  # alpha * rc in OpenMM
    assert gmx._EWALD_TOLERANCE == pytest.approx(tol)

    options = gmx._ewald_options(1.2, ljpme=True)
    # erfc(beta rc) = ewald-rtol  ->  beta == alpha
    assert float(options["ewald-rtol"]) == pytest.approx(math.erfc(x), rel=1e-9)
    assert float(options["ewald-rtol"]) == pytest.approx(2.0166e-4, rel=1e-3)
    # exp(-x^2)(1 + x^2 + x^4/2) = ewald-rtol-lj  ->  beta_lj == alpha
    assert float(options["ewald-rtol-lj"]) == pytest.approx(
        math.exp(-x * x) * (1 + x * x + x ** 4 / 2), rel=1e-9
    )
    assert float(options["ewald-rtol-lj"]) == pytest.approx(3.1766e-2, rel=1e-3)
    # L / n = 3 tol^0.2 / (2 alpha) with alpha = x / rc
    assert float(options["fourierspacing"]) == pytest.approx(3 * tol ** 0.2 * 1.2 / (2 * x), rel=1e-9)
    assert float(options["fourierspacing"]) == pytest.approx(0.14976, rel=1e-3)
    # The spacing scales with the cutoff, the tolerances do not
    shorter = gmx._ewald_options(1.0, ljpme=False)
    assert float(shorter["fourierspacing"]) == pytest.approx(0.14976 / 1.2, rel=1e-3)
    assert shorter["ewald-rtol"] == options["ewald-rtol"]
    assert "ewald-rtol-lj" not in shorter

    prod = GMXCalculator("start.pdb", rcut_nm=1.2).mdp_options("prod")
    assert prod["ewald-rtol"] == options["ewald-rtol"]
    assert prod["fourierspacing"] == options["fourierspacing"]


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
    assert record["gmx_bin"] is None


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
    assert set(gmx.__all__) == {"GMXCalculator", "gmx_sample", "GMX_STAGES", "GMX_PCOUPL", "GMX_ENV"}


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
    """No arguments, no environment: plain gmx, no launcher, threads left to mdrun"""
    argv = GMXCalculator("start.pdb").mdrun_command("gmxfiles/s_prod")
    assert isinstance(argv, list) and all(isinstance(x, str) for x in argv)
    assert argv[:4] == ["gmx", "mdrun", "-deffnm", "gmxfiles/s_prod"]
    assert argv[argv.index("-nb") + 1] == "cpu"
    assert "-ntmpi" not in argv and "-ntomp" not in argv


def test_mdrun_command_with_mpi_threads_and_gpu():
    calc = GMXCalculator(
        "start.pdb", mpi_command=["mpirun", "-np", "4"], gmx_bin="gmx_mpi",
        ntomp=8, device="CUDA",
    )
    argv = calc.mdrun_command("s_prod")
    assert argv[:3] == ["mpirun", "-np", "4"]
    assert argv[3:5] == ["gmx_mpi", "mdrun"]
    assert argv[argv.index("-nb") + 1] == "gpu"
    assert "-ntmpi" not in argv
    assert argv[argv.index("-ntomp") + 1] == "8"


def test_mdrun_command_thread_mpi_without_a_launcher():
    argv = GMXCalculator("start.pdb", ntmpi=2, ntomp=4).mdrun_command("s_prod")
    assert argv[:2] == ["gmx", "mdrun"]
    assert argv[argv.index("-ntmpi") + 1] == "2"
    assert argv[argv.index("-ntomp") + 1] == "4"


def test_mpi_command_given_as_a_string_is_split_like_a_shell():
    calc = GMXCalculator("start.pdb", mpi_command="mpirun -np 4 --bind-to 'core x'")
    argv = calc.mdrun_command("s_prod")
    assert argv[:6] == ["mpirun", "-np", "4", "--bind-to", "core x", "gmx"]
    # the record keeps the string as given, the resolver does the splitting
    assert calc.to_dict()["mpi_command"] == "mpirun -np 4 --bind-to 'core x'"
    assert calc._resolve_execution()[0]["mpi_command"] == ["mpirun", "-np", "4", "--bind-to", "core x"]
    # a list is taken as is, an empty string as no launcher
    assert GMXCalculator("start.pdb", mpi_command=["srun", "-n", "8"]).mdrun_command("x")[:3] == ["srun", "-n", "8"]
    assert GMXCalculator("start.pdb", mpi_command="").mdrun_command("x")[0] == "gmx"


@pytest.mark.parametrize("value, match", [
    ("abc", r"\$IMOLCRAFT_GMX_NTOMP='abc' is not an integer"),
    ("0", r"\$IMOLCRAFT_GMX_NTOMP='0' must be a positive integer"),
    ("-2", r"\$IMOLCRAFT_GMX_NTOMP='-2' must be a positive integer"),
])
def test_thread_counts_from_the_environment_are_validated(monkeypatch, value, match):
    monkeypatch.setenv("IMOLCRAFT_GMX_NTOMP", value)
    with pytest.raises(ValueError, match=match):
        GMXCalculator("start.pdb").mdrun_command("s_prod")


@pytest.mark.parametrize("name, value, match", [
    ("ntmpi", 0, r"ntmpi=0 must be a positive integer"),
    ("ntomp", -1, r"ntomp=-1 must be a positive integer"),
    ("ntomp", "four", r"ntomp='four' is not an integer"),
])
def test_thread_counts_from_arguments_are_validated(name, value, match):
    with pytest.raises(ValueError, match=match):
        GMXCalculator("start.pdb", **{name: value}).mdrun_command("s_prod")


def test_thread_counts_accept_numeric_strings(monkeypatch):
    monkeypatch.setenv("IMOLCRAFT_GMX_NTMPI", " 2 ")
    values, _ = GMXCalculator("start.pdb", ntomp="4")._resolve_execution()
    assert values["ntmpi"] == 2 and values["ntomp"] == 4


def test_execution_resolves_to_defaults_without_environment():
    values, sources = GMXCalculator("start.pdb")._resolve_execution()
    assert values == {"gmx_bin": "gmx", "mpi_command": None, "ntmpi": None, "ntomp": None}
    assert set(sources.values()) == {"default"}


def test_execution_resolves_from_the_environment(monkeypatch):
    monkeypatch.setenv("IMOLCRAFT_GMX_BIN", "gmx_mpi")
    monkeypatch.setenv("IMOLCRAFT_GMX_MPI_COMMAND", "srun --mpi=pmix -n 8")
    monkeypatch.setenv("IMOLCRAFT_GMX_NTOMP", "6")
    calc = GMXCalculator("start.pdb")
    values, sources = calc._resolve_execution()
    assert values["gmx_bin"] == "gmx_mpi"
    assert values["mpi_command"] == ["srun", "--mpi=pmix", "-n", "8"]
    assert values["ntomp"] == 6 and values["ntmpi"] is None
    assert sources == {"gmx_bin": "environment", "mpi_command": "environment",
                       "ntmpi": "default", "ntomp": "environment"}

    argv = calc.mdrun_command("s_prod")
    assert argv[:6] == ["srun", "--mpi=pmix", "-n", "8", "gmx_mpi", "mdrun"]
    assert argv[argv.index("-ntomp") + 1] == "6" and "-ntmpi" not in argv
    # grompp uses the binary from the environment but never the launcher
    assert calc.grompp_command("a.mdp", "a.gro", "a.top", "a.tpr")[:2] == ["gmx_mpi", "grompp"]


def test_execution_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("IMOLCRAFT_GMX_BIN", "gmx_mpi")
    monkeypatch.setenv("IMOLCRAFT_GMX_NTOMP", "6")
    values, sources = GMXCalculator("start.pdb", gmx_bin="gmx_d", ntomp=2)._resolve_execution()
    assert values["gmx_bin"] == "gmx_d" and sources["gmx_bin"] == "argument"
    assert values["ntomp"] == 2 and sources["ntomp"] == "argument"


def test_execution_treats_an_empty_variable_as_unset(monkeypatch):
    monkeypatch.setenv("IMOLCRAFT_GMX_MPI_COMMAND", "   ")
    monkeypatch.setenv("IMOLCRAFT_GMX_BIN", "")
    values, sources = GMXCalculator("start.pdb")._resolve_execution()
    assert values["mpi_command"] is None and values["gmx_bin"] == "gmx"
    assert sources["mpi_command"] == "default" and sources["gmx_bin"] == "default"


@pytest.mark.parametrize("via_env", [False, True])
def test_launcher_with_ntmpi_is_refused(monkeypatch, via_env):
    """A launcher means gmx_mpi, which does not take -ntmpi"""
    if via_env:
        monkeypatch.setenv("IMOLCRAFT_GMX_MPI_COMMAND", "mpirun -np 4")
        monkeypatch.setenv("IMOLCRAFT_GMX_NTMPI", "4")
        calc = GMXCalculator("start.pdb")
    else:
        calc = GMXCalculator("start.pdb", mpi_command=["mpirun", "-np", "4"], ntmpi=4)
    with pytest.raises(ValueError, match="does not accept -ntmpi"):
        calc.mdrun_command("s_prod")


def test_to_dict_records_arguments_not_the_environment(monkeypatch):
    """A checkpoint carries the recipe of the state, not the launcher of the job"""
    monkeypatch.setenv("IMOLCRAFT_GMX_BIN", "gmx_mpi")
    monkeypatch.setenv("IMOLCRAFT_GMX_MPI_COMMAND", "srun -n 8")
    monkeypatch.setenv("IMOLCRAFT_GMX_NTOMP", "6")
    calc = GMXCalculator("start.pdb", ntomp=2)
    record = calc.to_dict()
    assert record["gmx_bin"] is None and record["mpi_command"] is None
    assert record["ntmpi"] is None and record["ntomp"] == 2
    assert GMXCalculator.from_dict(record).to_dict() == record
    # ...and the restored calculator still resolves the environment at run time
    assert GMXCalculator.from_dict(record)._resolve_execution()[0]["gmx_bin"] == "gmx_mpi"


def test_missing_binary_message_names_the_variable(monkeypatch):
    monkeypatch.setenv("IMOLCRAFT_GMX_BIN", "definitely-not-a-gmx")
    with pytest.raises(FileNotFoundError, match="IMOLCRAFT_GMX_BIN"):
        GMXCalculator("start.pdb", md_log="none").run("ff.xml", "s_0.xtc")


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


def _openmm_energies(calc, ffxml, gro, zero_charges=False):
    """
    Potential energy terms of the System of ``calc`` at the coordinates of a
    ``.gro`` (so both codes see the same 3-decimal positions), in kJ/mol.

    Returns bonded terms by force class, the direct and reciprocal parts of
    the NonbondedForce, and the total. With ``zero_charges`` the nonbonded
    part is Lennard-Jones only.
    """
    import openmm
    from openmm import app, unit

    _, _, system = calc._build_system(ffxml)
    positions = app.GromacsGroFile(gro).getPositions()
    groups = {}
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
        groups[type(force).__name__] = i
        if isinstance(force, openmm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(31)
            groups["recip"] = 31
            if zero_charges:
                for k in range(force.getNumParticles()):
                    _, sigma, eps = force.getParticleParameters(k)
                    force.setParticleParameters(k, 0.0, sigma, eps)
                for k in range(force.getNumExceptions()):
                    a, b, _, sigma, eps = force.getExceptionParameters(k)
                    force.setExceptionParameters(k, a, b, 0.0, sigma, eps)
    context = openmm.Context(
        system, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions)

    def energy(**kwargs):
        return context.getState(getEnergy=True, **kwargs).getPotentialEnergy(
        ).value_in_unit(unit.kilojoule_per_mole)

    out = {name: energy(groups={g}) for name, g in groups.items()}
    out["total"] = energy()
    return out


def _gromacs_energies(calc, ffxml, workdir):
    """
    Energy terms GROMACS reports for a zero-step run of ``calc`` on the
    exported .top / .gro, in kJ/mol, read back with ``gmx dump -e``.
    Also returns the path of the .gro the run used.
    """
    import subprocess

    top, gro = calc._export_inputs(ffxml, workdir)
    options = calc.mdp_options("prod")
    options.update({
        "nsteps": "0", "tcoupl": "no", "pcoupl": "no", "gen-vel": "no",
        "continuation": "no", "nstcalcenergy": "1", "nstenergy": "1",
        "nstxout-compressed": "0", "nstlist": "1",
    })
    for key in gmx._PCOUPL_KEYS:
        options.pop(key, None)
    mdp = os.path.join(workdir, "sp.mdp")
    with open(mdp, "w") as handle:
        handle.write(gmx._format_mdp(options))
    deffnm = os.path.join(workdir, "sp")
    env = dict(os.environ, GMX_MAXBACKUP="-1")
    subprocess.run(
        calc.grompp_command(mdp, gro, top, f"{deffnm}.tpr"),
        check=True, env=env, capture_output=True,
    )
    subprocess.run(
        calc.mdrun_command(deffnm) + ["-ntmpi", "1", "-ntomp", "1"],
        check=True, env=env, capture_output=True,
    )
    dump = subprocess.run(
        [calc._resolve_execution()[0]["gmx_bin"], "dump", "-e", f"{deffnm}.edr"],
        check=True, env=env, capture_output=True, text=True,
    ).stdout
    import re

    terms = {}
    for line in dump.splitlines():
        match = re.match(r"\s{2,}(\S.*?\S)\s{2,}(-?\d\.\d+e[+-]\d+)\s*$", line)
        if match and match.group(1) not in terms:   # first frame = step 0
            terms[match.group(1)] = float(match.group(2))
    return terms, gro


@pytest.mark.gmx
@pytest.mark.skipif(shutil.which("gmx") is None, reason="gmx binary not on PATH")
@pytest.mark.parametrize("method", ["PME", "LJPME"])
@pytest.mark.parametrize("dispcorr", [False, True])
def test_single_point_energy_matches_openmm(method, dispcorr, tmp_path):
    """
    OpenMM and GROMACS agree on the potential energy of the exported system.

    Measured on this fixture (10 atoms, 4 nm box, rc = 1.2 nm), both codes at
    the 3-decimal .gro coordinates, in kJ/mol:

    - Bond / Angle / Dihedral agree to < 1e-4: same parameters, same
      coordinates.
    - Lennard-Jones with a plain cutoff (PME) agrees to < 1e-4 (3.9686 both),
      the dispersion correction to 1e-4 (-0.0028 vs -0.0029).
    - Coulomb (direct + reciprocal + 1-4) differs by ~2e-3 (-55.463 vs
      -55.464, 3e-5 relative): same alpha (ewald-rtol matched), but a 27^3
      order-5 mesh in OpenMM against 28^3 order-6 in GROMACS.
    - Lennard-Jones under LJPME differs by ~4e-3 (3.9642 vs 3.9686): GROMACS
      applies the potential shift OpenMM does not, and the dispersion mesh is
      14^3 in OpenMM against 28^3 (the Coulomb grid) in GROMACS. OpenMM
      ignores the dispersion correction under LJPME; GROMACS's is ~0 here.

    The tolerances below are these differences with a margin.
    """
    calc = GMXCalculator(
        PDB, nonbondedmethod=method, dispcorr=dispcorr, useHbondConstraint=False,
        rcut_nm=1.2, md_log="none",
    )
    gmx_terms, gro = _gromacs_energies(calc, FFXML, str(tmp_path))
    omm = _openmm_energies(calc, FFXML, gro)
    omm_lj = _openmm_energies(calc, FFXML, gro, zero_charges=True)

    # bonded: exact up to coordinate precision
    assert gmx_terms["Bond"] == pytest.approx(omm["HarmonicBondForce"], abs=1e-3)
    assert gmx_terms["Angle"] == pytest.approx(omm["HarmonicAngleForce"], abs=1e-3)
    assert gmx_terms["Proper Dih."] == pytest.approx(omm["PeriodicTorsionForce"], abs=1e-3)

    # Lennard-Jones (1-4 included), dispersion correction included on both sides
    gmx_lj = gmx_terms["LJ-14"] + gmx_terms["LJ (SR)"] + gmx_terms.get("LJ recip.", 0.0) \
        + gmx_terms.get("Disper. corr.", 0.0)
    omm_lj_total = omm_lj["NonbondedForce"] + omm_lj["recip"]
    lj_tolerance = 2e-2 if method == "LJPME" else 2e-3
    assert gmx_lj == pytest.approx(omm_lj_total, abs=lj_tolerance)
    if method == "PME" and dispcorr:
        assert gmx_terms["Disper. corr."] == pytest.approx(
            omm_lj["NonbondedForce"] - _openmm_energies(
                GMXCalculator(PDB, nonbondedmethod=method, dispcorr=False,
                              useHbondConstraint=False, rcut_nm=1.2),
                FFXML, gro, zero_charges=True,
            )["NonbondedForce"], abs=1e-3,
        )

    # Coulomb (1-4, direct, reciprocal): same alpha, slightly different mesh
    gmx_coulomb = gmx_terms["Coulomb-14"] + gmx_terms["Coulomb (SR)"] + gmx_terms["Coul. recip."]
    omm_coulomb = omm["total"] - omm_lj["total"]
    assert gmx_coulomb == pytest.approx(omm_coulomb, rel=1e-3)

    assert gmx_terms["Potential"] == pytest.approx(omm["total"], abs=1e-2)
