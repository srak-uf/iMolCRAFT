"""Unit tests for OpenMMCalculator and the shared annealing / logging parts"""
import os
import sys

import pytest
from openmm import unit

from imolcraft.calculator import _mdcommon, omm

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def test_anneal_schedule_expands_the_corners_into_legs():
    """N temperature points and N-1 leg step counts become (T_from, T_to, nsteps)"""
    assert _mdcommon._anneal_schedule([100, 1000, 300], [50000, 30000]) == [
        (100.0, 1000.0, 50000),
        (1000.0, 300.0, 30000),
    ]


def test_anneal_schedule_without_a_schedule_is_empty():
    """No annealing spec means zero legs, i.e. nothing happens"""
    assert _mdcommon._anneal_schedule(None, None) == []
    assert _mdcommon._anneal_schedule([], []) == []
    assert _mdcommon._anneal_schedule([300, 400], None) == []


@pytest.mark.parametrize("anneal_T, anneal_steps", [
    (300.0, [1000]),
    ([300, 400], 1000),
    ("300,400", [1000]),
])
def test_anneal_schedule_rejects_a_bare_value(anneal_T, anneal_steps):
    """Temperatures and step counts are lists; a bare number is not silently wrapped"""
    with pytest.raises(TypeError, match="must be a list or a tuple"):
        _mdcommon._anneal_schedule(anneal_T, anneal_steps)


def test_anneal_schedule_takes_tuples_too():
    """A tuple, as used when calling from Python, is accepted as a list too"""
    assert _mdcommon._anneal_schedule((100, 1000, 300), (50000, 30000)) == [
        (100.0, 1000.0, 50000),
        (1000.0, 300.0, 30000),
    ]


@pytest.mark.parametrize("anneal_T, anneal_steps, match", [
    ([300], [1000], "at least two"),
    ([300, 400, 300], [1000], "one entry fewer"),
    ([300, 400], [-1], "negative"),
])
def test_anneal_schedule_rejects_an_inconsistent_schedule(
    anneal_T, anneal_steps, match
):
    with pytest.raises(ValueError, match=match):
        _mdcommon._anneal_schedule(anneal_T, anneal_steps)


class _FakeIntegrator:
    def __init__(self):
        self.temperatures = []

    def setTemperature(self, T):
        self.temperatures.append(T.value_in_unit(unit.kelvin))


class _FakeSimulation:
    def __init__(self):
        self.steps = []

    def step(self, n):
        self.steps.append(n)


def test_ramp_temperature_lands_exactly_on_the_target():
    """A leg ends exactly at T_to and the total MD step count matches"""
    integrator, simulation = _FakeIntegrator(), _FakeSimulation()
    omm._ramp_temperature(simulation, integrator, 300.0, 400.0, 1000, 250)

    assert integrator.temperatures == [325.0, 350.0, 375.0, 400.0]
    assert simulation.steps == [250, 250, 250, 250]
    assert sum(simulation.steps) == 1000


def test_ramp_temperature_keeps_the_last_chunk_short():
    """Even when interval does not divide evenly, the remainder keeps the total"""
    integrator, simulation = _FakeIntegrator(), _FakeSimulation()
    omm._ramp_temperature(simulation, integrator, 300.0, 400.0, 250, 100)

    assert sum(simulation.steps) == 250
    assert simulation.steps[-1] == 50
    assert integrator.temperatures[-1] == pytest.approx(400.0)


def test_ramp_temperature_skips_a_zero_length_leg():
    """A zero-step leg runs no MD and does not touch the set temperature"""
    integrator, simulation = _FakeIntegrator(), _FakeSimulation()
    omm._ramp_temperature(simulation, integrator, 300.0, 400.0, 0, 100)

    assert integrator.temperatures == []
    assert simulation.steps == []


def test_open_md_log_none_yields_no_stream():
    with _mdcommon._open_md_log("none", None, "sample_0.xtc") as stream:
        assert stream is None
    with _mdcommon._open_md_log(None, None, "sample_0.xtc") as stream:
        assert stream is None


def test_open_md_log_stdout_yields_stdout():
    with _mdcommon._open_md_log("stdout", None, "sample_0.xtc") as stream:
        assert stream is sys.stdout


def test_open_md_log_file_names_the_log_after_the_trajectory(tmp_path, monkeypatch):
    """The default log name derives from the trajectory, so replicas do not clash"""
    monkeypatch.chdir(tmp_path)
    with _mdcommon._open_md_log("file", None, "sample_0.xtc") as stream:
        stream.write("hello")
    assert (tmp_path / "mdlogs" / "sample_0.log").read_text() == "hello"


def test_open_md_log_file_honours_an_explicit_path(tmp_path):
    path = tmp_path / "logs" / "omm.log"
    with _mdcommon._open_md_log("file", str(path), "sample_0.xtc") as stream:
        stream.write("hello")
    assert path.read_text() == "hello"


def test_open_md_log_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="md_log"):
        with _mdcommon._open_md_log("quiet", None, "sample_0.xtc"):
            pass


def test_calculator_records_every_setting_including_the_defaults():
    """to_dict writes omitted settings with their defaults, so runs stay reproducible"""
    calc = omm.OpenMMCalculator("start.pdb", temperature_K=350.0, prod_steps=1000)
    record = calc.to_dict()

    assert record["init_structure"] == "start.pdb"
    assert record["temperature_K"] == 350.0
    assert record["prod_steps"] == 1000
    # The untouched settings are all included as well
    assert set(record) == {"init_structure"} | set(omm.OpenMMCalculator.SETTINGS)
    assert record["dt_fs"] == omm.OpenMMCalculator.SETTINGS["dt_fs"]


def test_calculator_round_trips_through_a_record():
    calc = omm.OpenMMCalculator(
        "start.pdb", temperature_K=350.0, ensemble="isonpt",
        anneal_T=[300, 400, 300], anneal_steps=[10, 20], md_log="none",
    )
    assert omm.OpenMMCalculator.from_dict(calc.to_dict()).to_dict() == calc.to_dict()


def test_calculator_rejects_an_unknown_setting():
    """A misspelled setting name is not silently ignored"""
    with pytest.raises(TypeError, match="unknown MD settings: temperature"):
        omm.OpenMMCalculator("start.pdb", temperature=350.0)


def test_calculator_rejects_an_unknown_ensemble():
    with pytest.raises(ValueError, match="Invalid ensemble"):
        omm.OpenMMCalculator("start.pdb", ensemble="npt")


def test_calculator_expands_the_annealing_schedule_on_construction():
    """A broken schedule fails at construction time, without waiting for run"""
    calc = omm.OpenMMCalculator("start.pdb", anneal_T=[300, 400], anneal_steps=[50])
    assert calc.anneal_legs == [(300.0, 400.0, 50)]

    with pytest.raises(ValueError, match="one entry fewer"):
        omm.OpenMMCalculator("start.pdb", anneal_T=[300, 400, 300], anneal_steps=[50])


@pytest.mark.parametrize("name, expected", [
    ("PME", _mdcommon.NONBONDED_METHODS["PME"]),
    ("LJPME", _mdcommon.NONBONDED_METHODS["LJPME"]),
])
def test_resolve_nonbondedmethod(name, expected):
    assert _mdcommon.resolve_nonbondedmethod(name) is expected
    # Passing an already resolved constant through again is a no-op
    assert _mdcommon.resolve_nonbondedmethod(expected) is expected


def test_resolve_nonbondedmethod_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Invalid nonbonded method"):
        _mdcommon.resolve_nonbondedmethod("cutoff")


@pytest.mark.parametrize(
    "ensemble, expected",
    [
        ("nvt", type(None)),
        ("nve", type(None)),
        ("isonpt", "MonteCarloBarostat"),
        ("anisonpt", "MonteCarloAnisotropicBarostat"),
        ("trinpt", "MonteCarloFlexibleBarostat"),
    ],
)
def test_make_barostat(ensemble, expected):
    barostat = omm._make_barostat(ensemble, 300.0)
    if expected is type(None):
        assert barostat is None
    else:
        assert type(barostat).__name__ == expected


@pytest.mark.parametrize("ensemble", ["isonpt", "anisonpt", "trinpt"])
def test_make_barostat_sets_the_requested_pressure(ensemble):
    """pressure_bar reaches the barostat, on every axis of the anisotropic one"""
    barostat = omm._make_barostat(ensemble, 300.0, 50.0)
    pressure = barostat.getDefaultPressure()
    if ensemble == "anisonpt":
        assert [p for p in pressure.value_in_unit(unit.bar)] == [50.0, 50.0, 50.0]
    else:
        assert pressure.value_in_unit(unit.bar) == 50.0
    assert barostat.getDefaultTemperature().value_in_unit(unit.kelvin) == 300.0


def test_make_barostat_defaults_to_one_bar():
    """Called as before pressure_bar existed, the barostat is the 1 bar one"""
    assert omm._make_barostat("isonpt", 300.0).getDefaultPressure().value_in_unit(
        unit.bar
    ) == 1.0
    assert omm._make_barostat("nvt", 300.0, 50.0) is None
    assert omm._make_barostat("nve", 300.0, 50.0) is None


def test_pressure_bar_is_a_setting_with_the_gromacs_default():
    assert omm.OpenMMCalculator.SETTINGS["pressure_bar"] == 1.0
    assert omm.OpenMMCalculator("start.pdb", pressure_bar=50.0).pressure_bar == 50.0


@pytest.mark.parametrize("pressure_bar, expected", [(None, 1.0), (50.0, 50.0)])
def test_build_simulation_puts_pressure_bar_on_the_barostat(pressure_bar, expected):
    """The System of an NPT run carries one barostat at the requested pressure"""
    openmm = pytest.importorskip("openmm")
    pdb = os.path.join(DATA, "supercell_bonds.pdb")
    ffxml = os.path.join(DATA, "supercell_bonds_novsite.xml")
    settings = {} if pressure_bar is None else {"pressure_bar": pressure_bar}
    calc = omm.OpenMMCalculator(pdb, ensemble="isonpt", **settings)
    simulation, _, _ = calc._build_simulation(ffxml, lambda message: None)
    barostats = [
        f for f in simulation.system.getForces()
        if isinstance(f, openmm.MonteCarloBarostat)
    ]
    assert len(barostats) == 1
    assert barostats[0].getDefaultPressure().value_in_unit(unit.bar) == expected
    assert barostats[0].getDefaultTemperature().value_in_unit(unit.kelvin) == 300.0


def test_modules_export_the_calculator_and_the_shared_names_only():
    """__all__ keeps openmm, app, unit, os and sys out of imolcraft.calculator"""
    import imolcraft.calculator

    assert omm.__all__ == ["OpenMMCalculator"]
    assert set(_mdcommon.__all__) == {
        "VALID_ENSEMBLES", "MD_LOG_MODES", "NONBONDED_METHODS", "resolve_nonbondedmethod",
    }
    assert not hasattr(imolcraft.calculator, "openmm")
    assert not hasattr(imolcraft.calculator, "app")
    assert not hasattr(imolcraft.calculator, "unit")
    assert not hasattr(imolcraft.calculator, "contextlib")


def test_omm_and_gmx_share_the_common_parts_without_importing_each_other():
    """omm and gmx are peers: both take the shared names from _mdcommon"""
    import imolcraft.calculator.gmx as gmx

    assert omm.VALID_ENSEMBLES is _mdcommon.VALID_ENSEMBLES
    assert gmx.VALID_ENSEMBLES is _mdcommon.VALID_ENSEMBLES
    assert omm._anneal_schedule is _mdcommon._anneal_schedule
    assert gmx._anneal_schedule is _mdcommon._anneal_schedule


def test_valid_ensembles_cover_the_barostat_flavours():
    assert "nvt" in _mdcommon.VALID_ENSEMBLES and "isonpt" in _mdcommon.VALID_ENSEMBLES


def test_calculator_settings_are_named_as_the_sampling_section_is():
    """The setting names match the YAML sampling keys, so no translation table"""
    sampling = {
        "init_structure": "start.pdb",
        "temperature_K": 233.15,
        "rcut_nm": 1.2,
        "ensemble": "nvt",
        "nonbondedmethod": "LJPME",
        "dispcorr": True,
        "dt_fs": 2.0,
        "nstxout": 20,
        "relax_steps": 100,
        "prod_steps": 200,
        "anneal_T": [233.15, 400.0],
        "anneal_steps": [50],
        "anneal_interval": 25,
        # Not an MD setting, so it is not picked up
        "neff": 30,
        # The barostat set point is one, so it is
        "pressure_bar": 2.0,
    }
    settings = {k: v for k, v in sampling.items() if k in omm.OpenMMCalculator.SETTINGS}
    assert "neff" not in settings and settings["pressure_bar"] == 2.0

    calc = omm.OpenMMCalculator(sampling["init_structure"], **settings)
    assert calc.temperature_K == 233.15
    assert calc.dt_fs == 2.0
    assert calc.dispcorr is True
    assert calc.pressure_bar == 2.0
    assert calc.anneal_legs == [(233.15, 400.0, 50)]
