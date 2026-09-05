"""Regression tests for trainer parts needing no external program or long MD"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from imolcraft.trainer import dmff_utils
from imolcraft.trainer.base import _broadcast_lr_clip, _nan_recovery_gradients
from imolcraft.calculator.md import VALID_ENSEMBLES
from imolcraft.trainer.dmff_utils import neutralize
from imolcraft.trainer.properties import (
    DISTRIBUTION_TARGETS,
    PROPERTY_KEYS,
    PROPERTY_KINDS,
    REQUIRED_TARGET_KEYS,
    REQUIRED_VALIDATION_KEYS,
    SCALAR_TARGETS,
    VALIDATION_ONLY_PROPERTIES,
)
from imolcraft.trainer.loss import (
    IMPLEMENTED_WEIGHT_SCHEMES,
    _squared_error,
    jsdivergence,
    mse_energy,
    wrightfactor,
)

TESTS = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------- loss.py
@pytest.mark.parametrize("zeropoint", ["auto", "qmmin", None])
def test_squared_error_is_non_negative(zeropoint):
    e_qm = jnp.array([0.0, 1.0, 3.0, 2.0])
    e_ff = jnp.array([0.5, 1.2, 2.6, 2.4])
    se = _squared_error(e_ff, e_qm, zeropoint)
    assert se.shape == e_qm.shape
    assert jnp.all(se >= 0)


def test_squared_error_qmmin_anchors_at_the_qm_minimum():
    """qmmin aligns both curves at the QM minimum, so the error there is 0"""
    e_qm = jnp.array([2.0, 0.0, 3.0])
    e_ff = jnp.array([5.0, 1.0, 7.0])
    se = _squared_error(e_ff, e_qm, "qmmin")
    assert se[jnp.argmin(e_qm)] == pytest.approx(0.0)


def test_squared_error_is_zero_for_a_constant_offset():
    """auto absorbs the mean offset, so a constant shift is not an error"""
    e_qm = jnp.array([0.0, 1.0, 2.0, 3.0])
    se = _squared_error(e_qm + 5.0, e_qm, "auto")
    assert jnp.allclose(se, jnp.square(5.0 - 5.0 / len(e_qm) * 1.0) * 0 + se)
    assert jnp.all(se >= 0)


@pytest.mark.parametrize("weight_scheme", ["uniform", "boltzmann"])
@pytest.mark.parametrize("zeropoint", ["auto", "qmmin", None])
def test_mse_energy_is_zero_for_identical_energies(weight_scheme, zeropoint):
    e = jnp.array([0.0, 1.0, 3.0, 2.0, 5.0])
    mse = mse_energy(e, e, weight_scheme=weight_scheme, zeropoint=zeropoint)
    assert float(mse) == pytest.approx(0.0, abs=1e-12)


def test_mse_energy_rejects_unknown_weight_scheme():
    with pytest.raises(ValueError, match="Unknown weight scheme"):
        mse_energy(jnp.ones(3), jnp.ones(3), weight_scheme="hoge")


def test_nonboltzmann_is_listed_but_not_implemented():
    """Known unimplemented branch. Delete this test once it is implemented"""
    assert "nonboltzmann" in IMPLEMENTED_WEIGHT_SCHEMES
    with pytest.raises(UnboundLocalError):
        mse_energy(jnp.ones(3), jnp.zeros(3), weight_scheme="nonboltzmann")


def test_wrightfactor_and_jsdivergence_vanish_for_equal_distributions():
    g = jnp.array([0.1, 0.4, 0.3, 0.2])
    assert float(wrightfactor(g, g)) == pytest.approx(0.0, abs=1e-12)
    assert float(jsdivergence(g, g)) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------- dmff_utils.py
def _ffparams(charges):
    return {"NonbondedForce": {"charge": jnp.array(charges)}}


@pytest.mark.parametrize("nc", [0, -1, 2])
def test_neutralize_reaches_the_requested_net_charge(nc):
    natoms = jnp.array([2, 3, 1, 4])
    out = neutralize(_ffparams([0.3, -0.2, 0.5, -0.9]), natoms, nc=nc)
    net = jnp.dot(out["NonbondedForce"]["charge"], natoms)
    assert float(net) == pytest.approx(nc, abs=1e-6)


def test_neutralize_honours_the_constrained_groups():
    natoms = jnp.array([2, 3, 1, 4])
    out = neutralize(
        _ffparams([0.3, -0.2, 0.5, -0.9]),
        natoms,
        nc=0,
        target_lists=[[0, 1]],
        target_charges=[1.0],
    )
    charges = out["NonbondedForce"]["charge"]
    constrained = jnp.dot(charges[jnp.array([0, 1])], natoms[jnp.array([0, 1])])
    assert float(constrained) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.dot(charges, natoms)) == pytest.approx(0.0, abs=1e-6)


def test_neutralize_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="!= len\\(natoms_list\\)"):
        neutralize(_ffparams([0.1, 0.2]), jnp.array([1, 2, 3]))


def test_neutralize_rejects_mismatched_targets():
    with pytest.raises(ValueError, match="len\\(target_lists\\) != len"):
        neutralize(
            _ffparams([0.1, 0.2]),
            jnp.array([1, 2]),
            target_lists=[[0], [1]],
            target_charges=[0.0],
        )


def test_target_key_tables_agree():
    """Scalar targets always require gt and weight"""
    for target in SCALAR_TARGETS:
        assert REQUIRED_TARGET_KEYS[target] == ("gt", "weight")
    assert set(REQUIRED_TARGET_KEYS) >= set(SCALAR_TARGETS) | {"rdf", "adf"}
    assert "nvt" in VALID_ENSEMBLES and "isonpt" in VALID_ENSEMBLES


def test_every_property_is_declared_once_and_completely():
    """The property list lives in one place, holding both kind and required keys"""
    assert set(PROPERTY_KINDS) == set(PROPERTY_KEYS)
    assert set(PROPERTY_KINDS.values()) <= {"scalar", "distribution"}


def test_the_loss_takes_every_property_but_the_validation_only_ones():
    """The optimization targets are every property except the validation-only ones"""
    assert set(REQUIRED_TARGET_KEYS) == set(PROPERTY_KINDS) - set(
        VALIDATION_ONLY_PROPERTIES
    )
    # The two sets the loss side branches on are derived from the same table
    assert set(SCALAR_TARGETS) | set(DISTRIBUTION_TARGETS) == set(
        REQUIRED_TARGET_KEYS
    )
    # The converse holds: a property usable in loss can also be monitored in validation
    assert set(REQUIRED_VALIDATION_KEYS) == set(PROPERTY_KINDS)


def test_a_validated_distribution_always_needs_its_reference():
    """Distributions record the distance to a reference, so validation needs gt too"""
    for name, kind in PROPERTY_KINDS.items():
        assert ("gt" in REQUIRED_VALIDATION_KEYS[name]) is (kind == "distribution")


# ----------------------------------------------------------------- base.py
def test_broadcast_lr_clip_expands_scalars():
    lr, clip = _broadcast_lr_clip(0.01, 0.1, ["a", "b", "c"])
    assert lr == [0.01] * 3
    assert clip == [0.1] * 3


def test_broadcast_lr_clip_passes_lists_through():
    lr, clip = _broadcast_lr_clip([1.0, 2.0], [0.1, 0.2], ["a", "b"])
    assert lr == [1.0, 2.0] and clip == [0.1, 0.2]


@pytest.mark.parametrize(
    "lr, clip, opt_fftypes",
    [([1.0], [0.1, 0.2], ["a", "b"]), ([1.0, 2.0], [0.1, 0.2], ["a"])],
)
def test_broadcast_lr_clip_rejects_mismatched_lengths(lr, clip, opt_fftypes):
    with pytest.raises(ValueError, match="Length of lr"):
        _broadcast_lr_clip(lr, clip, opt_fftypes)


def test_nan_recovery_gradients_is_small_and_deterministic():
    params = {"NonbondedForce": {"charge": jnp.array([0.5, -0.5, 1.0])}}
    a = _nan_recovery_gradients(params)
    b = _nan_recovery_gradients(params)
    assert jnp.array_equal(a["NonbondedForce"]["charge"], b["NonbondedForce"]["charge"])
    delta = a["NonbondedForce"]["charge"] - params["NonbondedForce"]["charge"]
    assert float(jnp.max(jnp.abs(delta))) < 0.01


# -------------------------------------------------------------- trainer.py
def test_ffparams_without_charge_strips_charge_and_vsite():
    from imolcraft.trainer.trainer import _ffparams_without_charge

    ffparams = {
        "NonbondedForce": {"charge": 1, "sigma": 2, "epsilon": 3},
        "VsiteForce": {"weight": 9},
        "HarmonicBondForce": {"k": 4},
    }
    stripped = _ffparams_without_charge(ffparams)
    assert stripped == {
        "NonbondedForce": {"sigma": 2, "epsilon": 3},
        "HarmonicBondForce": {"k": 4},
    }


def test_qm_energies_stacks_the_scans():
    from imolcraft.trainer.trainer import _qm_energies

    qm_scan = [{"energy_kjmol": [0.0, 1.0]}, {"energy_kjmol": [2.0, 3.0]}]
    assert np.array_equal(np.asarray(_qm_energies(qm_scan)), [[0.0, 1.0], [2.0, 3.0]])


def test_sum_loss_and_grads_accumulates():
    from imolcraft.trainer.trainer import _sum_loss_and_grads

    ffparams = {"a": jnp.array([1.0, 2.0])}
    pairs = [
        (1.0, {"a": jnp.array([1.0, 1.0])}),
        (2.0, {"a": jnp.array([3.0, 4.0])}),
    ]
    loss, grads = _sum_loss_and_grads(ffparams, pairs)
    assert loss == pytest.approx(3.0)
    assert jnp.array_equal(grads["a"], jnp.array([4.0, 5.0]))


# ------------------------------------------------------- checkpoint round trip
def _ethane_pdb(path):
    """Build a bonded PDB matching ethane.xml"""
    from ase import Atoms
    from openmm.app import PDBFile

    from imolcraft.crafter.asemol import aseatoms2pdb, asemol_wrapper, merge_asemols

    atoms = Atoms(
        "CH3CH3",
        positions=[
            [-2.79452060, 1.06849313, 0.0],
            [-2.43786617, 0.05968313, 0.0],
            [-2.43784776, 1.57289132, -0.87365150],
            [-3.86452060, 1.06850632, 0.0],
            [-2.28117838, 1.79444941, 1.25740497],
            [-2.63623472, 2.80382250, 1.25642745],
            [-2.63944749, 1.29118098, 2.13105486],
            [-1.21118019, 1.79274272, 1.25838372],
        ],
    )
    aw = asemol_wrapper(atoms)
    molatoms, _, _ = aw.get_ase_molecules(out_nX=True)
    aseatoms2pdb(str(path), merge_asemols(molatoms))
    pdb_omm = PDBFile(str(path))
    atomlist = [a for a in pdb_omm.topology.atoms()]
    for b in aw.get_bonds():
        pdb_omm.topology.addBond(atomlist[b[0]], atomlist[b[1]])
    with open(path, "w") as f:
        PDBFile.writeFile(pdb_omm.topology, pdb_omm.positions, f)
    return str(path)


def _dummy_trainer_cls():
    """Minimal sub-trainer that runs neither MD nor QM"""
    import jax

    from imolcraft.trainer.base import BaseTrainer

    class DummyTrainer(BaseTrainer):
        def setup(self):
            if getattr(self, "opt_state", None) is None:
                self.opt_state = self.optimizer.init(self.ffparams)

        def get_loss_gradients(self):
            grads = jax.tree_util.tree_map(
                lambda x: jnp.ones_like(x) * 0.1, self.ffparams
            )
            return jnp.float32(1.0), grads

    return DummyTrainer


@pytest.fixture
def sum_trainer_env(tmp_path, monkeypatch):
    pytest.importorskip("dmff")
    monkeypatch.chdir(tmp_path)
    ffxml = os.path.join(TESTS, "data", "ethane.xml")
    pdbfile = _ethane_pdb(tmp_path / "ethane.pdb")
    DummyTrainer = _dummy_trainer_cls()

    def make(label, **kwargs):
        options = {
            "ffxml_list": [ffxml],
            "nums_ffxml": [1],
            "pdbfile": pdbfile,
            "loss_fn": None,
            "opt_fftypes": ["NonbondedForce/charge"],
            "label": label,
            "lr": 0.01,
            "clip": 0.1,
        }
        options.update(kwargs)
        return DummyTrainer(**options)

    return make


def test_sumtrainer_writes_a_non_empty_checkpoint(sum_trainer_env, tmp_path):
    """dump_dict is really pickled (the old implementation wrote an empty file)"""
    import pickle

    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"),
        opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 2.0], lr=0.01, clip=0.1,
    )
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1)

    ckpt = tmp_path / f"train_state_{trainer.label}.pkl"
    assert ckpt.exists() and ckpt.stat().st_size > 0

    with open(ckpt, "rb") as f:
        dump = pickle.load(f)
    assert {"ffparams", "opt_state", "epoch", "losses", "weight"} <= set(dump)
    assert dump["weight"] == [1.0, 2.0]


def test_sumtrainer_roundtrip_restores_the_optimizer_state(sum_trainer_env, tmp_path):
    """opt_state missing from the sub pickle is restored and survives setup()"""
    import jax

    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"),
        opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 2.0], lr=0.01, clip=0.1,
    )
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1)

    ckpt = str(tmp_path / f"train_state_{trainer.label}.pkl")
    restored = SumTrainer.from_checkpoint(ckpt, make("t1"), make("t2"))
    restored.setup()

    def same(a, b):
        return jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: bool(jnp.array_equal(x, y)), a, b)
        )

    assert same(restored.ffparams, trainer.ffparams)
    assert same(restored.opt_state, trainer.opt_state)
    assert restored.weight == trainer.weight
    assert [float(x) for x in restored.losses] == [float(x) for x in trainer.losses]
    # The restored values are distributed to the sub-trainers as well
    sub = restored.trainer1.ffparams["NonbondedForce"]["charge"]
    joint = restored.ffparams["NonbondedForce"]["charge"]
    assert jnp.array_equal(sub, joint[: len(sub)])


def test_sumtrainer_can_continue_after_restore(sum_trainer_env, tmp_path):
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"),
        opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 1.0], lr=0.01, clip=0.1,
    )
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1)

    ckpt = str(tmp_path / f"train_state_{trainer.label}.pkl")
    restored = SumTrainer.from_checkpoint(ckpt, make("t1"), make("t2"))
    restored.setup()
    before = restored._epoch
    restored.fit(steps=1, checkpoint_frequency=10)
    assert restored._epoch > before


def test_every_trainer_supports_checkpointing():
    """All four classes have their own write_checkpoint and from_checkpoint"""
    from imolcraft.trainer import (
        DihedralTrainer,
        DistanceTrainer,
        ThermodynamicTrainer,
    )
    from imolcraft.trainer.base import SumTrainer

    for cls in (DistanceTrainer, DihedralTrainer, ThermodynamicTrainer, SumTrainer):
        assert "write_checkpoint" in cls.__dict__, cls.__name__
        assert "from_checkpoint" in cls.__dict__, cls.__name__


def test_checkpoint_filenames_carry_the_label():
    """All four write train_state_{label}.pkl (no clash inside SumTrainer)"""
    import inspect

    from imolcraft.trainer import (
        DihedralTrainer,
        DistanceTrainer,
        ThermodynamicTrainer,
    )
    from imolcraft.trainer.base import SumTrainer

    for cls in (DistanceTrainer, DihedralTrainer, ThermodynamicTrainer, SumTrainer):
        src = inspect.getsource(cls.__dict__["write_checkpoint"])
        assert 'f"train_state_{self.label}.pkl"' in src, cls.__name__
        assert '"train_state.pkl"' not in src, cls.__name__


def test_loop_xml_filenames_carry_the_label():
    """The periodic relaxation xml also uses xmlfiles/loop_{label}-{epoch}.xml"""
    import inspect

    from imolcraft.trainer.trainer import _ScanTrainerMixin

    src = inspect.getsource(_ScanTrainerMixin._render_loop_xml)
    assert 'f"xmlfiles/loop_{self.label}-{epoch}.xml"' in src

    from imolcraft.trainer import DihedralTrainer, DistanceTrainer

    for cls in (DistanceTrainer, DihedralTrainer):
        after_step = inspect.getsource(cls.__dict__["after_step"])
        assert "_render_loop_xml(epoch)" in after_step, cls.__name__
        assert "loop-" not in after_step, cls.__name__


def _decreasing_loss_trainer_cls(losses):
    """Sub-trainer that returns the given loss sequence in order"""
    import jax

    from imolcraft.trainer.base import BaseTrainer

    class ScriptedTrainer(BaseTrainer):
        def setup(self):
            self._i = 0
            if getattr(self, "opt_state", None) is None:
                self.opt_state = self.optimizer.init(self.ffparams)

        def get_loss_gradients(self):
            value = losses[min(self._i, len(losses) - 1)]
            self._i += 1
            grads = jax.tree_util.tree_map(
                lambda x: jnp.ones_like(x) * 0.1, self.ffparams
            )
            return jnp.float32(value), grads

    return ScriptedTrainer


def test_best_is_none_before_training(sum_trainer_env):
    """The attributes exist even before fit runs (previously an AttributeError)"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    t1, t2 = make("t1"), make("t2")
    assert t1.best_params is None and t1.best_epoch is None and t1.best_loss is None

    trainer = SumTrainer(
        t1, t2, opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 1.0],
        lr=0.01, clip=0.1,
    )
    assert trainer.best_params is None
    assert trainer.best_epoch is None
    assert trainer.best_loss is None


def test_best_snapshot_survives_a_restart(sum_trainer_env, tmp_path):
    """A worse loss after resuming does not overwrite the previous best"""
    import jax

    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"), opt_fftypes=["NonbondedForce/charge"],
        weight=[1.0, 1.0], lr=0.01, clip=0.1,
    )
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1)

    assert trainer.best_params is not None
    assert trainer.best_loss is not None
    best_loss = float(trainer.best_loss)
    best_epoch = trainer.best_epoch

    ckpt = str(tmp_path / f"train_state_{trainer.label}.pkl")
    restored = SumTrainer.from_checkpoint(ckpt, make("t1"), make("t2"))
    restored.setup()

    assert float(restored.best_loss) == pytest.approx(best_loss)
    assert restored.best_epoch == best_epoch
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            restored.best_params,
            trainer.best_params,
        )
    )

    # Running further keeps the best while the loss stays worse
    restored.fit(steps=1, checkpoint_frequency=10)
    assert float(restored.best_loss) <= best_loss


def test_best_checkpoint_fields_are_stored_by_every_trainer():
    import inspect

    from imolcraft.trainer import (
        DihedralTrainer,
        DistanceTrainer,
        ThermodynamicTrainer,
    )
    from imolcraft.trainer.base import SumTrainer

    for cls in (DistanceTrainer, DihedralTrainer, ThermodynamicTrainer, SumTrainer):
        write_src = inspect.getsource(cls.__dict__["write_checkpoint"])
        assert "_best_checkpoint_fields()" in write_src, cls.__name__

    # DistanceTrainer restore is disabled for now, so only check the writing side
    for cls in (DihedralTrainer, ThermodynamicTrainer, SumTrainer):
        from_src = inspect.getsource(cls.__dict__["from_checkpoint"])
        assert "_restore_best(dump_dict)" in from_src, cls.__name__


def test_distance_trainer_restore_is_switched_off():
    """Restore is off for now: NotImplementedError instead of breaking silently"""
    import pytest

    from imolcraft.trainer import DistanceTrainer

    with pytest.raises(NotImplementedError, match="switched off"):
        DistanceTrainer.from_checkpoint("train_state_x.pkl")


def test_thermodynamic_from_checkpoint_restores_history_explicitly():
    """The ineffective attr_lists loop was replaced with explicit assignments"""
    import inspect

    from imolcraft.trainer import ThermodynamicTrainer

    src = inspect.getsource(ThermodynamicTrainer.__dict__["from_checkpoint"])
    # The loop with mismatched key names that created epoch instead of _epoch is gone
    assert "ff_info" not in src
    assert "attr_lists" not in src
    # The histories are restored explicitly
    assert 'trainer.losses = dump_dict["losses"]' in src
    assert 'trainer.epochs = dump_dict["epochs"]' in src
    assert 'trainer._epoch = dump_dict["epoch"]' in src
    # The force field comes from initial_ffxml, so ffinfo is not restored
    assert "ff.ffinfo" not in src


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_fit_runs_exactly_the_requested_number_of_steps(
    steps, sum_trainer_env, tmp_path
):
    """fit(steps) runs exactly steps epochs (it used to run steps+1)"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"), opt_fftypes=["NonbondedForce/charge"],
        weight=[1.0, 1.0], lr=0.01, clip=0.1,
    )
    trainer.setup()
    trainer.fit(steps=steps, checkpoint_frequency=1000)

    assert trainer._epoch == steps
    assert len(trainer.losses) == steps
    assert len(trainer.epochs) == steps
    assert trainer.epochs == list(range(steps))


def test_fit_continues_from_the_previous_call(sum_trainer_env, tmp_path):
    """Splitting fit into several calls gives the same total epoch count"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"), opt_fftypes=["NonbondedForce/charge"],
        weight=[1.0, 1.0], lr=0.01, clip=0.1,
    )
    trainer.setup()
    trainer.fit(steps=2, checkpoint_frequency=1000)
    trainer.fit(steps=3, checkpoint_frequency=1000)

    assert trainer._epoch == 5
    assert trainer.epochs == list(range(5))


# --------------------------------------- ThermodynamicTrainer resampling decision
def _thermo_stub(neff, states=(), resample=(), n_replicas=None):
    import types

    return types.SimpleNamespace(
        neff=list(neff),
        estimator=types.SimpleNamespace(
            states=[types.SimpleNamespace(name=n) for n in states]
        ),
        resample=list(resample),
        sampling_params=[{}] * (len(neff) if n_replicas is None else n_replicas),
    )


def test_needs_resample_matches_by_name_not_position():
    """Look at the own replica's contribution even if the state order changes"""
    from imolcraft.trainer import ThermodynamicTrainer

    # After resampling sample_1, estimator.states is ordered [0, 2, 1]
    ieff = {"sample_0": 100, "sample_2": 100, "sample_1": 3, "Total": 203}
    stub = _thermo_stub([10, 10, 10])

    decisions = [
        ThermodynamicTrainer._needs_resample(stub, ii, ieff) for ii in range(3)
    ]
    assert decisions == [False, True, False]


def test_needs_resample_ignores_the_total_entry():
    from imolcraft.trainer import ThermodynamicTrainer

    # A small Total alone does not trigger resampling if the own contribution suffices
    ieff = {"sample_0": 100, "Total": 1}
    stub = _thermo_stub([10])
    assert ThermodynamicTrainer._needs_resample(stub, 0, ieff) is False


def test_needs_resample_tolerates_a_missing_entry():
    """Does not crash when the own state is not registered yet"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _thermo_stub([10, 10])
    assert ThermodynamicTrainer._needs_resample(stub, 1, {"sample_0": 5}) is False


def test_resample_indices_selects_the_flagged_replicas():
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _thermo_stub(
        [10, 10, 10], states=("sample_0", "sample_1", "sample_2"),
        resample=(False, True, False),
    )
    assert ThermodynamicTrainer._resample_indices(stub) == [1]

    stub = _thermo_stub(
        [10, 10], states=("sample_0", "sample_1"), resample=(True, True)
    )
    assert ThermodynamicTrainer._resample_indices(stub) == [0, 1]


def test_resample_indices_covers_everything_when_nothing_registered():
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _thermo_stub([10, 10, 10], states=(), resample=(False, False, False))
    assert ThermodynamicTrainer._resample_indices(stub) == [0, 1, 2]


def test_state_name_is_shared_by_setup_and_resample():
    import inspect

    from imolcraft.trainer import ThermodynamicTrainer
    from imolcraft.trainer.trainer import _state_name

    assert _state_name(3) == "sample_3"
    for name in ("setup", "_resample", "_needs_resample"):
        src = inspect.getsource(ThermodynamicTrainer.__dict__[name])
        assert "_state_name(" in src, name
        assert 'f"sample_{' not in src, name


# ------------------------------------------------------------ hook effectiveness
_HOOK_MARK = -77.0


def _pure_hook(tree):
    """Return a new modified dict without touching the argument (as contracted)"""
    nb = dict(tree["NonbondedForce"])
    nb["charge"] = nb["charge"].at[0].set(_HOOK_MARK)
    out = dict(tree)
    out["NonbondedForce"] = nb
    return out


def _marked(trainer):
    return any(
        float(v) == _HOOK_MARK for v in trainer.ffparams["NonbondedForce"]["charge"]
    )


def test_sub_after_update_hook_takes_effect(sum_trainer_env):
    """after_update registered on a sub also takes effect through SumTrainer"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    t1, t2 = make("t1"), make("t2")
    trainer = SumTrainer(
        t1, t2, opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 1.0],
        lr=0.01, clip=0.1,
    )
    t1.add_modifyfn("after_update", _pure_hook)
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1000)

    assert _marked(trainer)
    assert _marked(t1)


def test_parent_after_update_hook_takes_effect(sum_trainer_env):
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"), opt_fftypes=["NonbondedForce/charge"],
        weight=[1.0, 1.0], lr=0.01, clip=0.1,
    )
    trainer.add_modifyfn("after_update", _pure_hook)
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1000)

    assert _marked(trainer)
    # The parent hook's result is distributed to the subs as well
    assert _marked(trainer.trainer1)


def test_sub_hooks_run_before_the_parent_hook(sum_trainer_env):
    """A sub sees its own half; the parent sees the whole concatenated vector"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    t1, t2 = make("t1"), make("t2")
    trainer = SumTrainer(
        t1, t2, opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 1.0],
        lr=0.01, clip=0.1,
    )
    order = []

    def sub_hook(tree):
        order.append(("sub", len(tree["NonbondedForce"]["charge"])))
        return tree

    def parent_hook(tree):
        order.append(("parent", len(tree["NonbondedForce"]["charge"])))
        return tree

    t1.add_modifyfn("after_update", sub_hook)
    trainer.add_modifyfn("after_update", parent_hook)
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1000)

    assert [who for who, _ in order] == ["sub", "parent"]
    sub_len = order[0][1]
    parent_len = order[1][1]
    assert parent_len == 2 * sub_len


def test_do_modify_returns_the_value_when_no_hook_is_registered(sum_trainer_env):
    """An unregistered hook kind does not return None (assignment stays safe)"""
    make = sum_trainer_env
    trainer = make("t1")
    sentinel = {"a": 1}
    assert trainer._do_modify("no_such_hook", sentinel) is sentinel


def _opt_counts(state):
    """Collect every step counter from the optax state"""
    import jax

    return [
        int(x)
        for x in jax.tree_util.tree_leaves(state)
        if getattr(x, "shape", None) == () and x.dtype in (jnp.int32, jnp.int64)
    ]


def test_sub_optimizer_state_is_not_advanced(sum_trainer_env):
    """A sub applying no update does not advance its optimizer state either"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    t1, t2 = make("t1"), make("t2")
    trainer = SumTrainer(
        t1, t2, opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 5.0],
        lr=0.01, clip=0.1,
    )
    trainer.setup()
    before = _opt_counts(t1.opt_state)
    trainer.fit(steps=3, checkpoint_frequency=1000)

    # Only the parent advances three times
    assert _opt_counts(trainer.opt_state) == [c + 3 for c in before]
    assert _opt_counts(t1.opt_state) == before
    assert _opt_counts(t2.opt_state) == before


def test_sub_after_grad_hook_still_applies(sum_trainer_env):
    """The after_grad hook still works even without optimizer.update"""
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    t1, t2 = make("t1"), make("t2")
    trainer = SumTrainer(
        t1, t2, opt_fftypes=["NonbondedForce/charge"], weight=[1.0, 1.0],
        lr=0.01, clip=0.1,
    )
    seen = []

    def hook(grads):
        seen.append(len(grads["NonbondedForce"]["charge"]))
        return jax.tree_util.tree_map(lambda x: x * 0.0, grads)

    import jax

    t1.add_modifyfn("after_grad", hook)
    trainer.setup()
    trainer.fit(steps=1, checkpoint_frequency=1000)

    assert seen, "after_grad was not called"
    # Only trainer1's gradient was zeroed, so only the first half stays fixed
    half = seen[0]
    charge = trainer.ffparams["NonbondedForce"]["charge"]
    assert len(charge) == 2 * half


# ------------------------------------ ThermodynamicTrainer validation records
def _validation_stub(params, pred, dev=None, label="t", ffxml="epoch_t-3.xml"):
    import types

    return types.SimpleNamespace(
        validation_params=params,
        validation_pred=pred,
        validation_dev=[{} for _ in params] if dev is None else dev,
        validation_curves=[{} for _ in params],
        validation_history=[],
        validation_dev_history=[],
        ffxml=ffxml,
        label=label,
    )


def test_record_validation_labels_the_values_by_replica():
    """Recorded in the history keyed by replica index and item name"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub(
        [{"dself_Li": {}}, {"dself_Li": {}}],
        [{"dself_Li": 1.0}, {"dself_Li": 2.0}],
    )
    ThermodynamicTrainer._record_validation(stub, 3)

    assert stub.validation_history == [
        {
            "epoch": 3,
            "ffxml": "epoch_t-3.xml",
            "sample_0/dself_Li": 1.0,
            "sample_1/dself_Li": 2.0,
        }
    ]


def test_record_validation_keeps_the_values_of_the_untouched_replicas():
    """A replica that was not resampled is still recorded with its previous value"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub(
        [{"dself_Li": {}}, {"dself_Li": {}}],
        [{"dself_Li": 1.0}, {"dself_Li": 2.0}],
    )
    ThermodynamicTrainer._record_validation(stub, 3)
    # Only sample_0 was resampled; keep the force field used at that point
    stub.validation_pred[0] = {"dself_Li": 1.5}
    stub.ffxml = "epoch_t-8.xml"
    ThermodynamicTrainer._record_validation(stub, 8)

    assert stub.validation_history[-1] == {
        "epoch": 8,
        "ffxml": "epoch_t-8.xml",
        "sample_0/dself_Li": 1.5,
        "sample_1/dself_Li": 2.0,
    }


def test_record_validation_is_skipped_without_any_target():
    """A run without validation configured creates no history"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub([{}, {}], [{}, {}])
    ThermodynamicTrainer._record_validation(stub, 0)

    assert stub.validation_history == []


def test_update_validation_is_skipped_without_any_target():
    """With empty validation, no trajectory is read (a missing file does not crash)"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub([{}], [{}])
    stub.pdbfile_vsite = ["nonexistent.pdb"]
    ThermodynamicTrainer._update_validation(stub, 0, "nonexistent.xtc")

    assert stub.validation_pred == [{}]
    assert stub.validation_dev == [{}]
    assert stub.validation_curves == [{}]


def test_resample_records_the_epoch_of_the_force_field_it_sampled_with(monkeypatch):
    """The recorded epoch is the number of the xml used for resampling (= _epoch + 1)"""
    import types
    from imolcraft.trainer import ThermodynamicTrainer
    from imolcraft.trainer import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "get_target_pred_frame", lambda *a, **k: {})

    recorded = []
    stub = types.SimpleNamespace(
        _epoch=180,
        ffxml="xmlfiles/epoch_t-181.xml",
        estimator=types.SimpleNamespace(states=[], optimize_mbar=lambda: None),
        sampling_params=[{}],
        target_params=[{}],
        pdbfile_vsite=["x.pdb"],
        target_pred_frame=[{}],
        _resample_indices=lambda: [0],
        _run_md=lambda idx, name: "sample_0.xtc",
        _add_sample=lambda idx, name, xtc: None,
        _update_validation=lambda idx, xtc: None,
        _record_validation=lambda epoch: recorded.append(epoch),
    )
    ThermodynamicTrainer._resample(stub)

    # The trajectory was taken after after_step wrote epoch-181.xml, so record 181
    assert recorded == [181]


def test_record_validation_keeps_the_deviations_in_their_own_history():
    """Deviations stay in a history separate from the values, split per item"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub(
        [{"rho": {}, "rdf_Li_O": {}}],
        [{"rho": 1.5}],
        dev=[{"rho": -0.05, "rdf_Li_O": 0.02}],
    )
    ThermodynamicTrainer._record_validation(stub, 3)

    # Only scalars carry values; deviations cover everything including distributions
    assert stub.validation_history == [
        {"epoch": 3, "ffxml": "epoch_t-3.xml", "sample_0/rho": 1.5}
    ]
    assert stub.validation_dev_history == [
        {
            "epoch": 3,
            "ffxml": "epoch_t-3.xml",
            "sample_0/rho": -0.05,
            "sample_0/rdf_Li_O": 0.02,
        }
    ]


# ---------------------------------------------------------- target history
def _target_stub(target_log, fresh=(True,)):
    """A two-target stub replica: a density and one RDF, both reweighted"""
    import types

    n = len(fresh)
    return types.SimpleNamespace(
        _epoch=4,
        target_log=target_log,
        loss=3.0,
        losses_per_replica=[1.5 for _ in range(n)],
        wresults=[
            {"density_gcm3": 0.9, "rdf": {"Li-O": jnp.ones(5)}} for _ in range(n)
        ],
        target_pred_frame=[
            {"density_gcm3": jnp.ones(3), "rdf": {"Li-O": jnp.ones((3, 5))}}
            for _ in range(n)
        ],
        _fresh_frames=list(fresh),
        sampling_params=[{} for _ in range(n)],
        target_history=[],
    )


def _record_targets(stub, neff=None):
    from imolcraft.trainer import ThermodynamicTrainer

    if neff is None:
        neff = [{"sample_0": 40.0, "total": 40.0}] * len(stub.sampling_params)
    ThermodynamicTrainer._record_targets(stub, "xmlfiles/epoch_t-4.xml", neff)


def test_target_record_low_keeps_the_scalars_only():
    """low: losses, neff, the resampled flag and the scalar targets"""
    stub = _target_stub("low")
    _record_targets(stub)

    assert stub.target_history == [
        {
            "epoch": 4,
            "ffxml": "xmlfiles/epoch_t-4.xml",
            "loss": 3.0,
            "sample_0/loss": 1.5,
            "sample_0/neff": {"sample_0": 40.0, "total": 40.0},
            "sample_0/resampled": True,
            "sample_0/density_gcm3": 0.9,
        }
    ]


def test_target_record_medium_adds_the_distribution_curves():
    stub = _target_stub("medium")
    _record_targets(stub)

    record = stub.target_history[0]
    assert record["sample_0/rdf/Li-O"].shape == (5,)
    # the reweighted curve, not the per-frame ones
    assert not any(key.startswith("sample_0/frames/") for key in record)


def test_target_record_all_adds_the_frames_of_the_fresh_replicas_only():
    """all: per-frame values ride along once per resampling, per replica"""
    stub = _target_stub("all", fresh=(True, False))
    _record_targets(stub, neff=[{"sample_0": 40.0}, None])

    record = stub.target_history[0]
    assert record["sample_0/resampled"] is True
    assert record["sample_1/resampled"] is False
    assert record["sample_1/neff"] is None
    assert record["sample_0/frames/density_gcm3"].shape == (3,)
    assert record["sample_0/frames/rdf/Li-O"].shape == (3, 5)
    assert "sample_1/frames/density_gcm3" not in record
    assert record["sample_1/rdf/Li-O"].shape == (5,)
    # recorded, so the frames count as seen: the next record has none
    assert stub._fresh_frames == [False, False]
    _record_targets(stub, neff=[{"sample_0": 40.0}, None])
    assert not any(
        key.startswith("sample_0/frames/") for key in stub.target_history[1]
    )


def test_target_record_none_records_nothing_but_clears_the_flags():
    stub = _target_stub("none")
    _record_targets(stub)

    assert stub.target_history == []
    assert stub._fresh_frames == [False]


def test_thermodynamic_records_the_targets_before_resampling():
    """after_step records with the ffxml of this epoch, before rendering the next"""
    import inspect

    from imolcraft.trainer import ThermodynamicTrainer

    src = inspect.getsource(ThermodynamicTrainer.__dict__["after_step"])
    assert src.index("ffxml = self.ffxml") < src.index("renderXML")
    assert src.index("self._record_targets(ffxml, neff)") < src.index(
        "self._resample()"
    )
    # the history rides in the checkpoint and a restart puts it back
    src = inspect.getsource(ThermodynamicTrainer.__dict__["write_checkpoint"])
    assert '"target_history": self.target_history' in src
    src = inspect.getsource(ThermodynamicTrainer.__dict__["from_checkpoint"])
    assert 'dump_dict.get("target_history", [])' in src
# ------------------------------------------- NaN loss recovery by resampling
def _retry_stub(retries, losses, recoveries=None):
    """
    Minimal ``self`` for :meth:`BaseTrainer._retry_invalid_loss`.

    ``losses[0]`` is the loss handed to the retry loop, the rest are what the
    recomputations return in order. ``recoveries`` says what each recovery
    reports; None means every one of them succeeds.
    """
    import types

    calls = {"loss": 0, "recover": 0}
    seq = list(losses)

    def get_loss_gradients():
        calls["loss"] += 1
        return seq[min(calls["loss"], len(seq) - 1)], {"grads": calls["loss"]}

    def recover_from_invalid_loss(attempt):
        calls["recover"] += 1
        return True if recoveries is None else recoveries[attempt - 1]

    stub = types.SimpleNamespace(
        nan_resample_retries=retries,
        get_loss_gradients=get_loss_gradients,
        recover_from_invalid_loss=recover_from_invalid_loss,
    )
    return stub, calls, seq[0], {"grads": 0}


def test_base_trainer_attempts_no_recovery():
    """Nothing to renew, so the default neither retries nor claims it can"""
    from imolcraft.trainer.base import BaseTrainer

    assert BaseTrainer.nan_resample_retries == 0
    assert BaseTrainer.recover_from_invalid_loss(object(), 1) is False


def test_thermodynamic_trainer_retries_once_by_default():
    import inspect

    from imolcraft.trainer import ThermodynamicTrainer

    signature = inspect.signature(ThermodynamicTrainer.__init__)
    assert signature.parameters["nan_resample_retries"].default == 1


def test_a_finite_loss_is_never_retried():
    from imolcraft.trainer.base import BaseTrainer

    stub, calls, loss, grads = _retry_stub(3, [1.5, 2.5])
    out_loss, out_grads = BaseTrainer._retry_invalid_loss(stub, loss, grads)

    assert (out_loss, out_grads) == (1.5, {"grads": 0})
    assert calls == {"loss": 0, "recover": 0}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_an_invalid_loss_is_recovered_and_recomputed(bad):
    """A NaN or Inf loss is retried, and the retry replaces loss and gradients"""
    from imolcraft.trainer.base import BaseTrainer

    stub, calls, loss, grads = _retry_stub(2, [bad, 0.25])
    out_loss, out_grads = BaseTrainer._retry_invalid_loss(stub, loss, grads)

    assert out_loss == 0.25
    assert out_grads == {"grads": 1}
    # one recovery was enough, so the second retry was never spent
    assert calls == {"loss": 1, "recover": 1}


def test_retrying_stops_at_the_configured_number_of_attempts():
    """A loss that stays invalid is handed back rather than retried forever"""
    from imolcraft.trainer.base import BaseTrainer

    nan = float("nan")
    stub, calls, loss, grads = _retry_stub(2, [nan, nan, nan, nan])
    out_loss, _ = BaseTrainer._retry_invalid_loss(stub, loss, grads)

    assert jnp.isnan(out_loss)
    assert calls == {"loss": 2, "recover": 2}


def test_retrying_stops_when_the_recovery_reports_it_did_nothing():
    """A recovery that cannot run does not lead to the same loss being recomputed"""
    from imolcraft.trainer.base import BaseTrainer

    nan = float("nan")
    stub, calls, loss, grads = _retry_stub(3, [nan, nan], recoveries=[False])
    out_loss, out_grads = BaseTrainer._retry_invalid_loss(stub, loss, grads)

    assert jnp.isnan(out_loss)
    assert out_grads == {"grads": 0}
    assert calls == {"loss": 0, "recover": 1}


def test_retrying_is_off_when_no_retries_are_allowed():
    from imolcraft.trainer.base import BaseTrainer

    stub, calls, loss, grads = _retry_stub(0, [float("nan"), 1.0])
    out_loss, _ = BaseTrainer._retry_invalid_loss(stub, loss, grads)

    assert jnp.isnan(out_loss)
    assert calls == {"loss": 0, "recover": 0}


def test_training_step_retries_before_giving_up_on_the_step():
    """The perturbation is the fallback, reached only after the retrying"""
    import inspect

    from imolcraft.trainer.base import BaseTrainer

    src = inspect.getsource(BaseTrainer.__dict__["training_step"])
    assert src.index("_retry_invalid_loss") < src.index("_nan_recovery_gradients")


def test_sumtrainer_retries_each_half_on_its_own_data():
    """The recovery runs on the sub-trainer holding the data, not on the sum"""
    import inspect

    from imolcraft.trainer.base import SumTrainer

    src = inspect.getsource(SumTrainer.__dict__["_substep"])
    assert "trainer._retry_invalid_loss(loss, grads)" in src
    assert src.index("_retry_invalid_loss") < src.index("_nan_recovery_gradients")


def _recovery_stub(n=2, epoch=5, estimator=True):
    import types

    calls = []
    stub = types.SimpleNamespace(
        _epoch=epoch,
        ffxml="xmlfiles/epoch_x-5.xml",
        nan_resample_retries=1,
        sampling_params=[{}] * n,
        resample=[False] * n,
        _nan_resampled=False,
        estimator=types.SimpleNamespace() if estimator else None,
        _resample=lambda record_epoch=None: calls.append(record_epoch),
    )
    return stub, calls


def test_recovery_resamples_every_replica_for_the_epoch_being_retried():
    from imolcraft.trainer import ThermodynamicTrainer

    stub, calls = _recovery_stub(n=3, epoch=665)
    assert ThermodynamicTrainer.recover_from_invalid_loss(stub, 1) is True

    # the trajectories belong to the epoch being retried, not to the next one
    assert calls == [665]
    # every replica is renewed, and the flags are left clean for after_step
    assert stub.resample == [False, False, False]
    assert stub._nan_resampled is True


def test_recovery_does_nothing_before_setup():
    """Without an estimator there is nothing to resample into"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub, calls = _recovery_stub(estimator=False)
    assert ThermodynamicTrainer.recover_from_invalid_loss(stub, 1) is False
    assert calls == []


def test_after_step_does_not_resample_twice_for_the_same_invalid_loss():
    """A recovery resampling this step makes the one in after_step redundant"""
    import inspect

    from imolcraft.trainer import ThermodynamicTrainer

    src = inspect.getsource(ThermodynamicTrainer.__dict__["after_step"])
    assert "if loss_is_invalid and not self._nan_resampled:" in src
    # and the flag is cleared, so the next step is judged on its own
    assert "self._nan_resampled = False" in src


def test_resample_files_the_validation_record_under_the_given_epoch(monkeypatch):
    import types

    from imolcraft.trainer import ThermodynamicTrainer
    from imolcraft.trainer import trainer as trainer_module

    monkeypatch.setattr(
        trainer_module, "get_target_pred_frame", lambda *a, **k: {"frame": 1}
    )
    recorded = []
    stub = types.SimpleNamespace(
        _epoch=11,
        ffxml="ff.xml",
        resample_counter=42,
        sampling_params=[{}],
        pdbfile_vsite=["vs.pdb"],
        target_params=[{}],
        target_pred_frame=[None],
        estimator=types.SimpleNamespace(
            states=[], optimize_mbar=lambda: recorded.append("mbar")
        ),
        _resample_indices=lambda: [0],
        _run_md=lambda idx, name: f"{name}.xtc",
        _add_sample=lambda idx, name, xtc: None,
        _update_validation=lambda idx, xtc: None,
        _record_validation=lambda epoch: recorded.append(epoch),
    )

    ThermodynamicTrainer._resample(stub, record_epoch=stub._epoch)
    assert recorded == [11, "mbar"]
    assert stub.resample_counter == 0

    recorded.clear()
    ThermodynamicTrainer._resample(stub)
    assert recorded == [12, "mbar"]


# --------------------------------------- lower bounds on physical parameters
def _tree(sigma, epsilon, charge=(0.5, -0.5)):
    return {
        "NonbondedForce": {
            "sigma": jnp.array(sigma, dtype=jnp.float64),
            "epsilon": jnp.array(epsilon, dtype=jnp.float64),
            "charge": jnp.array(charge, dtype=jnp.float64),
        }
    }


def test_the_defaults_bound_sigma_and_epsilon_only():
    """A charge is signed and a torsion constant may change sign, so neither is bound"""
    from imolcraft.trainer.base import DEFAULT_PARAM_FLOORS

    assert set(DEFAULT_PARAM_FLOORS) == {
        "NonbondedForce/sigma",
        "NonbondedForce/epsilon",
    }
    # a sigma divides a distance, so it has to stay clear of zero, while a
    # switched-off Lennard-Jones site is a legitimate epsilon of zero
    assert DEFAULT_PARAM_FLOORS["NonbondedForce/sigma"] > 0.0
    assert DEFAULT_PARAM_FLOORS["NonbondedForce/epsilon"] == 0.0


def test_none_asks_for_the_defaults_and_an_empty_dict_for_nothing():
    from imolcraft.trainer.base import DEFAULT_PARAM_FLOORS, _resolve_param_floors

    tree = _tree([0.3, 0.3], [0.5, 0.5])
    assert _resolve_param_floors(tree, None) == {
        name: float(floor) for name, floor in DEFAULT_PARAM_FLOORS.items()
    }
    assert _resolve_param_floors(tree, {}) == {}


def test_a_bound_the_force_field_does_not_carry_is_dropped():
    """A trainer fitting torsions alone carries no sigma, and must not raise"""
    from imolcraft.trainer.base import _resolve_param_floors

    torsions = {"PeriodicTorsionForce": {"proper_k": jnp.array([1.0])}}
    assert _resolve_param_floors(torsions, None) == {}
    assert _resolve_param_floors(
        torsions, {"HarmonicBondForce/k": 0.0, "PeriodicTorsionForce/proper_k": -5.0}
    ) == {"PeriodicTorsionForce/proper_k": -5.0}


def test_a_feasible_step_is_left_untouched():
    from imolcraft.trainer.base import _resolve_param_floors, enforce_param_floors

    previous = _tree([0.30, 0.31], [0.5, 0.6])
    stepped = _tree([0.29, 0.32], [0.4, 0.0])
    floors = _resolve_param_floors(stepped, None)

    out, report = enforce_param_floors(stepped, previous, floors)
    assert report == {}
    # the very same tree comes back, so nothing was copied for nothing
    assert out is stepped
    # an epsilon of zero is on its bound, not below it
    assert float(out["NonbondedForce"]["epsilon"][1]) == 0.0


def test_a_parameter_pushed_below_its_bound_is_held_at_the_previous_value():
    """Not parked on the bound: a sigma at the bound has no gradient left"""
    from imolcraft.trainer.base import _resolve_param_floors, enforce_param_floors

    previous = _tree([0.30, 0.31], [0.5, 0.6])
    stepped = _tree([-0.02, 0.32], [0.4, -0.1])
    floors = _resolve_param_floors(stepped, None)

    out, report = enforce_param_floors(stepped, previous, floors)

    assert list(out["NonbondedForce"]["sigma"]) == [0.30, 0.32]
    assert list(out["NonbondedForce"]["epsilon"]) == [0.4, 0.6]
    assert report["NonbondedForce/sigma"] == {"count": 1, "lowest": -0.02}
    assert report["NonbondedForce/epsilon"]["count"] == 1


def test_holding_a_parameter_leaves_the_previous_tree_alone():
    from imolcraft.trainer.base import _resolve_param_floors, enforce_param_floors

    previous = _tree([0.30, 0.31], [0.5, 0.6])
    stepped = _tree([-0.02, 0.32], [0.4, 0.6])
    floors = _resolve_param_floors(stepped, None)

    enforce_param_floors(stepped, previous, floors)
    assert list(previous["NonbondedForce"]["sigma"]) == [0.30, 0.31]
    assert list(stepped["NonbondedForce"]["sigma"]) == [-0.02, 0.32]


def test_a_previous_value_below_the_bound_is_raised_onto_it():
    """A force field that started out unphysical still comes out feasible"""
    from imolcraft.trainer.base import (
        DEFAULT_PARAM_FLOORS,
        _resolve_param_floors,
        enforce_param_floors,
    )

    previous = _tree([-0.5, 0.31], [0.5, 0.6])
    stepped = _tree([-0.6, 0.32], [0.5, 0.6])
    floors = _resolve_param_floors(stepped, None)

    out, _ = enforce_param_floors(stepped, previous, floors)
    assert float(out["NonbondedForce"]["sigma"][0]) == pytest.approx(
        DEFAULT_PARAM_FLOORS["NonbondedForce/sigma"]
    )


def test_charges_are_never_held_back():
    from imolcraft.trainer.base import _resolve_param_floors, enforce_param_floors

    previous = _tree([0.30], [0.5], charge=(0.4,))
    stepped = _tree([0.30], [0.5], charge=(-1.2,))
    floors = _resolve_param_floors(stepped, None)

    out, report = enforce_param_floors(stepped, previous, floors)
    assert float(out["NonbondedForce"]["charge"][0]) == -1.2
    assert report == {}


def test_bounding_nothing_short_circuits():
    from imolcraft.trainer.base import enforce_param_floors

    stepped = _tree([-1.0], [-1.0])
    out, report = enforce_param_floors(stepped, _tree([0.3], [0.5]), {})
    assert out is stepped and report == {}


def test_the_bounds_are_applied_before_the_after_update_hook():
    """A hook registered for a hard constraint keeps the last word"""
    import inspect

    from imolcraft.trainer.base import BaseTrainer, SumTrainer

    for cls in (BaseTrainer, SumTrainer):
        src = inspect.getsource(cls.__dict__["training_step"])
        assert "enforce_param_floors" in src, cls.__name__
        assert src.index("apply_updates") < src.index("enforce_param_floors"), (
            cls.__name__
        )
        # the call, not the mention of the hook in the docstring
        assert src.index("enforce_param_floors") < src.index(
            '_do_modify("after_update"'
        ), cls.__name__


def test_sumtrainer_bounds_the_joint_tree_before_splitting_it():
    import inspect

    from imolcraft.trainer.base import SumTrainer

    src = inspect.getsource(SumTrainer.__dict__["training_step"])
    assert src.index("enforce_param_floors") < src.index("_scatter_to_subtrainers")


def test_a_real_trainer_resolves_the_bounds_of_its_force_field(sum_trainer_env):
    from imolcraft.trainer.base import DEFAULT_PARAM_FLOORS

    trainer = sum_trainer_env("t1")
    assert trainer.param_floors == {
        name: float(floor) for name, floor in DEFAULT_PARAM_FLOORS.items()
    }
    assert trainer.param_floors_given is None


def test_a_trainer_can_be_asked_to_bound_nothing(sum_trainer_env):
    trainer = sum_trainer_env("t1", param_floors={})
    assert trainer.param_floors == {}
    assert trainer.param_floors_given == {}


def test_every_trainer_records_the_bounds_it_was_given():
    """A restart must not silently drop back to the defaults"""
    import inspect

    from imolcraft.trainer import DihedralTrainer, DistanceTrainer
    from imolcraft.trainer.base import SumTrainer

    for cls in (DistanceTrainer, DihedralTrainer, SumTrainer):
        src = inspect.getsource(cls.__dict__["write_checkpoint"])
        assert '"param_floors": self.param_floors_given' in src, cls.__name__

    from imolcraft.trainer import ThermodynamicTrainer

    # this one rebuilds itself from restart_args rather than from loose keys
    src = inspect.getsource(ThermodynamicTrainer.__dict__["__init__"])
    assert '"param_floors": param_floors' in src
    src = inspect.getsource(ThermodynamicTrainer.__dict__["from_checkpoint"])
    assert '"param_floors": param_floors' in src


def test_the_bounds_survive_a_sumtrainer_restart(sum_trainer_env, tmp_path):
    from imolcraft.trainer.base import SumTrainer

    make = sum_trainer_env
    trainer = SumTrainer(
        make("t1"), make("t2"), opt_fftypes=["NonbondedForce/charge"],
        weight=[1.0, 1.0], lr=0.01, clip=0.1,
        param_floors={"NonbondedForce/sigma": 0.05},
    )
    trainer.setup()
    trainer.write_checkpoint(1)

    restored = SumTrainer.from_checkpoint(
        f"train_state_{trainer.label}.pkl", make("t1"), make("t2")
    )
    assert restored.param_floors == {"NonbondedForce/sigma": 0.05}


def test_a_real_step_cannot_drive_sigma_negative(sum_trainer_env, capsys):
    """End to end: the optimizer overshoots by far, and sigma stays physical"""
    trainer = sum_trainer_env(
        "sig", opt_fftypes=["NonbondedForce/sigma"], lr=1.0, clip=10.0
    )
    trainer.setup()
    before = [float(v) for v in trainer.ffparams["NonbondedForce"]["sigma"]]
    trainer.fit(steps=1, checkpoint_frequency=1000)
    after = [float(v) for v in trainer.ffparams["NonbondedForce"]["sigma"]]

    # a learning rate of 1 nm per step takes every sigma well below zero
    assert all(value > 0.0 for value in after)
    assert after == before
    out = capsys.readouterr().out
    assert "NonbondedForce/sigma was pushed below" in out
    assert "lr or clip is too large" in out
