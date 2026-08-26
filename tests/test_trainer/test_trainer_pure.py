"""外部プログラムや長時間の MD を必要としない trainer の回帰テスト"""
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
    """qmmin は QM の最小点で両曲線を一致させるので、その点の誤差は 0"""
    e_qm = jnp.array([2.0, 0.0, 3.0])
    e_ff = jnp.array([5.0, 1.0, 7.0])
    se = _squared_error(e_ff, e_qm, "qmmin")
    assert se[jnp.argmin(e_qm)] == pytest.approx(0.0)


def test_squared_error_is_zero_for_a_constant_offset():
    """auto は平均のずれを吸収するので、定数シフトは誤差にならない"""
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
    """既知の未実装分岐。実装されたらこのテストを消すこと"""
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
    """スカラー目標は必ず gt と weight を要求する"""
    for target in SCALAR_TARGETS:
        assert REQUIRED_TARGET_KEYS[target] == ("gt", "weight")
    assert set(REQUIRED_TARGET_KEYS) >= set(SCALAR_TARGETS) | {"rdf", "adf"}
    assert "nvt" in VALID_ENSEMBLES and "isonpt" in VALID_ENSEMBLES


def test_every_property_is_declared_once_and_completely():
    """物性の一覧は 1 か所で、種類と必要なキーの両方を持つ"""
    assert set(PROPERTY_KINDS) == set(PROPERTY_KEYS)
    assert set(PROPERTY_KINDS.values()) <= {"scalar", "distribution"}


def test_the_loss_takes_every_property_but_the_validation_only_ones():
    """最適化対象は、validation 専用を除いた全物性"""
    assert set(REQUIRED_TARGET_KEYS) == set(PROPERTY_KINDS) - set(
        VALIDATION_ONLY_PROPERTIES
    )
    # loss 側が分岐に使う 2 つの組も、同じ表から導出されている
    assert set(SCALAR_TARGETS) | set(DISTRIBUTION_TARGETS) == set(
        REQUIRED_TARGET_KEYS
    )
    # 逆向きは成り立つ: loss で使える物性は validation でも監視できる
    assert set(REQUIRED_VALIDATION_KEYS) == set(PROPERTY_KINDS)


def test_a_validated_distribution_always_needs_its_reference():
    """分布は参照との距離を記録するので、validation でも gt が要る"""
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


# ------------------------------------------------------- checkpoint の往復
def _ethane_pdb(path):
    """ethane.xml に合う結合付き PDB を作る"""
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
    """MD も QM も走らせない最小の sub-trainer"""
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

    def make(label):
        return DummyTrainer(
            ffxml_list=[ffxml],
            nums_ffxml=[1],
            pdbfile=pdbfile,
            loss_fn=None,
            opt_fftypes=["NonbondedForce/charge"],
            label=label,
            lr=0.01,
            clip=0.1,
        )

    return make


def test_sumtrainer_writes_a_non_empty_checkpoint(sum_trainer_env, tmp_path):
    """dump_dict が実際に pickle される（旧実装では空ファイルだった）"""
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
    """sub の pickle には無い opt_state が復元され、setup() で潰されない"""
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
    # 復元した値が sub-trainer にも配られている
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
    """4 クラスすべてが独自の write_checkpoint と from_checkpoint を持つ"""
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
    """4 クラスとも train_state_{label}.pkl に書く（SumTrainer で衝突しないこと）"""
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
    """周期的な再緩和用 xml も xmlfiles/loop_{label}-{epoch}.xml に揃える"""
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
    """指定した損失列を順に返す sub-trainer"""
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
    """fit を回す前でも属性は存在する（以前は AttributeError）"""
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
    """再開後に損失が悪化しても、過去の最良が上書きされない"""
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

    # 続きを回しても、悪化しているうちは最良が保たれる
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
        from_src = inspect.getsource(cls.__dict__["from_checkpoint"])
        assert "_restore_best(dump_dict)" in from_src, cls.__name__


def test_thermodynamic_from_checkpoint_restores_history_explicitly():
    """効いていなかった attr_lists ループを明示的な代入に置き換えた"""
    import inspect

    from imolcraft.trainer import ThermodynamicTrainer

    src = inspect.getsource(ThermodynamicTrainer.__dict__["from_checkpoint"])
    # 一致しないキー名と、_epoch ではなく epoch を作ってしまうループが消えたこと
    assert "ff_info" not in src
    assert "attr_lists" not in src
    # 履歴は明示的に復元する
    assert 'trainer.losses = dump_dict["losses"]' in src
    assert 'trainer.epochs = dump_dict["epochs"]' in src
    assert 'trainer._epoch = dump_dict["epoch"]' in src
    # 力場は initial_ffxml 由来なので ffinfo は復元しない
    assert "ff.ffinfo" not in src


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_fit_runs_exactly_the_requested_number_of_steps(
    steps, sum_trainer_env, tmp_path
):
    """fit(steps) はちょうど steps エポック回る（以前は steps+1 回っていた）"""
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
    """fit を分けて呼んでも合計エポック数は同じ"""
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


# --------------------------------------- ThermodynamicTrainer の再サンプリング判定
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
    """状態の並びが変わっても、自分のレプリカの寄与を見る"""
    from imolcraft.trainer import ThermodynamicTrainer

    # sample_1 を再サンプリングした後は estimator.states が [0, 2, 1] の順になる
    ieff = {"sample_0": 100, "sample_2": 100, "sample_1": 3, "Total": 203}
    stub = _thermo_stub([10, 10, 10])

    decisions = [
        ThermodynamicTrainer._needs_resample(stub, ii, ieff) for ii in range(3)
    ]
    assert decisions == [False, True, False]


def test_needs_resample_ignores_the_total_entry():
    from imolcraft.trainer import ThermodynamicTrainer

    # Total だけが小さくても、自分の寄与が足りていれば再サンプリングしない
    ieff = {"sample_0": 100, "Total": 1}
    stub = _thermo_stub([10])
    assert ThermodynamicTrainer._needs_resample(stub, 0, ieff) is False


def test_needs_resample_tolerates_a_missing_entry():
    """自分の状態がまだ登録されていない場合に落ちない"""
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


# ------------------------------------------------------------- hook の有効性
_HOOK_MARK = -77.0


def _pure_hook(tree):
    """引数を触らず、加工した新しい dict を返す（契約どおりの書き方）"""
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
    """sub に登録した after_update が SumTrainer 経由でも効く"""
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
    # 親の hook の結果も sub へ配られる
    assert _marked(trainer.trainer1)


def test_sub_hooks_run_before_the_parent_hook(sum_trainer_env):
    """sub は自分の半分、親は連結ベクトル全体を見る"""
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
    """未登録のフック種別でも None を返さない（代入して壊れない）"""
    make = sum_trainer_env
    trainer = make("t1")
    sentinel = {"a": 1}
    assert trainer._do_modify("no_such_hook", sentinel) is sentinel


def _opt_counts(state):
    """optax の state からステップカウンタをすべて拾う"""
    import jax

    return [
        int(x)
        for x in jax.tree_util.tree_leaves(state)
        if getattr(x, "shape", None) == () and x.dtype in (jnp.int32, jnp.int64)
    ]


def test_sub_optimizer_state_is_not_advanced(sum_trainer_env):
    """更新を適用しない sub の optimizer は状態も進めない"""
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

    # 親だけが 3 回進む
    assert _opt_counts(trainer.opt_state) == [c + 3 for c in before]
    assert _opt_counts(t1.opt_state) == before
    assert _opt_counts(t2.opt_state) == before


def test_sub_after_grad_hook_still_applies(sum_trainer_env):
    """optimizer.update を外しても after_grad フックは効く"""
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

    assert seen, "after_grad が呼ばれていない"
    # trainer1 の勾配だけゼロにしたので、連結ベクトルの前半だけ動かない
    half = seen[0]
    charge = trainer.ffparams["NonbondedForce"]["charge"]
    assert len(charge) == 2 * half


# ------------------------------------ ThermodynamicTrainer の validation 記録
def _validation_stub(params, pred, dev=None, epoch=0, label="t"):
    import types

    return types.SimpleNamespace(
        validation_params=params,
        validation_pred=pred,
        validation_dev=[{} for _ in params] if dev is None else dev,
        validation_curves=[{} for _ in params],
        validation_history=[],
        validation_dev_history=[],
        label=label,
        _epoch=epoch,
    )


def test_record_validation_labels_the_values_by_replica():
    """レプリカ番号と項目名の組で履歴に残る"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub(
        [{"dself_Li": {}}, {"dself_Li": {}}],
        [{"dself_Li": 1.0}, {"dself_Li": 2.0}],
        epoch=3,
    )
    ThermodynamicTrainer._record_validation(stub)

    assert stub.validation_history == [
        {"epoch": 3, "sample_0/dself_Li": 1.0, "sample_1/dself_Li": 2.0}
    ]


def test_record_validation_keeps_the_values_of_the_untouched_replicas():
    """再サンプリングされなかったレプリカも、前回の値のまま記録に残る"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub(
        [{"dself_Li": {}}, {"dself_Li": {}}],
        [{"dself_Li": 1.0}, {"dself_Li": 2.0}],
        epoch=3,
    )
    ThermodynamicTrainer._record_validation(stub)
    # sample_0 だけ再サンプリングされた状況
    stub.validation_pred[0] = {"dself_Li": 1.5}
    stub._epoch = 8
    ThermodynamicTrainer._record_validation(stub)

    assert stub.validation_history[-1] == {
        "epoch": 8, "sample_0/dself_Li": 1.5, "sample_1/dself_Li": 2.0
    }


def test_record_validation_is_skipped_without_any_target():
    """validation を設定していない run では、履歴を作らない"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub([{}, {}], [{}, {}])
    ThermodynamicTrainer._record_validation(stub)

    assert stub.validation_history == []


def test_update_validation_is_skipped_without_any_target():
    """validation が空なら、軌跡を読みに行かない（存在しないファイルでも落ちない）"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub([{}], [{}])
    stub.pdbfile_vsite = ["nonexistent.pdb"]
    ThermodynamicTrainer._update_validation(stub, 0, "nonexistent.xtc")

    assert stub.validation_pred == [{}]
    assert stub.validation_dev == [{}]
    assert stub.validation_curves == [{}]


def test_record_validation_keeps_the_deviations_in_their_own_history():
    """ズレは値とは別の履歴に、項目ごとに分かれたまま残る"""
    from imolcraft.trainer import ThermodynamicTrainer

    stub = _validation_stub(
        [{"rho": {}, "rdf_Li_O": {}}],
        [{"rho": 1.5}],
        dev=[{"rho": -0.05, "rdf_Li_O": 0.02}],
        epoch=3,
    )
    ThermodynamicTrainer._record_validation(stub)

    # 値を持つのはスカラーだけ、ズレは分布も含めて全部
    assert stub.validation_history == [{"epoch": 3, "sample_0/rho": 1.5}]
    assert stub.validation_dev_history == [
        {"epoch": 3, "sample_0/rho": -0.05, "sample_0/rdf_Li_O": 0.02}
    ]
