"""
Tests of imolcraft.trainer.selection: no MD, no external program.

The regression fixture ``tests/data/train_state_history_s_opt.json.gz`` is
the validation-relevant part of a real ``train_state_s_opt.pkl`` (2000
epochs, one replica, ``dself_S`` validated). It was made with::

    import gzip, json, numpy as np
    from imolcraft.trainer.selection import load_history
    h = load_history("train_state_s_opt.pkl")
    with gzip.open("tests/data/train_state_history_s_opt.json.gz", "wt",
                   encoding="utf-8", compresslevel=9) as f:
        f.write(json.dumps(h, default=lambda o: o.item()))

and ``select_epoch`` on the full pkl and on the fixture agree exactly.
"""
import gzip
import json
import os
import pickle
import re
import types

import jax.numpy as jnp
import numpy as np
import pytest

import imolcraft.trainer
from imolcraft.trainer import selection
from imolcraft.trainer.selection import (
    Band,
    EpochSelection,
    RunSelection,
    SelectionDiagnostics,
    extract_history,
    load_history,
    main,
    select_epoch,
    select_run,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "data", "train_state_history_s_opt.json.gz"
)


def _history(epochs, dev, loss, gt=1.0, entry="dself_S", extra_entries=None,
             with_target_history=True):
    """
    A checkpoint-shaped dict with the same records ``ThermodynamicTrainer``
    writes: ``validation_history[k]`` holds ``gt * (1 + dev)`` under
    ``sample_0/<entry>``, ``validation_dev_history`` holds ``dev`` itself
    (metric ``relerr``), ``target_history`` has one record per epoch with
    ``sample_0/resampled`` set at the validation epochs, and ``epochs`` /
    ``losses`` cover every epoch with the loss held between two validation
    points (so the nearest-neighbour lookup gives the same loss with or
    without ``target_history``).
    """
    epochs = [int(e) for e in epochs]
    dev = np.asarray(dev, dtype=float)
    loss = np.asarray(loss, dtype=float)
    extra_entries = extra_entries or {}
    params = {entry: {"property": "dself_cm2s", "select": "element S", "gt": gt}}
    params.update({name: {"property": "density_gcm3", "gt": g}
                   for name, g in extra_entries.items()})
    vh, vdh = [], []
    for e, d in zip(epochs, dev):
        ffxml = f"xmlfiles/epoch_t-{e}.xml"
        rec = {"epoch": e, "ffxml": ffxml, f"sample_0/{entry}": gt * (1.0 + d)}
        drec = {"epoch": e, "ffxml": ffxml, f"sample_0/{entry}": d}
        for name, g in extra_entries.items():
            rec[f"sample_0/{name}"] = g * (1.0 + 0.5 * d)
            drec[f"sample_0/{name}"] = 0.5 * d
        vh.append(rec)
        vdh.append(drec)
    all_epochs = list(range(max(epochs) + 1))
    step = np.searchsorted(epochs, all_epochs, side="right") - 1
    all_losses = [float(loss[max(s, 0)]) for s in step]
    state = {
        "label": "t",
        "validation_params": [params],
        "validation_history": vh,
        "validation_dev_history": vdh,
        "epochs": all_epochs,
        "losses": all_losses,
    }
    if with_target_history:
        resampled = set(epochs)
        state["target_history"] = [
            {"epoch": e, "ffxml": f"xmlfiles/epoch_t-{e}.xml", "loss": l,
             "sample_0/loss": l, "sample_0/neff": 100.0,
             "sample_0/resampled": e in resampled}
            for e, l in zip(all_epochs, all_losses)
        ]
    return state


def _synthetic(band_means=(-0.6, -0.3, -0.1, -0.2, -0.25), per_band=10, seed=1):
    """
    5 bands x 10 points: band 1 is burn-in (loss 10x), bands 2-5 have the
    given mean deviations plus sd 0.01 noise; one point of band 3 (k = 25,
    epoch 100) has the smallest loss of that band.
    """
    rng = np.random.default_rng(seed)
    n_bands = len(band_means)
    m = n_bands * per_band
    epochs = [4 * k for k in range(m)]
    dev = np.repeat(band_means, per_band) + rng.normal(0.0, 0.01, m)
    loss = 1e-3 * (1.0 + 0.1 * rng.uniform(size=m))
    loss[:per_band] *= 10.0
    loss[25] = 0.5e-3
    return _history(epochs, dev, loss)


@pytest.fixture(scope="module")
def real_history():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ T1, T2
def test_synthetic_picks_the_band_closest_to_zero_and_its_smallest_loss():
    sel = select_epoch(_synthetic())
    assert isinstance(sel, EpochSelection)
    assert sel.n_points == 50
    assert sel.band_count == 5
    assert len(sel.bands) == 5 and all(isinstance(b, Band) for b in sel.bands)
    assert sel.bands[0].admissible is False
    assert [b.admissible for b in sel.bands] == [False, True, True, True, True]
    assert sel.best.index == 3
    assert sel.best is sel.bands[2]
    assert sel.epoch == 100                       # k = 25, the smallest loss of band 3
    assert sel.ffxml == "xmlfiles/epoch_t-100.xml"
    assert sel.loss_at_epoch == pytest.approx(0.5e-3)
    assert sel.score == pytest.approx(-0.1, abs=0.02)
    assert sel.score_se == pytest.approx(sel.sigma / np.sqrt(10))
    assert sel.best.dev_mean == sel.score and sel.best.dev_se == sel.score_se
    assert sel.monitor == "sample_0/dself_S"
    assert sel.label == "t"
    assert sel.notes == []


def test_admissibility_depends_on_the_loss_only():
    """With a huge loss_tol the burn-in band is admissible; the best is unchanged."""
    sel = select_epoch(_synthetic(), loss_tol=100.0)
    assert all(b.admissible for b in sel.bands)
    assert sel.best.index == 3
    assert sel.epoch == 100
    # the burn-in band has a large |dev| and is not chosen although admissible
    assert abs(sel.bands[0].dev_mean) > abs(sel.best.dev_mean)


def test_within_the_best_band_the_epoch_is_the_smallest_loss_not_the_smallest_dev():
    state = _synthetic()
    # make the point of smallest |dev| in band 3 different from the smallest loss
    k_min_loss = 25
    idx = [k for k in range(20, 30) if k != k_min_loss]
    k_zero = idx[0]
    state["validation_history"][k_zero]["sample_0/dself_S"] = 1.0    # dev exactly 0
    sel = select_epoch(state)
    assert sel.best.index == 3
    assert sel.epoch == 4 * k_min_loss
    assert sel.epoch != 4 * k_zero


# ---------------------------------------------------------------------- T3
def test_monitor_is_found_when_unique_and_required_otherwise():
    state = _synthetic()
    assert select_epoch(state).monitor == "sample_0/dself_S"

    rng = np.random.default_rng(1)
    two = _history([4 * k for k in range(50)], rng.normal(0, 0.01, 50),
                   1e-3 * np.ones(50), extra_entries={"rho": 1.5})
    with pytest.raises(ValueError) as err:
        select_epoch(two)
    assert "sample_0/dself_S" in str(err.value) and "sample_0/rho" in str(err.value)
    sel = select_epoch(two, monitor="sample_0/rho")
    assert sel.monitor == "sample_0/rho"
    # dev of rho is half that of dself_S in _history
    ref = select_epoch(two, monitor="sample_0/dself_S")
    assert sel.score == pytest.approx(0.5 * ref.score)


# ---------------------------------------------------------------------- T4
def test_without_gt_the_recorded_deviation_is_used_and_noted():
    state = _synthetic()
    ref = select_epoch(state)
    del state["validation_params"][0]["dself_S"]["gt"]
    sel = select_epoch(state)
    fallback = [n for n in sel.notes if "validation_dev_history" in n]
    assert len(fallback) == 1
    assert "metric" in fallback[0]
    # validation_dev_history holds relerr here, so the numbers do not change
    assert sel.epoch == ref.epoch
    assert sel.score == pytest.approx(ref.score)
    assert ref.notes == []


def test_dev_is_recomputed_from_gt():
    """The recorded deviation is ignored when gt is present."""
    state = _synthetic()
    for rec in state["validation_dev_history"]:
        rec["sample_0/dself_S"] = 99.0
    ep, dev, loss, monitor, notes = extract_history(state)
    assert ep.shape == dev.shape == loss.shape == (50,)
    assert np.all(np.abs(dev) < 1.0)
    assert monitor == "sample_0/dself_S" and notes == []


# ---------------------------------------------------------------------- T5
def test_too_few_points_and_bad_band_count_are_rejected():
    rng = np.random.default_rng(2)
    nine = _history(range(9), rng.normal(0, 0.01, 9), 1e-3 * np.ones(9))
    with pytest.raises(ValueError, match="10"):
        select_epoch(nine, band_count=5)
    with pytest.raises(ValueError, match="band_count"):
        select_epoch(_synthetic(), band_count=1)

    twelve = _history(range(12), rng.normal(0, 0.01, 12), 1e-3 * np.ones(12))
    sel = select_epoch(twelve, band_count=5)
    assert sel.n_points == 12
    assert any("12" in n and "band_count=5" in n for n in sel.notes)
    fifteen = _history(range(15), rng.normal(0, 0.01, 15), 1e-3 * np.ones(15))
    assert select_epoch(fifteen, band_count=5).notes == []


# ---------------------------------------------------------------------- T6
def test_nan_points_are_dropped_and_counted():
    state = _synthetic()
    for k in (3, 17, 41):
        state["validation_history"][k]["sample_0/dself_S"] = float("nan")
    state["validation_history"][17]["sample_0/dself_S"] = float("inf")
    sel = select_epoch(state)
    assert sel.n_points == 47
    assert any("3" in n and "NaN" in n for n in sel.notes)


# ---------------------------------------------------------------------- T7
def test_every_input_form_gives_the_same_result(tmp_path):
    state = _synthetic()
    ref = select_epoch(state)

    trainer_like = types.SimpleNamespace(**state)
    pkl = tmp_path / "train_state_t.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(state, f)

    for source in (trainer_like, pkl, str(pkl), tmp_path):
        sel = select_epoch(source)
        assert sel.epoch == ref.epoch
        assert sel.score == ref.score and sel.score_se == ref.score_se
        assert sel.bands == ref.bands
    assert select_epoch(pkl).label == "train_state_t.pkl"
    assert select_epoch(tmp_path).label == tmp_path.name
    assert select_epoch(pkl, label="lr=1e-4").label == "lr=1e-4"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        select_epoch(empty)
    with pytest.raises(FileNotFoundError):
        load_history(empty)

    no_validation = {k: v for k, v in state.items() if k != "validation_history"}
    with pytest.raises(ValueError, match="validation_history"):
        select_epoch(no_validation)
    with pytest.raises(ValueError, match="validation_history"):
        load_history({**state, "validation_history": []})


def test_load_history_keeps_only_the_history_keys_as_floats(real_history):
    state = _synthetic()
    state["losses"] = [jnp.asarray(x) for x in state["losses"]]
    for rec in state["target_history"]:
        rec["loss"] = jnp.asarray(rec["loss"])
        rec["sample_0/loss"] = jnp.asarray(rec["sample_0/loss"])
    state["opt_state"] = object()
    state["topology"] = object()
    h = load_history(state)
    assert set(h) == set(selection.HISTORY_KEYS)
    assert all(type(x) is float for x in h["losses"])
    assert all(type(r["loss"]) is float for r in h["target_history"])
    assert all(set(r) == {"epoch", "loss", "sample_0/resampled"} for r in h["target_history"])
    assert all(type(r["sample_0/resampled"]) is bool for r in h["target_history"])
    assert select_epoch(h).epoch == select_epoch(state).epoch
    # idempotent, and the fixture is such a reduced dict
    assert load_history(h) == h
    assert set(load_history(real_history)) <= set(selection.HISTORY_KEYS)


# ---------------------------------------------------------------------- T8
def test_loss_lookup_falls_back_to_epochs_and_losses_and_uses_the_nearest_epoch():
    state = _synthetic()
    ref = select_epoch(state)
    without = _synthetic()
    del without["target_history"]
    sel = select_epoch(without)
    assert sel.epoch == ref.epoch
    assert sel.bands == ref.bands
    assert sel.loss_at_epoch == ref.loss_at_epoch

    # a validation record one epoch past the last recorded loss
    last = max(without["epochs"])
    without["validation_history"][-1]["epoch"] = last + 1
    sel = select_epoch(without)
    assert sel.bands[-1].epoch_hi == last + 1
    assert sel.bands[-1].loss_mean == pytest.approx(ref.bands[-1].loss_mean)


def test_points_that_are_not_resampled_epochs_are_noted():
    state = _synthetic()
    for rec in state["target_history"]:
        if rec["epoch"] == 100:
            rec["sample_0/resampled"] = False
    sel = select_epoch(state)
    assert any("resampled" in n for n in sel.notes)
    assert sel.epoch == 100                        # the note does not change the choice


# ---------------------------------------------------------------------- T9
def test_select_run_ties_and_breaks_the_tie_by_the_band_mean_loss():
    a = select_epoch(_synthetic(), label="a")
    cheaper = _synthetic()
    for rec in cheaper["target_history"]:
        rec["loss"] *= 0.9
    cheaper["losses"] = [0.9 * x for x in cheaper["losses"]]
    b = select_epoch(cheaper, label="b")
    assert a.score == b.score

    run = select_run([a, b])
    assert isinstance(run, RunSelection)
    assert run.tied == [a, b] or run.tied == [b, a]
    assert run.winner is b
    assert run.epoch == b.epoch
    assert run.z == selection.DEFAULT_Z

    shifted = _synthetic()
    for rec in shifted["validation_history"]:
        rec["sample_0/dself_S"] += 1.0             # dev + 1.0 everywhere
    c = select_epoch(shifted, label="c")
    run = select_run([c, a, b])
    assert run.ranked[-1] is c
    assert c not in run.tied
    assert run.winner is b

    # a large z makes everything tied; the tie-break is still the loss
    run = select_run([c, a, b], z=1e6)
    assert len(run.tied) == 3 and run.winner is b

    with pytest.raises(ValueError):
        select_run([])


# --------------------------------------------------------------------- T10
def test_diagnostics_are_deterministic_and_optional():
    state = _synthetic()
    a = select_epoch(state, seed=0)
    b = select_epoch(state, seed=0)
    assert isinstance(a.diagnostics, SelectionDiagnostics)
    assert a.diagnostics == b.diagnostics
    assert a.diagnostics.n_boot == selection.DEFAULT_BOOTSTRAP
    assert a.diagnostics.f_df == (3, 36)          # 4 admissible bands of 10 points
    assert a.diagnostics.f_stat > 1.0             # the band means really differ

    c = select_epoch(state, seed=1)
    assert c.diagnostics.selection_bias != a.diagnostics.selection_bias
    assert c.diagnostics.selection_bias == pytest.approx(
        a.diagnostics.selection_bias, abs=0.01)
    assert (c.epoch, c.score) == (a.epoch, a.score)   # the seed never touches the decision

    assert select_epoch(state, diagnostics=False).diagnostics is None
    assert select_epoch(state, n_boot=0).diagnostics.selection_bias == 0.0


def test_autocorrelated_residuals_are_noted():
    """A slow oscillation inside the bands breaks the white-noise assumption."""
    m = 50
    epochs = [4 * k for k in range(m)]
    dev = -0.2 + 0.1 * np.sin(np.arange(m) * 2 * np.pi / 10)
    sel = select_epoch(_history(epochs, dev, 1e-3 * np.ones(m)))
    assert abs(sel.autocorr1) > selection.AUTOCORR_WARN
    assert any("autocorrelation" in n for n in sel.notes)


# --------------------------------------------------------------------- T11
def test_regression_on_the_real_run(real_history):
    sel = select_epoch(real_history, band_count=5)
    assert sel.monitor == "sample_0/dself_S"
    assert sel.n_points == 171
    assert sel.label == "s_opt"
    assert sel.epoch == 882
    assert sel.ffxml == "xmlfiles/epoch_s_opt-882.xml"
    assert (sel.best.epoch_lo, sel.best.epoch_hi) == (572, 951)
    assert sel.best.n == 34
    assert sel.score == pytest.approx(-0.189697, abs=1e-5)
    assert sel.score_se == pytest.approx(0.021632, abs=1e-5)
    assert sel.sigma == pytest.approx(0.126137, abs=1e-5)
    assert sel.best.loss_mean == pytest.approx(3.887463e-4, rel=1e-4)
    assert sel.loss_at_epoch == pytest.approx(2.596065e-4, rel=1e-4)
    assert sel.autocorr1 == pytest.approx(-0.0802, abs=1e-3)
    assert [b.admissible for b in sel.bands] == [False, True, True, True, True]
    assert sel.diagnostics.f_stat == pytest.approx(4.2117, abs=1e-3)
    assert sel.diagnostics.f_df == (3, 132)
    assert sel.diagnostics.selection_bias == pytest.approx(0.00956, abs=1e-4)
    assert sel.notes == []


@pytest.mark.parametrize("band_count, epoch, lo, hi", [
    (3, 882, 442, 1041),
    (5, 882, 572, 951),
    (8, 578, 560, 751),
])
def test_regression_band_count_sensitivity(real_history, band_count, epoch, lo, hi):
    sel = select_epoch(real_history, band_count=band_count, diagnostics=False)
    assert sel.epoch == epoch
    assert (sel.best.epoch_lo, sel.best.epoch_hi) == (lo, hi)


def test_regression_fallbacks_agree_on_the_real_run(real_history):
    ref = select_epoch(real_history)
    no_target = {k: v for k, v in real_history.items() if k != "target_history"}
    assert select_epoch(no_target).epoch == 882
    assert select_epoch(no_target).score == pytest.approx(ref.score)
    no_gt = json.loads(json.dumps(real_history))
    del no_gt["validation_params"][0]["dself_S"]["gt"]
    sel = select_epoch(no_gt)
    assert sel.epoch == 882
    assert sel.score == pytest.approx(ref.score)
    assert any("validation_dev_history" in n for n in sel.notes)


# --------------------------------------------------------------------- T12
def test_replica_key_pattern_matches_the_trainer_convention():
    from imolcraft.trainer.trainer import _state_name

    match = selection._MONITOR_RE.match(_state_name(7) + "/x")
    assert match is not None and match.groups() == ("7", "x")
    assert selection._split_monitor(_state_name(7) + "/dself_S") == (7, "dself_S")
    assert selection._RESAMPLED_RE.match(_state_name(3) + "/resampled")
    assert not selection._RESAMPLED_RE.match(_state_name(3) + "/loss")


# --------------------------------------------------------------------- T13
def test_public_names_are_exported_from_the_trainer_package():
    for name in selection.__all__:
        assert hasattr(imolcraft.trainer, name), name
        assert getattr(imolcraft.trainer, name) is getattr(selection, name)
    for name in ("np", "os", "pickle", "re", "argparse", "Any", "Sequence"):
        assert name not in selection.__all__
    assert "main" not in selection.__all__
    assert selection.DEFAULT_BAND_COUNT == 5
    assert selection.DEFAULT_LOSS_TOL == 1.5
    assert selection.DEFAULT_Z == 2.5
    assert selection.DEFAULT_BOOTSTRAP == 2000


# --------------------------------------------------------------------- T14
def test_cli_prints_the_selection_and_the_sensitivity(tmp_path, capsys, real_history):
    pkl = tmp_path / "train_state_s_opt.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(real_history, f)

    assert main([str(pkl), "--sensitivity", "3", "5"]) == 0
    out = capsys.readouterr().out
    assert "adopted epoch = 882" in out
    assert "monitor=sample_0/dself_S" in out
    assert "sensitivity to band_count [3, 5]" in out
    assert "442-1041" in out and "572-951" in out
    assert "F(3,132)" in out
    assert "comparison of the runs" not in out

    other = tmp_path / "other"
    other.mkdir()
    with open(other / "train_state_o.pkl", "wb") as f:
        pickle.dump(real_history, f)
    assert main([str(pkl), str(other), "--no-diagnostics", "--band-count", "8",
                 "--z", "2.5", "--loss-tol", "1.5", "--bootstrap", "10",
                 "--monitor", "sample_0/dself_S"]) == 0
    out = capsys.readouterr().out
    assert "comparison of the runs" in out
    assert "tied set (2/2)" in out
    assert "adopted epoch = 578" in out
    assert "F(" not in out


# --------------------------------------------------------------------- T15
def test_module_source_is_ascii():
    with open(selection.__file__, encoding="utf-8") as f:
        source = f.read()
    assert source.isascii()
    assert not re.search(r"sched_setaffinity|NUM_THREADS", source)
