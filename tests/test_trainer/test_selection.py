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
import inspect
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
    EpochSelection,
    PermutationDiagnostics,
    extract_history,
    format_epoch_selection,
    load_history,
    main,
    select_epoch,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "data", "train_state_history_s_opt.json.gz"
)

#: Points, burn-in points and averaging window of ``_synthetic``.
SYN_POINTS, SYN_BURN, SYN_WINDOW = 100, 20, 16
#: Epoch ``_synthetic`` is built to have the selection adopt.
SYN_EPOCH = 320


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


def _flat_loss(m=SYN_POINTS, burn=SYN_BURN, base=1e-3):
    """A loss that is ``10 * base`` over the burn-in and exactly ``base`` after."""
    loss = np.full(m, base)
    loss[:burn] = 10.0 * base
    return loss


def _synthetic(m=SYN_POINTS, burn=SYN_BURN, zero_at=80.5, slope=-0.02,
               seed=1, loss=None, noise=0.005):
    """
    A run whose deviation crosses zero at a known point.

    ``dev`` is the straight line ``slope * (k - zero_at)`` plus noise, so the
    mean over a window of ``SYN_WINDOW = 16`` points is the value at the
    middle of the window and the best window is the one centred on
    ``zero_at``. With ``burn = 20`` points of ten times the loss the burn-in
    ends at index 20, ``n_used = 80``, ``w = 16`` and the adopted point is
    index ``k + 7`` of the best window, that is epoch ``SYN_EPOCH = 320``.
    The noise (sd 0.005) is far smaller than the step between neighbouring
    window means (``|slope| = 0.02``), so the choice is not a coin toss.
    """
    rng = np.random.default_rng(seed)
    epochs = [4 * k for k in range(m)]
    dev = slope * (np.arange(m) - zero_at) + rng.normal(0.0, noise, m)
    if loss is None:
        loss = _flat_loss(m, burn)
    return _history(epochs, dev, loss)


@pytest.fixture(scope="module")
def real_history():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------- S1
def test_window_size_is_round_n_over_five_clipped_in_integer_arithmetic():
    assert [selection._window_size(n) for n in
            (10, 24, 25, 27, 28, 125, 136, 174, 175, 300)] == \
           [5, 5, 5, 5, 6, 25, 27, 35, 35, 35]
    for n in range(10, 1001):
        w = selection._window_size(n)
        assert type(w) is int
        assert w == min(selection.WINDOW_MAX,
                        max(selection.WINDOW_MIN, (2 * n + 5) // 10))
        assert w == min(35, max(5, int(np.floor(n / 5 + 0.5))))   # half up
    body = inspect.getsource(selection._window_size).split('"""')[2]
    assert "round" not in body          # no banker's rounding, no float


# ---------------------------------------------------------------------- S2
def test_median_smooth_shrinks_the_window_at_the_ends():
    got = selection._median_smooth(np.array([9.0, 8, 7, 1, 1, 1, 1]), 5)
    np.testing.assert_allclose(got, [8.0, 7.5, 7.0, 1.0, 1.0, 1.0, 1.0])
    # shorter than the window: every index is still defined
    np.testing.assert_allclose(selection._median_smooth(np.array([3.0, 1.0]), 5),
                               [2.0, 2.0])
    np.testing.assert_allclose(selection._median_smooth(np.array([5.0]), 5), [5.0])


# ---------------------------------------------------------------------- S3
def test_burn_in_drops_the_front_by_the_smoothed_loss():
    sel = select_epoch(_synthetic(), diagnostics=False)
    assert sel.n_points == SYN_POINTS
    assert sel.burn_in_points == SYN_BURN
    assert sel.burn_in_epoch == 4 * (SYN_BURN - 1)
    assert sel.n_used == SYN_POINTS - SYN_BURN


def test_a_single_loss_spike_does_not_move_the_burn_in():
    """The running median is blind to one outlier among equal neighbours."""
    loss = _flat_loss()
    loss[60] = 1.0                       # a thousand times the plateau
    sel = select_epoch(_synthetic(loss=loss), diagnostics=False)
    assert sel.burn_in_points == SYN_BURN
    assert sel.epoch == SYN_EPOCH


def test_a_flat_loss_drops_nothing():
    sel = select_epoch(_synthetic(loss=np.full(SYN_POINTS, 1e-3)),
                       diagnostics=False)
    assert sel.burn_in_points == 0
    assert sel.burn_in_epoch is None
    assert sel.n_used == SYN_POINTS


def test_the_burn_in_shrinks_as_loss_tol_grows():
    # the loss decays over the first 40 points and is flat afterwards
    loss = np.concatenate([np.geomspace(1e-1, 1e-3, 40), np.full(60, 1e-3)])
    state = _synthetic(loss=loss)
    dropped = [select_epoch(state, loss_tol=t, diagnostics=False).burn_in_points
               for t in (1.0, 1.5, 2.0, 5.0, 50.0)]
    assert dropped == sorted(dropped, reverse=True)
    assert dropped[0] > dropped[-1]


# ---------------------------------------------------------------------- S4
def test_the_window_closest_to_zero_is_adopted_and_the_loss_dip_is_ignored():
    loss = _flat_loss()
    loss[25] = 1e-9                      # a deep dip far from the zero crossing
    loss[95] = 1e-9
    sel = select_epoch(_synthetic(loss=loss), diagnostics=False)
    assert isinstance(sel, EpochSelection)
    assert sel.window == SYN_WINDOW
    assert sel.n_windows == sel.n_used - sel.window + 1 == 65
    assert sel.epoch == SYN_EPOCH
    assert sel.epoch not in (4 * 25, 4 * 95)
    assert sel.window_lo == 4 * 73 and sel.window_hi == 4 * 88
    assert sel.epoch == sel.window_lo + 4 * ((SYN_WINDOW - 1) // 2)
    assert abs(sel.score) < 0.005
    assert sel.ffxml == f"xmlfiles/epoch_t-{SYN_EPOCH}.xml"
    assert sel.monitor == "sample_0/dself_S"
    assert sel.label == "t"


def test_the_adopted_epoch_is_the_centre_point_of_the_window():
    sel = select_epoch(_synthetic(), diagnostics=False)
    ep, dev, _, _, _ = extract_history(_synthetic())
    used = ep[sel.burn_in_points:]
    k = int(np.where(used == sel.window_lo)[0][0])
    assert sel.epoch == used[k + (sel.window - 1) // 2]
    # the ends can never be adopted
    assert sel.epoch not in used[:(sel.window - 1) // 2].tolist()
    assert sel.epoch not in used[len(used) - sel.window // 2:].tolist()
    np.testing.assert_allclose(
        sel.score, dev[sel.burn_in_points:][k:k + sel.window].mean(), atol=1e-12)


# ---------------------------------------------------------------------- S5
def test_the_loss_never_chooses_the_epoch():
    """Any loss with the same burn-in gives the same window."""
    ref = select_epoch(_synthetic(), diagnostics=False)
    rng = np.random.default_rng(5)
    others = []
    for factor in (np.ones(SYN_POINTS),
                   1.0 + rng.uniform(0.0, 0.4, SYN_POINTS),
                   1.0 + 0.3 * (np.arange(SYN_POINTS) % 7)):
        loss = _flat_loss() * factor
        loss[SYN_BURN:] = np.minimum(loss[SYN_BURN:], 1.4e-3)   # keep the gate
        others.append(select_epoch(_synthetic(loss=loss), diagnostics=False))
    for sel in others:
        assert sel.burn_in_points == ref.burn_in_points
        assert (sel.epoch, sel.window_lo, sel.window_hi) == \
               (ref.epoch, ref.window_lo, ref.window_hi)
        assert sel.score == ref.score


# ---------------------------------------------------------------------- S6
def test_the_ends_of_the_run_cannot_be_adopted_and_are_noted():
    dev = np.full(SYN_POINTS, 0.4)
    dev[SYN_BURN:SYN_BURN + 2] = 0.0      # the points closest to zero sit
    dev[-2:] = 0.0                        # at the very ends of the used range
    state = _history([4 * k for k in range(SYN_POINTS)], dev, _flat_loss())
    sel = select_epoch(state, diagnostics=False)
    assert sel.epoch not in (4 * SYN_BURN, 4 * (SYN_BURN + 1),
                             4 * (SYN_POINTS - 2), 4 * (SYN_POINTS - 1))
    assert sel.window_lo == 4 * SYN_BURN          # the first full window
    assert sel.epoch == 4 * (SYN_BURN + (SYN_WINDOW - 1) // 2)
    assert len([n for n in sel.notes if "first or the last full window" in n]) == 1


# ---------------------------------------------------------------------- S7
def test_sigma_is_the_adjacent_difference_estimate_and_the_se_follows_it():
    m = 40
    dev = np.tile([0.0, 1.0], m // 2)
    sel = select_epoch(_history([4 * k for k in range(m)], dev, np.full(m, 1e-3)),
                       diagnostics=False)
    assert sel.n_used == m and sel.window == 8
    np.testing.assert_allclose(sel.sigma, np.sqrt(np.mean(np.diff(dev) ** 2) / 2),
                               atol=1e-12)
    np.testing.assert_allclose(sel.sigma, 1.0 / np.sqrt(2.0), atol=1e-12)
    np.testing.assert_allclose(sel.score_se, sel.sigma / np.sqrt(sel.window),
                               atol=1e-12)
    np.testing.assert_allclose(sel.optimism, 2.0 * sel.score_se, atol=1e-12)


# ---------------------------------------------------------------------- S8
def test_a_costly_window_is_warned_about_but_the_epoch_does_not_move():
    ref = select_epoch(_synthetic(), diagnostics=False)
    costly, cheap = _flat_loss(), _flat_loss(base=1.2e-3)
    costly[70:90] = 1.2e-3                # the adopted window is 73-88
    cheap[70:90] = 1.0e-3
    hot = select_epoch(_synthetic(loss=costly), diagnostics=False)
    cold = select_epoch(_synthetic(loss=cheap), diagnostics=False)
    assert hot.epoch == cold.epoch == ref.epoch
    assert hot.loss_ratio > 1.0 and cold.loss_ratio < 1.0
    assert len([n for n in hot.notes if "higher-loss" in n]) == 1
    assert [n for n in cold.notes if "higher-loss" in n] == []
    np.testing.assert_allclose(hot.window_loss_mean, 1.2e-3, rtol=1e-12)
    np.testing.assert_allclose(hot.loss_median, 1.0e-3, rtol=1e-12)
    np.testing.assert_allclose(hot.loss_at_epoch, 1.2e-3, rtol=1e-12)
    np.testing.assert_allclose(ref.loss_ratio, 1.0, rtol=1e-12)
    assert ref.notes == []                # a flat loss says nothing at all


# ---------------------------------------------------------------------- S9
def test_too_few_points_and_bad_arguments_are_rejected():
    rng = np.random.default_rng(2)
    nine = _history(range(9), rng.normal(0, 0.01, 9), np.full(9, 1e-3))
    with pytest.raises(ValueError, match="loss_tol"):
        select_epoch(nine)
    # 29 points of which 20 are burn-in leave 9
    short = _synthetic(m=29, zero_at=24.5)
    with pytest.raises(ValueError, match="loss_tol"):
        select_epoch(short)
    with pytest.raises(ValueError, match="window"):
        select_epoch(_synthetic(), window=2)
    with pytest.raises(ValueError, match="window"):
        select_epoch(_synthetic(), window=SYN_POINTS - SYN_BURN + 1)
    with pytest.raises(ValueError, match="loss_tol"):
        select_epoch(_synthetic(), loss_tol=0.9)
    assert select_epoch(_synthetic(), window=9, diagnostics=False).window == 9


def test_a_short_run_is_noted_when_the_window_hits_its_limit():
    sel = select_epoch(_synthetic(m=40, zero_at=30.5), diagnostics=False)
    assert sel.n_used == 20 and sel.window == selection.WINDOW_MIN
    assert len([n for n in sel.notes if "lower limit" in n]) == 1


# --------------------------------------------------------------------- S10
def test_the_permutation_test_is_reproducible_and_never_decides(monkeypatch):
    state = _synthetic()
    a = select_epoch(state)
    b = select_epoch(state)
    assert isinstance(a.diagnostics, PermutationDiagnostics)
    assert a.diagnostics == b.diagnostics
    assert a.diagnostics.n_perm == selection.DEFAULT_PERMUTATION
    assert a.diagnostics.seed == 0
    assert 0.0 < a.diagnostics.p_value <= 1.0

    c = select_epoch(state, n_perm=137, seed=3)
    assert c.diagnostics.n_perm == 137 and c.diagnostics.seed == 3
    assert (c.epoch, c.score, c.window_lo) == (a.epoch, a.score, a.window_lo)

    off = select_epoch(state, diagnostics=False)
    assert off.diagnostics is None
    assert off.epoch == a.epoch
    none = select_epoch(state, n_perm=0)
    assert none.diagnostics is None
    assert any("permutation" in n for n in none.notes)

    def boom(*args, **kwargs):            # the decision draws no random number
        raise AssertionError("default_rng must not be called")

    monkeypatch.setattr(np.random, "default_rng", boom)
    assert select_epoch(state, diagnostics=False).epoch == a.epoch


def test_the_permutation_p_value_sees_a_trend():
    state = _synthetic()
    with_trend = select_epoch(state, n_perm=200)
    assert with_trend.diagnostics.p_value < 0.5


# --------------------------------------------------------------------- S11
def test_regression_on_the_real_run(real_history):
    sel = select_epoch(real_history)
    assert sel.monitor == "sample_0/dself_S"
    assert sel.label == "s_opt"
    assert sel.n_points == 171
    assert sel.burn_in_points == 35
    assert sel.burn_in_epoch == 187
    assert sel.n_used == 136
    assert sel.window == 27
    assert sel.n_windows == 110
    assert sel.epoch == 618
    assert (sel.window_lo, sel.window_hi) == (543, 759)
    assert sel.ffxml == "xmlfiles/epoch_s_opt-618.xml"
    np.testing.assert_allclose(sel.score, -0.154686, atol=1e-6)
    np.testing.assert_allclose(sel.sigma, 0.127056, atol=1e-6)
    np.testing.assert_allclose(sel.score_se, 0.024452, atol=1e-6)
    np.testing.assert_allclose(sel.optimism, 2 * sel.score_se, atol=1e-12)
    np.testing.assert_allclose(sel.dev_at_epoch, -0.143042, atol=1e-5)
    np.testing.assert_allclose(sel.loss_at_epoch, 3.836178e-04, rtol=1e-6)
    np.testing.assert_allclose(sel.window_loss_mean, 4.223878e-04, rtol=1e-6)
    np.testing.assert_allclose(sel.loss_median, 3.844891e-04, rtol=1e-6)
    np.testing.assert_allclose(sel.loss_ratio, 1.0986, atol=1e-3)
    assert len([n for n in sel.notes if "higher-loss" in n]) == 1
    assert sel.diagnostics.p_value <= 0.02
    assert (sel.diagnostics.n_perm, sel.diagnostics.seed) == (2000, 0)


def test_the_real_run_keeps_its_epoch_whatever_the_loss_says(real_history):
    """With the burn-in pinned, the loss cannot move the window."""
    ref = select_epoch(real_history, burn_in=187, diagnostics=False)
    assert ref.epoch == 618 and ref.n_used == 136
    scrambled = json.loads(json.dumps(real_history))
    for rec in scrambled["target_history"]:
        if "loss" in rec:
            rec["loss"] = float(rec["loss"]) * (1.0 + 0.3 * (rec["epoch"] % 7))
    sel = select_epoch(scrambled, burn_in=187, diagnostics=False)
    assert (sel.epoch, sel.window_lo, sel.window_hi) == (618, 543, 759)
    assert sel.score == ref.score


# --------------------------------------------------------------------- S12
def test_regression_with_a_hand_set_burn_in(real_history):
    sel = select_epoch(real_history, burn_in=300, diagnostics=False)
    assert sel.n_used == 125
    assert sel.window == 25
    assert sel.epoch == 635
    assert (sel.window_lo, sel.window_hi) == (557, 776)
    np.testing.assert_allclose(sel.score, -0.153765, atol=1e-6)
    np.testing.assert_allclose(sel.sigma, 0.121295, atol=1e-6)
    np.testing.assert_allclose(sel.score_se, 0.024259, atol=1e-6)


def test_regression_fallbacks_agree_on_the_real_run(real_history):
    ref = select_epoch(real_history, diagnostics=False)
    no_target = {k: v for k, v in real_history.items() if k != "target_history"}
    sel = select_epoch(no_target, diagnostics=False)
    assert sel.epoch == 618
    np.testing.assert_allclose(sel.score, ref.score, atol=1e-12)
    no_gt = json.loads(json.dumps(real_history))
    del no_gt["validation_params"][0]["dself_S"]["gt"]
    sel = select_epoch(no_gt, diagnostics=False)
    assert sel.epoch == 618
    np.testing.assert_allclose(sel.score, ref.score, atol=1e-12)
    assert any("validation_dev_history" in n for n in sel.notes)


# ---------------------------------------- kept from the previous rule (T3)
def test_monitor_is_found_when_unique_and_required_otherwise():
    state = _synthetic()
    assert select_epoch(state, diagnostics=False).monitor == "sample_0/dself_S"

    two = _synthetic()
    for key in ("validation_history", "validation_dev_history"):
        for rec in two[key]:
            rec["sample_0/rho"] = 0.5 * rec["sample_0/dself_S"]
    two["validation_params"][0]["rho"] = {"property": "density_gcm3", "gt": 1.0}
    with pytest.raises(ValueError) as err:
        select_epoch(two)
    assert "sample_0/dself_S" in str(err.value) and "sample_0/rho" in str(err.value)
    sel = select_epoch(two, monitor="sample_0/rho", diagnostics=False)
    assert sel.monitor == "sample_0/rho"


# ------------------------------------------------------------------- (T4)
def test_without_gt_the_recorded_deviation_is_used_and_noted():
    state = _synthetic()
    ref = select_epoch(state, diagnostics=False)
    del state["validation_params"][0]["dself_S"]["gt"]
    sel = select_epoch(state, diagnostics=False)
    fallback = [n for n in sel.notes if "validation_dev_history" in n]
    assert len(fallback) == 1 and "metric" in fallback[0]
    # validation_dev_history holds relerr here, so the numbers do not change
    assert sel.epoch == ref.epoch
    np.testing.assert_allclose(sel.score, ref.score, atol=1e-12)
    assert ref.notes == []


def test_dev_is_recomputed_from_gt():
    """The recorded deviation is ignored when gt is present."""
    state = _synthetic()
    for rec in state["validation_dev_history"]:
        rec["sample_0/dself_S"] = 99.0
    ep, dev, loss, monitor, notes = extract_history(state)
    assert ep.shape == dev.shape == loss.shape == (SYN_POINTS,)
    assert np.all(np.abs(dev) < 2.0)
    assert monitor == "sample_0/dself_S" and notes == []


# ------------------------------------------------------------------- (T6)
def test_nan_points_are_dropped_and_counted():
    state = _synthetic()
    for k in (3, 41, 77):
        state["validation_history"][k]["sample_0/dself_S"] = float("nan")
    state["validation_history"][41]["sample_0/dself_S"] = float("inf")
    sel = select_epoch(state, diagnostics=False)
    assert sel.n_points == SYN_POINTS - 3
    assert any("3" in n and "NaN" in n for n in sel.notes)


# ------------------------------------------------------------------- (T7)
def test_every_input_form_gives_the_same_result(tmp_path):
    state = _synthetic()
    ref = select_epoch(state, diagnostics=False)

    trainer_like = types.SimpleNamespace(**state)
    pkl = tmp_path / "train_state_t.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(state, f)

    for source in (trainer_like, pkl, str(pkl), tmp_path):
        sel = select_epoch(source, label="t", diagnostics=False)
        assert sel == ref
    assert select_epoch(pkl, diagnostics=False).label == "train_state_t.pkl"
    assert select_epoch(tmp_path, diagnostics=False).label == tmp_path.name
    assert select_epoch(pkl, label="lr=1e-4", diagnostics=False).label == "lr=1e-4"

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
    assert select_epoch(h, diagnostics=False).epoch == \
        select_epoch(state, diagnostics=False).epoch
    # idempotent, and the fixture is such a reduced dict
    assert load_history(h) == h
    assert set(load_history(real_history)) <= set(selection.HISTORY_KEYS)


# ------------------------------------------------------------------- (T8)
def test_loss_lookup_falls_back_to_epochs_and_losses_and_uses_the_nearest_epoch():
    state = _synthetic()
    ref = select_epoch(state, diagnostics=False)
    without = _synthetic()
    del without["target_history"]
    sel = select_epoch(without, diagnostics=False)
    assert sel.epoch == ref.epoch
    assert sel.loss_at_epoch == ref.loss_at_epoch
    assert sel.window_loss_mean == ref.window_loss_mean

    # a validation record one epoch past the last recorded loss
    last = max(without["epochs"])
    without["validation_history"][-1]["epoch"] = last + 1
    ep, _, loss, _, _ = extract_history(without)
    assert ep[-1] == last + 1
    assert loss[-1] == without["losses"][last]


def test_points_that_are_not_resampled_epochs_are_noted():
    state = _synthetic()
    for rec in state["target_history"]:
        if rec["epoch"] == SYN_EPOCH:
            rec["sample_0/resampled"] = False
    sel = select_epoch(state, diagnostics=False)
    assert any("resampled" in n for n in sel.notes)
    assert sel.epoch == SYN_EPOCH              # the note does not change the choice


# ------------------------------------------------------------------ (T12)
def test_replica_key_pattern_matches_the_trainer_convention():
    from imolcraft.trainer.trainer import _state_name

    match = selection._MONITOR_RE.match(_state_name(7) + "/x")
    assert match is not None and match.groups() == ("7", "x")
    assert selection._split_monitor(_state_name(7) + "/dself_S") == (7, "dself_S")
    assert selection._RESAMPLED_RE.match(_state_name(3) + "/resampled")
    assert not selection._RESAMPLED_RE.match(_state_name(3) + "/loss")


# --------------------------------------------------------------------- S13
def test_public_names_are_exported_from_the_trainer_package():
    for name in selection.__all__:
        assert hasattr(imolcraft.trainer, name), name
        assert getattr(imolcraft.trainer, name) is getattr(selection, name)
    for name in ("np", "os", "pickle", "re", "argparse", "Any", "Sequence"):
        assert name not in selection.__all__
    assert "main" not in selection.__all__
    assert selection.DEFAULT_LOSS_TOL == 1.5
    assert selection.DEFAULT_PERMUTATION == 2000
    assert selection.LOSS_MEDIAN_WINDOW == 5
    assert (selection.WINDOW_MIN, selection.WINDOW_MAX) == (5, 35)
    assert selection.MIN_POINTS == 10
    assert selection.LOSS_WARN_RATIO == 1.0


def test_the_retired_rule_is_gone():
    for name in ("Band", "SelectionDiagnostics", "RunSelection", "select_run",
                 "format_run_selection", "DEFAULT_BAND_COUNT", "DEFAULT_Z",
                 "DEFAULT_BOOTSTRAP"):
        assert not hasattr(selection, name), name
    with open(selection.__file__, encoding="utf-8") as f:
        source = f.read()
    assert "band" not in source.lower()


# --------------------------------------------------------------------- S14
def test_cli_prints_the_selection(tmp_path, capsys, real_history):
    pkl = tmp_path / "train_state_s_opt.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(real_history, f)

    assert main([str(pkl)]) == 0
    out = capsys.readouterr().out
    assert "adopted epoch = 618" in out
    assert "543-759" in out
    assert "monitor=sample_0/dself_S" in out
    assert "+/-" in out and "optimism" in out
    assert "permutation test, 2000 shuffles" in out

    assert main([str(pkl), "--burn-in", "300", "--no-diagnostics"]) == 0
    out = capsys.readouterr().out
    assert "adopted epoch = 635" in out
    assert "permutation" not in out

    assert main([str(pkl), "--window", "27", "--loss-tol", "1.5",
                 "--permutations", "100", "--monitor", "sample_0/dself_S"]) == 0
    out = capsys.readouterr().out
    assert "adopted epoch = 618" in out
    assert "permutation test, 100 shuffles" in out

    for gone in ("--band-count", "--z", "--bootstrap", "--sensitivity"):
        with pytest.raises(SystemExit):
            main([str(pkl), gone, "5"])


def test_the_report_always_states_the_optimism_and_the_error(real_history):
    text = format_epoch_selection(select_epoch(real_history, diagnostics=False))
    assert "[optimism]" in text
    assert "flatters" in text and "worse" in text
    assert "-0.155 +/- 0.024" in text
    assert "sigma of one point 0.127" in text
    assert "[reference only] dev at epoch 618 alone" in text
    assert text.isascii()


# --------------------------------------------------------------------- S15
def test_module_source_is_ascii():
    with open(selection.__file__, encoding="utf-8") as f:
        source = f.read()
    assert source.isascii()
    assert not re.search(r"sched_setaffinity|NUM_THREADS", source)
