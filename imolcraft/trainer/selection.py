"""
Pick the epoch of a :class:`~imolcraft.trainer.ThermodynamicTrainer` run from
its validation history, after the fact.

The trainer's ``best_epoch`` / ``best_params`` are the epoch of smallest
*loss*. The validation properties (the ``validation`` section of the YAML)
are measured on freshly sampled trajectories and never enter the loss, so
they say something else; this module turns them into a choice of epoch.

Statistical model
-----------------
At every resampled epoch the signed relative deviation of the monitored
property, ``dev = (pred - gt) / gt``, is a smooth trend plus noise of
standard deviation ``sigma`` (about 0.13 for a self-diffusion coefficient
out of a few nanoseconds). One point therefore says almost nothing: the
mean over a window of neighbouring points is the best estimate of the trend
there, and its error is ``sigma / sqrt(window)``.

The loss does **not** say where the validation is good. A property listed in
``imolcraft.trainer.properties.VALIDATION_ONLY_PROPERTIES`` (``dself_cm2s``
and the like) never enters the loss, and on a real run the two are
uncorrelated, so a single-epoch dip of the loss carries no information about
the validated property. The loss is used here for one thing only: to mark
the end of the burn-in.

Rule
----
1. The evaluation points are the records of ``validation_history`` (the
   resampled epochs). NaN/inf values are dropped and noted.
2. Burn-in: the loss is smoothed with a centred running median of
   ``LOSS_MEDIAN_WINDOW`` points and the points before it first falls to
   ``loss_tol`` times its smallest value are dropped. This is the only step
   that looks at the loss.
3. The averaging window is ``w = clip(round(n / 5), WINDOW_MIN, WINDOW_MAX)``
   points, with ``n`` the number of points left.
4. ``dev`` is averaged over every full window of ``w`` consecutive points and
   the window of smallest ``|mean|`` is taken.
5. The adopted epoch is the centre point of that window (the lower of the two
   middle points when ``w`` is even). No loss, no tie-break inside the
   window: the points of a window are equivalent as far as the validation
   can tell.

Everything else -- the noise, the standard error, the mean over all the
points used, the loss of the window -- is reported and never enters the
decision, which is completely deterministic: the module draws no random
number anywhere.

How to read the score
---------------------
``score`` is the mean ``dev`` of the adopted window and is **optimistic**: it
is the smallest ``|mean|`` out of many windows, so even with no epoch
dependence at all a minimum comes out about ``2 * score_se`` below the
typical window. Quote it as ``score +/- score_se`` and expect the true
deviation of the adopted force field to be worse. Read it next to
``used_mean``, the mean over every point the selection kept: the window is
only worth something if it beats ``used_mean`` by more than ``optimism``.
``dev_at_epoch``, the single point of the adopted epoch, scatters by
``sigma`` -- not by ``score_se`` -- and is reference only.

Units
-----
``dev`` is recomputed from ``validation_params[i][entry]["gt"]`` and is a
dimensionless signed relative deviation, whatever ``metric`` the YAML asked
for. Only when there is no ``gt`` does the module fall back to
``validation_dev_history``, whose values depend on that ``metric``:
``diff`` is in the unit of the property (cm^2/s for ``dself_cm2s``, g/cm^3
for ``density_gcm3``), so scores of different runs are then comparable only
if they used the same metric. ``score``, ``score_se``, ``sigma``,
``optimism`` and ``dev_at_epoch`` are all in the unit of ``dev``. The loss is
dimensionless. No unit conversion is performed anywhere.

Usage
-----
::

    from imolcraft.trainer.selection import select_epoch
    sel = select_epoch("train_state_s_opt.pkl")     # or a directory, a dict, the trainer
    print(sel.epoch, sel.ffxml, f"{sel.score:+.3f} +/- {sel.score_se:.3f}")

    python -m imolcraft.trainer.selection lr_1e-4 lr_2e-4
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "EpochSelection",
    "load_history", "extract_history", "select_epoch", "format_epoch_selection",
    "HISTORY_KEYS",
    "DEFAULT_LOSS_TOL",
    "LOSS_MEDIAN_WINDOW", "WINDOW_MIN", "WINDOW_MAX", "MIN_POINTS",
    "LOSS_WARN_RATIO",
]

#: Burn-in gate: the smoothed loss must fall to this times its smallest value.
DEFAULT_LOSS_TOL = 1.5
#: Points of the centred running median the burn-in gate is applied to.
LOSS_MEDIAN_WINDOW = 5
#: Smallest averaging window, in points.
WINDOW_MIN = 5
#: Largest averaging window, in points.
WINDOW_MAX = 35
#: Fewest points that may be left after the burn-in.
MIN_POINTS = 10
#: Warn when the mean loss of the adopted window exceeds this times the median.
LOSS_WARN_RATIO = 1.0

#: Keys of a checkpoint (or attributes of a trainer) the selection reads.
HISTORY_KEYS = ("label", "validation_params", "validation_history",
                "validation_dev_history", "epochs", "losses", "target_history")

# The replica part of a history key, ``sample_{i}/{entry}``. It must agree
# with ``imolcraft.trainer.trainer._state_name``, which is not imported here
# so that this module stays free of dmff/openmm.
_MONITOR_RE = re.compile(r"^sample_(\d+)/(.+)$")
_RESAMPLED_RE = re.compile(r"^sample_(\d+)/resampled$")
_RESERVED_KEYS = ("epoch", "ffxml")


# --------------------------------------------------------------------------
# Reading the training state
# --------------------------------------------------------------------------
def _get(state: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a checkpoint dict or from a trainer instance alike."""
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _resolve_pkl(path: str) -> str:
    """The pkl file ``path`` points at; a directory yields its first train_state*.pkl."""
    if os.path.isdir(path):
        cands = sorted(f for f in os.listdir(path)
                       if f.startswith("train_state") and f.endswith(".pkl"))
        if not cands:
            raise FileNotFoundError(f"no train_state*.pkl found in {path}")
        path = os.path.join(path, cands[0])
    return path


def _as_float(value: Any) -> float:
    """A Python float out of a scalar, a numpy array or a jax array."""
    return float(np.asarray(value))


def _reduce_target_record(record: dict) -> dict:
    """The part of a ``target_history`` record the selection uses."""
    out: dict = {"epoch": int(record["epoch"])}
    if "loss" in record:
        out["loss"] = _as_float(record["loss"])
    for key, value in record.items():
        if _RESAMPLED_RE.match(key):
            out[key] = bool(np.asarray(value))
    return out


def load_history(source: str | os.PathLike | dict | object) -> dict:
    """
    The part of a training state the selection needs, as a plain dict.

    Parameters
    ----------
    source : str, os.PathLike, dict or object
        The path of a ``train_state_*.pkl``, a directory holding one (the
        first in sorted order is taken), the unpickled dict, or a trainer
        instance (read through ``getattr``).

    Returns
    -------
    dict
        Only the keys of :data:`HISTORY_KEYS` that ``source`` has. The
        values of ``losses`` and the ``loss`` of every ``target_history``
        record are converted to ``float``, so no jax array leaks out; a
        ``target_history`` record keeps only ``epoch``, ``loss`` and its
        ``sample_{i}/resampled`` flags. The other keys are passed through
        as they are. The result can be fed back into this function or into
        :func:`select_epoch`.

    Raises
    ------
    FileNotFoundError
        ``source`` is a directory without a ``train_state*.pkl``.
    ValueError
        ``validation_history`` is missing or empty (the run had no
        ``validation`` section, or the checkpoint is of another trainer).
    """
    if isinstance(source, (str, os.PathLike)):
        with open(_resolve_pkl(os.fspath(source)), "rb") as fh:
            source = pickle.load(fh)
    history: dict = {}
    for key in HISTORY_KEYS:
        value = _get(source, key)
        if value is None:
            continue
        if key == "losses":
            value = [_as_float(x) for x in value]
        elif key == "epochs":
            value = [int(e) for e in value]
        elif key == "target_history":
            value = [_reduce_target_record(r) for r in value]
        history[key] = value
    if not history.get("validation_history"):
        raise ValueError(
            "validation_history is empty: train with a validation section "
            "(ThermodynamicTrainer only) before selecting an epoch from it")
    return history


def _auto_monitor(state: Any) -> str:
    """The only ``sample_{i}/{entry}`` key of ``validation_history``."""
    hist = _get(state, "validation_history") or []
    if not hist:
        raise ValueError(
            "validation_history is empty: train with a validation section "
            "before selecting an epoch from it")
    keys = [k for k in hist[0] if k not in _RESERVED_KEYS]
    if not keys:
        raise ValueError("validation_history has no entry to monitor")
    if len(keys) != 1:
        raise ValueError(
            f"more than one validation entry, choose one with monitor=: {keys}")
    return keys[0]


def _split_monitor(monitor: str) -> tuple[int, str]:
    """``(replica index, entry name)`` of a ``sample_{i}/{entry}`` key."""
    match = _MONITOR_RE.match(monitor)
    if match:
        return int(match.group(1)), match.group(2)
    _, _, entry = monitor.partition("/")
    return 0, entry or monitor


def _ground_truth(state: Any, monitor: str) -> float | None:
    """The ``gt`` of ``monitor`` in ``validation_params``, or None."""
    idx, entry = _split_monitor(monitor)
    params = _get(state, "validation_params") or []
    if idx < len(params) and isinstance(params[idx], dict):
        block = params[idx].get(entry)
        if isinstance(block, dict) and block.get("gt") is not None:
            return float(block["gt"])
    return None


def _loss_lookup(state: Any) -> Callable[[int], float]:
    """
    A function ``epoch -> loss`` by nearest recorded epoch.

    ``target_history`` is the first choice; without it ``epochs`` /
    ``losses`` are used. Nearest rather than exact, because a validation
    record can carry the epoch of the force field written *after* the last
    recorded loss (see ``ThermodynamicTrainer._record_validation``).
    """
    th = _get(state, "target_history")
    if th:
        table = {int(h["epoch"]): _as_float(h["loss"]) for h in th if "loss" in h}
        if table:
            keys = np.array(sorted(table))
            vals = np.array([table[k] for k in keys], dtype=float)
            return lambda e: float(vals[int(np.argmin(np.abs(keys - e)))])
    epochs = np.asarray(_get(state, "epochs", []), dtype=float)
    losses = np.asarray(_get(state, "losses", []), dtype=float)
    if epochs.size == 0 or losses.size == 0:
        raise ValueError(
            "no loss series found: target_history and epochs/losses are empty")
    n = min(epochs.size, losses.size)
    epochs, losses = epochs[:n], losses[:n]
    return lambda e: float(losses[int(np.argmin(np.abs(epochs - e)))])


def _resampled_epochs(state: Any) -> set[int]:
    """Epochs at which any replica was resampled, from ``target_history``."""
    th = _get(state, "target_history") or []
    return {
        int(h["epoch"]) for h in th
        if any(bool(np.asarray(v)) for k, v in h.items() if _RESAMPLED_RE.match(k))
    }


def extract_history(
    state: Any, monitor: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[str]]:
    """
    The validation points of a run: ``(epochs, dev, loss, monitor, notes)``.

    Parameters
    ----------
    state : dict or object
        A checkpoint dict, the dict of :func:`load_history`, or a trainer.
    monitor : str, optional
        The history key to follow, ``sample_{i}/{entry}``. Required when the
        run validates more than one entry.

    Returns
    -------
    epochs : ndarray of int, shape (n_points,)
        Epoch of every finite validation record, in increasing order.
    dev : ndarray of float, shape (n_points,)
        Signed relative deviation ``(pred - gt) / gt`` (dimensionless),
        recomputed from the ``gt`` of ``validation_params`` so that it does
        not depend on the ``metric`` the YAML asked for. Without ``gt`` the
        values of ``validation_dev_history`` are used instead -- in the
        unit that metric gives, e.g. the unit of the property for ``diff`` --
        and a note says so.
    loss : ndarray of float, shape (n_points,)
        Loss at the nearest recorded epoch of every point.
    monitor : str
        The key actually followed.
    notes : list of str
        Remarks about the input: the fallback to ``validation_dev_history``,
        the number of NaN/inf points dropped, points that are not resampled
        epochs.
    """
    monitor = monitor or _auto_monitor(state)
    hist = _get(state, "validation_history") or []
    gt = _ground_truth(state, monitor)
    notes: list[str] = []

    if gt is not None and gt != 0.0:
        rows = [(int(h["epoch"]), (float(h[monitor]) - gt) / gt)
                for h in hist if monitor in h and h[monitor] is not None]
    else:
        dev_hist = _get(state, "validation_dev_history") or []
        if not dev_hist:
            raise ValueError(
                f"{monitor} has neither a gt nor a validation_dev_history")
        notes.append(
            "no gt found, so the values of validation_dev_history are used; "
            "they depend on the metric of the YAML (in the unit of the "
            "property for diff), take care when comparing runs")
        rows = [(int(h["epoch"]), float(h[monitor]))
                for h in dev_hist if monitor in h and h[monitor] is not None]

    if not rows:
        raise ValueError(f"no history for {monitor}")
    rows.sort()
    ep = np.array([r[0] for r in rows], dtype=int)
    dev = np.array([r[1] for r in rows], dtype=float)
    ok = np.isfinite(dev)
    if not ok.all():
        notes.append(f"{int((~ok).sum())} NaN/inf validation point(s) dropped")
        ep, dev = ep[ok], dev[ok]

    at_loss = _loss_lookup(state)
    loss = np.array([at_loss(e) for e in ep], dtype=float)

    # Diagnostic only: are the validation points really resampled epochs?
    resampled = _resampled_epochs(state)
    if resampled and not set(ep.tolist()) <= resampled:
        notes.append(
            "some validation points are not resampled epochs; they are not "
            "out of sample, check the validation settings")
    return ep, dev, loss, monitor, notes


def _ffxml_at(state: Any, epoch: int) -> str | None:
    """The ``ffxml`` of the ``validation_history`` record of ``epoch``."""
    for h in _get(state, "validation_history") or []:
        if int(h["epoch"]) == epoch:
            return h.get("ffxml")
    return None


# --------------------------------------------------------------------------
# Pieces of the rule
# --------------------------------------------------------------------------
def _median_smooth(x: np.ndarray, width: int) -> np.ndarray:
    """
    Centred running median of ``x``, with the window shrunk at the ends.

    ``out[i] = median(x[i - width // 2 : i + width // 2 + 1])`` clipped to the
    array, so every index is defined (``width // 2`` points at each end use a
    shorter window, and an even count takes the mean of the two middle
    values, as :func:`numpy.median` does). This is the convention of
    ``pandas.Series.rolling(width, center=True, min_periods=1).median()``.
    Leaving the ends undefined instead would push the start of the usable
    range to ``width // 2`` even for a run without any burn-in.

    Parameters
    ----------
    x : ndarray of float, shape (n,)
    width : int
        Window in points.

    Returns
    -------
    ndarray of float, shape (n,)
        Same unit as ``x``.
    """
    half = int(width) // 2
    n = int(x.size)
    return np.array([float(np.median(x[max(0, i - half):min(n, i + half + 1)]))
                     for i in range(n)], dtype=float)


def _burn_in_start(loss: np.ndarray,
                   loss_tol: float) -> tuple[int, np.ndarray, float]:
    """
    First index at which the run counts as converged, with the gate it passed.

    The loss is smoothed with a centred running median of
    :data:`LOSS_MEDIAN_WINDOW` points and compared with ``loss_tol`` times the
    smallest value of that smoothed series **over the whole run**. Taking the
    minimum over the whole run rather than over the kept part keeps the
    definition from being circular; the burn-in is above the minimum anyway.
    Points after the returned index are never dropped, so the kept range stays
    contiguous.

    Parameters
    ----------
    loss : ndarray of float, shape (n,)
        Loss at every validation point, in time order (dimensionless).
    loss_tol : float
        Multiple of the smallest smoothed loss the gate sits at (>= 1).

    Returns
    -------
    start : int
        Index of the first point that passes the gate (0 when nothing is
        dropped).
    smoothed : ndarray of float, shape (n,)
        The running median the gate was applied to, same unit as ``loss``.
    gate : float
        ``loss_tol * min(smoothed)``, the value the gate sits at.
    """
    smoothed = _median_smooth(loss, LOSS_MEDIAN_WINDOW)
    gate = float(loss_tol) * float(smoothed.min())
    return int(np.argmax(smoothed <= gate)), smoothed, gate


def _window_size(n: int) -> int:
    """
    Averaging window in points: ``round(n / 5)`` clipped to the limits.

    Computed as ``(2 * n + 5) // 10`` in integer arithmetic (half up), which
    equals ``round(n / 5)`` exactly -- ``n / 5`` is never a half-integer -- and
    so depends neither on the banker's rounding of :func:`round` nor on
    floating point.

    Parameters
    ----------
    n : int
        Number of points the window is taken over.

    Returns
    -------
    int
        Window in points, in ``[WINDOW_MIN, WINDOW_MAX]``.
    """
    return min(WINDOW_MAX, max(WINDOW_MIN, (2 * int(n) + 5) // 10))


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    Mean over every full window of ``w`` points, along the last axis.

    Parameters
    ----------
    x : ndarray of float, shape (..., n)
    w : int
        Window in points, ``w <= n``.

    Returns
    -------
    ndarray of float, shape (..., n - w + 1)
        ``out[..., k] = mean(x[..., k:k + w])``, same unit as ``x``. Partial
        windows at the ends are not produced (``mode="valid"``).
    """
    kernel = np.full(int(w), 1.0 / float(w))
    return np.apply_along_axis(np.convolve, -1, x, kernel, "valid")


def _noise_sigma(x: np.ndarray) -> float:
    """
    Noise of a single point from adjacent differences.

    ``sqrt(mean(diff(x) ** 2) / 2)``: for a smooth trend plus white noise the
    differences cancel the trend, so this is insensitive to the drift the
    window means are meant to follow.

    Parameters
    ----------
    x : ndarray of float, shape (n,)
        Points in time order, ``n >= 2``.

    Returns
    -------
    float
        Standard deviation of one point, in the unit of ``x``.
    """
    return float(np.sqrt(np.mean(np.diff(x) ** 2) / 2.0))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class EpochSelection:
    """
    The epoch chosen for one run, with everything it was chosen from.

    The decision rests on ``epoch`` alone; the rest is there to be read with
    it. ``score`` is the smallest ``|mean|`` out of ``n_windows`` windows and
    is therefore optimistic by about ``optimism`` even when the validation
    does not depend on the epoch at all, so it is only meaningful together
    with ``score_se``: the true deviation of the adopted force field is worse
    than ``score``. ``used_mean``, the mean over every point kept, is what no
    decision at all would give, so ``|used_mean| - |score|`` is what the
    window claims to have gained and is worth reading against ``optimism``.
    ``dev_at_epoch`` is a single point and scatters by ``sigma``, not by
    ``score_se``.

    ``score``, ``score_se``, ``used_mean``, ``sigma``, ``optimism`` and
    ``dev_at_epoch`` are in the unit of ``dev``: a dimensionless relative deviation when the entry
    has a ``gt``, otherwise whatever ``metric`` the YAML asked for (see
    :func:`extract_history` and the ``notes``). The losses are dimensionless.
    """
    label: str
    monitor: str                 #: history key followed, e.g. ``sample_0/dself_S``
    n_points: int                #: finite validation points in the run, before the burn-in
    loss_tol: float              #: multiple of the smallest smoothed loss the gate sat at
    burn_in_request: int | None  #: the ``burn_in`` asked for (None: the loss decided)
    burn_in_points: int          #: points dropped at the front
    burn_in_lo: int | None       #: first dropped epoch (None when nothing was dropped)
    burn_in_epoch: int | None    #: last dropped epoch (None when nothing was dropped)
    n_used: int                  #: ``n_points - burn_in_points``
    used_lo: int                 #: first epoch used, the one that passed the burn-in
    used_hi: int                 #: last epoch used, the last of the run
    window: int                  #: averaging window, in points
    window_request: int | None   #: the ``window`` asked for (None: the rule set it)
    n_windows: int               #: ``n_used - window + 1`` full windows
    epoch: int                   #: adopted epoch: centre point of the best window
    ffxml: str | None            #: force-field file of that epoch, from validation_history
    window_lo: int               #: first epoch of the adopted window
    window_hi: int               #: last epoch of the adopted window
    score: float                 #: mean dev over the adopted window, signed
    score_se: float              #: ``sigma / sqrt(window)``
    used_mean: float             #: mean dev over every point used, the no-choice baseline
    sigma: float                 #: noise of one point, from adjacent differences
    optimism: float              #: ``2 * score_se``, how much a minimum flatters with no signal
    dev_at_epoch: float          #: reference only: the single point at the adopted epoch
    loss_at_epoch: float         #: reference only
    window_loss_mean: float      #: reference only: mean loss over the adopted window
    loss_median: float           #: median loss of the points used
    loss_ratio: float            #: ``window_loss_mean / loss_median``
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Selection of the epoch
# --------------------------------------------------------------------------
def _default_label(source: Any, state: dict) -> str:
    if isinstance(source, (str, os.PathLike)):
        return os.path.basename(os.path.normpath(os.fspath(source)))
    return str(state.get("label") or "run")


def select_epoch(
    source: str | os.PathLike | dict | object,
    *,
    label: str | None = None,
    monitor: str | None = None,
    loss_tol: float = DEFAULT_LOSS_TOL,
    burn_in: int | None = None,
    window: int | None = None,
) -> EpochSelection:
    """
    Choose the epoch of one run from its validation history.

    The burn-in is dropped by the loss, the signed relative deviation of the
    monitored property is averaged over every window of ``w`` consecutive
    validation points, and the centre point of the window whose mean is
    closest to zero is adopted. The loss takes no part in that last step: a
    validation-only property does not enter the loss and need not follow it,
    so a single-epoch dip of the loss says nothing about the validation.

    Parameters
    ----------
    source : str, os.PathLike, dict or object
        What :func:`load_history` accepts: a ``train_state_*.pkl``, a
        directory holding one, the checkpoint dict, or the trainer itself.
    label : str, optional
        Name of the run in the report. Defaults to the base name of the
        path, else the ``label`` of the state, else ``"run"``.
    monitor : str, optional
        History key to follow, ``sample_{i}/{entry}``. May be omitted when
        the run validates a single entry.
    loss_tol : float
        Burn-in gate (>= 1): the points before the running median of the loss
        first falls to ``loss_tol`` times its smallest value are dropped.
        This is the only use the decision makes of the loss.
    burn_in : int, optional
        Escape hatch: keep only the points of epoch **greater** than this,
        instead of the automatic burn-in. The default (None) changes nothing.
    window : int, optional
        Escape hatch: averaging window in points, overriding
        ``clip(round(n / 5), WINDOW_MIN, WINDOW_MAX)``. Must be at least 3 and
        at most the number of points kept.

    Returns
    -------
    EpochSelection
        ``score`` is the mean deviation of the adopted window, in the unit of
        ``dev`` (dimensionless unless the run has no ``gt``; see
        :func:`extract_history`). It is the smallest of ``n_windows`` window
        means, so it flatters the epoch by about ``optimism``: report it as
        ``score +/- score_se``.

    Raises
    ------
    ValueError
        ``loss_tol < 1``, fewer than :data:`MIN_POINTS` points left after the
        burn-in, ``window`` outside ``[3, n_used]``, no or ambiguous
        ``monitor``, no validation history, no loss series.
    """
    if loss_tol < 1.0:
        raise ValueError(
            f"loss_tol must be at least 1 (it is a multiple of the smallest "
            f"smoothed loss), got {loss_tol}")
    state = load_history(source)
    ep, dev, loss, monitor, notes = extract_history(state, monitor)
    n_points = int(ep.size)

    # ---- the decision ----------------------------------------------------
    # 1. Burn-in. The only step that looks at the loss. The gate is worked out
    #    even when the caller pins the burn-in, so that the same remarks about
    #    the loss are reported either way; only `start` decides anything.
    gate_start, smoothed_loss, gate = _burn_in_start(loss, loss_tol)
    start = gate_start if burn_in is None \
        else int(np.searchsorted(ep, int(burn_in), side="right"))
    ep_u, dev_u, loss_u = ep[start:], dev[start:], loss[start:]
    n_used = int(ep_u.size)
    if n_used < MIN_POINTS:
        raise ValueError(
            f"only {n_used} validation point(s) left after dropping {start} "
            f"burn-in point(s) out of {n_points}, at least {MIN_POINTS} are "
            f"needed: validate more often, or raise loss_tol (now {loss_tol}) "
            f"if the run converged slowly")

    # 2. Window, in points.
    w = _window_size(n_used) if window is None else int(window)
    if w < 3 or w > n_used:
        raise ValueError(
            f"window must be between 3 and the {n_used} point(s) kept, got {w}")

    # 3. The window of smallest |mean dev|, and its centre point. No loss and
    #    no tie-break here: inside a window the epochs are equivalent as far
    #    as the validation can tell.
    smoothed_dev = _moving_average(dev_u, w)
    k = int(np.argmin(np.abs(smoothed_dev)))
    centre = k + (w - 1) // 2
    epoch = int(ep_u[centre])

    # The decision ends here. Everything below is reported only and cannot
    # move the adopted epoch.
    n_windows = int(smoothed_dev.size)
    sigma = _noise_sigma(dev_u)
    score = float(smoothed_dev[k])
    used_mean = float(dev_u.mean())
    score_se = sigma / float(np.sqrt(w))
    window_loss_mean = float(loss_u[k:k + w].mean())
    loss_median = float(np.median(loss_u))
    loss_ratio = window_loss_mean / loss_median if loss_median else float("nan")

    # "Climbing back" only means something once the loss has been below the
    # gate, so the search starts at the later of the two boundaries: a
    # hand-set burn-in in front of the gate would otherwise report the tail of
    # the burn-in itself as a loss that rose again.
    off = max(start, gate_start)
    above = np.where(smoothed_loss[off:] > gate)[0]
    if above.size:
        notes.append(
            f"the smoothed loss climbs back above {loss_tol:g} x its smallest "
            f"value at epoch {int(ep[off + above[0]])}; those points are kept all "
            f"the same, so that the range used stays contiguous")
    if start < gate_start:
        notes.append(
            f"burn_in={burn_in} keeps {gate_start - start} point(s) from before the "
            f"smoothed loss first falls to {loss_tol:g} x its smallest value (epoch "
            f"{int(ep[gate_start])}), so the run had not converged over part of the "
            f"range used")
    if window is None and n_used < 5 * WINDOW_MIN:
        notes.append(
            f"{n_used} point(s) is few: the window is clipped at its lower limit "
            f"of {WINDOW_MIN} points, so the score is noisier than usual")
    if window is None and n_used > 5 * WINDOW_MAX:
        notes.append(
            f"{n_used} point(s) is many: the window is clipped at its upper limit "
            f"of {WINDOW_MAX} points, so the smoothing is relatively weaker")
    if k == 0 or k == n_windows - 1:
        notes.append(
            "the best window is the first or the last full window of the run: "
            "the deviation may still be moving there, and the epochs within "
            f"{(w - 1) // 2} point(s) of the ends can never be adopted")
    if loss_ratio > LOSS_WARN_RATIO:
        notes.append(
            f"the adopted window sits in a higher-loss part of the run: its mean "
            f"loss is {loss_ratio:.2f} x the median loss of the points used "
            f"({loss_median:.3e}); the epoch is not moved, because the loss is not "
            f"what this rule optimises, but look at the force field before "
            f"shipping it")

    return EpochSelection(
        label=label or _default_label(source, state), monitor=monitor,
        n_points=n_points, loss_tol=float(loss_tol),
        burn_in_request=None if burn_in is None else int(burn_in),
        burn_in_points=start,
        burn_in_lo=int(ep[0]) if start > 0 else None,
        burn_in_epoch=int(ep[start - 1]) if start > 0 else None,
        n_used=n_used, used_lo=int(ep_u[0]), used_hi=int(ep_u[-1]),
        window=w, window_request=None if window is None else int(window),
        n_windows=n_windows,
        epoch=epoch, ffxml=_ffxml_at(state, epoch),
        window_lo=int(ep_u[k]), window_hi=int(ep_u[k + w - 1]),
        score=score, score_se=score_se, used_mean=used_mean,
        sigma=sigma, optimism=2.0 * score_se,
        dev_at_epoch=float(dev_u[centre]), loss_at_epoch=float(loss_u[centre]),
        window_loss_mean=window_loss_mean, loss_median=loss_median,
        loss_ratio=loss_ratio, notes=notes)


# --------------------------------------------------------------------------
# Report (used by the CLI; a library caller need not call it)
# --------------------------------------------------------------------------
def format_epoch_selection(sel: EpochSelection) -> str:
    """
    The choice of one run and how to read it, as printed by the CLI.

    Parameters
    ----------
    sel : EpochSelection
        A result of :func:`select_epoch`.

    Returns
    -------
    str
        Several lines, without a trailing newline. The score is always
        printed with its standard error and with the warning that it is the
        smallest of many windows and therefore optimistic.
    """
    out = [f"# {sel.label}   monitor={sel.monitor}   points={sel.n_points}"]
    if sel.burn_in_request is None:
        why_dropped = (f"the running median of the loss first falls to "
                       f"{sel.loss_tol:g} x its smallest value at epoch {sel.used_lo}")
        why_kept = (f"the running median of the loss is already within "
                    f"{sel.loss_tol:g} x its smallest value at epoch {sel.used_lo}, "
                    f"the first validation point")
    else:
        why_dropped = f"requested by burn_in={sel.burn_in_request}, not by the loss"
        why_kept = (f"burn_in={sel.burn_in_request} is at or before the first "
                    f"validation point, epoch {sel.used_lo}; the loss was not asked")
    if sel.burn_in_points:
        out.append(f"  burn-in: {sel.burn_in_points} point(s) dropped, epochs "
                   f"{sel.burn_in_lo}-{sel.burn_in_epoch} ({why_dropped})")
    else:
        out.append(f"  burn-in: no point dropped ({why_kept})")
    how = ("set by hand" if sel.window_request is not None else
           f"round(n / 5), clipped to [{WINDOW_MIN}, {WINDOW_MAX}]")
    out.append(f"  used {sel.n_used} point(s), epochs {sel.used_lo}-{sel.used_hi}   "
               f"window = {sel.window} points ({how}), "
               f"{sel.n_windows} full windows")
    out.append(f"  adopted epoch = {sel.epoch}   window epochs {sel.window_lo}-{sel.window_hi} "
               f"({sel.window} points; the adopted epoch is the centre point)   "
               f"ffxml {sel.ffxml}")
    out.append(f"  adopted window mean = {sel.score:+.3f} +/- {sel.score_se:.3f}   "
               f"(sigma of one point {sel.sigma:.3f}, SE = sigma / sqrt({sel.window}))")
    gain = abs(sel.used_mean) - abs(sel.score)
    out.append(f"  all {sel.n_used} points used = {sel.used_mean:+.3f}   "
               f"(what taking no decision at all would give; the adopted window is "
               f"{gain:+.3f} closer to zero than that)")
    out.append(f"  [optimism] the window mean is the smallest |window mean| out of "
               f"{sel.n_windows} windows, so it flatters the epoch: even with no epoch "
               f"dependence at all a minimum runs about 2 x SE = {sel.optimism:.3f} low. "
               f"Read the {gain:+.3f} above against that {sel.optimism:.3f}. The true "
               f"deviation of this epoch is worse than the window mean; quote it with "
               f"its SE")
    out.append(f"  [reference only] dev at epoch {sel.epoch} alone = {sel.dev_at_epoch:+.3f} "
               f"(one point scatters by sigma = {sel.sigma:.3f}, not by the SE)")
    out.append(f"  [reference only] loss at epoch {sel.epoch} = {sel.loss_at_epoch:.3e}; "
               f"mean loss over the window {sel.window_loss_mean:.3e}; median loss of the "
               f"points used {sel.loss_median:.3e} (ratio {sel.loss_ratio:.2f})")
    out.append("  [note] the loss never chooses the epoch here: it only marks the end of "
               "the burn-in. A validation-only property does not enter the loss, so a "
               "dip of the loss says nothing about it")
    for note in sel.notes:
        out.append(f"  [note] {note}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command line: ``python -m imolcraft.trainer.selection PATH [PATH ...]``.

    ``PATH`` is a ``train_state_*.pkl`` or a directory holding one; each is
    read, reported and dropped in turn. Options: ``--monitor``,
    ``--loss-tol``, ``--burn-in`` and ``--window``.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        0.
    """
    p = argparse.ArgumentParser(
        prog="python -m imolcraft.trainer.selection",
        description="Choose the epoch of a run from its validation history.")
    p.add_argument("paths", nargs="+", help="run directory or train_state_*.pkl")
    p.add_argument("--monitor", default=None,
                   help="history key to follow, e.g. sample_0/dself_S "
                        "(only needed when the run validates several entries)")
    p.add_argument("--loss-tol", type=float, default=DEFAULT_LOSS_TOL,
                   help="burn-in gate on the running median of the loss "
                        "(default 1.5): the points before it first falls to this "
                        "multiple of its smallest value are dropped")
    p.add_argument("--burn-in", type=int, default=None, metavar="EPOCH",
                   help="use only the points after EPOCH instead of the automatic burn-in")
    p.add_argument("--window", type=int, default=None, metavar="N",
                   help="override the averaging window, in points")
    a = p.parse_args(argv)

    # Runs are read one at a time and dropped: a checkpoint with
    # target_log "all" is tens of MB.
    for path in a.paths:
        sel = select_epoch(path, monitor=a.monitor, loss_tol=a.loss_tol,
                           burn_in=a.burn_in, window=a.window)
        print(format_epoch_selection(sel), end="\n\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
