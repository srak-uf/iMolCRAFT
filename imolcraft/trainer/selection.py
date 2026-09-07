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
property, ``dev = (pred - gt) / gt``, is a smooth trend plus white noise of
standard deviation ``sigma``. At a resampled epoch the MBAR weights are
uniform, so both the loss and the validation are out of sample. The scatter
between neighbouring points is measurement noise, and the mean over a band
(a block of consecutive evaluation points) is the best estimate of the trend.

Rule
----
1. The evaluation points are the records of ``validation_history`` (the
   resampled epochs). NaN/inf values are dropped and noted.
2. The points are split, in time order, into ``band_count`` bands of equal
   size (``numpy.array_split``); a fixed number of bands keeps the number
   of trials equal between runs.
3. Every band gets ``mean(dev) +/- SE`` and ``mean(loss)``, with
   ``SE = sigma / sqrt(n)`` and ``sigma`` the within-band standard
   deviation pooled over the admissible bands only.
4. The admissible bands are those with
   ``mean(loss) <= loss_tol * min(mean(loss))``: the burn-in falls out by
   itself, and admissibility is decided by the loss alone, never by ``dev``.
5. Among the admissible bands, the one of smallest ``|mean(dev)|`` is the
   best band; its band mean and SE are the validation score of the run.
6. Inside the best band the epochs are equivalent as far as the validation
   can tell, so the epoch of smallest loss in it is adopted.
7. Between runs (a learning-rate sweep, say), the runs whose ``|score|`` is
   within ``z * sqrt(SE_a^2 + SE_b^2)`` of the best one are tied, and the
   tie is broken by the mean loss of the best band.

Diagnostics that never enter the decision but are always reported: the
lag-1 autocorrelation of the within-band residuals (the white-noise
assumption; above ``AUTOCORR_WARN`` the SE is not to be trusted), a one-way F
statistic between the admissible bands (is there an epoch dependence at
all), and a parametric-bootstrap estimate of the selection bias of taking
the smallest ``|mean(dev)|`` of ``band_count`` bands.

Units
-----
``dev`` is recomputed from ``validation_params[i][entry]["gt"]`` and is a
dimensionless signed relative deviation, whatever ``metric`` the YAML asked
for. Only when there is no ``gt`` does the module fall back to
``validation_dev_history``, whose values depend on that ``metric``:
``diff`` is in the unit of the property (cm^2/s for ``dself_cm2s``, g/cm^3
for ``density_gcm3``), so scores of different runs are then comparable only
if they used the same metric. No unit conversion is performed anywhere.

Usage
-----
::

    from imolcraft.trainer.selection import select_epoch, select_run
    sel = select_epoch("train_state_s_opt.pkl")     # or a directory, a dict, the trainer
    print(sel.epoch, sel.ffxml, f"{sel.score:+.3f} +/- {sel.score_se:.3f}")
    best = select_run([select_epoch(p, label=p) for p in ("lr_1e-4", "lr_2e-4")])

    python -m imolcraft.trainer.selection lr_1e-4 lr_2e-4 --sensitivity 3 5 8
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
    "Band", "EpochSelection", "RunSelection", "SelectionDiagnostics",
    "load_history", "extract_history", "select_epoch", "select_run",
    "format_epoch_selection", "format_run_selection",
    "DEFAULT_BAND_COUNT", "DEFAULT_LOSS_TOL", "DEFAULT_Z", "DEFAULT_BOOTSTRAP",
    "HISTORY_KEYS",
]

DEFAULT_BAND_COUNT = 5
DEFAULT_LOSS_TOL = 1.5
DEFAULT_Z = 2.5
DEFAULT_BOOTSTRAP = 2000
#: |lag-1 autocorrelation| of the within-band residuals above which the
#: white-noise assumption behind the SE is questionable.
AUTOCORR_WARN = 0.30

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
# Results
# --------------------------------------------------------------------------
@dataclass
class Band:
    """One block of consecutive validation points."""
    index: int          #: 1-based position in time order
    epoch_lo: int       #: first epoch of the band
    epoch_hi: int       #: last epoch of the band
    n: int              #: number of validation points
    dev_mean: float     #: band mean of the deviation
    dev_se: float       #: sigma / sqrt(n)
    loss_mean: float    #: band mean of the loss
    admissible: bool    #: loss_mean <= loss_tol * min over bands


@dataclass
class SelectionDiagnostics:
    """
    Reported quantities that **never enter the decision**.

    They are kept apart so that a reader can tell which numbers the choice
    rests on: those are ``score`` / ``score_se`` / ``best.loss_mean`` of
    :class:`EpochSelection`, and ``autocorr1`` as a check of the assumption.

    ``f_stat`` / ``f_df``
        One-way F between the admissible bands: is there any epoch
        dependence at all. It was once used to branch the rule ("take the
        last band when not significant") and was dropped: a band mean +/- SE
        is valid whether or not the bands differ, and if the true mean is
        constant the best estimate is the mean over all admissible bands,
        not the last band.
    ``selection_bias``
        How much choosing the smallest ``|dev|`` of ``band_count`` bands
        pulls the score down, measured by a parametric bootstrap that takes
        the estimated band means as truth. The textbook correction
        ``c_B * SE`` (expected maximum of ``B`` standard normals) assumes
        equal true means and overcorrects by a factor 2-15 on real runs, so
        it is not applied. Below the SE the bias is inside the error bar.
    """
    f_stat: float
    f_df: tuple[int, int]
    selection_bias: float
    n_boot: int


@dataclass
class EpochSelection:
    """The epoch chosen for one run, with everything it was chosen from."""
    label: str
    monitor: str                 #: history key followed, e.g. ``sample_0/dself_S``
    n_points: int                #: validation points used (finite ones)
    band_count: int
    sigma: float                 #: pooled within-band sd of dev, the noise of one point
    bands: list[Band]
    best: Band                   #: admissible band of smallest ``|dev_mean|``
    epoch: int                   #: epoch to adopt: smallest loss inside ``best``
    ffxml: str | None            #: force-field file of that epoch, from validation_history
    loss_at_epoch: float
    score: float                 #: ``best.dev_mean``; signed, dimensionless when from gt
    score_se: float              #: ``best.dev_se``
    autocorr1: float             #: lag-1 autocorrelation of the within-band residuals
    notes: list[str] = field(default_factory=list)
    diagnostics: SelectionDiagnostics | None = None

    @property
    def abs_score(self) -> float:
        """``|score|``, what the runs are ranked by."""
        return abs(self.score)


@dataclass
class RunSelection:
    """The run chosen among several :class:`EpochSelection`."""
    ranked: list[EpochSelection]   #: by ``|score|`` ascending
    tied: list[EpochSelection]     #: not distinguishable from ``ranked[0]`` at ``z``
    winner: EpochSelection         #: smallest ``best.loss_mean`` among ``tied``
    z: float

    @property
    def epoch(self) -> int:
        """The epoch of the winning run."""
        return self.winner.epoch


# --------------------------------------------------------------------------
# Selection of the epoch
# --------------------------------------------------------------------------
def _bootstrap_bias(mu: np.ndarray, se: np.ndarray, n_boot: int, seed: int) -> float:
    """Selection bias of ``argmin |mu|`` by parametric bootstrap around ``mu``."""
    if n_boot <= 0 or mu.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    sim = mu + rng.normal(0.0, se, size=(n_boot, mu.size))
    k = np.argmin(np.abs(sim), axis=1)
    rep = np.abs(sim[np.arange(n_boot), k])
    tru = np.abs(mu[k])
    return float((tru - rep).mean())


def _diagnose(mu: np.ndarray, se: np.ndarray, n: np.ndarray, cand: np.ndarray,
              ss: float, dof: int, n_boot: int, seed: int) -> SelectionDiagnostics:
    """The reported-only quantities, from the same band decomposition."""
    nc = n[cand].astype(float)
    if len(cand) > 1 and dof > 0:
        gm = float((nc * mu[cand]).sum() / nc.sum())
        f_stat = float((nc * (mu[cand] - gm) ** 2).sum() / (len(cand) - 1) / (ss / dof))
    else:
        f_stat = float("nan")
    return SelectionDiagnostics(
        f_stat=f_stat, f_df=(len(cand) - 1, dof),
        selection_bias=_bootstrap_bias(mu[cand], se[cand], n_boot, seed),
        n_boot=n_boot)


def _default_label(source: Any, state: dict) -> str:
    if isinstance(source, (str, os.PathLike)):
        return os.path.basename(os.path.normpath(os.fspath(source)))
    return str(state.get("label") or "run")


def select_epoch(
    source: str | os.PathLike | dict | object,
    *,
    label: str | None = None,
    monitor: str | None = None,
    band_count: int = DEFAULT_BAND_COUNT,
    loss_tol: float = DEFAULT_LOSS_TOL,
    diagnostics: bool = True,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> EpochSelection:
    """
    Choose the epoch of one run from its validation history.

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
    band_count : int
        Number of bands of equal size the points are split into (>= 2).
        Together with ``loss_tol`` the only parameter of the decision.
    loss_tol : float
        A band is admissible when its mean loss is at most ``loss_tol`` times
        the smallest band mean loss.
    diagnostics : bool
        Compute the F statistic and the bootstrap selection bias (they do
        not affect the result). The autocorrelation check is always done.
    n_boot, seed : int
        Size and seed (``numpy.random.default_rng``) of the bootstrap; the
        result is deterministic for a given seed.

    Returns
    -------
    EpochSelection
        ``score`` is the band mean of ``dev`` in the best band, a
        dimensionless signed relative deviation unless the run has no ``gt``
        (then see :func:`extract_history` and the ``notes``).

    Raises
    ------
    ValueError
        ``band_count < 2``, fewer than ``2 * band_count`` finite validation
        points, no or ambiguous ``monitor``, no loss series.
    """
    state = load_history(source)
    ep, dev, loss, monitor, notes = extract_history(state, monitor)
    m, B = len(dev), int(band_count)
    if B < 2:
        raise ValueError("band_count must be at least 2")
    if m < 2 * B:
        raise ValueError(
            f"only {m} validation points for band_count={B}; "
            f"at least 2*B={2 * B} are needed")
    if m < 3 * B:
        notes.append(
            f"{m} validation points is few for band_count={B} ({m // B} per "
            f"band): the SE is large and the choice weak")

    idx = np.array_split(np.arange(m), B)
    n = np.array([len(b) for b in idx])
    mu = np.array([dev[b].mean() for b in idx])
    ml = np.array([loss[b].mean() for b in idx])

    # Admissible bands (converged loss). Decided by the loss alone, not dev.
    adm = ml <= loss_tol * float(ml.min())
    cand = np.where(adm)[0]
    if cand.size == 0:                       # numerical safety; cannot happen
        cand = np.array([int(np.argmin(ml))])

    # The noise of one point is pooled over the admissible bands only: a
    # burn-in band moves steeply within itself and would inflate sigma.
    # Since admissibility does not look at dev, this pooling is unbiased.
    ss = float(sum(((dev[idx[i]] - mu[i]) ** 2).sum() for i in cand))
    dof = int(sum(len(idx[i]) for i in cand) - len(cand))
    sigma = float(np.sqrt(ss / dof)) if dof > 0 else float("nan")
    se = sigma / np.sqrt(n)

    # Lag-1 autocorrelation of the within-band residuals: a check of the
    # white-noise assumption the SE (hence the decision) rests on. Measured
    # on the residuals the model actually uses, without pairs across bands,
    # because detrending the whole history with a line leaves the curvature
    # of the burn-in in and inflates the autocorrelation.
    pairs = [(dev[idx[i]] - mu[i]) for i in cand if len(idx[i]) > 2]
    if pairs:
        x = np.concatenate([p[:-1] for p in pairs])
        y = np.concatenate([p[1:] for p in pairs])
        ac1 = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else 0.0
    else:
        ac1 = 0.0
    if abs(ac1) > AUTOCORR_WARN:
        notes.append(
            f"lag-1 autocorrelation of the within-band residuals is {ac1:+.2f}, "
            f"the white-noise assumption is doubtful; correct the SE with the "
            f"effective sample size n*(1-r)/(1+r)")

    j = int(cand[np.argmin(np.abs(mu[cand]))])

    bands = [Band(i + 1, int(ep[b][0]), int(ep[b][-1]), len(b),
                  float(mu[i]), float(se[i]), float(ml[i]), bool(adm[i]))
             for i, b in enumerate(idx)]

    # Inside the best band the points are equivalent in dev (that is what the
    # band resolution means), so the epoch of smallest loss is adopted, not
    # the one of smallest |dev|.
    inb = idx[j]
    pick = int(inb[int(np.argmin(loss[inb]))])

    # The decision ends here. What follows is reported only.
    diag = None
    if diagnostics:
        diag = _diagnose(mu, se, n, cand, ss, dof, n_boot, seed)
        if diag.selection_bias > se[j]:
            notes.append(
                f"the measured selection bias {diag.selection_bias:.3f} exceeds "
                f"the SE {se[j]:.3f}: the score is flattered by that much, say so "
                f"when reporting it")

    epoch = int(ep[pick])
    return EpochSelection(
        label=label or _default_label(source, state), monitor=monitor,
        n_points=m, band_count=B, sigma=sigma, bands=bands, best=bands[j],
        epoch=epoch, ffxml=_ffxml_at(state, epoch),
        loss_at_epoch=float(loss[pick]),
        score=float(mu[j]), score_se=float(se[j]),
        autocorr1=ac1, notes=notes, diagnostics=diag)


# --------------------------------------------------------------------------
# Selection between runs
# --------------------------------------------------------------------------
def select_run(selections: Sequence[EpochSelection], z: float = DEFAULT_Z) -> RunSelection:
    """
    Choose one run among several, e.g. the runs of a learning-rate sweep.

    The runs are ranked by ``|score|``; those within
    ``z * sqrt(SE_a^2 + SE_b^2)`` of the first are tied with it, and the tie
    is broken by the mean loss of the best band. The smallest single-epoch
    loss is deliberately not used for the tie-break: taking the minimum over
    many epochs would bring the selection bias back.

    Parameters
    ----------
    selections : sequence of EpochSelection
        One per run, from :func:`select_epoch`. Their scores must be
        comparable (same ``monitor`` and, when falling back to
        ``validation_dev_history``, the same ``metric``).
    z : float
        Width of the tie in units of the combined SE.

    Raises
    ------
    ValueError
        ``selections`` is empty.
    """
    if not selections:
        raise ValueError("no selections to compare")
    ranked = sorted(selections, key=lambda r: r.abs_score)
    top = ranked[0]
    tied = [top]
    for r in ranked[1:]:
        d = r.abs_score - top.abs_score
        sed = float(np.hypot(r.score_se, top.score_se))
        if d <= z * sed:
            tied.append(r)
    winner = min(tied, key=lambda r: r.best.loss_mean)
    return RunSelection(ranked=ranked, tied=tied, winner=winner, z=z)


# --------------------------------------------------------------------------
# Reports (used by the CLI; a library caller need not call them)
# --------------------------------------------------------------------------
def format_epoch_selection(sel: EpochSelection) -> str:
    """A table of the bands and the choice of one run, as printed by the CLI."""
    out = [f"# {sel.label}   monitor={sel.monitor}   points={sel.n_points}  "
           f"B={sel.band_count}  within-band sd={sel.sigma:.3f}"]
    out.append(f'  {"band":>4}{"epochs":>14}{"n":>5}{"dev":>9}{"+/-SE":>7}'
               f'{"loss(mean)":>13}{"adm.":>6}')
    for b in sel.bands:
        star = " <<" if b is sel.best else ""
        out.append(f'  {b.index:>4}{f"{b.epoch_lo}-{b.epoch_hi}":>14}{b.n:>5}'
                   f'{b.dev_mean:>9.3f}{b.dev_se:>7.3f}{b.loss_mean:>13.3e}'
                   f'{("o" if b.admissible else "x"):>6}{star}')
    out.append(f"  adopted epoch = {sel.epoch}  (band {sel.best.epoch_lo}-{sel.best.epoch_hi}, "
               f"loss at that epoch {sel.loss_at_epoch:.3e}, ffxml {sel.ffxml})")
    out.append(f"  validation score = {sel.score:+.3f} +/- {sel.score_se:.3f}")
    out.append(f"  [assumption] lag-1 autocorrelation of the within-band residuals = "
               f"{sel.autocorr1:+.2f} (SE invalid if |r| > {AUTOCORR_WARN})")
    if sel.diagnostics is not None:
        d = sel.diagnostics
        out.append(f"  [diagnostics, not used] F({d.f_df[0]},{d.f_df[1]}) = {d.f_stat:.2f} "
                   f"(epoch dependence)  selection bias = {d.selection_bias:.3f} "
                   f"(no correction needed if <= SE {sel.score_se:.3f})")
    for note in sel.notes:
        out.append(f"  [note] {note}")
    return "\n".join(out)


def format_run_selection(sel: RunSelection) -> str:
    """The ranking of the runs and the winner, as printed by the CLI."""
    top = sel.ranked[0]
    out = [f'{"run":<22}{"score":>9}{"+/-SE":>7}{"loss(mean)":>12}'
           f'{"epoch":>8}{"z vs 1st":>10}{"verdict":>12}']
    for r in sel.ranked:
        if r is top:
            zs, verdict = "-", "1st"
        else:
            sed = float(np.hypot(r.score_se, top.score_se))
            zs = f"{(r.abs_score - top.abs_score) / sed:.2f}"
            verdict = "tied" if r in sel.tied else "worse"
        out.append(f'{r.label:<22}{r.score:>9.3f}{r.score_se:>7.3f}'
                   f'{r.best.loss_mean:>12.3e}{r.epoch:>8}{zs:>10}{verdict:>12}')
    out.append(f"\n  tied set ({len(sel.tied)}/{len(sel.ranked)}): "
               f"{[r.label for r in sel.tied]}")
    out.append(f"  smallest best-band mean loss in the tied set -> {sel.winner.label}")
    out.append(f"  adopted: {sel.winner.label} / epoch {sel.winner.epoch}  "
               f"score {sel.winner.score:+.3f} +/- {sel.winner.score_se:.3f}")
    if len(sel.tied) > 1:
        out.append("  note: the validation does not separate the tied runs; "
                   "the choice among them is the loss tie-break")
    return "\n".join(out)


def _format_sensitivity(label: str, state: dict, band_counts: Sequence[int],
                        **kw: Any) -> list[str]:
    """One row per ``band_count`` for the sensitivity table of the CLI."""
    rows = []
    for i, B in enumerate(band_counts):
        head = label if i == 0 else ""
        try:
            s = select_epoch(state, label=label, band_count=B, diagnostics=False, **kw)
        except ValueError as exc:
            rows.append(f"{head:<22}{B:>3}  {exc}")
            continue
        rows.append(f'{head:<22}{B:>3}{s.score:>9.3f}{s.score_se:>7.3f}'
                    f'{f"{s.best.epoch_lo}-{s.best.epoch_hi}":>14}'
                    f'{s.epoch:>8}{s.best.loss_mean:>12.3e}')
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command line: ``python -m imolcraft.trainer.selection PATH [PATH ...]``.

    ``PATH`` is a ``train_state_*.pkl`` or a directory holding one. Options:
    ``--band-count``, ``--loss-tol``, ``--z``, ``--monitor``, ``--bootstrap``,
    ``--no-diagnostics`` and ``--sensitivity B [B ...]`` (repeat the analysis
    with other band counts). With several paths the runs are compared.
    """
    p = argparse.ArgumentParser(
        prog="python -m imolcraft.trainer.selection",
        description="Choose the epoch of a run from its validation history.")
    p.add_argument("paths", nargs="+", help="run directory or train_state_*.pkl")
    p.add_argument("--band-count", type=int, default=DEFAULT_BAND_COUNT)
    p.add_argument("--loss-tol", type=float, default=DEFAULT_LOSS_TOL)
    p.add_argument("--z", type=float, default=DEFAULT_Z)
    p.add_argument("--monitor", default=None, help="e.g. sample_0/dself_S")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--no-diagnostics", action="store_true",
                   help="skip the F statistic and the selection bias (not used for the decision)")
    p.add_argument("--sensitivity", type=int, nargs="*", default=None, metavar="B",
                   help="also analyse with these band counts (e.g. 3 5 8)")
    a = p.parse_args(argv)

    # Runs are read one at a time and dropped: a checkpoint with
    # target_log "all" is tens of MB.
    runs = []
    for path in a.paths:
        sel = select_epoch(path, monitor=a.monitor, band_count=a.band_count,
                           loss_tol=a.loss_tol, n_boot=a.bootstrap,
                           diagnostics=not a.no_diagnostics)
        runs.append(sel)
        print(format_epoch_selection(sel), end="\n\n", flush=True)

    if len(runs) > 1:
        print("=== comparison of the runs ===")
        print(format_run_selection(select_run(runs, z=a.z)))

    if a.sensitivity:
        print(f"\n=== sensitivity to band_count {a.sensitivity} ===")
        print(f'{"run":<22}{"B":>3}{"score":>9}{"+/-SE":>7}{"best band":>14}'
              f'{"epoch":>8}{"loss(mean)":>12}')
        for path in a.paths:
            state = load_history(path)
            label = os.path.basename(os.path.normpath(path))
            for row in _format_sensitivity(label, state, a.sensitivity,
                                           monitor=a.monitor, loss_tol=a.loss_tol):
                print(row)
            del state
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
