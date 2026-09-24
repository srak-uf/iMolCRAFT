# Changelog

All notable changes to iMolCRAFT are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions follow `0.MINOR.PATCH`: a release that breaks compatibility in any
way — a renamed or retired YAML key, a moved import path, a changed argument
name, a default that changes results, a new shape for a saved pickle, a
feature switched off — moves the minor number. A release of nothing but bug
fixes and backwards-compatible additions moves the patch number.

## [0.4.3] — 2026-09-24

### Changed

- `ThermodynamicTrainer` writes its sampling trajectories as
  `xtcfiles/{label}_sample_{i}.xtc` instead of `xtcfiles/sample_{i}.xtc`. 

### Fixed

- `SumTrainer` now sets the `.loss` of its sub-trainers. Their own `after_step`
  reads it (the target history and the NaN recovery), so with `target_log`
  other than `"none"` a `ThermodynamicTrainer` inside a `SumTrainer` stopped
  with an `AttributeError` at the end of the first epoch.

## [0.4.2] — 2026-09-14

### Changed

- `imolcraft.trainer.selection.select_epoch` now chooses the epoch by smoothing
  the validation deviation instead of by the loss inside a band. The monitored
  property is validation-only (`dself_cm2s` and the like never enter the loss)
  and is uncorrelated with the loss, so picking the smallest loss inside the
  best band was latching onto single-epoch loss dips. The new rule drops the
  burn-in (the points before the running median of the loss first falls to
  `loss_tol` times its smallest value), averages the signed relative deviation
  over a window of `w = clip(round(n / 5), 5, 35)` consecutive points and adopts
  the centre point of the window whose mean is closest to zero. On a held-out
  check this halves the error of the adopted epoch (|dev| 0.153 against 0.227
  for the old rule and 0.242 for no selection). The same run that used to give
  epoch 882 now gives epoch 618.
- `--loss-tol` (and the `loss_tol` argument) keeps its name and its default 1.5
  but now gates the burn-in on the smoothed loss instead of admitting bands.
- `EpochSelection` reports the window instead of the bands: `window`,
  `n_windows`, `window_lo`, `window_hi`, `n_used`, `used_lo`, `used_hi`,
  `burn_in_points`, `burn_in_lo`, `burn_in_epoch`, `burn_in_request`,
  `window_request`, `loss_tol`, `sigma`, `optimism`, `dev_at_epoch`,
  `used_mean`, `window_loss_mean`, `loss_median` and `loss_ratio`. The
  `abs_score` property is gone with `select_run`, the only thing that ranked
  by it. `n_points` now counts every finite validation point of the run,
  before the burn-in. `score` is the window mean and comes with `score_se`; it
  is the smallest of many windows, so it flatters the epoch by about
  `2 * score_se` even with no signal, and the report says so. It is printed
  next to `used_mean`, the mean deviation of every point kept, which is what
  taking no decision at all would give, so that what the window claims to have
  gained can be read against that optimism. `select_epoch` also takes
  `burn_in` (an epoch threshold) and `window` (a window in points) to override
  the two automatic steps. Nothing in the module draws a random number.

### Removed

- The band rule and everything that belonged to it: `Band`,
  `SelectionDiagnostics`, `RunSelection`, `select_run`,
  `format_run_selection`, `DEFAULT_BAND_COUNT`, `DEFAULT_Z`,
  `DEFAULT_BOOTSTRAP`, the `band_count` / `n_boot` arguments of `select_epoch`
  and the `--band-count`, `--z`, `--bootstrap` and `--sensitivity` options of
  `python -m imolcraft.trainer.selection`. Comparing several runs is no longer
  part of the module.
- The permutation test that had been reported alongside the choice, and with
  it `PermutationDiagnostics`, `DEFAULT_PERMUTATION`, the `diagnostics` field
  of `EpochSelection`, the `diagnostics` / `n_perm` / `seed` arguments of
  `select_epoch` and the `--permutations` / `--no-diagnostics` options of the
  command line. It answered a question (is there any epoch dependence at all)
  that is easier to read off `score` against `used_mean` and `optimism`, which
  the report now prints. No part of the module uses `numpy.random` any more.

## [0.4.1] — 2026-09-08

### Added

- `imolcraft.trainer.selection` (new module), which picks the epoch of a
  `ThermodynamicTrainer` run from its validation history, after the fact.
  `select_epoch(train_state)` reads a checkpoint (path, directory, dict or the
  trainer itself), splits the resampled epochs into `band_count` (default 5)
  bands of equal size, drops the bands whose mean loss is above `loss_tol`
  (default 1.5) times the smallest, takes the band whose mean signed relative
  deviation `(pred - gt) / gt` of the monitored property is closest to zero,
  and adopts the epoch of smallest loss inside it; the band mean and its
  standard error are the validation score of the run, and the `ffxml` of the
  adopted epoch is reported with it. `select_run` compares several runs (a
  learning-rate sweep, say): the runs within `z * sqrt(SE_a^2 + SE_b^2)`
  (default `z = 2.5`) of the best are tied, and the tie is broken by the mean
  loss of the best band. Both return dataclasses (`EpochSelection`,
  `RunSelection`) that carry the bands, the notes on the assumptions (lag-1
  autocorrelation of the within-band residuals, points that are not
  resampled epochs, missing `gt`) and, apart from the decision, a one-way F
  statistic and a bootstrap estimate of the selection bias.
  `python -m imolcraft.trainer.selection train_state_x.pkl [...]
  [--sensitivity 3 5 8]` prints the same as tables. The `best_epoch` /
  `best_params` of the trainer are unchanged and still mean the smallest loss.

## [0.4.0] — 2026-09-07

### Added

- `imolcraft.calculator.GMXCalculator` (new module
  `imolcraft/calculator/gmx.py`), which samples one thermodynamic state with
  GROMACS through `gmx grompp` / `gmx mdrun`. It takes the same `.pdb` as
  `MDCalculator`, names its settings exactly as `MDCalculator` does
  (`rcut_nm`, `temperature_K`, `dt_fs`, `nstxout`, `relax_steps`,
  `prod_steps`, `ensemble`, ...), and follows the same `to_dict` /
  `from_dict` and `run(ffxml, trajectory)` contract, so it is a drop-in
  replacement. `run` takes the same `.ffxml` as `MDCalculator.run`: the
  OpenMM `System` is built exactly as `MDCalculator` builds it (same
  `Modeller.addExtraParticles`, `nonbondedMethod`, `nonbondedCutoff` and
  dispersion correction), serialized, and turned into the `.top` / `.gro`
  through `imolcraft.io.exporter(..., fmt="gmx")` in a temporary directory
  that is removed afterwards, while the `.mdp`, `.tpr`, `.log`, `.edr` and
  `.cpt` files of every stage are kept under `workdir` (default
  `gmxfiles/`). Hydrogen constraints are applied from the mdp
  (`constraints = h-bonds`) rather than baked into the exported System. The
  Ewald parameters are OpenMM's: `ewald-rtol`, `ewald-rtol-lj` and
  `fourierspacing` are derived from OpenMM's error tolerance (5e-4) so that
  the splitting parameter is identical; `"LJPME"` writes
  `lj-pme-comb-rule = Geometric` and `vdw-modifier = Potential-Shift`
  explicitly. The pair-list radius is left to GROMACS:
  `verlet-buffer-tolerance` is written at the GROMACS default
  (0.005 kJ/mol/ps) and no `rlist` is, so grompp and mdrun set `rlist` and
  `nstlist` from that tolerance; pass both `verlet-buffer-tolerance = -1`
  and `rlist` through `mdp_extra` to fix the list manually. A single-point energy test against OpenMM (bonded terms agree
  to < 1e-3 kJ/mol, Lennard-Jones to < 1e-3 kJ/mol with PME and < 5e-3
  kJ/mol with LJPME, Coulomb to 3e-5 relative) is
  part of the `gmx`-marked tests. A
  force field with virtual sites is refused with a `ValueError`: the GROMACS
  exporter does not translate OpenMM virtual sites to `[ virtual_sites2 ]`,
  a known limitation of this first version. `anneal_interval` and `rigidWater`
  are accepted for compatibility and have no effect. GROMACS-specific
  settings (`pressure_bar`, `compressibility_bar`,
  `tau_t_ps`, `tau_p_ps`, `tcoupl`, `pcoupl`, `min_steps`, `emtol`,
  `gmx_bin`, `mpi_command`, `ntmpi`, `ntomp`, `maxwarn`, `mdp_templates`,
  `mdp_extra`, `workdir`) are additions. The production trajectory is
  written as `.xtc` under `xtcfiles/` (no `.trr`). The `.mdp` files of
  srak-uf/gromacs_tutorial ship as package data under `imolcraft/data/mdp/`
  and are used as templates.
  The four execution settings (`gmx_bin`, `mpi_command`, `ntmpi`, `ntomp`)
  default to `None` and are resolved at run time as argument > environment
  variable (`IMOLCRAFT_GMX_BIN`, `IMOLCRAFT_GMX_MPI_COMMAND`,
  `IMOLCRAFT_GMX_NTMPI`, `IMOLCRAFT_GMX_NTOMP`; listed in
  `imolcraft.calculator.gmx.GMX_ENV`) > default (`gmx`, no launcher, threads
  left to mdrun), so a batch script can point a trainer at `gmx_mpi` under
  `srun`/`mpirun` without the checkpoint recording the job's launcher. A
  launcher combined with `ntmpi` is refused (`gmx_mpi` does not take
  `-ntmpi`). `mpi_command` may be a list or a string; a string (argument or
  variable) is split like a shell would when the command is built. `ntmpi`
  and `ntomp` must be positive integers, from either source. The resolved
  mdrun command and the source of each value are written to the MD log at
  the start of a run.
- `gmx_sample(...)`, a thin wrapper over `GMXCalculator`.
- pytest marker `gmx` for tests that need the `gmx` binary.
- **The MD software is chosen with one setting**: `MDCalculator(init_structure,
  software="openmm")` (default) or `software="gromacs"` runs the sampling with
  `OpenMMCalculator` or with `GMXCalculator`, which `MDCalculator` now holds
  (`MDCalculator.backend`) and delegates `run`, `to_dict`, `device`, `md_log`
  and `md_logfile` to. The key is named as the QM `software` of the crafter
  is, takes the same lower-case literal names and refuses anything else with
  `Unknown software`. A sampling block of `ThermodynamicTrainer` therefore
  selects the software with `software: gromacs`, per replica, and the choice
  is recorded in the checkpoint like any other setting. `MDCalculator.SETTINGS`
  is the union of both and `imolcraft.calculator.SOFTWARE_SETTINGS` says which
  setting applies to which; a setting of the other software given a
  non-default value is refused with a `ValueError` rather than silently
  ignored (a default value, such as the `anneal_interval: 100` of the
  existing YAML files, is dropped). A force field with virtual sites combined
  with `software="gromacs"` is refused when the trainer is built. `md_sample`
  takes `software` as well.
- `imolcraft.calculator.OpenMMCalculator` (new module
  `imolcraft/calculator/omm.py`), the OpenMM calculator itself, usable
  directly as `GMXCalculator` is.
- `pressure_bar` is now a setting of the OpenMM sampling too (default 1.0 bar),
  so both softwares read the same key.

### Changed

- **The OpenMM implementation moved from `imolcraft/calculator/md.py` to
  `imolcraft/calculator/omm.py` as `OpenMMCalculator`.** `MDCalculator` stays
  in `md.py` as the software-selecting entry point and delegates instead of
  running the MD itself: constructing it, `run`, `to_dict` / `from_dict` and
  the `device` / `md_log` / `md_logfile` attributes are unchanged, but code
  that reached into internals such as `anneal_legs` or `_build_simulation`
  now finds them on `MDCalculator.backend`. The parts shared by both
  softwares (`VALID_ENSEMBLES`, `MD_LOG_MODES`, `NONBONDED_METHODS`,
  `resolve_nonbondedmethod`, the annealing schedule and the MD log) live in
  the new `imolcraft/calculator/_mdcommon.py`, which `omm.py` and `gmx.py`
  both import so that neither depends on the other; the public names are
  still importable from `imolcraft.calculator.md`. `omm.py`, `_mdcommon.py`
  and `md.py` define `__all__`, so `openmm`, `app`, `unit`, `sys` and
  `contextlib` no longer leak into the `imolcraft.calculator` namespace.
- **`MDCalculator.to_dict()` records `software` and `pressure_bar`**, and a
  `gromacs` record carries the GROMACS settings instead of the OpenMM-only
  ones (`anneal_interval`, `rigidWater`). A record written before this
  release restores as `software="openmm"` with `pressure_bar=1.0`, which is
  what it used to do; a record written now cannot be read by an earlier
  version.

### Fixed

- **The OpenMM barostat ignored `pressure_bar` and always ran at 1 bar**, while
  the reweighting used the requested pressure as the PV term, so a sampling
  block with `pressure_bar: 100` sampled 1 bar and analysed 100 bar. All three
  NPT barostats now take `pressure_bar`. Runs at the default 1 bar are
  unaffected; an NPT run with any other `pressure_bar` changes.

## [0.3.2] — 2026-09-05

### Added

- **A `ThermodynamicTrainer` epoch whose loss is NaN or Inf is resampled and
  recomputed** before being given up on. `nan_resample_retries` (default `1`,
  `0` disables) sets how many rounds; it is stored in the checkpoint and
  accepted by `from_checkpoint`. A loss still NaN afterwards falls back to the
  previous perturb-and-continue behaviour.
- **A Lennard-Jones sigma can no longer be driven to zero or below, nor an
  epsilon negative, by an optimizer step.** A parameter that fell below its
  bound is put back to its previous value, with a printed warning. The bounds
  are the new `param_floors` argument of every trainer: `None` (default) is
  `imolcraft.trainer.base.DEFAULT_PARAM_FLOORS`, sigma at `1e-3` nm and
  epsilon at `0`; `{}` bounds nothing. They are stored in the checkpoint and
  accepted by `from_checkpoint`; a checkpoint written before 0.3.2 gets the
  defaults. This changes the trajectory of a run that previously walked
  through a negative epsilon and kept going.
- `ThermodynamicTrainer` keeps a **target history**: one record per epoch of
  what the force field of that epoch gives for every target, next to the
  loss. A record carries `epoch`, `ffxml`, `loss` and, per replica, the loss,
  the effective sample sizes, whether the frames were freshly sampled, and
  the MBAR-reweighted targets keyed `sample_{i}/{target}` (a distribution as
  `sample_{i}/{target}/{kind}`). It lives in `trainer.target_history`, is
  saved in the checkpoint as `target_history` and comes back through
  `from_checkpoint`. The new `target_log` argument picks how much is kept:
  `"low"` the scalars only, `"medium"` (default) the RDF and ADF curves as
  well, `"all"` also the per-frame values behind them whenever a replica was
  resampled, `"none"` nothing.
- Every validation record of `ThermodynamicTrainer` now carries an `ffxml`
  key next to `epoch`: the force field file the values were measured on. It
  traces a record back to its parameters even when the XML files have been
  renamed or a restart broke the numbering. `plot_validation` skips it, as it
  does `epoch`, so it draws no panel for it.

### Fixed

- **The validation history of `ThermodynamicTrainer` labelled every record one
  epoch behind the force field it was measured on.** `after_step` renders the
  force field of the next epoch, `xmlfiles/epoch_<label>-<N+1>.xml`, and
  resamples with it, but the record was stamped with the epoch being run, `N`.
  A record now carries the epoch of the force field that produced it, so it
  matches both the number in the XML file name printed by `Resampling ... by`
  and the epoch of the loss measured on that same force field. The record
  written by `setup` keeps epoch 0, the initial force field, which also
  removes the duplicate epoch 0 that appeared when the first epoch resampled.
  Histories restored from a checkpoint written before this fix keep their old
  labels, so a run continued across it mixes the two conventions.
- **A parameter that came out NaN or Inf ended the run**, the next epoch
  writing a force field OpenMM refuses. Non-finite gradients now take the
  same perturbation route a NaN loss does, and a parameter that is NaN or Inf
  after the update is put back to its previous value, with a printed warning.
- **A NaN loss froze the best force field for the rest of the run.** The best
  is now tracked over the finite losses only; NaN epochs stay in `losses`.
- **A checkpoint without the best-so-far snapshot made a restart report
  `Best Loss: None at epoch None`.** The best loss and its epoch are read off
  the restored history instead; `best_params` stays None.
- Two packmol scratch files are untracked again and, with `.claude/`, ignored.

## [0.3.1] — 2026-08-26

### Added

- Four Li<sup>+</sup> ion force fields shipped with iMolCRAFT and selectable
  by name through the existing `iontype` key: `Gmanr`, `Madrid`, `Wu-Wick` and
  `SMM`, with parameters from <https://doi.org/10.1021/acs.jpcb.3c05591>. The
  names are matched case insensitively and are resolved one ion at
  a time: they supply lithium alone, and every other ion falls back to
  `amber/ions/ionsff99_tip3p.xml`, which stays the default when `iontype` is
  left out entirely. `iontype` still accepts a path — one that exists is used
  as it stands, otherwise it is read relative to the `ffxml` directory of
  openmmforcefields, as before.

### Fixed

- **Every single-atom ion but the first one in the `iontype` library was
  given the wrong Lennard-Jones parameters.** The writer narrowed the atom
  types of the library down to the ion at hand but left the parameter arrays
  full length, and DMFF pairs the two positionally, so Na<sup>+</sup>,
  K<sup>+</sup>, Cl<sup>-</sup> and the rest were all written out with the
  sigma and epsilon of Li<sup>+</sup>, the first entry of
  `amber/ions/ionsff99_tip3p.xml`. Charges were never affected.
- Assigning an ion that the chosen `iontype` file has no parameters for raised
  an `UnboundLocalError` from inside the writer. It now says which element is
  missing from which file.

## [0.3.0] — 2026-08-26

### Added

- `imolcraft.calculator.MDCalculator`, which samples one thermodynamic state
  and remembers how. Its state is fixed at construction while the force field
  and the trajectory are arguments of `run()`, so `to_dict()` hands over the
  complete recipe of a run, defaults included, and `from_dict()` rebuilds it.
- `ThermodynamicTrainer.from_checkpoint()` restores from the pickle alone. The
  checkpoint records the recipe of every replica's MD alongside the arguments
  the trainer was built with, so a restart reads
  `ThermodynamicTrainer.from_checkpoint("train_state_x.pkl")`. Every other
  argument stays available and overrides what the checkpoint says. It runs
  `setup()` itself unless given `setup=False`. A loss function that cannot be
  pickled is recorded as None and asked for; anything registered with
  `add_modifyfn` still has to be registered again.
- A choice of where the MD progress and per-step state data go: `md_log`
  (`"stdout"`, `"file"` or `"none"`) and `md_logfile`, on both `MDCalculator`
  and `ThermodynamicTrainer`. `"file"` writes one log per trajectory under
  `mdlogs/`, and `"none"` attaches no `StateDataReporter` at all.
- Annealing along an arbitrary temperature schedule: `anneal_T` lists the
  temperature corners and `anneal_steps` the MD steps of each leg between
  them, so `anneal_T: [300, 500, 300]` with `anneal_steps: [50000, 100000]`
  heats and then cools. `anneal_interval` sets how often the thermostat set
  point is refreshed along a leg.
- A `validation` section for the thermodynamic trainer. The properties in it
  are computed on every resampled trajectory and recorded, but never enter the
  loss.
- Mean squared displacement and the self-diffusion coefficient in the
  analyzer, with options for saving and plotting them.
- A provenance stamp in every pickle iMolCRAFT writes: the package version and
  the git commit of the source that produced it.
- Sampling settings that may now be left out entirely: `pressure_bar` and
  `dispcorr`, and the MD settings `dt_fs`, `nstxout`, `relax_steps`,
  `prod_steps`, `anneal_T`, `anneal_steps` and `anneal_interval`, whose
  defaults live in `MDCalculator.SETTINGS`.

### Changed

- **`md_sample` moved** from `imolcraft.trainer.dmff_utils` to
  `imolcraft.calculator.md`, where it is a thin wrapper over `MDCalculator`.
- **`md_sample` arguments renamed** after the sampling keys, so a sampling
  block needs no translation: `initialpdb` → `init_structure`, `T` →
  `temperature_K`, `rc` → `rcut_nm`, `dt` → `dt_fs`,
  `useDispersionCorrection` → `dispcorr`.
- **`pressure_bar` is required by the NPT ensembles** and raises when absent,
  where the YAML parser used to inject 1 bar silently. A fixed-volume replica
  may omit it and gets no PV term, where it previously defaulted to 1.0. MBAR
  weights, the loss and the effective sample size are unaffected — a per-state
  constant is absorbed by the free-energy offset — but the absolute `utarget`
  values shift.
- **Trainer checkpoints changed shape.** They gained `md_params`,
  `restart_args`, `initial_ffxml`, `resample_counter` and `loss_fn`, and lost
  the per-field copies of `T_K`, `P_bar`, `rc_nm`, `dt_fs`, `ensemble`,
  `neff`, `relax_steps`, `prod_steps`, `nstxout` and the annealing settings,
  which duplicated `sampling_params` and were never read back. A checkpoint
  written by an earlier version says so rather than restoring half a trainer.
- `VALID_ENSEMBLES`, `MD_LOG_MODES` and the nonbonded-method resolver moved to
  `imolcraft.calculator.md` alongside the MD they describe.
- The annealing ramps continuously to each corner instead of jumping to the
  peak temperature and stepping down in stages, and it lands exactly on the
  temperature it is aimed at.
- The Langevin integrator damping coefficient was adjusted.
- The conda environment moved to NumPy 2.

### Removed

- **The sampling keys `anneal_Tmax`, `anneal_totalsteps` and
  `anneal_totaltime`**, replaced by the schedule above. A YAML still carrying
  them anneals not at all.
- **`DistanceTrainer.from_checkpoint`** raises `NotImplementedError`. It
  restored validation attributes that only `ThermodynamicTrainer` ever sets
  up, so it had been raising `AttributeError` since the validation section
  arrived, and nothing depends on restoring that trainer. `write_checkpoint`
  is untouched and still records everything a restore would need.

### Fixed

- The YAML parser filled in a default named `anneal_totaltime` while the
  trainer read `anneal_totalsteps`, so the setting was never actually
  defaulted.
- `.gitignore` named run artefacts one path at a time and missed every new run
  label, leaving checkpoints, plots and trajectories untracked in the working
  tree. It matches them by the templates the code formats instead.

## 0.2.1 and earlier

Not itemized. `0.2.1` was set on 2026-04-08 and the releases before it were
not tagged, so their history lives in `git log` alone.

[Unreleased]: https://github.com/srak-uf/iMolCRAFT/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/srak-uf/iMolCRAFT/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/srak-uf/iMolCRAFT/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/srak-uf/iMolCRAFT/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/srak-uf/iMolCRAFT/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/srak-uf/iMolCRAFT/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/srak-uf/iMolCRAFT/releases/tag/v0.3.0
