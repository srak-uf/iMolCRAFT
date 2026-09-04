# Changelog

All notable changes to iMolCRAFT are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions follow `0.MINOR.PATCH`: a release that breaks compatibility in any
way — a renamed or retired YAML key, a moved import path, a changed argument
name, a default that changes results, a new shape for a saved pickle, a
feature switched off — moves the minor number. A release of nothing but bug
fixes and backwards-compatible additions moves the patch number.

## [Unreleased]

### Added

- **A `ThermodynamicTrainer` epoch whose loss comes out NaN or Inf is now
  resampled and computed again** instead of being given up on right away.
  The loss is an MBAR-reweighted average over the stored trajectories, so it
  degenerates once the parameters have drifted away from the state those were
  sampled in; the recovery runs the MD of every replica again with `ffxml`,
  the force field the loss was measured with, and recomputes the loss of the
  same epoch, leaving the parameters untouched. The new
  `nan_resample_retries` argument says how many such rounds are allowed
  (default `1`, `0` switches the recovery off); `from_checkpoint` takes it as
  an override and it rides in the checkpoint with the other restart
  arguments. A loss that survives the retrying falls back to the previous
  behaviour, the parameters being perturbed and the epoch recorded as NaN,
  and `after_step` no longer resamples a second time for a NaN the recovery
  has already resampled for. Note that this cannot rescue a NaN coming out of
  the parameters themselves, such as a sigma that has crossed zero: the MD is
  reseeded on every run, but the force field decides whether the energies are
  finite.
  `BaseTrainer` gained the hook behind it, `recover_from_invalid_loss`, whose
  default does nothing, so the scan trainers are unaffected. `SumTrainer`
  retries each half through the sub-trainer that owns the data.
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

- Removed a dead NaN check in `ThermodynamicTrainer.get_loss_gradients`. It
  read `if jnp.isnan(loss) is True:`, which compares a JAX array with `True`
  by identity and is therefore never taken, and it duplicated what
  `after_step` already does. Behaviour is unchanged, the recovery described
  above having taken its place.
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

[Unreleased]: https://github.com/srak-uf/iMolCRAFT/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/srak-uf/iMolCRAFT/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/srak-uf/iMolCRAFT/releases/tag/v0.3.0
