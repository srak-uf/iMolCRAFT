# Changelog

All notable changes to iMolCRAFT are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions follow `0.MINOR.PATCH`: a release that breaks compatibility in any
way — a renamed or retired YAML key, a moved import path, a changed argument
name, a default that changes results, a new shape for a saved pickle, a
feature switched off — moves the minor number. A release of nothing but bug
fixes and backwards-compatible additions moves the patch number.

## [Unreleased]

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

[Unreleased]: https://github.com/srak-uf/iMolCRAFT/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/srak-uf/iMolCRAFT/releases/tag/v0.3.0
