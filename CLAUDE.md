# iMolCRAFT — notes for Claude Code

iMolCRAFT is a Python package for automated force-field development for
liquid electrolytes and molecular crystal electrolytes, built on DMFF/JAX,
OpenMM and Psi4. Package code lives in `imolcraft/` (`analyzer`, `calculator`,
`crafter`, `io`, `trainer`); tests in `tests/`; docs in `docs/`.

## Branch workflow (always)

- `main` is the release branch. `dev` is the integration branch.
- **Every task gets its own branch, created from `dev`.** Never commit
  directly to `dev` or `main`.
  - Name it `feat/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`
    or `refactor/<topic>` (short kebab-case).
  - Open the pull request against `dev`, not `main`.
- One task, one branch, one PR. If a request contains unrelated changes, make
  separate branches.
- Finish by reporting the branch name and a summary of the changes.

## Coding conventions

- Python 3.11+. Comments, docstrings, log messages and commit messages in
  English.
- Keep the public API stable. Renaming a YAML key, a class/argument name, a
  default that changes results, or the shape of a saved pickle is a breaking
  change and needs an explicit note (see Changelog below).
- Do not add runtime dependencies to `pyproject.toml`. Scientific
  dependencies (openmm, psi4, ambertools, dmff, ...) come from the conda
  environment (`env.yml`, `env_cuda.yml`).

## Changelog and versioning

- `CHANGELOG.md` follows Keep a Changelog. Add an entry under
  `## [Unreleased]` for every user-visible change (`Added` / `Changed` /
  `Fixed` / `Removed`).
- Version is `imolcraft.__version__`, scheme `0.MINOR.PATCH`: a breaking
  change bumps MINOR; bug fixes and compatible additions bump PATCH. Do not
  bump the version unless asked to cut a release.

## Tests

- `pytest` with markers `g16` (needs Gaussian 16), `qm` (needs Psi4 etc.),
  `tp`, `dihedral`, `distance`. CI (`.github/workflows/pytest.yml`) runs each
  `tests/test_*` directory with `-m "not g16"` on a PR to `dev` or `main`.
- Run locally in the `imc_cpu` conda environment, e.g.
  `pytest tests/test_io -m "not g16"`.
- The full stack (openmm, psi4, DMFF) is not pip-installable, so in a
  sandbox without the conda environment most tests cannot run. In that case
  say so plainly in the report rather than claiming the tests passed; at
  minimum run `python -m compileall imolcraft` and any pure-Python tests
  that import cleanly.
