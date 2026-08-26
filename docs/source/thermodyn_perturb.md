# Trainer: Thermodynamic Gradient
## Overview

The **Thermodynamic Trainer** is a specialized trainer class in iMolCRAFT designed to optimize force field parameters by directly matching thermodynamic properties computed from molecular dynamics (MD) simulations. Unlike conventional force field fitting approaches that rely on isolated geometries or single-state calculations, the thermodynamic gradient method leverages ensemble reweighting techniques to efficiently compute gradients with respect to force field parameters for multiple thermodynamic states simultaneously.

Built on top of [DMFF](https://github.com/deepmodeling/DMFF) (Differentiable Molecular Force Field), the thermodynamic gradient method utilizes automatic differentiation through JAX to enable efficient backpropagation of gradients through complex MD simulations and trajectory reweighting operations.

This approach is particularly powerful for liquid electrolytes and molecular crystal electrolytes, where accurate prediction of thermodynamic properties (density, radial distribution functions, lattice parameters, etc.) is critical for capturing the correct ionic interactions and structural features.

## Theoretical Foundation

### Ensemble Reweighting via MBAR

The core innovation of the thermodynamic gradient method is the use of the **Multistate Bennett Acceptance Ratio (MBAR)** algorithm to reweight MD trajectories. MBAR enables efficient estimation of thermodynamic properties across multiple force field states by reweighting configurations from reference trajectories, avoiding the need for expensive separate simulations for each parameter perturbation.

Key capabilities provided by MBAR:

1. **Observable reweighting**: Estimate thermodynamic properties (density, RDF, etc.) for perturbed force field parameters using trajectories from reference states
2. **Free energy estimation**: Compute relative free energies between different force field states
3. **Effective sample size monitoring**: Assess the quality and reliability of reweighting during optimization

### Differentiable Trajectory Reweighting with DMFF

The key advantage of iMolCRAFT's approach is that all MBAR operations and energy calculations are implemented in a fully differentiable manner using DMFF and JAX. DMFF provides:

- **Differentiable force field evaluation**: Energy functions are implemented with automatic differentiation support through JAX
- **Efficient energy computation**: Neighbor list and cutoff management optimized for GPU execution
- **OpenMM integration**: Seamless compatibility with OpenMM force field specifications (XML format)

Combined with trajectory reweighting, this enables:

- Automatic differentiation through the entire trajectory reweighting pipeline
- Direct computation of gradients of thermodynamic properties with respect to force field parameters
- End-to-end optimization connecting high-accuracy calculations with thermodynamic property matching

## Key Features of iMolCRAFT Thermodynamic Fitting

### Molecular Symmetry Consideration

iMolCRAFT leverages molecular graph analysis to identify equivalent atoms, bonds, and dihedrals within molecules. This symmetry-aware fitting is expected to reduce the number of independent parameters while ensuring chemical consistency and improving transferability across similar molecular systems.

### Charge Scaling Strategy

One of the critical features for electrolyte systems is the **charge scaling** capability, which enables direct optimization of relative charge distributions while maintaining key chemical constraints. iMolCRAFT allows:

- **Independent charge optimization**: Scale charges for different molecule types (ions, solvents) independently during the same optimization run
- **Parameter constraints**: Enforce net charge constraints for ionic species and other chemical requirements using parameter modification functions

Through custom gradient and parameter modification functions, iMolCRAFT can determine the **optimal charge scaling factors** that best reproduce experimental or high-accuracy calculated thermodynamic properties. This is particularly valuable for liquid electrolytes where the correct relative electrostatic strength between cations, anions, and solvent molecules is critical for accurate property prediction.

## ThermodynamicTrainer Architecture

### Workflow

1. **Initialization**: User specifies force field XML files (OpenMM format), PDB structure, thermodynamic states (temperature, pressure, ensemble), and target properties

2. **MD Sampling**: For each thermodynamic state:
   - Generate reference MD trajectories using DMFF-powered force field evaluation through OpenMM
   - Apply annealing and equilibration protocols
   - Store configurations and energies

3. **MBAR Setup**: 
   - Construct MBAR estimator with reference states and trajectories using DMFF's differentiable energy function backend
   - Optimize MBAR weights using PyMBAR backend
   - Prepare energy functions via DMFF for all states with automatic differentiation support

4. **Gradient Computation**:
   - For each thermodynamic state, compute energy function outputs using DMFF's differentiable potential functions
   - Reweight trajectories using MBAR weights
   - Compute observables (density, RDF, ADF) from weighted configurations
   - Compute loss function comparing predicted vs. target properties
   - Backpropagate through DMFF's JAX-based computational graph to obtain parameter gradients

5. **Parameter Update**:
   - Update force field parameters using optimizer (Adam, SGD, etc.)
   - Monitor effective sample sizes
   - Trigger resampling if convergence degrades

6. **Resampling** (when needed):
   - Re-run MD simulations with updated force field parameters
   - Update MBAR estimator with new trajectories and states
   - Ensure consistent sampling across the parameter landscape

### Key Components

#### Loss Functions

The thermodynamic trainer supports multiple observable types and computes a combined loss function:

```{math}
\mathcal{L}_{\text{total}} = \sum_{\text{scalar}} w_i \mathcal{L}_{i} + \sum_{\text{distribution}} w_j \mathcal{L}_{j}
```

where {math}`w_i` and {math}`w_j` are the weight factors (specified in `target_params`) that balance the importance of different properties, and {math}`\mathcal{L}_i`, {math}`\mathcal{L}_j` are the individual loss functions for scalar and distribution observables respectively.

**Scalar Properties** - Direct comparison of thermodynamic properties:

For scalar quantities like density and lattice parameters, the loss is computed as:

```{math}
\mathcal{L}_{\text{scalar}} = \frac{(A_{\text{target}} - A_{\text{pred}})^2}{A_{\text{target}}^2}
```

where {math}`A_{\text{pred}}` is the predicted property value and {math}`A_{\text{target}}` is the target value.

Supported scalar observables:
- **Density** ({math}`\rho`): Mean density of the system
- **Lattice Parameters** ({math}`L_a`, {math}`L_b`, {math}`L_c`): Dimensions of periodic box for crystal systems

**Distribution Similarity Metrics** - Measures for comparing probability distributions (RDF, ADF):

**Wright Factor** - Normalized mean squared error between distributions:

```{math}
W = \frac{\sum_i (g_{\text{ff},i} - g_{\text{gt},i})^2}{\sum_i g_{\text{gt},i}^2}
```

where {math}`g_{\text{ff}}` is the predicted distribution and {math}`g_{\text{gt}}` is the target distribution.

**Jensen-Shannon Divergence** - Symmetric divergence between probability distributions:

```{math}
D_{\text{JS}} = \frac{1}{2} D_{\text{KL}}(g_{\text{ff}} \| M) + \frac{1}{2} D_{\text{KL}}(g_{\text{gt}} \| M)
```

where {math}`M = \frac{1}{2}(g_{\text{ff}} + g_{\text{gt}})` and {math}`D_{\text{KL}}` is the Kullback-Leibler divergence.

Supported distribution observables:
- **Radial Distribution Functions (RDF)**: Pair correlations between atoms
- **Angular Distribution Functions (ADF)**: Angle distributions in molecular systems

#### Usage Examples
You can find the [example](https://github.com/srak-uf/iMolCRAFT/tree/dev/examples/trainer/LiSLBF4)
**Basic Example: Scalar Properties and Distributions**

```python
from functools import partial
from imolcraft.trainer.loss import loss_thermodynamicperturbation

# Choose loss function for distribution metrics:
# Option 1: "wrightfactor" - Normalized mean squared error (default for RDF)
lossfn = partial(loss_thermodynamicperturbation, 
                 losstype_distribfn="wrightfactor")

# Option 2: "jsdivergence" - Symmetric divergence (more robust for ADF)
# lossfn = partial(loss_thermodynamicperturbation, 
#                  losstype_distribfn="jsdivergence")

# Combine scalar properties and distribution functions
target_params = [{
    # Scalar properties
    "density_gcm3": {"gt": 1.2, "weight": 0.1},
    "La_A": {"gt": 20.624, "weight": 0.1},
    "Lb_A": {"gt": 17.707, "weight": 0.1},
    "Lc_A": {"gt": 18.337, "weight": 0.1},
    
    # Distribution functions (reference data stored in txt files)
    "rdf": {
        "Li-O": {"gt": "rdf_Li_O.txt", "elem1": "Li", "elem2": "O", "weight": 1.0, "rcut12_A": 8.0},
        "Li-F": {"gt": "rdf_Li_F.txt", "elem1": "Li", "elem2": "F", "weight": 1.0, "rcut12_A": 8.0},
        "Li-Li": {"gt": "rdf_Li_Li.txt", "elem1": "Li", "elem2": "Li", "weight": 1.0, "rcut12_A": 8.0},
    },
}]
```

iMolCRAFT provides the parser from the [yml file](https://github.com/srak-uf/iMolCRAFT/blob/dev/examples/trainer/LiSLBF4/dmff.yml). 
```
params = parser_dmffyaml("dmff.yml")
```

File format for RDF/ADF data (space or tab-separated, 2 columns):
```
# distance_or_angle   distribution_value
0.0    0.0
0.1    0.0
0.2    0.1
...
```

#### Ensemble Support

The trainer supports multiple MD ensembles:
- **NVT** (Canonical): Fixed volume and temperature
- **NPT** (Isothermal-Isobaric): Fixed pressure and temperature
- **Anisotropic NPT**: Separate pressure control for each axis (for crystal systems)

#### Resampling Strategy

Adaptive resampling is triggered when:
- Loss becomes NaN (indicates poor reweighting)
- Effective sample size drops below user-specified threshold
- Predefined resampling frequency is reached

This ensures the method remains stable across the optimization landscape.

#### Validation Properties

Properties listed in the optional `validation` section of the YAML are computed
on every freshly sampled trajectory and recorded, but they never enter the loss.
They are an independent score, measured on a trajectory run with the current
parameters rather than reweighted from the stored ones.

The properties themselves are declared once, in `imolcraft.trainer.properties`,
whether they are fitted or monitored: `PROPERTY_KINDS` says whether a property is a **scalar**
(one number per frame) or a **distribution** (a curve per frame),
`PROPERTY_KEYS` what it needs to be computed, and `VALIDATION_ONLY_PROPERTIES`
which of them the loss cannot fit. The tables `targets` and `validation` are
checked against are derived from those three.

| `property` | kind | computed value | keys of the property | optional keys |
| --- | --- | --- | --- | --- |
| `density_gcm3` | scalar | density, g/cm^3 | -- | -- |
| `La_A`, `Lb_A`, `Lc_A` | scalar | cell length, A | -- | -- |
| `rdf` | distribution | radial distribution function | `elem1`, `elem2`, `rcut12_A` | `dr_A`, `intermolecular` |
| `adf` | distribution | angle distribution function | `elem1`, `elem2`, `elem3`, `rcut12_A`, `rcut23_A` | -- |
| `dself_cm2s` (validation only) | scalar | self-diffusion coefficient, cm^2/s | `select` | `msd_type`, `fit_range`, `start`, `stop`, `step` |

Every property the loss fits can also be validated -- useful to watch one on a
replica it is not fitted on. The reverse does not hold: `dself_cm2s` is
**validation only** and is rejected in `targets`, because the thermodynamic
perturbation reweights the configurations a trajectory stored, which says
nothing about how fast they interconvert. It has to be measured on the
trajectory as it was run, i.e. at a resampling.

##### Metrics

Every entry with a reference is also recorded as its deviation from it, one
number per entry: they are kept apart rather than summed, so what a run is
judged by stays your call. How that deviation is measured is the `metric` key
of the entry, written next to `gt`:

```yaml
    rho:
        property: density_gcm3
        gt: 1.568
        metric: relerr        # this is the whole syntax
```

Each kind of property has its own set of metric names, and all of them are zero
for a perfect match:

| `metric` | kind | recorded value | |
| --- | --- | --- | --- |
| `relerr` | scalar | `(pred - gt) / gt` | **default**; signed, so the record says in which direction the property is off |
| `absrelerr` | scalar | `abs(pred - gt) / abs(gt)` | the same without its sign |
| `sqrelerr` | scalar | `(pred - gt)^2 / gt^2` | the form the scalar targets are fitted with |
| `diff` | scalar | `pred - gt` | in the unit of the property, not dimensionless |
| `wrightfactor` | distribution | `sum((pred - gt)^2) / sum(gt^2)` | **default**; the metric the loss uses for a fitted distribution |
| `jsdivergence` | distribution | Jensen-Shannon divergence | the loss's other distribution metric |

The rules `parser_dmffyaml` enforces:

- a name from the wrong kind, or one that does not exist, is rejected with the
  list of the names that kind accepts -- there is no silently ignored `metric`
- `metric` without `gt` is rejected too: a deviation needs something to deviate
  from
- for a distribution `gt` is mandatory in any case, and is the same two-column
  file the targets use; a curve cannot be followed epoch by epoch, so its
  deviation is all that is recorded
- for a scalar `gt` is optional: without it the value is still followed, it is
  just not turned into a deviation

```yaml
validation:
    dself_Li:                 # your own name for the entry
        property: dself_cm2s
        select: element Li    # MDAnalysis selection to follow
        msd_type: xyz         # optional, default xyz
        fit_range: [0.1, 0.5] # optional, fraction of the MSD used for the fit
        gt: 1.0e-6            # optional reference value, in cm^2/s
        metric: relerr        # optional, how far it is from gt
    rho:
        property: density_gcm3
        gt: 1.568
    rdf_Li_O:
        property: rdf
        elem1: Li
        elem2: O
        rcut12_A: 8.0
        gt: reference_rdf_Li_O.txt
        metric: wrightfactor  # optional, or jsdivergence
```

```python
params = parser_dmffyaml("dmff.yml")
trainer = ThermodynamicTrainer(
    ...,
    target_params=params["targets"],
    validation_params=params["validation"],
)
```

The results live in the checkpoint, like the rest of the training state:

- `validation_history` / `validation_dev_history`: one record per resampling of
  the values and of their deviations, keyed `sample_{i}/{entry}`
- `validation_pred` / `validation_dev` / `validation_curves`: the latest values,
  the latest deviations, and the curves they came from, the MSD as
  `(lagtime_ps, msd_A2)` columns and a distribution as `(x, pred, gt)` columns

and are restored by `from_checkpoint`. Only the scalars appear among the values,
a distribution having none; the deviations hold every entry that has a
reference. Three figures are written next to the learning curve:
`validation_LABEL.png`, the values against the epoch with their references,
`validation_dev_LABEL.png`, the deviations against the epoch with a line at
zero, and `validation_curves_LABEL_{i}.png`, the curves behind them -- the one
to look at to check that the MSD is straight over the fitted window.

## Usage Example
### 1. Parameter optimization
```python
from imolcraft.trainer import ThermodynamicTrainer
from imolcraft.trainer.loss import loss_thermodynamicperturbation
from imolcraft.trainer.dmff_utils import neutralize
from functools import partial
import jax.numpy as jnp

# Define loss function
lossfn = partial(loss_thermodynamicperturbation, 
                 losstype_distribfn="wrightfactor")

# Initialize trainer
trainer = ThermodynamicTrainer(
    ffxml_list=["Li.xml", "BF4.xml", "SL.xml"],
    nums_ffxml=[1],
    pdbfile="supercell_bonds.pdb",
    loss_fn=lossfn,
    sampling_params=[{
        "init_structure": "supercell_bonds.pdb",
        "temperature_K": 300.0,
        "pressure_bar": 1.0,
        "rcut_nm": 0.8,
        "dt_fs": 1.0,
        "ensemble": "anisonpt",
        "relax_steps": 20000,
        "prod_steps": 100000,
        "nstxout": 1000,
        "neff": 50,
        "dispcorr": True,
        # Optional: heat to 500 K over 50000 steps, then cool back over 100000
        "anneal_T": [300.0, 500.0, 300.0],
        "anneal_steps": [50000, 100000],
    }],
    target_params=[{
        "density_gcm3": {"gt": 0.8, "weight": 0.01},
        "La_A":{...},
        "rdf": {...},
    }],
    opt_fftypes=["NonbondedForce/charge", 
                 "NonbondedForce/epsilon",
                 "NonbondedForce/sigma"],
    label="ff_opt",
    lr=1e-4
)

# If you use parser_dmffyaml,
# trainer = ThermodynamicTrainer(
#     ffxml_list=["BF4_bond.xml", "SL.xml", "Li.xml"],
#     nums_ffxml=[1, 1, 1],
#     pdbfile="supercell_bonds.pdb",
#     loss_fn=lossfn,
#     sampling_params=params["sampling"],
#     target_params=params["targets"],
#     validation_params=params["validation"],
#     opt_fftypes=["NonbondedForce/charge",
#                  "NonbondedForce/epsilon",
#                  "NonbondedForce/sigma"],
#     label="lbs_opt",
#     lr=1e-4
# )

# Optional: Define custom charge scaling with gradient and parameter modification
def grad_modify(grads):
    """Couple gradients between different ionic species for charge neutrality"""
    grads_Li = grads["NonbondedForce"]["charge"][20]
    total_grads_BF4 = jnp.sum(grads["NonbondedForce"]["charge"][0:4])
    scale_grads_BF4 = -1.0 * (grads_Li / total_grads_BF4)
    grads["NonbondedForce"]["charge"] = grads["NonbondedForce"]["charge"].at[0:4].set(
        grads["NonbondedForce"]["charge"][0:4] * scale_grads_BF4
    )
    return grads

def ffparams_modify(ffparams):
    """Maintain charge neutrality constraints"""
    ffparams = neutralize(ffparams, trainer.natoms_list,
                          target_lists=[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]],
                          target_charges=[0.0])
    return ffparams

# Register modification functions
trainer.add_modifyfn("after_grad", grad_modify)
trainer.add_modifyfn("after_update", ffparams_modify)

# Setup and train
trainer.setup()
trainer.fit(num_epochs=100, checkpoint_freq=10)
```

#### Sampling Parameters

`sampling_params` carries one block per replica. Only the settings that say
which state is being sampled are required, since guessing one would quietly fit
something else:

| Key | Meaning |
| --- | --- |
| `init_structure` | PDB or CIF the replica starts from. |
| `temperature_K` | Temperature of the state. |
| `rcut_nm` | Nonbonded cutoff. |
| `nonbondedmethod` | `"PME"` or `"LJPME"`. |
| `ensemble` | One of `nve`, `nvt`, `isonpt`, `anisonpt`, `trinpt`. |
| `neff` | Effective sample count below which the replica is resampled. |

Everything else may be left out:

| Key | Left out |
| --- | --- |
| `pressure_bar` | No PV term, which is what a fixed volume means. Required for the NPT ensembles, where a barostat needs it. |
| `dispcorr` | No dispersion correction. |
| `dt_fs`, `nstxout`, `relax_steps`, `prod_steps` | `MDCalculator` uses its own defaults. |
| `anneal_T`, `anneal_steps`, `anneal_interval` | No annealing. |

Each block becomes one `imolcraft.calculator.MDCalculator`, which is what
actually runs the MD. The calculator names its settings exactly as the keys
above are named, so a sampling block needs no translation and the defaults of
the second half live in `MDCalculator.SETTINGS` rather than being restated
here. `neff` and `pressure_bar` are not settings of the MD and stay with the
trainer.

#### Restarting from a Checkpoint

A checkpoint records the complete recipe of every replica's MD, defaults
filled in, alongside the arguments the trainer was built with. Restarting
therefore needs nothing but the checkpoint:

```python
trainer = ThermodynamicTrainer.from_checkpoint("train_state_ff_opt.pkl")
trainer.fit(500, 2)
```

`from_checkpoint` runs `setup()` itself, so the returned trainer samples the
restored force field and is ready to fit. Pass `setup=False` to get it back
without running the MD. Any argument given overrides what the checkpoint says,
which is how a run is resumed with a different learning rate or on a different
device.

Two things cannot be recorded and have to be supplied again:

- a `loss_fn` written as a lambda, since it cannot be pickled. It is stored as
  None and `from_checkpoint` then asks for it. A `partial` of a module-level
  loss, as in the example above, is recorded fine.
- anything registered with `add_modifyfn`, which has to be registered again
  after the restart.

The MD is rebuilt from the recorded recipe rather than re-derived from
`sampling_params`, so a default that changed since the checkpoint was written
cannot silently change what gets sampled.

#### Simulated Annealing

Each replica may walk its thermostat through an arbitrary temperature schedule
before the relaxation at `temperature_K` begins. Two sampling keys describe it:

| Key | Meaning |
| --- | --- |
| `anneal_T` | Temperature corners of the schedule, in kelvin. |
| `anneal_steps` | MD steps spent on each leg between them, so one entry fewer than `anneal_T`. |
| `anneal_interval` | How often, in MD steps, the set point is refreshed along a leg (default 100). |

`anneal_T: [300, 500, 300]` with `anneal_steps: [50000, 100000]` heats from
300 K to 500 K over 50000 steps and cools back over 100000. The set point moves
linearly along each leg and lands exactly on the corner, so the ramp reads as
continuous rather than as a handful of jumps. Leaving `anneal_T` or
`anneal_steps` out, or setting either to `null`, skips the annealing entirely.
Both are sequences, a list in the YAML or a list or tuple from Python: a ramp
needs a temperature to start from and one to end at, so a bare number is
rejected rather than read as a one-leg schedule.

The schedule is per replica, so a multi-state fit can anneal each state
differently, and it is recorded in the checkpoint alongside the other sampling
settings.

#### MD Logging

The MD of every replica reports its progress and per-step state data through
one destination, chosen with the `md_log` argument of `ThermodynamicTrainer`:

| `md_log` | Destination |
| --- | --- |
| `"stdout"` | The terminal (default). |
| `"file"` | `md_logfile`, or `mdlogs/<state name>.log` when that is left out, giving one file per replica. |
| `"none"` | Nothing is written and no reporter is attached at all. |

```python
trainer = ThermodynamicTrainer(..., md_log="file")
```

Passing an explicit `md_logfile` makes every replica share, and overwrite, that
one file; leave it out to keep the per-replica default.

#### Gradient and Parameter Modification Functions

The example above demonstrates two key custom modification functions that enable advanced charge scaling strategies:

**`grad_modify(grads)` - Gradient modification for charge coupling**

This function is called after computing gradients but before parameter updates. It couples gradients between different ionic species to maintain charge relationships:

- **Input**: Dictionary of gradients with structure `grads["NonbondedForce"]["charge"][atom_index]`
- **Operation**: 
  - Extract gradient of Li charge at atom index 20
  - Sum gradients for all BF₄⁻ atoms (indices 0:4)
  - Compute scaling factor to ensure opposite charge changes (e.g., Li⁺ increases → BF₄⁻ decreases proportionally)
  - Apply scaled gradients back to BF₄ atoms
- **Purpose**: Ensures charge neutrality is maintained during optimization while allowing independent scaling of cation vs. anion charges

**`ffparams_modify(ffparams)` - Parameter modification with hard constraints**

This function is called after parameter updates to enforce chemical constraints:

- **Input**: Dictionary of force field parameters
- **Operation**: Uses `neutralize()` utility to restore net charge constraints
  - `target_lists`: List of atom indices for each constraint group (e.g., all BF₄ atoms)
  - `target_charges`: Target net charge for each group (e.g., -1.0 for BF₄⁻)
- **Purpose**: Ensures that any numerical drift from charge updates is corrected, maintaining chemical validity

**`trainer.add_modifyfn(hook_point, function)` - API for registering modifications**

This method registers custom modification functions at specific points in the optimization loop:

- **Hook points available**:
  - `"after_grad"`: Called after gradient computation, before parameter update (modify gradients)
  - `"after_update"`: Called after parameter update (enforce hard constraints)

- **Usage**:
  ```python
  trainer.add_modifyfn("after_grad", grad_modify)
  trainer.add_modifyfn("after_update", ffparams_modify)
  ```

- **Multiple modifications**: You can register multiple functions for the same hook point; they will be called in order of registration

This approach enables sophisticated parameter optimization strategies while maintaining chemical validity throughout training.

### 2. Results and restarts
The optimization results are stored in the `train_state_....pkl` file.  
For example, you can plot the training curve by the following script.
```
import pickle
import matplotlib.pyplot as plt

with open("train_state_2rdf_1adf.pkl", mode="rb") as f:
    l = pickle.load(f)

plt.plot(l["epochs"], l["losses"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
```

You can restart the calculation by using the `train_state_....pkl` and `chkpoint....xml` files.  
```
params = parser_dmffyaml("dmff_prod.yml")
lossfn = partial(loss_thermodynamicperturbation, losstype_distribfn="wrightfactor")

trainer = ThermodynamicTrainer.from_checkpoint(trainer_checkpoint="train_state.pkl",
                                     ffxml_list=["BF4_bond.xml", "SL.xml", "Li.xml"],
                                     nums_ffxml=[1,1,1],
                                     pdbfile="supercell_bonds.pdb",
                                     initial_ffxml="chkpoint.xml",
                                     loss_fn=lossfn,
                                     sampling_params=params["sampling"],
                                     target_params=params["targets"],
                                     validation_params=params["validation"],
                                     lr=lr_prod,
)

trainer.add_modifyfn("after_grad", grad_modify)
trainer.add_modifyfn("after_update", ffparams_modify)
trainer.setup()
trainer.fit(100, 2)
```

## Advantages

1. **Thermodynamic Accuracy**: Directly optimizes force fields to match experimental or high-accuracy calculated properties
2. **Differentiability**: Full gradient information through DMFF enables efficient optimization with modern optimizers
3. **Symmetry-Aware Fitting**: Automatically respects molecular symmetry, reducing parameter space and improving transferability
4. **Optimal Charge Scaling**: Determines the optimal charge scaling factors for different molecule types (ions vs. neutral molecules) through gradient-based optimization
5. **Multi-State Efficiency**: MBAR reweighting avoids expensive re-simulation for each parameter change
6. **Adaptive Sampling**: Automatic resampling ensures robust convergence
7. **Specialized for Electrolytes**: Optimized for systems with strong electrostatic interactions via charge scaling method

## References

- X. Wang, L. Zhang,\* K. Yu,\*  *et al.* , "DMFF: An Open-Source Automatic Differentiable Platform for Molecular Force Field Development." *J. Chem. Theory and Comput.*, 19(17), 5897-5909 (2023). doi: [10.1021/acs.jctc.2c01297](https://doi.org/10.1021/acs.jctc.2c01297)
