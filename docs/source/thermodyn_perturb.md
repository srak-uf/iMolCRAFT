# Thermodynamic Gradient
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
        "ensemble": "anisonpt",
        "relax_steps": 20000
        "prod_steps": 100000,
        "nstxout": 1000,
        "neff": 50,
        "dispcorr": True
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
