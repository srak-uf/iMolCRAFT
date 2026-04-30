.. index:: Methods
.. index:: Background

Methods
=========

Force Field
----------------
iMolCRAFT uses classical force fields to perform molecular dynamics simulations of organic molecules.

1. GAFF (General AMBER Force Field)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. math::
    :nowrap:

    \begin{align}
    U(\mathbf{r}) &= \sum_{\mathrm{bonds}} K_{r}(r - r_{\mathrm{eq}})^2 + \sum_{\mathrm{bonds}} K_{\theta}(\theta - \theta_{\mathrm{eq}})^2 +  \\ 
    & \sum_{\mathrm{dihedrals}} K_{\mathrm{d}} [1+\mathrm{cos}(n\phi - \gamma)] + \sum_{i<j} 4\epsilon_{ij} \left[\left(\frac{\sigma_{ij}}{r_{ij}} \right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}} \right)^{6}\right] +\\
    & \sum_{i<j} \frac{q_i q_j}{4\pi \epsilon_0 r_{ij}}
    \end{align}


* Point charges
    The structure is optimized at the DFT level (default: ωB97X-D/6-311+G(2d,p)). After that, the point charges are calculated by the RESP (Restrained ElectroStatic Potential) method.
    The charge of ionic species is usually scaled, e.g., by a factor of 0.8 for +/- 1.0 ions, to account for polarization effects in condensed phases.

* Dihedral parameters
    Sometimes, the default GAFF parameters may not accurately reproduce the conformational preferences of certain molecules.
    The dihedral parameters are determined by fitting to the quantum mechanical (QM) potential energy surface (PES) obtained from relaxed scans of the dihedral angles at the DFT level as mentioned in :ref:`autodiff`.

2. OPLS-AA (Optimized Potentials for Liquid Simulations All Atom)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
   Support for OPLS-AA force field is planned for a future release. This will enable optimization of OPLS-AA parameters alongside GAFF, expanding the range of systems that can be modeled with iMolCRAFT.


3. AMOEBA (Atomic Multipole Optimized Energetics for Biomolecular Applications)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
   Support for AMOEBA force field is planned for a future release. AMOEBA's polarizable model will enable more accurate simulations of systems with significant polarization effects.

.. _autodiff:

Automatic differentiation
-------------------------
iMolCRAFT is a wrapper of DMFF or JAX-MD software to perform force field calculations including automatic-differentiation functions.
This allows users to easily implement new force field functional forms and the parameter optimization.
One of the examples is the dihedral parameter fitting. The sample jupyter notebook is provided in the examples folder.

Thermodynamic gradient
-----------------------
iMolCRAFT can optimize force field parameters by directly matching thermodynamic properties computed from molecular dynamics simulations, using a method called **Thermodynamic gradient**. This approach is particularly powerful for liquid electrolytes and molecular crystal electrolytes.

The method combines two key innovations:

* **Trajectory reweighting via MBAR**: The Multistate Bennett Acceptance Ratio (MBAR) algorithm reweights MD trajectories to efficiently estimate thermodynamic properties (density, RDF, lattice parameters) for perturbed force field parameters without expensive separate simulations.

* **Differentiable computation with DMFF**: All energy calculations and reweighting operations are implemented as differentiable operations using DMFF (backend) and JAX, enabling automatic differentiation through the entire optimization pipeline.

Key advantages of this approach include:

* **Molecular symmetry consideration**: Automatically identifies equivalent atoms and dihedrals, reducing parameter space while ensuring chemical consistency.
* **Optimal charge scaling**: Determines the optimal charge scaling factors for different molecule types through gradient-based optimization, crucial for electrolyte systems.
* **Multi-state efficiency**: Avoids re-simulation for each parameter change by reweighting configurations from reference trajectories.
