.. index:: theory
.. index:: Background

Theory
=========

Force Field
----------------
iMolCRAFT uses classical force fields to perform molecular dynamics simulations of organic molecules.

1. GAFF (General AMBER Force Field)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. math::
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

.. _autodiff:

Automatic differentiation
-------------------------
iMolCRAFT is a wrapper of DMFF or JAX-MD software to perform force field calculations including automatic-differentiation functions.
This allows users to easily implement new force field functional forms and the parameter optimization.
One of the examples is the dihedral parameter fitting. The sample jupyter notebook is provided in the examples folder.

Thermodynamic gradient
-----------------------
bbb
