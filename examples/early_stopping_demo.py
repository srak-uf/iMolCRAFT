#!/usr/bin/env python3
"""
Demonstration of early stopping functionality in iMolCRAFT trainers.

This example shows how to use the new early stopping parameters to automatically
stop training when the loss function converges, eliminating the need for manual
monitoring.
"""

# Example usage of early stopping with a trainer

# Previously, you had to manually monitor the loss and stop training:
"""
trainer = ThermodynamicTrainer(
    ffxml_list=ffxml_list,
    nums_ffxml=nums_ffxml,
    pdbfile=pdbfile,
    loss_fn=lossfn,
    opt_fftypes=opt_fftypes,
)

# Manual training with visual inspection needed
trainer.fit(100, 10)  # Run for 100 epochs, hope it converges
# ... check loss curve manually ...
trainer.fit(50, 10)   # Run more if needed
# ... repeat until satisfied ...
"""

# Now with early stopping, you can set it up once and let it run:
"""
trainer = ThermodynamicTrainer(
    ffxml_list=ffxml_list,
    nums_ffxml=nums_ffxml,
    pdbfile=pdbfile,
    loss_fn=lossfn,
    opt_fftypes=opt_fftypes,
    # New early stopping parameters:
    early_stopping=True,       # Enable early stopping
    patience=10,               # Wait 10 epochs after convergence before stopping
    min_delta=1e-6,           # Minimum improvement to count as progress
    convergence_window=5,     # Look at last 5 epochs for convergence analysis
)

# Training will automatically stop when converged
trainer.fit(1000, 10)  # Set a high number, will stop early when converged

# Check if training converged naturally
if trainer.converged:
    print(f"Training converged at epoch {trainer._epoch}")
else:
    print("Training completed all epochs without convergence")
"""

# The early stopping algorithm works by:
# 1. Calculating the gradient (slope) of the loss function over recent epochs
# 2. Checking if the gradient magnitude is below min_delta threshold
# 3. Waiting for 'patience' epochs to confirm convergence
# 4. Stopping training when both gradient and patience criteria are met

print("Early stopping demonstration")
print("=" * 50)
print("Early stopping parameters:")
print("- early_stopping: Enable/disable the feature (default: False)")
print("- patience: Number of epochs to wait after convergence criteria met (default: 10)")
print("- min_delta: Minimum loss improvement to qualify as progress (default: 1e-6)")
print("- convergence_window: Number of recent epochs for gradient analysis (default: 5)")
print()
print("Benefits:")
print("- Automatic convergence detection")
print("- No need for manual monitoring")
print("- Prevents overfitting")
print("- Saves computational time")
print()
print("Usage: Add early stopping parameters to any trainer constructor.")