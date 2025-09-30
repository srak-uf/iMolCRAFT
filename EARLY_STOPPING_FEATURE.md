# Early Stopping Feature for Loss Function Convergence

## Problem Statement

The original issue was:
> 現在は、目で見てloss関数の収束判定をjudge. -> loss関数の微分量を求めて、ある値になったら収束判定する。

Translation: "Currently, loss function convergence is judged by visual inspection. We need to calculate the derivative of the loss function and determine convergence when it reaches a certain value."

## Solution

This implementation provides automatic early stopping based on mathematical analysis of the loss function's derivative, eliminating the need for manual visual inspection.

### Key Features

1. **Automatic Convergence Detection**: Uses gradient analysis instead of visual inspection
2. **Configurable Parameters**: Flexible thresholds and patience settings
3. **Backward Compatibility**: Disabled by default, existing code unchanged
4. **Universal Support**: Works with all trainer types

### Algorithm

The early stopping mechanism:

1. **Gradient Calculation**: Computes the slope of the loss function over recent epochs using linear regression:
   ```
   slope = (n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)
   ```

2. **Convergence Check**: Determines if `|slope| < min_delta`

3. **Patience Mechanism**: Waits for specified epochs to confirm stable convergence

4. **Early Termination**: Stops training when criteria are met

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `early_stopping` | `False` | Enable/disable the feature |
| `patience` | `10` | Epochs to wait after convergence detected |
| `min_delta` | `1e-6` | Minimum improvement threshold |
| `convergence_window` | `5` | Window size for gradient analysis |

### Usage Examples

#### ThermodynamicTrainer
```python
trainer = ThermodynamicTrainer(
    ffxml_list=ffxml_list,
    nums_ffxml=nums_ffxml,
    pdbfile=pdbfile,
    loss_fn=lossfn,
    sampling_params=sampling_params,
    target_params=target_params,
    opt_fftypes=opt_fftypes,
    early_stopping=True,      # Enable early stopping
    patience=10,              # Wait 10 epochs after convergence
    min_delta=1e-6,          # Minimum improvement threshold
    convergence_window=5,    # Look at last 5 epochs
)

trainer.fit(1000, 10)  # Will stop early when converged

if trainer.converged:
    print(f"Training converged at epoch {trainer._epoch}")
```

#### DistanceTrainer
```python
trainer = DistanceTrainer(
    ffxml_list=ffxml_list,
    nums_ffxml=nums_ffxml,
    pdbfile=pdbfile,
    calculator=calculator,
    loss_fn=loss_fn,
    early_stopping=True,
    patience=15,
    min_delta=1e-5,
    convergence_window=8,
)
```

#### DihedralTrainer
```python
trainer = DihedralTrainer(
    ffxml=ffxml,
    pdbfile=pdbfile,
    calculator=calculator,
    loss_fn=loss_fn,
    early_stopping=True,
    patience=20,
    min_delta=1e-4,
    convergence_window=10,
)
```

### Benefits

- **Efficiency**: Automatically stops when converged, saving computational time
- **Objectivity**: Mathematical criteria replace subjective visual inspection
- **Consistency**: Reproducible convergence determination
- **Flexibility**: Configurable parameters for different optimization scenarios
- **Prevention**: Helps prevent overfitting by stopping at optimal points

### Implementation Details

- **Location**: `imolcraft/trainer/base.py` (core logic)
- **Integration**: All trainer classes updated to support parameters
- **Testing**: Unit tests verify convergence logic correctness
- **Documentation**: Examples and usage demonstrations provided

This solution directly addresses the original issue by replacing manual visual inspection ("目で見て") with automated mathematical analysis of the loss function's derivative ("loss関数の微分量を求めて").