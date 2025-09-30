# PyTorch Lightning-Style Early Stopping Feature

## Problem Statement

The original issue was:
> 現在は、目で見てloss関数の収束判定をjudge. -> loss関数の微分量を求めて、ある値になったら収束判定する。

Translation: "Currently, loss function convergence is judged by visual inspection. We need to calculate the derivative of the loss function and determine convergence when it reaches a certain value."

## Solution

This implementation provides automatic early stopping following **PyTorch Lightning's proven approach**, which is the standard in deep learning. The implementation directly monitors the loss metric without complex gradient calculations, making it simpler, more robust, and aligned with deep learning best practices.

### Key Features

1. **Direct Metric Monitoring**: Monitors loss values directly (no gradient calculation needed)
2. **PyTorch Lightning Compatibility**: Follows the same pattern as PyTorch Lightning's EarlyStopping
3. **Mode Support**: Supports both 'min' (for loss) and 'max' (for accuracy) modes
4. **Robust Safety Features**: Includes stopping/divergence thresholds and finite value checking
5. **Backward Compatibility**: Disabled by default, existing code unchanged

### Algorithm (PyTorch Lightning Style)

The early stopping mechanism follows the standard deep learning pattern:

1. **Direct Monitoring**: Monitors the loss value directly each epoch
2. **Improvement Check**: Compares current loss with best score: `current < best - min_delta`
3. **Patience Counter**: Increments when no improvement, resets when improved
4. **Early Termination**: Stops when patience counter reaches limit
5. **Safety Checks**: Additional checks for thresholds and finite values

```python
# Simplified algorithm
if current_loss < best_score - min_delta:
    best_score = current_loss
    wait_count = 0  # Reset patience
else:
    wait_count += 1
    if wait_count >= patience:
        stop_training()
```

### Parameters (PyTorch Lightning Compatible)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `early_stopping` | `False` | Enable/disable the feature |
| `patience` | `7` | Number of epochs with no improvement before stopping |
| `min_delta` | `0.0` | Minimum change to qualify as improvement |
| `mode` | `'min'` | 'min' for loss, 'max' for accuracy |
| `check_finite` | `True` | Stop training on NaN or infinite values |
| `stopping_threshold` | `None` | Stop immediately when metric reaches this value |
| `divergence_threshold` | `None` | Stop if metric becomes worse than this threshold |
| `verbose` | `False` | Print early stopping messages |

### Usage Examples

#### Basic Usage (Loss Minimization)
```python
trainer = ThermodynamicTrainer(
    ffxml_list=ffxml_list,
    nums_ffxml=nums_ffxml,
    pdbfile=pdbfile,
    loss_fn=lossfn,
    sampling_params=sampling_params,
    target_params=target_params,
    opt_fftypes=opt_fftypes,
    # PyTorch Lightning-style early stopping
    early_stopping=True,         # Enable early stopping
    patience=7,                  # Standard patience value
    min_delta=0.0,              # Any improvement counts
    mode='min',                 # Minimize loss
    verbose=True,               # Show stopping messages
)

trainer.fit(1000, 10)  # Will stop early when appropriate

if trainer.should_stop:
    print(f"Stopped early at epoch {trainer.stopped_epoch}")
    print(f"Best score: {trainer.best_score}")
```

#### Advanced Usage with Thresholds
```python
trainer = ThermodynamicTrainer(
    # ... other parameters ...
    early_stopping=True,
    patience=15,
    min_delta=0.001,            # Require meaningful improvement
    mode='min',
    stopping_threshold=0.01,    # Stop immediately if loss <= 0.01
    divergence_threshold=100.0, # Stop if loss >= 100.0 (diverged)
    check_finite=True,          # Stop on NaN/Inf
    verbose=True,
)
```

#### For Maximizing Metrics (e.g., Accuracy)
```python
trainer = SomeTrainer(
    # ... other parameters ...
    early_stopping=True,
    mode='max',                 # Maximize the metric
    patience=10,
    min_delta=0.01,            # Minimum accuracy improvement
    stopping_threshold=0.95,    # Stop when accuracy >= 95%
    verbose=True,
)
```

### Comparison with Previous Implementation

| Aspect | Previous (Gradient-based) | Current (PyTorch Lightning) |
|--------|--------------------------|----------------------------|
| **Algorithm** | Linear regression slope calculation | Direct metric comparison |
| **Complexity** | Complex gradient math | Simple comparison logic |
| **Pattern** | Custom approach | Standard deep learning pattern |
| **Robustness** | Sensitive to noise | More robust to fluctuations |
| **Parameters** | `convergence_window`, gradient thresholds | Standard `patience`, `min_delta` |
| **Compatibility** | Custom implementation | PyTorch Lightning compatible |
| **Safety** | Basic NaN checking | Comprehensive threshold checks |

### Benefits Over Previous Implementation

- **Simplicity**: No complex gradient calculations needed
- **Robustness**: More reliable in noisy loss landscapes
- **Standards Compliance**: Follows established deep learning practices
- **Flexibility**: Support for both minimization and maximization
- **Safety**: Better handling of edge cases (NaN, divergence)
- **Familiarity**: Uses patterns familiar to deep learning practitioners

### Implementation Details

- **Location**: `imolcraft/trainer/base.py` (core logic)
- **Integration**: All trainer classes updated with new parameters
- **Testing**: Comprehensive tests verify PyTorch Lightning compatibility
- **Documentation**: Updated examples and usage demonstrations

This solution directly addresses the original issue by replacing manual visual inspection ("目で見て") with automated, mathematically sound early stopping that follows deep learning best practices, making it both more reliable and more familiar to practitioners in the field.