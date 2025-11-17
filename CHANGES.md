# Quick Reference: What Changed

## Core Improvements

### Base Models (L1) - More Aggressive Training
| Model | Parameter | Old Value | New Value | Change |
|-------|-----------|-----------|-----------|--------|
| XGBoost | n_estimators | 800 | 1500 | +87.5% |
| XGBoost | learning_rate | 0.02 | 0.01 | -50% |
| XGBoost | max_depth | 7 | 8 | +1 |
| XGBoost | early_stopping | None | 150 | NEW |
| LightGBM | n_estimators | 1000 | 2000 | +100% |
| LightGBM | learning_rate | 0.015 | 0.008 | -47% |
| LightGBM | max_depth | 9 | 10 | +1 |
| LightGBM | num_leaves | 127 | 255 | +101% |
| CatBoost | iterations | 1000 | 2000 | +100% |
| CatBoost | learning_rate | 0.015 | 0.008 | -47% |
| CatBoost | depth | 8 | 9 | +1 |
| CatBoost | early_stopping | None | 150 | NEW |

### Meta Models (L2) - More Aggressive Training
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| n_estimators | 1200 | 2000 | +67% |
| learning_rate | 0.01 | 0.008 | -20% |
| max_depth (XGB) | 6 | 7 | +1 |
| max_depth (LGB) | 7 | 8 | +1 |
| early_stopping | 100 | 150 | +50% |

### Neural Meta-Learner - Enhanced Architecture
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| hidden_layers | (128, 64) | (256, 128, 64) | Deeper |
| max_iter | 500 | 1000 | +100% |
| learning_rate | fixed | adaptive | NEW |
| learning_rate_init | - | 0.001 | NEW |
| early_stopping | False | True | NEW |
| validation_fraction | - | 0.1 | NEW |
| n_iter_no_change | - | 50 | NEW |
| alpha (L2 reg) | 0.0001 | 0.001 | +900% |

### Feature Engineering - Better Signal
| Aspect | Old | New | Change |
|--------|-----|-----|--------|
| Total Features | 41 | 64 | +23 (+56%) |
| Interaction Pairs | 5 | 8 | +3 |
| Polynomial Types | 2 (square, sqrt) | 4 (square, sqrt, cbrt, log) | +2 |
| Features/Variable | 2-3 | 4-6 | Doubled |

### Ensemble Configuration
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| Default Seeds | 6 | 8 | +2 (+33%) |

## Bug Fixes

1. **Calibration Sampling Bug**: Fixed mismatch between sampled training data and full dataset in calibration step
2. **CatBoost Early Stopping**: Added proper evaluation set integration
3. **XGBoost Early Stopping**: Conditional fit() call based on early_stopping_rounds parameter

## New Feature Transforms

### Added Interaction Pairs
- `interest_rate` × `credit_score`
- `debt_to_income_ratio` × `annual_income`  
- `loan_amount` × `debt_to_income_ratio`

### Added Polynomial Transforms (for each key feature)
- Cube root: `cbrt(x)`
- Log transform: `log1p(x)`

### Extended Feature Set
Applied transforms to 2 additional variables:
- `interest_rate`
- `debt_to_income_ratio`

## Performance Expectations

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Base AUC (L1) | ~88% | ~90-91% | +2-3% |
| Meta AUC (L2) | ~89% | ~92-93% | +3-4% |
| Final AUC (calibrated) | ~89% | **93%+** | **+4-5%** |
| Training Time | 10-15 min | 15-25 min | +50% |

## Key Principles

1. **Slower Learning, More Iterations** = Better convergence to optimal solution
2. **Deeper Models** = More capacity to learn complex patterns
3. **Early Stopping** = Prevent overfitting, auto-tune iteration count
4. **More Features** = More signal for models to learn from
5. **Better Ensemble** = More diversity reduces prediction variance

## Command Line Reference

```bash
# Default (recommended) - uses all improvements
python loan_meta_optimized_script.py

# Quick test
python loan_meta_optimized_script.py --sample-n 5000 --n-splits 3

# Maximum ensemble diversity
python loan_meta_optimized_script.py --more-seeds 30

# Custom configuration
python loan_meta_optimized_script.py --seeds 42,43,44 --n-splits 7 --use-neural
```

## Files Modified

- `loan_meta_optimized_script.py` - Main script with all improvements

## Files Added

- `IMPROVEMENT_SUMMARY.md` - Detailed documentation
- `CHANGES.md` - This quick reference (you are here)
