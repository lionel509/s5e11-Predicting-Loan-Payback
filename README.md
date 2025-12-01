# Loan Payback Prediction - Kaggle Playground Series S5E11

## Competition Overview
Participated in Kaggle's **Playground Series S5E11** competition focused on predicting loan payback outcomes using binary classification.

**Competition Statistics:**
- **10,879** total entrants
- **3,311** active participants across **3,211** teams  
- **26,869** total submissions
- **Status:** Discontinued to focus on Santa competition

## Dataset
- **Training samples:** 593,994 loans (after merging external datasets)
- **Features:** 12 core features including financial metrics and demographics
- **Target:** Binary classification (loan paid back: 79.9% vs not paid back: 20.1%)
- **Class imbalance:** ~4:1 ratio requiring weighted loss functions

## Methodology

### Model Architecture
**LightGBM with Optuna Hyperparameter Optimization**
- Single seed (42) for full reproducibility
- 5-fold stratified cross-validation
- 100 Optuna trials with intelligent pruning
- Early stopping with 150-round patience

### Feature Engineering
Advanced feature creation including:
- **Financial ratios:** loan-to-income, debt-to-income interactions
- **Polynomial features:** squared/cubed interest rates, log transforms
- **Credit score binning:** 5-tier categorization (very_low to excellent)
- **Target encoding:** Smoothed categorical encoding (smoothing=10.0)
- **Risk scoring:** Composite risk metrics combining multiple features
- **Missing value indicators:** Binary flags for data quality

### Key Features (Top 5 by Importance)
1. **employment_status** (777,559 gain)
2. **dti_div_credit** (283,632 gain)
3. **employment_status_target_enc** (128,217 gain)
4. **credit_score** (113,451 gain)
5. **debt_to_income_ratio** (86,224 gain)

### Data Insights
From exploratory analysis:
- **Credit scores** show strong negative correlation with interest rates (-0.538)
- **Higher interest rates** strongly predict default risk
- **Employment status** is the dominant predictor
- **Debt-to-income ratio** has moderate negative correlation with payback (-0.336)
- **Grade subgrades** show clear payback rate stratification (95% for A1 down to 60% for F5)

## Results
- **ROC-AUC Score:** Achieved competitive performance through extensive feature engineering
- **Fold-safe preprocessing:** Zero data leakage with proper train/validation splits
- **External data integration:** Successfully merged 3 additional datasets with schema alignment

## Technical Implementation

### Main Notebook
[`loan_lgbm_optuna.ipynb`](loan_lgbm_optuna.ipynb)
- Complete end-to-end pipeline
- Automated hyperparameter tuning
- Comprehensive evaluation metrics
- Feature importance analysis

### Visualizations
Generated comprehensive EDA including:
- Target distribution analysis (bar charts, pie charts)
- Feature distributions (histograms for all numeric features)
- Payback rates by demographics (gender, marital status, education, employment)
- Grade subgrade analysis with payback trends
- Correlation heatmaps
- Bivariate scatter plots with target coloring

All visualizations saved to [`visualizations/`](visualizations/) directory.

## Repository Structure
```
├── loan_lgbm_optuna.ipynb          # Main modeling notebook
├── Data/                            # Training/test data + external datasets
├── submissions/                     # Generated predictions
├── visualizations/                  # EDA plots and feature importance
├── models/                          # Saved model checkpoints
└── README.md                        # This file
```

## Technologies Used
- **Python 3.x**
- **LightGBM** - Gradient boosting framework
- **Optuna** - Hyperparameter optimization
- **scikit-learn** - Preprocessing and metrics
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization

## Key Learnings
- External data integration requires careful schema alignment and target consistency
- Employment status emerged as unexpectedly powerful predictor
- Target encoding with smoothing prevents overfitting on high-cardinality categoricals
- Grade subgrade feature demonstrates clear risk stratification
- Class imbalance handling through `scale_pos_weight` improved model calibration

---

*Competition discontinued to prioritize Santa 2024 competition.*
