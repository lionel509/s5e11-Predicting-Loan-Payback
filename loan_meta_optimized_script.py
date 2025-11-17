import os, sys, json, math, warnings, gc, time, random, re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Optional display for notebooks; fallback to print
try:
    from IPython.display import display as ipy_display
except Exception:
    ipy_display = None

def display_df(df: pd.DataFrame, head: int = 12, title: Optional[str] = None):
    if title:
        print(title)
    if ipy_display is not None:
        ipy_display(df.head(head))
    else:
        try:
            print(df.head(head).to_string())
        except Exception:
            print(df.head(head))

# Try XGBoost for meta
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("xgboost not installed; meta-XGB will be skipped.")

# Try LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    print("lightgbm not installed; LightGBM will be skipped.")

# Try CatBoost
try:
    import catboost as cb
    CB_AVAILABLE = True
except Exception:
    CB_AVAILABLE = False
    print("catboost not installed; CatBoost will be skipped.")

warnings.filterwarnings('ignore')
RANDOM_BASE = 42
np.random.seed(RANDOM_BASE)
random.seed(RANDOM_BASE)

ROOT = Path.cwd()
DATA_DIR = ROOT / 'Data'
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SAMPLE_SUB_PATH = DATA_DIR / 'sample_submission.csv'
SUB_DIR = ROOT / 'submissions'
SUB_DIR.mkdir(exist_ok=True, parents=True)

print(f'ROOT: {ROOT}')
print(f'Files exist? train={TRAIN_PATH.exists()} test={TEST_PATH.exists()} sample={SAMPLE_SUB_PATH.exists()}')

# -------------------------
# USER CONFIG / QUICK TUNES
# Put your preferred seeds, flags, and quick-run switches here.
# This is the single place to edit when experimenting locally.
# -------------------------
# Example: SEEDS = [42, 43, 44]
# ULTRA AGGRESSIVE: More seeds for maximum ensemble diversity
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 99, 123, 456, 789, 888, 999, 1337]  # 12 seeds for max diversity
# If >0, append this many sequential seeds after DEFAULT_SEEDS (useful to expand ensemble quickly)
DEFAULT_MORE_SEEDS = 15  # Increased from 10 to 15

# Training options
DEFAULT_USE_NEURAL = True   # enable MLP blending at L2
DEFAULT_N_SPLITS = 7         # Increased from 5 to 7 for better generalization
DEFAULT_SAMPLE_N = 0         # if >0, sample this many rows for fast smoke runs
DEFAULT_SAVE_MODELS = False  # save refit base models
DEFAULT_USE_ENABLE_CATEGORICAL = True  # try XGBoost categorical support when available
USE_CATBOOST = True  # quick toggle to enable/disable CatBoost

# Optional runtime profile
DEFAULT_MODE = 'FAST'  # FAST or ULTRA

# Monitoring / baseline
LAST_REPORTED_SCORE = 0.92849  # MUST BEAT THIS - current target

print(f"CONFIG: seeds={DEFAULT_SEEDS} more_seeds={DEFAULT_MORE_SEEDS} use_neural={DEFAULT_USE_NEURAL} n_splits={DEFAULT_N_SPLITS} sample_n={DEFAULT_SAMPLE_N}")
print(f"Last reported score: {LAST_REPORTED_SCORE:.5f} — aim to beat this (notebook was higher)")

# Config & target/id detection
TARGET_CANDIDATES = ['target','TARGET','label','Label','default','is_default','loan_status','loan_repaid']
ID_CANDIDATES = ['id','ID','loan_id','Loan_ID']

def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    id_col = None
    for c in ID_CANDIDATES:
        if c in cols:
            id_col = c
            break
    target_col = None
    for c in TARGET_CANDIDATES:
        if c in cols:
            target_col = c
            break
    if target_col is None and cols:
        last = cols[-1]
        try:
            if df[last].dropna().isin([0,1]).mean() > 0.9:
                target_col = last
        except Exception:
            pass
    return id_col, target_col

# Load data
assert TRAIN_PATH.exists(), f"Missing train file at {TRAIN_PATH}"
preview = pd.read_csv(TRAIN_PATH, nrows=100)
ID_COL, TARGET = detect_columns(preview)
print('Detected ID_COL=', ID_COL, ' TARGET=', TARGET)
assert TARGET is not None, 'Target column not detected; please set TARGET manually.'

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH) if TEST_PATH.exists() else None
print(train.shape, 'train shape')
if test is not None:
    print(test.shape, 'test shape')

# Leakage audit utilities
LEAK_MAX_FEATURES = 40  # cap evaluation for speed

def single_feature_auc_scan(df: pd.DataFrame, y: pd.Series, max_features=LEAK_MAX_FEATURES):
    aucs = []
    for col in df.columns[:max_features]:
        try:
            if df[col].nunique() < 2:
                continue
            vals = df[col].fillna(df[col].median() if df[col].dtype != 'O' else 'missing')
            if vals.dtype == 'O':
                mapping = vals.value_counts(normalize=True).to_dict()
                enc = vals.map(mapping).astype(float)
            else:
                enc = vals.astype(float)
            score = roc_auc_score(y, enc) if len(np.unique(enc)) > 1 else 0.5
            aucs.append((col, score))
        except Exception:
            continue
    aucs.sort(key=lambda x: x[1], reverse=True)
    return aucs

AGG_PATTERNS = [r'^total_', r'^sum_', r'^avg_', r'^mean_', r'^max_', r'^min_']

def looks_leaky(colname: str) -> bool:
    for pat in AGG_PATTERNS:
        if re.search(pat, colname):
            return True
    return False

# KS & PSI drift checks between train/test

def ks_stat(train_col, test_col):
    a = pd.Series(train_col).dropna()
    b = pd.Series(test_col).dropna()
    if a.nunique() < 2 or b.nunique() < 2:
        return 0.0
    try:
        stat, pval = stats.ks_2samp(a, b)
        return stat
    except Exception:
        return 0.0

# Population Stability Index for binned values

def psi(train_col, test_col, buckets=10):
    a = pd.Series(train_col).dropna()
    b = pd.Series(test_col).dropna()
    if a.nunique() < 2 or b.nunique() < 2:
        return 0.0
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = a.quantile(quantiles).to_numpy()
    # Ensure bins are a clean, sorted Python list without NaNs
    bins_list = sorted([float(v) for v in np.unique(cuts) if v == v])
    a_bins = pd.cut(a, bins=bins_list, include_lowest=True)
    b_bins = pd.cut(b, bins=bins_list, include_lowest=True)
    a_dist = a_bins.value_counts(normalize=True)
    b_dist = b_bins.value_counts(normalize=True)
    psi_val = 0.0
    for idx in a_dist.index:
        expected = a_dist.get(idx, 1e-6)
        actual = b_dist.get(idx, 1e-6)
        if expected > 0 and actual > 0:
            psi_val += (actual - expected) * math.log(actual / expected)
    return psi_val

DRIFT_REPORT_LIMIT = 40

def drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame):
    rows = []
    shared = [c for c in train_df.columns if c in test_df.columns]
    for col in shared[:DRIFT_REPORT_LIMIT]:
        try:
            k = ks_stat(train_df[col], test_df[col])
            p = psi(train_df[col], test_df[col])
            rows.append({'feature': col, 'ks': k, 'psi': p})
        except Exception:
            continue
    rep = pd.DataFrame(rows)
    if not rep.empty:
        rep.sort_values(['ks','psi'], ascending=False, inplace=True)
    return rep

BOOL_LIKE = ['y','n','yes','no','true','false']

def cast_types(df: pd.DataFrame):
    for c in df.columns:
        if df[c].dtype == 'O':
            low = df[c].astype(str).str.lower()
            if (low.isin(BOOL_LIKE)).mean() > 0.9:
                df[c] = low.map({'y':1,'yes':1,'true':1,'n':0,'no':0,'false':0}).astype('Int8')
    return df

print('Leakage & drift utilities ready.')

# Apply type casting, leakage audit, and drift checks
# Note: In the notebook these ran later; here we run BEFORE feature engineering for correctness.
if ID_COL:
    train[ID_COL] = train[ID_COL].astype(str)
    if test is not None and ID_COL in test.columns:
        test[ID_COL] = test[ID_COL].astype(str)

train = cast_types(train)
if test is not None:
    test = cast_types(test)

feat_cols = [c for c in train.columns if c not in [TARGET] + ([ID_COL] if ID_COL else [])]
scan_df = train[feat_cols].copy()
scan_aucs = single_feature_auc_scan(scan_df, train[TARGET], max_features=min(LEAK_MAX_FEATURES, len(feat_cols)))
leaky = [c for (c, auc) in scan_aucs if auc >= 0.92 or auc <= 0.08 or looks_leaky(c)]

if len(leaky) > 0:
    print('Dropping suspicious leakage features:', leaky)
    train.drop(columns=[c for c in leaky if c in train.columns], inplace=True)
    if test is not None:
        test.drop(columns=[c for c in leaky if c in test.columns], inplace=True)
else:
    print('No leakage features flagged by simple scan.')

if test is not None:
    tr_common = train.drop(columns=[TARGET] + ([ID_COL] if ID_COL else []), errors='ignore')
    te_common = test.drop(columns=[ID_COL] if ID_COL else [], errors='ignore')
    rep = drift_report(tr_common, te_common)
    if not rep.empty:
        display_df(rep.head(12), title='Drift report (top 12):')
    drop_drift = rep[(rep['ks'] >= 0.2) | (rep['psi'] >= 0.25)]['feature'].tolist() if not rep.empty else []
    if drop_drift:
        print('Dropping drift-heavy features:', drop_drift)
        train.drop(columns=[c for c in drop_drift if c in train.columns], inplace=True)
        test.drop(columns=[c for c in drop_drift if c in test.columns], inplace=True)
else:
    print('Test set not available; skipping drift check.')

print('Preprocessing audits complete.')

# Lightweight distribution-aware tweaks: clip outliers and add normalized features
def _clip_series(s, low, high):
    try:
        return s.clip(lower=low, upper=high)
    except Exception:
        return s

# Compute clip bounds on train and apply to both train/test
ai_bounds = None
la_bounds = None
if 'annual_income' in train.columns and pd.api.types.is_numeric_dtype(train['annual_income']):
    q1, q99 = np.percentile(train['annual_income'].dropna(), [1, 99])
    ai_bounds = (q1, q99)
if 'loan_amount' in train.columns and pd.api.types.is_numeric_dtype(train['loan_amount']):
    q1, q99 = np.percentile(train['loan_amount'].dropna(), [1, 99])
    la_bounds = (q1, q99)

for df in [train] + ([test] if test is not None else []):
    if df is None:
        continue
    if ai_bounds and 'annual_income' in df.columns:
        df['annual_income'] = _clip_series(df['annual_income'], ai_bounds[0], ai_bounds[1])
    if la_bounds and 'loan_amount' in df.columns:
        df['loan_amount'] = _clip_series(df['loan_amount'], la_bounds[0], la_bounds[1])
    if 'debt_to_income_ratio' in df.columns:
        df['debt_to_income_ratio'] = _clip_series(df['debt_to_income_ratio'], 0.01, 0.6)
    if 'interest_rate' in df.columns:
        df['interest_rate'] = _clip_series(df['interest_rate'], 3.0, 21.0)
    if 'credit_score' in df.columns:
        df['credit_score'] = _clip_series(df['credit_score'], 400, 850)

# Normalized/derived versions
for df in [train] + ([test] if test is not None else []):
    if df is None:
        continue
    if 'credit_score' in df.columns:
        df['credit_score_norm'] = (df['credit_score'] - 300.0) / (850.0 - 300.0)
    if 'debt_to_income_ratio' in df.columns:
        df['dti_pct'] = df['debt_to_income_ratio'] * 100.0
    if 'interest_rate' in df.columns:
        df['interest_rate_norm'] = (df['interest_rate'] - 3.0) / (21.0 - 3.0)

# EXTREME Feature Engineering + Target Encoding for 93%+
y = train[TARGET].astype(int)
X = train.drop(columns=[TARGET] + ([ID_COL] if ID_COL else []))
X_test = None
if test is not None:
    X_test = test.drop(columns=[ID_COL] if ID_COL else [])

num_cols_orig = X.select_dtypes(include=['number','float','int','Int8','Int16','Int32','Int64']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols_orig]

print(f'Original: {len(num_cols_orig)} numeric, {len(cat_cols)} categorical')

# 1. TARGET ENCODING for categorical features (10-fold Stratified CV with smoothing)
TARGET_ENCODED = {}
global_mean = y.mean()
for cat in cat_cols[:]:  # Encode all categoricals
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X[f'{cat}_target_enc'] = 0.0
    k = 50  # smoothing strength
    for tr_idx, va_idx in skf.split(X, y):
        g = train.iloc[tr_idx].groupby(cat)[TARGET].agg(['mean', 'count'])
        g['smoothed'] = (g['mean'] * g['count'] + global_mean * k) / (g['count'] + k)
        mapping = g['smoothed']
        X.loc[X.index[va_idx], f'{cat}_target_enc'] = (
            X.iloc[va_idx][cat].map(mapping).fillna(global_mean)
        )
    if X_test is not None:
        g_full = train.groupby(cat)[TARGET].agg(['mean', 'count'])
        g_full['smoothed'] = (g_full['mean'] * g_full['count'] + global_mean * k) / (g_full['count'] + k)
        mapping_full = g_full['smoothed']
        X_test[f'{cat}_target_enc'] = X_test[cat].map(mapping_full).fillna(global_mean)
    TARGET_ENCODED[cat] = f'{cat}_target_enc'
    print(f'Target encoded (smoothed): {cat}')

# 2. INTERACTION FEATURES (ratios + products + polynomials)
IMPORTANT_PAIRS = [
    ('loan_amount', 'annual_income'),
    ('loan_amount', 'credit_score'),
    ('debt_to_income_ratio', 'credit_score'),
    ('annual_income', 'credit_score'),
    ('interest_rate', 'loan_amount'),
    ('interest_rate', 'credit_score'),  # Added
    ('debt_to_income_ratio', 'annual_income'),  # Added
    ('loan_amount', 'debt_to_income_ratio'),  # Added
]

for c1, c2 in IMPORTANT_PAIRS:
    if c1 in num_cols_orig and c2 in num_cols_orig:
        # Ratio features
        X[f'{c1}_div_{c2}'] = X[c1] / (X[c2] + 1e-6)
        if X_test is not None:
            X_test[f'{c1}_div_{c2}'] = X_test[c1] / (X_test[c2] + 1e-6)
        # Product features
        X[f'{c1}_x_{c2}'] = X[c1] * X[c2]
        if X_test is not None:
            X_test[f'{c1}_x_{c2}'] = X_test[c1] * X_test[c2]
        # Difference features
        X[f'{c1}_minus_{c2}'] = X[c1] - X[c2]
        if X_test is not None:
            X_test[f'{c1}_minus_{c2}'] = X_test[c1] - X_test[c2]

# 3. POLYNOMIAL FEATURES (square, sqrt, cube root for key predictors)
for col in ['credit_score', 'annual_income', 'loan_amount', 'interest_rate', 'debt_to_income_ratio']:
    if col in num_cols_orig:
        X[f'{col}_squared'] = X[col] ** 2
        X[f'{col}_sqrt'] = np.sqrt(X[col].clip(lower=0))
        X[f'{col}_cbrt'] = np.cbrt(X[col])  # Added cube root
        X[f'{col}_log'] = np.log1p(X[col].clip(lower=0))  # Added log transform
        if X_test is not None:
            X_test[f'{col}_squared'] = X_test[col] ** 2
            X_test[f'{col}_sqrt'] = np.sqrt(X_test[col].clip(lower=0))
            X_test[f'{col}_cbrt'] = np.cbrt(X_test[col])
            X_test[f'{col}_log'] = np.log1p(X_test[col].clip(lower=0))

# 4. BINNING FEATURES (discretize continuous)
for col in ['credit_score', 'annual_income', 'loan_amount']:
    if col in num_cols_orig:
        X[f'{col}_bin'] = pd.qcut(X[col], q=10, labels=False, duplicates='drop')
        if X_test is not None:
            quantiles = X[col].quantile(np.linspace(0, 1, 11)).to_numpy()
            q_bins = sorted([float(v) for v in np.unique(quantiles) if v == v])
            X_test[f'{col}_bin'] = pd.cut(
                X_test[col], bins=q_bins, labels=False, include_lowest=True
            ).astype('float').fillna(5)

num_cols = X.select_dtypes(include=['number','float','int','Int8','Int16','Int32','Int64']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f'Engineered: {len(num_cols)} numeric, {len(cat_cols)} categorical')
print(f'Feature count: {X.shape[1]}')

# Convert any remaining object-typed columns to pandas 'category' so LightGBM accepts them.
# XGBoost will receive integer-encoded copies later (we keep mappings), but LGB/CB can use pandas categorical dtypes.
for c in list(X.columns):
    if X[c].dtype == 'O':
        X[c] = X[c].astype('category')
        if X_test is not None and c in X_test.columns:
            X_test[c] = X_test[c].astype('category')

# Simplified preprocessing (prepared but unused in current flow)
numeric_tf = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_tf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocess = ColumnTransformer(transformers=[
    ('num', numeric_tf, num_cols),
    ('cat', categorical_tf, cat_cols)
])

def get_cv(n_splits=5, seed=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def threshold_sweep(y_true, prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best = {'threshold': None, 'f1': -1, 'precision': None, 'recall': None}
    for t in thresholds:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best['f1']:
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            best = {'threshold': float(t), 'f1': float(f1), 'precision': float(prec), 'recall': float(rec)}
    return best

def fit_isotonic(y_true, prob):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(prob, y_true)
    return iso

print('Progress tracking ready ✓')

# L1 Base Models

def train_base_models(X, y, X_test=None, seed=42, n_splits=5, mode: str = 'FAST'):
    cv = get_cv(n_splits=n_splits, seed=seed)

    # Class imbalance handling (approx 80/20): compute scale_pos_weight for boosters
    pos_rate = float(np.mean(y))
    neg_rate = 1.0 - pos_rate
    scale_pos = (neg_rate / max(pos_rate, 1e-6)) if pos_rate > 0 else 1.0
    print(f"Class balance: pos={pos_rate:.4f}, neg={neg_rate:.4f}, scale_pos_weight≈{scale_pos:.2f}")

    base_models_config = []
    if XGB_AVAILABLE:
        # ULTRA AGGRESSIVE: Even more iterations, slower learning, deeper trees
        base_models_config.append(('xgb', {
            'n_estimators': 2500, 'learning_rate': 0.005, 'max_depth': 9,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_lambda': 2.5, 'reg_alpha': 0.6, 'min_child_weight': 2,
            'tree_method': 'hist', 'n_jobs': -1,
            'early_stopping_rounds': 200,  # More patience
            'scale_pos_weight': scale_pos
        }))
    if LGB_AVAILABLE:
        # ULTRA AGGRESSIVE: More iterations, slower learning, deeper model
        base_models_config.append(('lgb', {
            'n_estimators': 3000, 'learning_rate': 0.005, 'max_depth': 12, 'num_leaves': 511,
            'subsample': 0.75, 'colsample_bytree': 0.75,
            'reg_lambda': 2.5, 'reg_alpha': 0.5, 'min_child_samples': 15,
            'verbose': -1, 'n_jobs': -1, 'force_col_wise': True,
            'scale_pos_weight': scale_pos
        }))
    if CB_AVAILABLE and USE_CATBOOST:
        # Lighter CatBoost with early stopping in fit
        base_models_config.append(('cb', {
            'iterations': 1000 if mode == 'ULTRA' else 600,
            'learning_rate': 0.015 if mode == 'ULTRA' else 0.02,
            'depth': 6,
            'l2_leaf_reg': 5,
            'border_count': 128,
            'min_data_in_leaf': 16,
            'verbose': 0,
            'thread_count': -1,
            'scale_pos_weight': scale_pos
        }))

    # Cheap but diverse L1 models
    base_models_config.append(('lr', { 'C': 0.5 }))
    base_models_config.append(('et', {
        'n_estimators': 400,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }))

    if not base_models_config:
        raise ValueError('No gradient boosters available! Install XGBoost, LightGBM, or CatBoost.')

    base_names = [name for name, _ in base_models_config]
    print(f'🚀 Training {len(base_names)} L1 base models: {base_names}')

    oof = np.zeros((len(X), len(base_names)))
    test_preds = np.zeros((len(X_test), len(base_names))) if X_test is not None else None
    aucs = {name: [] for name in base_names}

    # Precompute helpers
    from pandas.api import types as _pd_types
    numeric_features = [c for c in X.columns if _pd_types.is_numeric_dtype(X[c])]

    for j, (name, params) in enumerate(base_models_config):
        fold_idx = 0
        # Precompute label mappings for non-numeric categorical columns so XGBoost can be given numeric data
        # Use columns that are not numeric (includes 'category' dtype we set earlier) to build mappings
        cat_obj_cols = [c for c in X.columns if not _pd_types.is_numeric_dtype(X[c])]
        cat_mappings = {}
        for c in cat_obj_cols:
            # Use training-wide unique values; unknowns map to 0
            uniques = [v for v in pd.Series(X[c].astype(str).fillna('__nan__')).unique()]
            cat_mappings[c] = {v: i + 1 for i, v in enumerate(uniques)}

        # For CatBoost: identify categorical features and precompute test conversion once per model
        cat_features_cb = None
        X_test_cb = None
        if name == 'cb':
            from pandas.api import types as _pd_types3
            cat_features_cb = [c for c in X.columns if (not _pd_types3.is_numeric_dtype(X[c])) or '_bin' in c]
            if X_test is not None:
                X_test_cb = X_test.copy()
                for c in cat_features_cb:
                    if c in X_test_cb.columns:
                        X_test_cb[c] = X_test_cb[c].astype(str).fillna('__nan__')
        for tr_idx, va_idx in cv.split(X, y):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            if name == 'xgb':
                # XGBoost currently requires numeric (or category + enable_categorical).
                # Keep original DataFrames untouched for LGB/CB; create numeric copies for XGB only.
                X_tr_enc = X_tr.copy()
                X_va_enc = X_va.copy()
                X_test_enc = None
                for c, mapping in cat_mappings.items():
                    if c in X_tr_enc.columns:
                        X_tr_enc[c] = X_tr_enc[c].astype(str).map(mapping).fillna(0).astype(int)
                    if c in X_va_enc.columns:
                        X_va_enc[c] = X_va_enc[c].astype(str).map(mapping).fillna(0).astype(int)
                if X_test is not None:
                    X_test_enc = X_test.copy()
                    for c, mapping in cat_mappings.items():
                        if c in X_test_enc.columns:
                            X_test_enc[c] = X_test_enc[c].astype(str).map(mapping).fillna(0).astype(int)

                model = xgb.XGBClassifier(random_state=seed+fold_idx, **params)
                # Add early stopping support
                if 'early_stopping_rounds' in params:
                    model.fit(X_tr_enc, y_tr, eval_set=[(X_va_enc, y_va)], verbose=False)
                else:
                    model.fit(X_tr_enc, y_tr)
                p = np.asarray(model.predict_proba(X_va_enc))[:, 1]
            elif name == 'lgb':
                # LightGBM accepts pandas 'category' dtype; pass those columns as categorical features
                from pandas.api import types as _pd_types2
                cat_features = [c for c in X.columns if (not _pd_types2.is_numeric_dtype(X[c])) or '_bin' in c]
                model = lgb.LGBMClassifier(random_state=seed+fold_idx, **params)
                model.fit(X_tr, y_tr, categorical_feature=cat_features if cat_features else 'auto')
                p = np.asarray(model.predict_proba(X_va))[:, 1]
            elif name == 'cb':
                # CatBoost requires categorical features as strings/ints. Convert per-fold train/val, test once.
                X_tr_cb = X_tr.copy()
                X_va_cb = X_va.copy()
                if cat_features_cb:
                    for c in cat_features_cb:
                        if c in X_tr_cb.columns:
                            X_tr_cb[c] = X_tr_cb[c].astype(str).fillna('__nan__')
                        if c in X_va_cb.columns:
                            X_va_cb[c] = X_va_cb[c].astype(str).fillna('__nan__')

                model = cb.CatBoostClassifier(
                    random_seed=seed+fold_idx,
                    **params
                )
                if cat_features_cb:
                    model.fit(
                        X_tr_cb, y_tr,
                        cat_features=cat_features_cb,
                        eval_set=(X_va_cb, y_va),
                        early_stopping_rounds=100,
                        verbose=False
                    )
                else:
                    model.fit(
                        X_tr_cb, y_tr,
                        eval_set=(X_va_cb, y_va),
                        early_stopping_rounds=100,
                        verbose=False
                    )
                p = np.asarray(model.predict_proba(X_va_cb))[:, 1]
            elif name == 'lr':
                # Cheap linear model on numeric features only
                lr_C = params.get('C', 0.5)
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(
                        penalty='l2', C=lr_C, max_iter=5000, class_weight='balanced'
                    ))
                ])
                pipe.fit(X_tr[numeric_features], y_tr)
                p = np.asarray(pipe.predict_proba(X_va[numeric_features]))[:, 1]
            elif name == 'et':
                et = ExtraTreesClassifier(**params)
                et.fit(X_tr[numeric_features], y_tr)
                p = np.asarray(et.predict_proba(X_va[numeric_features]))[:, 1]
            else:
                raise ValueError('Unknown model name')

            oof[va_idx, j] = p
            auc = roc_auc_score(y_va, p)
            aucs[name].append(auc)

            if X_test is not None and test_preds is not None:
                # Use encoded test set for XGB, original for others
                if name == 'xgb' and X_test_enc is not None:
                    test_preds[:, j] += np.asarray(model.predict_proba(X_test_enc))[:, 1] / cv.get_n_splits()
                elif name == 'cb' and X_test_cb is not None:
                    test_preds[:, j] += np.asarray(model.predict_proba(X_test_cb))[:, 1] / cv.get_n_splits()
                elif name in ('lr', 'et'):
                    test_preds[:, j] += np.asarray(
                        (pipe if name == 'lr' else et).predict_proba(
                            (X_test[numeric_features])
                        )
                    )[:, 1] / cv.get_n_splits()
                else:
                    test_preds[:, j] += np.asarray(model.predict_proba(X_test))[:, 1] / cv.get_n_splits()
            fold_idx += 1

        mean_auc = np.mean(aucs[name])
        print(f"  ✓ {name}: {np.round(aucs[name], 5)} → mean {mean_auc:.5f}")

    return oof, test_preds, aucs

# L2 / L3

def train_meta_l2(oof_feats, y, test_feats=None, seed=42, use_neural=False):
    cv = get_cv(n_splits=5, seed=seed)
    oof_meta = np.zeros(len(y))
    test_meta = np.zeros(len(test_feats)) if test_feats is not None else None
    fold_aucs = []

    expanded_oof = oof_feats.copy()
    expanded_test = test_feats.copy() if test_feats is not None else None

    n_base = oof_feats.shape[1]
    for i in range(n_base):
        for j in range(i+1, n_base):
            expanded_oof = np.column_stack([expanded_oof, oof_feats[:, i] * oof_feats[:, j]])
            if expanded_test is not None and test_feats is not None:
                test_arr = np.asarray(test_feats)
                expanded_test = np.column_stack([expanded_test, test_arr[:, i] * test_arr[:, j]])

    expanded_oof = np.column_stack([
        expanded_oof,
        np.mean(oof_feats, axis=1),
        np.std(oof_feats, axis=1),
        np.max(oof_feats, axis=1),
        np.min(oof_feats, axis=1),
    ])
    if expanded_test is not None and test_feats is not None:
        test_arr = np.asarray(test_feats)
        expanded_test = np.column_stack([
            expanded_test,
            np.mean(test_arr, axis=1),
            np.std(test_arr, axis=1),
            np.max(test_arr, axis=1),
            np.min(test_arr, axis=1),
        ])

    print(f'  📊 L2 features: {n_base} → {expanded_oof.shape[1]} (with interactions + stats)')

    for fold, (tr_idx, va_idx) in enumerate(cv.split(expanded_oof, y)):
        X_tr, X_va = expanded_oof[tr_idx], expanded_oof[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if XGB_AVAILABLE:
            # ULTRA AGGRESSIVE L2 meta: maximum iterations, very slow learning
            clf = xgb.XGBClassifier(
                max_depth=8, n_estimators=1600, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.5, reg_alpha=0.8, min_child_weight=4,
                objective='binary:logistic', eval_metric='auc',
                random_state=seed+fold, tree_method='hist',
                early_stopping_rounds=100, n_jobs=-1
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif LGB_AVAILABLE:
            # ULTRA AGGRESSIVE L2 meta: maximum iterations, very slow learning
            clf = lgb.LGBMClassifier(
                n_estimators=1600, learning_rate=0.01, max_depth=9, num_leaves=255,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.5, reg_alpha=0.8, min_child_samples=25,
                random_state=seed+fold, verbose=-1, n_jobs=-1,
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        else:
            clf = LogisticRegression(max_iter=10000, C=0.1, penalty='l2')
            clf.fit(X_tr, y_tr)
        p = np.asarray(clf.predict_proba(X_va))[:, 1]

        # Optional neural meta: train an ULTRA AGGRESSIVE MLP on the same L2 features and average probabilities.
        p_neural = None
        if use_neural:
            try:
                # ULTRA AGGRESSIVE: even deeper network, more iterations, lower learning rate
                mlp = MLPClassifier(
                    hidden_layer_sizes=(512, 256, 128, 64), activation='relu', 
                    solver='adam', alpha=0.0005,
                    learning_rate='adaptive', learning_rate_init=0.0005,
                    random_state=seed+fold, max_iter=2000, 
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=75
                )
                mlp.fit(X_tr, y_tr)
                p_neural = np.asarray(mlp.predict_proba(X_va))[:, 1]
            except Exception:
                p_neural = None

        if p_neural is not None:
            # Average the meta-model and neural meta
            p = 0.5 * p + 0.5 * p_neural
        oof_meta[va_idx] = p
        fold_aucs.append(roc_auc_score(y_va, p))

        if expanded_test is not None and test_meta is not None:
            test_meta += np.asarray(clf.predict_proba(expanded_test))[:, 1] / cv.get_n_splits()
            if use_neural and p_neural is not None:
                try:
                        # We avoid double-counting; the neural meta was averaged into `p` above for OOF.
                        # For test predictions, average by adding (mlp_pred)/n_splits * 0.5 to match OOF averaging.
                        test_meta += 0.5 * np.asarray(mlp.predict_proba(expanded_test))[:, 1] / cv.get_n_splits()
                except Exception:
                    # If neural couldn't predict on expanded_test shape, skip adding
                    pass

    meta_auc = roc_auc_score(y, oof_meta)
    print(f'  ✓ L2 Meta AUC: {meta_auc:.5f} | folds: {np.round(fold_aucs, 5)}')

    return oof_meta, test_meta


def train_meta_l3_with_pseudo(oof_l2, y, test_l2, X, X_test, seed=42):
    if test_l2 is not None and X_test is not None:
        high_conf_mask = (test_l2 > 0.95) | (test_l2 < 0.05)
        pseudo_labels = (test_l2 > 0.5).astype(int)

        n_pseudo = int(high_conf_mask.sum())
        if n_pseudo > 0:
            print(f'  🎭 Pseudo-labeling: {n_pseudo} high-confidence test samples')

            X_combined = pd.concat([X, X_test.iloc[high_conf_mask]], axis=0, ignore_index=True)
            y_combined = pd.concat([y, pd.Series(pseudo_labels[high_conf_mask])], axis=0, ignore_index=True)
            oof_combined = np.concatenate([oof_l2, test_l2[high_conf_mask]])

            cv = get_cv(n_splits=5, seed=seed)
            oof_l3 = np.zeros(len(oof_combined))

            for fold, (tr_idx, va_idx) in enumerate(cv.split(oof_combined, y_combined)):
                X_tr, X_va = oof_combined[tr_idx].reshape(-1, 1), oof_combined[va_idx].reshape(-1, 1)
                y_tr, y_va = y_combined.iloc[tr_idx], y_combined.iloc[va_idx]

                clf = LogisticRegression(max_iter=5000, C=0.5)
                clf.fit(X_tr, y_tr)
                oof_l3[va_idx] = np.asarray(clf.predict_proba(X_va))[:, 1]

            oof_l3_train = oof_l3[:len(y)]
            auc_l3 = roc_auc_score(y, oof_l3_train)
            print(f'  ✓ L3 Meta + Pseudo AUC: {auc_l3:.5f}')

            return oof_l3_train

    return oof_l2

# Training orchestrator

def run_training_extreme(seeds: List[int] = [42, 43], target_auc: float = 0.9285,
                         use_neural: bool = False, n_splits: int = 5, sample_n: int = 0,
                         save_models: bool = False, use_enable_categorical: bool = False,
                         mode: str = DEFAULT_MODE):
    results = []
    best = None

    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}\n🚀 SEED {seed} ({i+1}/{len(seeds)}) — ULTRA AGGRESSIVE - Target: {target_auc:.5f}\n{'='*70}")

        # Optionally sample the dataset for quick smoke runs
        X_work, y_work, X_test_work = X, y, X_test
        if sample_n and sample_n > 0 and sample_n < len(X):
            rs = np.random.RandomState(seed)
            idx = rs.choice(len(X), size=sample_n, replace=False)
            X_work = X.iloc[idx].reset_index(drop=True)
            y_work = y.iloc[idx].reset_index(drop=True)
            if X_test is not None:
                # keep full test for predictions (not strictly necessary to sample test)
                X_test_work = X_test.copy()

        print('\n[L1] Training base models...')
        oof_l1, test_l1, base_aucs = train_base_models(X_work, y_work, X_test_work, seed=seed, n_splits=n_splits, mode=mode)

        print('\n[L2] Training meta model...')
        oof_l2, test_l2 = train_meta_l2(oof_l1, y_work, test_l1, seed=seed, use_neural=use_neural)

        print('\n[L3] Pseudo-labeling...')
        oof_l3 = train_meta_l3_with_pseudo(oof_l2, y_work, test_l2, X_work, X_test_work, seed=seed)

        print('\n[CAL] Calibrating predictions...')
        # Fix: Use the same y_work for calibration that was used for training
        iso = fit_isotonic(y_work.values, oof_l3)
        oof_cal = iso.predict(oof_l3)
        auc_cal = roc_auc_score(y_work, oof_cal)
        test_cal = iso.predict(test_l2) if test_l2 is not None else None

        best_thr = threshold_sweep(y_work.values, oof_cal)

        print(f'\n{"="*70}')
        print(f'🎯 FINAL AUC (calibrated): {auc_cal:.5f}')
        print(f'📊 Best threshold: {best_thr["threshold"]:.3f} (F1={best_thr["f1"]:.4f})')
        print(f'{"="*70}')

        record = {
            'seed': seed,
            'auc_l2': roc_auc_score(y_work, oof_l2),
            'auc_l3': roc_auc_score(y_work, oof_l3),
            'auc_cal': auc_cal,
            'best_thr': best_thr,
        }
        results.append(record)

        if (best is None) or (auc_cal > best['auc_cal']):
            best = {
                **record,
                'oof_cal': oof_cal,
                'test_cal': test_cal,
                'base_aucs': base_aucs
            }
            print(f'✨ NEW BEST: {auc_cal:.5f}')

        if auc_cal >= target_auc:
            print(f'\n🏆 BREAKTHROUGH! Hit {target_auc:.1%} target!')
            break

    return pd.DataFrame(results), best

# CLI main

def build_submission(best):
    if test is not None and SAMPLE_SUB_PATH.exists() and best is not None:
        sub = pd.read_csv(SAMPLE_SUB_PATH)
        sub_id_col = sub.columns[0]
        sub_target_col = sub.columns[1] if len(sub.columns) > 1 else (TARGET if TARGET is not None else 'target')

        if ID_COL and sub_id_col != ID_COL and ID_COL in test.columns:
            sub[sub_id_col] = test[ID_COL].values

        preds = best.get('test_cal', None)
        if preds is not None:
            sub[sub_target_col] = preds
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            auc_str = f"{best['auc_cal']:.5f}".replace('.', '')
            out_path = SUB_DIR / f'ULTRA_AGGRESSIVE_AUC{auc_str}_{timestamp}.csv'
            sub.to_csv(out_path, index=False)

            print(f'\n🏆 SUBMISSION SAVED!')
            print(f'   File: {out_path.name}')
            print(f'   AUC: {best["auc_cal"]:.5f}')
            print(f'   Threshold: {best["best_thr"]["threshold"]:.3f}')
            print(f'   Samples: {len(sub):,}')

            if best['auc_cal'] >= 0.92849:
                print(f'\n🎊 TARGET BEATEN! {best["auc_cal"]:.5f} > 0.92849! Submit this! 🎊')
        else:
            print('⚠️  No test predictions available.')
    else:
        print('⚠️  Submission not created (missing test data or best result).')


def parse_args(argv: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description='Extreme stacking trainer for loan default prediction')
    parser.add_argument('--seeds', type=str, default=','.join([str(s) for s in DEFAULT_SEEDS]), help='Comma-separated list of integer seeds')
    parser.add_argument('--target-auc', type=float, default=0.9285, help='Target AUC to stop early (must beat 0.92849)')
    parser.add_argument('--use-neural', action='store_true', default=DEFAULT_USE_NEURAL, help='Enable neural meta (MLP) blending at L2')
    parser.add_argument('--n-splits', type=int, default=DEFAULT_N_SPLITS, help='Number of CV splits for base models and meta')
    parser.add_argument('--sample-n', type=int, default=DEFAULT_SAMPLE_N, help='If >0, sample this many training rows for a quick smoke run')
    parser.add_argument('--save-models', action='store_true', default=DEFAULT_SAVE_MODELS, help='Save refit base models on full data into models/ (joblib)')
    parser.add_argument('--more-seeds', type=int, default=DEFAULT_MORE_SEEDS, help='Append this many consecutive seeds after the provided seeds')
    parser.add_argument('--use-enable-categorical', action='store_true', default=DEFAULT_USE_ENABLE_CATEGORICAL, help='Attempt to use XGBoost categorical support when available')
    parser.add_argument('--mode', type=str, choices=['FAST','ULTRA'], default=DEFAULT_MODE, help='Runtime profile: FAST (default) or ULTRA')
    args = parser.parse_args(argv)

    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
        if not seeds:
            seeds = [42]
    except Exception:
        seeds = [42]

    # Append more sequential seeds if requested
    if args.more_seeds and args.more_seeds > 0:
        last = seeds[-1]
        extra = [last + i + 1 for i in range(args.more_seeds)]
        seeds = seeds + extra

    return seeds, args.target_auc, args.use_neural, args.n_splits, args.sample_n, args.save_models, args.use_enable_categorical, args.mode


def main(argv: Optional[List[str]] = None):
    print('🎯 TARGET: MUST BEAT 0.92849 AUC (92.85%)')
    print('📈 Strategy: ULTRA AGGRESSIVE L1→L2→L3 stacking + pseudo-labeling + calibration')
    print('⏱️  ETA: ~25-40 minutes with ultra aggressive optimization\n')

    seeds, target_auc, use_neural, n_splits, sample_n, save_models, use_enable_categorical, mode = parse_args(argv)
    results_df, best = run_training_extreme(seeds=seeds, target_auc=target_auc, use_neural=use_neural,
                                           n_splits=n_splits, sample_n=sample_n, save_models=save_models,
                                           use_enable_categorical=use_enable_categorical, mode=mode)

    print('\n' + '='*70)
    print('📊 RESULTS SUMMARY')
    print('='*70)
    try:
        display_df(results_df)
    except Exception:
        print(results_df)

    if best is not None:
        print(f'\n🏆 BEST RESULT:')
        print(f'   Seed: {best["seed"]}')
        print(f'   L2 AUC: {best["auc_l2"]:.5f}')
        print(f'   L3 AUC: {best["auc_l3"]:.5f}')
        print(f'   Calibrated AUC: {best["auc_cal"]:.5f}')
        print(f'   Threshold: {best["best_thr"]["threshold"]:.3f}')
        print(f'   F1 Score: {best["best_thr"]["f1"]:.4f}')

        if best['auc_cal'] >= 0.92849:
            print(f'\n🎉🎉🎉 TARGET ACHIEVED! {best["auc_cal"]:.5f} >= 0.92849 🎉🎉🎉')
        else:
            gap = 0.92849 - best['auc_cal']
            print(f'\n📍 Gap to 0.92849: {gap:.5f} ({gap*100:.3f} pp)')
            print('💡 Next steps: Add even more seeds or try different hyperparameters')

        build_submission(best)
    else:
        print('\n⚠️  No best result produced.')


if __name__ == '__main__':
    main()
