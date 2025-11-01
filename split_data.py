import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


def resolve_train_path() -> Path:
    # Prefer the capitalized folder used in this repo
    candidates = [
        Path('Data') / 'train.csv',
        Path('data') / 'train.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find train.csv in any of: {[str(p) for p in candidates]} (cwd={os.getcwd()})")


def main():
    path = resolve_train_path()
    df = pd.read_csv(path)

    target_col = 'loan_paid_back'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Print sizes to verify
    print("Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    # Optional: print class balance to confirm stratification
    print("\nClass balance (train):")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nClass balance (test):")
    print(y_test.value_counts(normalize=True).sort_index())

    # Apply SMOTE-like resampling (SMOTENC) ONLY on the training set to balance classes
    # Identify categorical feature columns (object dtype)
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    if len(cat_indices) > 0:
        smote = SMOTENC(categorical_features=cat_indices, random_state=42)
    else:
        # Fallback to standard SMOTE when there are no categorical features
        from imblearn.over_sampling import SMOTE  # local import to avoid unused-import when cat present
        smote = SMOTE(random_state=42)

    res = smote.fit_resample(X_train, y_train)
    if isinstance(res, tuple):
        if len(res) == 2:
            X_resampled, y_resampled = res
        elif len(res) == 3:
            # Some APIs may return (X, y, sample_weight)
            X_resampled, y_resampled, _sample_weight = res
        else:
            raise RuntimeError(f"Unexpected number of return values from fit_resample: {len(res)}")
    else:
        raise RuntimeError("Unexpected return type from fit_resample; expected a tuple")

    # Convert back to DataFrame/Series with original column names
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    # Ensure 1D array/Series for labels
    if isinstance(y_resampled, pd.DataFrame):
        y_resampled = y_resampled.iloc[:, 0]
    else:
        y_resampled = pd.Series(np.asarray(y_resampled).ravel(), name=y_train.name)

    print("\nAfter SMOTE (on training set only) class counts:")
    print(y_resampled.value_counts().sort_index())

    # Save outputs
    out_dir = Path('Data') / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = pd.concat([X_train, y_train], axis=1)
    test_out = pd.concat([X_test, y_test], axis=1)

    train_path = out_dir / 'train_split.csv'
    test_path = out_dir / 'test_split.csv'

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"\nWrote: {train_path} ({len(train_out)}) and {test_path} ({len(test_out)})")

    # Additionally, save the resampled training set for downstream model training if desired
    smote_train_path = out_dir / 'train_split_smote.csv'
    pd.concat([X_resampled, y_resampled], axis=1).to_csv(smote_train_path, index=False)
    print(f"Wrote SMOTE-resampled train: {smote_train_path} ({len(y_resampled)})")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
