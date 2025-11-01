import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
