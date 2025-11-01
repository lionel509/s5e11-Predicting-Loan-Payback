import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_csv_path(filename: str) -> Path:
    """Find the CSV file in Data/ or data/ folder."""
    candidates = [
        Path('Data') / filename,
        Path('data') / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {[str(p) for p in candidates]} (cwd={os.getcwd()})"
    )


def main():
    # Load train.csv
    train_path = resolve_csv_path('train.csv')
    df = pd.read_csv(train_path)
    print(f"Loaded training data from: {train_path}")
    
    # Load test.csv
    test_path = resolve_csv_path('test.csv')
    test_df = pd.read_csv(test_path)
    print(f"Loaded test data from: {test_path}")
    
    # Prepare features and target from training data
    X = df.drop("loan_paid_back", axis=1)
    y = df["loan_paid_back"]
    
    # Stratified split: 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    
    # Print shapes of each dataset
    print("\nDataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  test_df: {test_df.shape}")
    
    # Print class distribution to verify stratification
    print("\nClass distribution in y_train:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nClass distribution in y_val:")
    print(y_val.value_counts(normalize=True).sort_index())


if __name__ == '__main__':
    main()
