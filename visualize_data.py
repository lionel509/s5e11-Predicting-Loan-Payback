"""
Data Visualization Script for Loan Payback Prediction
This script creates comprehensive visualizations of the loan dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
print("Loading data...")
train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("\nTraining data info:")
print(train_df.info())
print("\nFirst few rows:")
print(train_df.head())

# Create output directory for plots
output_dir = Path('visualizations')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. Target Variable Distribution
# ============================================================================
print("\n1. Visualizing target variable distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
loan_counts = train_df['loan_paid_back'].value_counts()
axes[0].bar(['Not Paid Back', 'Paid Back'], loan_counts.values, color=['#ff6b6b', '#51cf66'])
axes[0].set_ylabel('Count')
axes[0].set_title('Loan Payback Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(loan_counts.values):
    axes[0].text(i, v + 500, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(loan_counts.values, labels=['Not Paid Back', 'Paid Back'], 
            autopct='%1.1f%%', startangle=90, colors=['#ff6b6b', '#51cf66'])
axes[1].set_title('Loan Payback Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '01_target_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '01_target_distribution.png'}")

# ============================================================================
# 2. Numerical Features Distribution
# ============================================================================
print("\n2. Visualizing numerical features distribution...")
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(train_df[col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(col.replace('_', ' ').title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'Distribution of {col.replace("_", " ").title()}', fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / '02_numerical_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '02_numerical_distributions.png'}")

# ============================================================================
# 3. Numerical Features vs Target
# ============================================================================
print("\n3. Visualizing numerical features vs target variable...")
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    train_df.boxplot(column=col, by='loan_paid_back', ax=axes[idx])
    axes[idx].set_xlabel('Loan Paid Back (0=No, 1=Yes)')
    axes[idx].set_ylabel(col.replace('_', ' ').title())
    axes[idx].set_title(f'{col.replace("_", " ").title()} by Loan Payback Status')
    axes[idx].get_figure().suptitle('')  # Remove default title
    
# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / '03_numerical_vs_target.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '03_numerical_vs_target.png'}")

# ============================================================================
# 4. Categorical Features Distribution
# ============================================================================
print("\n4. Visualizing categorical features distribution...")
categorical_cols = ['gender', 'marital_status', 'education_level', 
                   'employment_status', 'loan_purpose']

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    value_counts = train_df[col].value_counts()
    axes[idx].barh(range(len(value_counts)), value_counts.values, color='teal')
    axes[idx].set_yticks(range(len(value_counts)))
    axes[idx].set_yticklabels(value_counts.index)
    axes[idx].set_xlabel('Count')
    axes[idx].set_title(f'Distribution of {col.replace("_", " ").title()}', fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='x')

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / '04_categorical_distributions.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '04_categorical_distributions.png'}")

# ============================================================================
# 5. Categorical Features vs Target
# ============================================================================
print("\n5. Visualizing categorical features vs target variable...")
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    # Create crosstab for proportions
    ct = pd.crosstab(train_df[col], train_df['loan_paid_back'], normalize='index') * 100
    ct.plot(kind='bar', stacked=False, ax=axes[idx], color=['#ff6b6b', '#51cf66'])
    axes[idx].set_xlabel(col.replace('_', ' ').title())
    axes[idx].set_ylabel('Percentage (%)')
    axes[idx].set_title(f'Loan Payback Rate by {col.replace("_", " ").title()}', fontweight='bold')
    axes[idx].legend(['Not Paid Back', 'Paid Back'], loc='upper right')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(True, alpha=0.3, axis='y')

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(output_dir / '05_categorical_vs_target.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '05_categorical_vs_target.png'}")

# ============================================================================
# 6. Correlation Heatmap
# ============================================================================
print("\n6. Creating correlation heatmap...")
plt.figure(figsize=(10, 8))

# Select numerical columns including target
corr_cols = numerical_cols + ['loan_paid_back']
correlation_matrix = train_df[corr_cols].corr()

sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / '06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '06_correlation_heatmap.png'}")

# ============================================================================
# 7. Grade Subgrade Analysis
# ============================================================================
print("\n7. Visualizing grade subgrade distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribution
grade_counts = train_df['grade_subgrade'].value_counts().sort_index()
axes[0].bar(range(len(grade_counts)), grade_counts.values, color='coral')
axes[0].set_xticks(range(len(grade_counts)))
axes[0].set_xticklabels(grade_counts.index, rotation=90)
axes[0].set_xlabel('Grade Subgrade')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Grade Subgrades', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Payback rate by grade
grade_payback = train_df.groupby('grade_subgrade')['loan_paid_back'].agg(['mean', 'count'])
grade_payback = grade_payback[grade_payback['count'] > 100].sort_index()  # Filter low-count grades
axes[1].plot(grade_payback.index, grade_payback['mean'] * 100, marker='o', linewidth=2, markersize=6, color='green')
axes[1].set_xlabel('Grade Subgrade')
axes[1].set_ylabel('Payback Rate (%)')
axes[1].set_title('Loan Payback Rate by Grade Subgrade', fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(output_dir / '07_grade_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '07_grade_analysis.png'}")

# ============================================================================
# 8. Scatter Plots - Key Relationships
# ============================================================================
print("\n8. Creating scatter plots for key relationships...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Credit Score vs Interest Rate
scatter1 = axes[0, 0].scatter(train_df['credit_score'], train_df['interest_rate'], 
                              c=train_df['loan_paid_back'], cmap='RdYlGn', alpha=0.5, s=10)
axes[0, 0].set_xlabel('Credit Score')
axes[0, 0].set_ylabel('Interest Rate (%)')
axes[0, 0].set_title('Credit Score vs Interest Rate', fontweight='bold')
plt.colorbar(scatter1, ax=axes[0, 0], label='Loan Paid Back')

# Annual Income vs Loan Amount
scatter2 = axes[0, 1].scatter(train_df['annual_income'], train_df['loan_amount'], 
                              c=train_df['loan_paid_back'], cmap='RdYlGn', alpha=0.5, s=10)
axes[0, 1].set_xlabel('Annual Income ($)')
axes[0, 1].set_ylabel('Loan Amount ($)')
axes[0, 1].set_title('Annual Income vs Loan Amount', fontweight='bold')
plt.colorbar(scatter2, ax=axes[0, 1], label='Loan Paid Back')

# Debt to Income Ratio vs Credit Score
scatter3 = axes[1, 0].scatter(train_df['debt_to_income_ratio'], train_df['credit_score'], 
                              c=train_df['loan_paid_back'], cmap='RdYlGn', alpha=0.5, s=10)
axes[1, 0].set_xlabel('Debt to Income Ratio')
axes[1, 0].set_ylabel('Credit Score')
axes[1, 0].set_title('Debt to Income Ratio vs Credit Score', fontweight='bold')
plt.colorbar(scatter3, ax=axes[1, 0], label='Loan Paid Back')

# Credit Score vs Loan Amount
scatter4 = axes[1, 1].scatter(train_df['credit_score'], train_df['loan_amount'], 
                              c=train_df['loan_paid_back'], cmap='RdYlGn', alpha=0.5, s=10)
axes[1, 1].set_xlabel('Credit Score')
axes[1, 1].set_ylabel('Loan Amount ($)')
axes[1, 1].set_title('Credit Score vs Loan Amount', fontweight='bold')
plt.colorbar(scatter4, ax=axes[1, 1], label='Loan Paid Back')

plt.tight_layout()
plt.savefig(output_dir / '08_scatter_plots.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '08_scatter_plots.png'}")

# ============================================================================
# 9. Summary Statistics
# ============================================================================
print("\n9. Generating summary statistics...")
summary_stats = train_df.describe()
summary_stats.to_csv(output_dir / 'summary_statistics.csv')
print(f"Saved: {output_dir / 'summary_statistics.csv'}")

# Payback rate by categorical variables
print("\nPayback rates by categorical variables:")
for col in categorical_cols:
    payback_rate = train_df.groupby(col)['loan_paid_back'].mean() * 100
    print(f"\n{col.upper()}:")
    print(payback_rate.sort_values(ascending=False))

# ============================================================================
# 10. Missing Values Analysis
# ============================================================================
print("\n10. Analyzing missing values...")
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("\nMissing values in training data:")
print(missing_train[missing_train > 0] if missing_train.sum() > 0 else "No missing values")

print("\nMissing values in test data:")
print(missing_test[missing_test > 0] if missing_test.sum() > 0 else "No missing values")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print(f"\nAll visualizations saved to: {output_dir.absolute()}")
print("\nGenerated files:")
for file in sorted(output_dir.glob('*.png')):
    print(f"  - {file.name}")
print(f"  - summary_statistics.csv")
