"""
Feature Engineering and Statistical Selection Pipeline
Internship Track: Week 2 - Task 3
Author: Sayantani Adhikary
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_and_create_features(filepath: str) -> pd.DataFrame:
    """Load raw dataset and engineer domain-specific interaction features."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded raw dataset with shape: {df.shape}")

    # 1. Feature Creation: Financial ratios
    df['Debt_to_Income'] = df['ExistingDebt'] / df['AnnualIncome']
    df['Loan_to_Income'] = df['LoanAmount'] / df['AnnualIncome']
    
    # 2. Financial Burden Indicator
    df['Total_Liability'] = df['LoanAmount'] + df['ExistingDebt']
    
    print("[INFO] Successfully engineered 3 new features: Debt_to_Income, Loan_to_Income, Total_Liability")
    return df


def preprocess_and_encode(df: pd.DataFrame):
    """Handle categorical encoding and split features/target."""
    # Drop identifier
    X_raw = df.drop(columns=['CustomerID', 'LoanApproved'])
    y = df['LoanApproved']

    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X_raw, columns=['EmploymentType'], drop_first=True)
    print(f"[INFO] Dataset shape after One-Hot Encoding: {X_encoded.shape}")
    return X_encoded, y


def scale_features(X_encoded: pd.DataFrame):
    """Apply StandardScaler to all continuous numerical features."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_encoded),
        columns=X_encoded.columns
    )
    print("[INFO] Applied StandardScaler normalization across all features.")
    return X_scaled, scaler


def select_best_features(X_scaled: pd.DataFrame, y: pd.Series, k: int = 5):
    """Select the top k features using ANOVA F-value (f_classif)."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Extract feature scores and rankings
    scores_df = pd.DataFrame({
        'Feature': X_scaled.columns,
        'F_Score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values(by='F_Score', ascending=False)
    
    selected_cols = X_scaled.columns[selector.get_support()]
    print("\n--- Feature Selection Rankings (ANOVA F-Score) ---")
    print(scores_df.to_string(index=False))
    print(f"\n[INFO] Top {k} selected features: {list(selected_cols)}")
    
    return pd.DataFrame(X_selected, columns=selected_cols), scores_df


def main():
    raw_path = "data/raw_dataset.csv"
    
    # Run pipeline
    df_features = load_and_create_features(raw_path)
    X_encoded, y = preprocess_and_encode(df_features)
    X_scaled, _ = scale_features(X_encoded)
    X_selected, scores_df = select_best_features(X_scaled, y, k=5)
    
    # Export engineered dataset
    output_df = X_selected.copy()
    output_df['LoanApproved'] = y.values
    output_df.to_csv("data/engineered_dataset.csv", index=False)
    print("\n[SUCCESS] Exported engineered dataset to 'data/engineered_dataset.csv'")


if __name__ == "__main__":
    main()
