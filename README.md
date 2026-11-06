# Feature Engineering and Selection Pipeline

## Project Overview
This repository implements an end-to-end feature engineering and statistical selection workflow applied to consumer loan credit data. The objective is to engineer domain-specific interaction ratios, handle categorical encoding, standardize feature scales, and identify the most statistically predictive features using ANOVA F-tests.

## Repository Architecture
- README.md
- requirements.txt
- data/raw_dataset.csv
- src/feature_engineering.py
- notebooks/feature_engineering.ipynb
- docs/feature_engineering_report.md

## Key Methodologies Implemented
1. Feature Creation:
   - Engineered Debt_to_Income ratio to measure liquidity strain.
   - Engineered Loan_to_Income ratio to evaluate leverage relative to earning capacity.
   - Created Total_Liability feature aggregating overall debt exposure.
2. Preprocessing & Encoding:
   - Applied One-Hot Encoding to categorical employment dimensions (drop_first=True to avoid collinearity).
   - Standardized continuous variables using StandardScaler.
3. Statistical Selection (ANOVA F-Test):
   - Ranked features with SelectKBest (f_classif).
   - Identified CreditScore, Debt_to_Income, and AnnualIncome as the primary drivers of loan approval.
4. Model Performance Impact:
   - Boosted baseline Logistic Regression classification accuracy from 75.0% to 95.0% using the top 5 engineered features.

## How to Run
1. Clone this repository:
   git clone https://github.com/adhikarysayantani7-tony/feature-engineering-and-selection.git
2. Install dependencies:
   pip install -r requirements.txt
3. Run the modular Python script:
   python src/feature_engineering.py
4. Open the interactive notebook:
   jupyter notebook notebooks/feature_engineering.ipynb
