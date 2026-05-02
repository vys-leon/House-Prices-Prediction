# House Prices Prediction

<img src="https://img.shields.io/badge/RMSLE-0.13007-blue" /> <img src="https://img.shields.io/badge/Model-LightGBM-success" />

End-to-end ML pipeline for regression problem on tabular data: from exploratory data analysis (EDA) to model training and evaluation

Goal: Predict residential property prices based on their characteristics using classical machine learning methods

## Key features

* Fully reproducible pipeline built with `sklearn.pipeline` (data leakage prevention)
* Modular preprocessing (separate handling of numerical and categorical features)
* Quality evaluation with cross-validation (metric RMSLE)

## Results

| Model | Description | RMSLE (5-fold CV) |
| :--- | :--- | :--- |
| LightGBM (baseline) | Base model without feature engineering | 0.13054 |
| LightGBM (improved) | Model with added feature `TotalSF` and optimized preprocessing | 0.13007 |

## Approach and solution architecture

The project implements a full ML development cycle:

1. Exploratory Data Analysis (EDA): Analysis of distributions, missing values, outliers and correlations
2. Feature Engineering: Generation of the `TotalSF` feature (sum of areas) and selection of significant variables
3. Preprocessing Pipeline:
   * Numerical features: imputation of missing values with the median, standartization (`StandardScaler`)
   * Categorical features: filling missing values with a constant, one-hot encoding
4. Modeling: Training models (`Ridge`, `LightGBM`) with cross-validation. The target variable is log-transformed
5. Validation: RMSLE metric is chosen as the primary metric because it is robust to outliers in price

## Project structure

```text
|-- data/
|   |-- raw/
|   |   |-- data_description.txt
|   |   |-- train.csv
|   |   |-- test.csv
|   |   |__ sample_submission.csv
|   |__ output/
|   |   |__ my_submission.csv
|-- notebooks/           # Jupyter notebooks with EDA and experiments
|   |-- 01_eda.ipynb
|   |__ 02_modeling.ipynb
|-- src/
|   |__ data/
|   |   |__ preprocessing.py
|-- .gitignore
|-- README.md
|__ requirements.txt
```

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/vys-leon/House-Prices-Prediction.git
cd House-Prices-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks

Open 01_eda.ipynb or 02_modeling.ipynb and execute all cells

## Technology stack

* Language: Python 3.x
* Data manipulation: NumPy, pandas
* Visualization: matplotlib, seaborn
* Modeling: scikit-learn, LightGBM
* Tools: Git, Jupyter Notebook