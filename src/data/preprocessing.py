from typing import List, Optional
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Определяет численные и категориальные признаки
    """
    def __init__(self):
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        self.numeric_features = X.select_dtypes(include='number').columns.tolist()
        self.categorical_features = X.select_dtypes(include='object').columns.tolist()
        return self
    
    def transform(self, X):
        return X[self.numeric_features + self.categorical_features]



class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Создаёт sklearn ColumnTransformer для препроцессинга данных
    """
    def __init__(self):
        self.column_transformer: ColumnTransformer

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        numeric_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = X.select_dtypes(exclude="number").columns.tolist()

        self.column_transformer = self.build(
            numeric_features=numeric_features,
            categorical_features=categorical_features
        )

        self.column_transformer.set_output(transform="pandas")
        self.column_transformer.fit(X)

        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

    def build(self, numeric_features, categorical_features):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown='ignore',
                                          sparse_output=False))
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop"
        )


class TargetTransformer:
    """
    Логарифмическое преобразование целевой переменной
    """

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y)
