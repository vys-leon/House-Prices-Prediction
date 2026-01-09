from typing import List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class FeatureSelector:
    """
    Определяет численные и категориальные признаки
    """
    def __init__(self):
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    def fit(self, data: pd.DataFrame) -> 'FeatureSelector':
        self.numeric_features = data.select_dtypes(include='object').columns.tolist()
        self.categorical_features = data.select_dtypes(include=['number']).columns.tolist()
        return self


class DataPreprocessor:
    """
    Создаёт sklearn ColumnTransformer для препроцессинга данных
    """
    def __init__(self):
        self.column_transformer: ColumnTransformer | None = None

    def build(self, numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown=False))
            ]
        )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features)
            ]
        )

        return self.column_transformer


class TargetTransformer:
    """
    Логарифмическое преобразование целевой переменной
    """

    def fit(self, y: pd.Series) -> "TargetTransformer":
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        return np.log1p(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y)
