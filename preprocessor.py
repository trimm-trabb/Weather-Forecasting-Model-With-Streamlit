import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer   
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from typing import Dict, Any, Tuple, List

class Preprocessor:
    def __init__(self, numeric_cols: List[str], imputer: IterativeImputer, 
                 scaler: MinMaxScaler, encoder: OneHotEncoder, categorical_cols: List[str]):
        """
        Initializes the Preprocessor with pre-fitted preprocessors.

        Args:
            numeric_cols (List[str]): List of numeric column names.
            imputer (IterativeImputer): Pre-fitted imputer for missing values.
            scaler (MinMaxScaler): Pre-fitted scaler for numeric values.
            encoder (OneHotEncoder): Pre-fitted OneHotEncoder for categorical features.
            categorical_cols (List[str]): Names of categorical columns.
        """
        self.numeric_cols = numeric_cols
        self.imputer = imputer
        self.scaler = scaler
        self.encoder = encoder
        self.categorical_cols = categorical_cols

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data using pre-fitted preprocessors.

        Steps:
        - Impute missing values in numeric columns.
        - Apply OneHotEncoding to categorical columns.
        - Scale numeric values.
        - Combine scaled and encoded features.

        Args:
            data (pd.DataFrame): Input DataFrame to preprocess.

        Returns:
            pd.DataFrame: Fully preprocessed DataFrame.
        """
        # Impute missing numeric values
        data[self.numeric_cols] = self.imputer.transform(data[self.numeric_cols])

        # Encode categorical columns
        encoded_cols = list(self.encoder.get_feature_names_out(self.categorical_cols))
        data[encoded_cols] = self.encoder.transform(data[self.categorical_cols])

        # Scale numeric columns
        data[self.numeric_cols] = self.scaler.transform(data[self.numeric_cols])

        # Return final DataFrame in expected order
        return data[self.numeric_cols + encoded_cols]
