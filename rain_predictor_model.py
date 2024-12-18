import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from preprocessor import Preprocessor

# Class for Model Management
class RainPredictorModel:
    def __init__(self, model_path):
        """
        Initializes the model and its associated preprocessors.

        Args:
            model_path (str): Path to the saved joblib model file.
        """
        # Load saved model components
        self.model = joblib.load(model_path)
        self.input_cols = self.model['input_cols']
        self.numeric_cols = self.model['numeric_cols']
        self.categorical_cols = self.model['categorical_cols']

        # Initialize Preprocessor with pre-fitted components
        self.preprocessor = Preprocessor(
            numeric_cols=self.numeric_cols,
            imputer=self.model['imputer'],
            scaler=self.model['scaler'],
            encoder=self.model['encoder'],
            categorical_cols=self.categorical_cols
        )

    def prepare_input_data(self, input_fields: dict) -> pd.DataFrame:
        """
        Converts input fields into a DataFrame matching the model's expected input structure.

        Args:
            input_fields (dict): Raw input fields from the user.

        Returns:
            pd.DataFrame: Formatted input data.
        """
        data = pd.DataFrame(input_fields).T
        data.columns = self.input_cols
        data[self.numeric_cols] = data[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
        return data

    def predict(self, input_fields: dict) -> Tuple[int, float]:
        """
        Preprocesses input data and makes a prediction.

        Args:
            input_fields (dict): Raw input fields.

        Returns:
            Tuple[int, float]: Predicted class and its probability.
        """
        # Prepare input data
        data = self.prepare_input_data(input_fields)

        # Preprocess using Preprocessor class
        data = self.preprocessor.preprocess(data)

        # Predict probabilities and return the result
        pred_proba = self.model['model'].predict_proba(data)
        prediction = np.argmax(pred_proba)
        probability = round(np.max(pred_proba), 2)
        return prediction, probability
