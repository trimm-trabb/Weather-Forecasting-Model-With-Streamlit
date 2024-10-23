import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from typing import Dict, Any, Tuple, List

def drop_na_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop rows with NA values in the specified columns.

    Args:
        df (pd.DataFrame): The raw dataframe.
        columns (list): List of columns to check for NA values.

    Returns:
        pd.DataFrame: DataFrame with NA values dropped.
    """
    return df.dropna(subset=columns)

def split_data_by_year(df: pd.DataFrame, year_col: str) -> Dict[str, pd.DataFrame]:
    """
    Split the dataframe into training, validation, and test sets based on the year.

    Args:
        df (pd.DataFrame): The raw dataframe.
        year_col (str): The column containing year information.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing the train, validation, and test dataframes.
    """
    year = pd.to_datetime(df[year_col]).dt.year
    train_df = df[year < 2015]
    val_df = df[year == 2015]
    test_df = df[year > 2015]
    return {'train': train_df, 'val': val_df, 'test': test_df}

def create_inputs_targets(df_dict: Dict[str, pd.DataFrame], input_cols: list, target_col: str) -> Dict[str, Any]:
    """
    Create inputs and targets for training, validation, and test sets.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dictionary containing the train, validation, and test dataframes.
        input_cols (list): List of input columns.
        target_col (str): Target column.

    Returns:
        Dict[str, Any]: Dictionary containing inputs and targets for train, val, and test sets.
    """
    data = {}
    for split in df_dict:
        data[f'{split}_inputs'] = df_dict[split][input_cols].copy()
        data[f'{split}_targets'] = df_dict[split][target_col].copy()
    return data

def impute_missing_values(data: Dict[str, Any], numeric_cols: list) -> IterativeImputer:
    """
    Impute missing numeric values using IterativeImputer that estimates each feature from all the others.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.
        
    Returns:
         IterativeImputer: Fitted imputer used for transforming columns in the dataset.
    """
    imputer = IterativeImputer(max_iter=10, random_state=42).fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val', 'test']:
          data[f'{split}_inputs'][numeric_cols] = imputer.transform(data[f'{split}_inputs'][numeric_cols])

    return imputer

def scale_numeric_features(data: Dict[str, Any], numeric_cols: list) -> MinMaxScaler:
    """
    Scale numeric features using MinMaxScaler.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        numeric_cols (list): List of numerical columns.

    Returns:
        MinMaxScaler: Fitted scaler used to transform the numerical columns in the dataset.
    """
    scaler = MinMaxScaler().fit(data['train_inputs'][numeric_cols])
    for split in ['train', 'val', 'test']:
        data[f'{split}_inputs'][numeric_cols] = scaler.transform(data[f'{split}_inputs'][numeric_cols])

    return scaler

def encode_categorical_features(data: Dict[str, Any], categorical_cols: list) -> Tuple[OneHotEncoder, LabelEncoder]:
    """
    One-hot encode categorical features.

    Args:
        data (Dict[str, Any]): Dictionary containing inputs and targets for train, val, and test sets.
        categorical_cols (list): List of categorical columns.

    Returns:
        Tuple: Contains the following elements:       
        - OneHotEncoder: Fitted one-hot encoder for input features 
        - LabelEncoder: Fitted label encoder for the target variable.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data['train_inputs'][categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    for split in ['train', 'val', 'test']:
        encoded = encoder.transform(data[f'{split}_inputs'][categorical_cols])
        data[f'{split}_inputs'] = pd.concat([data[f'{split}_inputs'], pd.DataFrame(encoded, columns=encoded_cols, index=data[f'{split}_inputs'].index)], axis=1)
        data[f'{split}_inputs'].drop(columns=categorical_cols, inplace=True)
    data['encoded_cols'] = encoded_cols

    label_encoder = LabelEncoder()
    label_encoder.fit(data['train_targets'])
    for split in ['train', 'val', 'test']:
         data[f'{split}_targets'] = label_encoder.transform(data[f'{split}_targets'])

    return encoder, label_encoder

def preprocess_data(raw_df: pd.DataFrame) -> Tuple[Dict[str, Any], IterativeImputer, MinMaxScaler, OneHotEncoder, LabelEncoder, list, list, list, str]:
    """
    Preprocess the raw dataframe.

    Args:
        raw_df (pd.DataFrame): The raw dataframe.

    Returns:
        Tuple: Contains the following elements:
            - Dictionary containing processed inputs and targets for train, val, and test sets.
            - Fitted IterativeImputer used for missing value imputation.
            - Fitted MinMaxScaler used for scaling numeric features.
            - Fitted OneHotEncoder used for encoding categorical features.
            - Fitted LabelEncoder used for encoding the target variable.
            - List of numeric columns.
            - List of categorical columns.
            - List of input columns used for training.
            - Target column.
    """
    raw_df = drop_na_values(raw_df, ['RainToday', 'RainTomorrow'])
    split_dfs = split_data_by_year(raw_df, 'Date')
    input_cols = list(raw_df.columns)[1:-1]
    target_col = 'RainTomorrow'
    data = create_inputs_targets(split_dfs, input_cols, target_col)

    numeric_cols = data['train_inputs'].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data['train_inputs'].select_dtypes('object').columns.tolist()
    imputer = impute_missing_values(data, numeric_cols)
    scaler = scale_numeric_features(data, numeric_cols)
    encoder, label_encoder = encode_categorical_features(data, categorical_cols)

    # Extract X_train, X_val, X_test
    X_train = data['train_inputs'][numeric_cols + data['encoded_cols']]
    X_val = data['val_inputs'][numeric_cols + data['encoded_cols']]
    X_test = data['test_inputs'][numeric_cols + data['encoded_cols']]

    data_dict = {
        'train_X': X_train,
        'train_y': data['train_targets'],
        'val_X': X_val,
        'val_y': data['val_targets'],
        'test_X': X_test,
        'test_y': data['test_targets'],
    }

    return data_dict, imputer, scaler, encoder, label_encoder, numeric_cols, categorical_cols, input_cols, target_col

def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], imputer: IterativeImputer, 
                        scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocesses new data using the provided imputer, scaler and encoder.

    Args:
        new_data (pd.DataFrame): The new input dataframe to preprocess.
        input_cols (List[str]): List of input column names used in training.
        scaler (MinMaxScaler): The scaler fitted on the training data.
        imputer (IterativeImputer): The imputer fitted on the training data.
        encoder (OneHotEncoder): The encoder fitted on the training data.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Select numerical and categorical columns
    numeric_cols = [col for col in input_cols if col in new_data.select_dtypes(include=np.number).columns]
    categorical_cols = [col for col in new_data.columns if col in encoder.feature_names_in_]

    # Impute missing values
    new_data[numeric_cols] = imputer.transform(new_data[numeric_cols])

    # Encode categorical features
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_data[encoded_cols] = encoder.transform(new_data[categorical_cols])

    # Scale numerical features
    if scaler:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    return new_data[numeric_cols+encoded_cols]
