#Data cleaning, encoding, and feature engineering.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def load_and_inspect_data(file_path):
    """
    Load the dataset and display an overview.
    Args:
    file_path (str): Path to the dataset file.
    Returns:
    pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(file_path)
    print("Dataset Info:")
    print(data.info())  # Shows column data types and null values
    print("\nMissing Values:")
    print(data.isnull().sum())  # Check for missing values
    return data

def handle_missing_values(data):
    """
    Handle missing values flexibly for both categorical and numerical variables.
    Args:
    data (pd.DataFrame): The dataset to process.
    Returns:
    pd.DataFrame: The dataset with missing values filled.
    """
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Impute numerical features with the median
    num_imputer = SimpleImputer(strategy='median')
    data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])
    
    # Impute categorical features with the most frequent category (mode)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
    
    return data

def encode_categorical_data(data):
    """
    Encode categorical variables using one-hot encoding.
    Args:
    data (pd.DataFrame): The dataset to process.
    Returns:
    pd.DataFrame: The dataset with encoded categorical features.
    """
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)  # One-hot encoding
    return data

def feature_engineering(data):
    """
    Generate additional features like BMI (Body Mass Index).
    Args:
    data (pd.DataFrame): The dataset to process.
    Returns:
    pd.DataFrame: The dataset with added feature(s).
    """
    data['BMI'] = data['Weight'] / (data['Height'] ** 2)  # BMI calculation
    return data

def handle_outliers(data, continuous_vars):
    """
    Handle outliers using the IQR method.
    Args:
    data (pd.DataFrame): The dataset to process.
    continuous_vars (list): List of continuous variable columns.
    Returns:
    pd.DataFrame: The dataset with outliers handled.
    """
    for var in continuous_vars:
        Q1 = data[var].quantile(0.25)
        Q3 = data[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[var] = np.clip(data[var], lower_bound, upper_bound)
    return data

def normalize_data(data, continuous_vars):
    """
    Normalize continuous variables.
    Args:
    data (pd.DataFrame): The dataset to process.
    continuous_vars (list): List of continuous variable columns.
    Returns:
    pd.DataFrame: The dataset with normalized features.
    """
    scaler = MinMaxScaler()
    data[continuous_vars] = scaler.fit_transform(data[continuous_vars])
    return data
