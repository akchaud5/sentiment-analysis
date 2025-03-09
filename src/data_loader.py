import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the data for modeling.
    
    Args:
        df (pd.DataFrame): Raw data
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert text to lowercase (using .loc to avoid SettingWithCopyWarning)
    df.loc[:, 'text'] = df['text'].str.lower()
    
    return df

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input data
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Calculate the adjusted validation size
    val_ratio = val_size / (1 - test_size)
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        df['text'], 
        df['sentiment'],
        test_size=test_size,
        random_state=random_state
    )
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, 
        y_temp,
        test_size=val_ratio,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test