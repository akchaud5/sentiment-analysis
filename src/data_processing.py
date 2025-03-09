import pandas as pd
import os
import numpy as np
from src.data_loader import load_data, preprocess_data, split_data

def process_raw_data(raw_file_path, processed_file_path):
    """
    Process raw data and save it to the processed directory.
    
    Args:
        raw_file_path (str): Path to the raw data file
        processed_file_path (str): Path to save the processed data
    """
    print(f"Processing raw data from {raw_file_path}...")
    
    # Load the raw data
    df = load_data(raw_file_path)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    
    # Save the processed data
    df.to_csv(processed_file_path, index=False)
    
    print(f"Processed data saved to {processed_file_path}")
    
    return df

def create_train_val_test_split(processed_file_path, output_dir):
    """
    Split the processed data into train, validation, and test sets.
    
    Args:
        processed_file_path (str): Path to the processed data file
        output_dir (str): Directory to save the split data
    """
    print(f"Splitting data from {processed_file_path}...")
    
    # Load the processed data
    df = load_data(processed_file_path)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the splits
    train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    val_df = pd.DataFrame({'text': X_val, 'sentiment': y_val})
    test_df = pd.DataFrame({'text': X_test, 'sentiment': y_test})
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Train set ({len(train_df)} samples) saved to {os.path.join(output_dir, 'train.csv')}")
    print(f"Validation set ({len(val_df)} samples) saved to {os.path.join(output_dir, 'val.csv')}")
    print(f"Test set ({len(test_df)} samples) saved to {os.path.join(output_dir, 'test.csv')}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # This allows the script to be run directly
    raw_file_path = "data/raw/sample_reviews.csv"
    processed_file_path = "data/processed/reviews.csv"
    output_dir = "data/processed"
    
    # Process the raw data
    df = process_raw_data(raw_file_path, processed_file_path)
    
    # Split the data
    train_df, val_df, test_df = create_train_val_test_split(processed_file_path, output_dir)