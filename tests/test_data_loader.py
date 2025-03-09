import unittest
import pandas as pd
import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data, preprocess_data, split_data

class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        """
        Set up test data.
        """
        self.data = {
            'text': [
                'This is a positive review.',
                'This is a negative review.',
                'I love this product!',
                'I hate this product.',
                None,  # Test handling of None
                'Another positive review, I really enjoyed it.',
                'Another negative review, it was terrible.'
            ],
            'sentiment': [1, 0, 1, 0, 1, 1, 0]
        }
        self.df = pd.DataFrame(self.data)
        
        # Create a temporary CSV file for testing
        self.test_csv_path = 'tests/test_data.csv'
        self.df.to_csv(self.test_csv_path, index=False)
        
    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove the temporary CSV file
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
    
    def test_load_data(self):
        """
        Test loading data from a CSV file.
        """
        loaded_df = load_data(self.test_csv_path)
        self.assertEqual(len(loaded_df), len(self.df))
        self.assertEqual(list(loaded_df.columns), list(self.df.columns))
    
    def test_preprocess_data(self):
        """
        Test preprocessing data.
        """
        preprocessed_df = preprocess_data(self.df)
        
        # Check that missing values are handled
        self.assertEqual(len(preprocessed_df), len(self.df) - 1)
        
        # Check that text is converted to lowercase
        for text in preprocessed_df['text']:
            self.assertEqual(text, text.lower())
    
    def test_split_data(self):
        """
        Test splitting data into train, validation, and test sets.
        """
        # First preprocess the data
        preprocessed_df = preprocess_data(self.df)
        
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(preprocessed_df)
        
        # Check that the split sizes are approximately correct
        total_samples = len(preprocessed_df)
        
        # Instead of checking exact sizes, let's check proportions
        test_proportion = len(X_test) / total_samples
        val_proportion = len(X_val) / total_samples
        train_proportion = len(X_train) / total_samples
        
        # Check that the proportions roughly match the expected values
        # Since we have a very small dataset in the test, these can vary more
        self.assertAlmostEqual(test_proportion, 0.2, delta=0.2)
        self.assertAlmostEqual(val_proportion, 0.1, delta=0.2)  
        self.assertAlmostEqual(train_proportion, 0.7, delta=0.2)
        
        # Check that the labels match the features
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))

if __name__ == '__main__':
    unittest.main()