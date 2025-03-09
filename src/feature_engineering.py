import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class TextFeatureExtractor:
    """
    Simplified class for text feature extraction and transformation.
    """
    def __init__(self, max_features=5000, min_df=5, max_df=0.7):
        """
        Initialize the text feature extractor.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            min_df (int): Minimum document frequency for TF-IDF
            max_df (float): Maximum document frequency for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            preprocessor=self._preprocess_text
        )
        
    def _preprocess_text(self, text):
        """
        Simple text preprocessing - lowercase and remove special characters
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        return text
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform the texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse.csr.csr_matrix: Document-term matrix
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform the texts using the fitted vectorizer.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse.csr.csr_matrix: Document-term matrix
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get the feature names from the vectorizer.
        
        Returns:
            list: List of feature names
        """
        return self.vectorizer.get_feature_names_out()