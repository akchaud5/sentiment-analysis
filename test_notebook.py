#!/usr/bin/env python
"""
Script to test key components of the sentiment analysis notebook
"""
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import pickle

print("1. Initializing NLTK components...")
# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize NLTK components
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using TreebankWordTokenizer
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    return " ".join(tokens)

print("\n2. Testing preprocessing function...")
sample_text = "I absolutely loved this product! It's exactly what I was looking for and exceeded my expectations."
processed_text = preprocess_text(sample_text)
print(f"Original: {sample_text}")
print(f"Processed: {processed_text}")

print("\n3. Checking for dataset files...")
combined_dataset_path = 'data/processed/combined_sentiment.csv'

if os.path.exists(combined_dataset_path):
    print(f"✓ Combined dataset found at {combined_dataset_path}")
    
    print("\n4. Testing model and vectorizer loading...")
    model_path = 'models/random_forest_model.pkl'
    vectorizer_path = 'models/feature_extractor.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print(f"✓ Model found at {model_path}")
        print(f"✓ Vectorizer found at {vectorizer_path}")
        
        # Load model and vectorizer
        model = joblib.load(model_path)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("\n5. Testing prediction on sample texts...")
        test_examples = [
            "This is the best product I've ever purchased. Absolutely love it!",
            "Terrible experience. The product broke after one use and customer service was unhelpful.",
            "Average product, nothing special but gets the job done."
        ]
        
        for text in test_examples:
            # Preprocess the text
            processed = preprocess_text(text)
            
            # Extract features
            features = vectorizer.transform([processed])
            
            # Make prediction
            prediction = model.predict(features)[0]
            confidence = max(model.predict_proba(features)[0])
            
            sentiment = "positive" if prediction == 1 or prediction == "positive" else "negative"
            
            print(f"\nText: {text}")
            print(f"Predicted Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
    else:
        if not os.path.exists(model_path):
            print(f"✗ Model not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            print(f"✗ Vectorizer not found at {vectorizer_path}")
else:
    print(f"✗ Combined dataset not found at {combined_dataset_path}")
    print("Run src/download_datasets.py first to download and process datasets")

print("\nTest script completed!")