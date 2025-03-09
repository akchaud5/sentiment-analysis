#!/usr/bin/env python
"""
Train a sentiment analysis model using the combined dataset
"""
import os
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Use the punkt tokenizer directly
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using TreebankWordTokenizer instead of word_tokenize
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    return " ".join(tokens)

def load_and_prepare_data():
    """Load the combined dataset and prepare for training"""
    dataset_path = "data/processed/combined_sentiment.csv"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run download_datasets.py first.")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {df.shape[0]} samples")
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove empty processed texts
    df = df[df['processed_text'].str.strip().astype(bool)]
    
    # Split data
    X = df['processed_text']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def extract_features(X_train, X_test):
    """Extract TF-IDF features from text"""
    print("Extracting features...")
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Save the vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/feature_extractor.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Features extracted. Training set shape: {X_train_features.shape}")
    
    return X_train_features, X_test_features, vectorizer

def train_model(X_train_features, y_train):
    """Train the sentiment analysis model"""
    print("Training the model...")
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_features, y_train)
    
    # Save the model
    joblib.dump(model, 'models/random_forest_model.pkl')
    
    return model

def evaluate_model(model, X_test_features, y_test):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test_features)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(report)

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Extract features
    X_train_features, X_test_features, vectorizer = extract_features(X_train, X_test)
    
    # Train model
    model = train_model(X_train_features, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test_features, y_test)
    
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()