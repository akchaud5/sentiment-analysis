#!/usr/bin/env python
"""
Script to make predictions with the trained sentiment analysis model
"""
import os
import sys
import pickle
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Use the TreebankWordTokenizer instead of word_tokenize
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
    
    # Tokenize using TreebankWordTokenizer
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    return " ".join(tokens)

def load_model_and_vectorizer():
    """Load the trained model and feature extractor"""
    model_path = 'models/random_forest_model.pkl'
    vectorizer_path = 'models/feature_extractor.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or vectorizer not found. Please run train_model.py first.")
        sys.exit(1)
    
    # Load model and vectorizer
    model = joblib.load(model_path)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    if not processed_text:
        return "neutral", 0.5  # Default for empty text
    
    # Extract features
    features = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    sentiment = "positive" if prediction == "positive" else "negative"
    
    # Get probability
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return sentiment, confidence

def predict_from_input():
    """Predict sentiment from user input"""
    model, vectorizer = load_model_and_vectorizer()
    
    print("Sentiment Analysis Predictor")
    print("Enter text to analyze (or 'q' to quit):")
    
    while True:
        text = input("> ")
        
        if text.lower() in ['q', 'quit', 'exit']:
            break
        
        sentiment, confidence = predict_sentiment(text, model, vectorizer)
        
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print()

def predict_from_file(file_path):
    """Predict sentiment for texts in a CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    model, vectorizer = load_model_and_vectorizer()
    
    # Read input file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    if 'text' not in df.columns:
        print("Error: Input file must contain a 'text' column.")
        sys.exit(1)
    
    # Make predictions
    print(f"Making predictions for {df.shape[0]} texts...")
    results = []
    
    for _, row in df.iterrows():
        text = row['text']
        sentiment, confidence = predict_sentiment(text, model, vectorizer)
        results.append({
            'text': text,
            'predicted_sentiment': sentiment,
            'confidence': confidence
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save results
    output_path = f"{os.path.splitext(file_path)[0]}_predictions.csv"
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")

def main():
    if len(sys.argv) > 1:
        # Predict from file
        file_path = sys.argv[1]
        predict_from_file(file_path)
    else:
        # Example texts
        texts = [
            "This is a fantastic product. Very happy with my purchase!",
            "Worst purchase ever. Doesn't work at all.",
            "Decent quality, but overpriced for what you get."
        ]
        
        model, vectorizer = load_model_and_vectorizer()
        
        print("\nPrediction Results:")
        print("==================")
        for text in texts:
            sentiment, confidence = predict_sentiment(text, model, vectorizer)
            print(f"Text: {text}")
            print(f"Predicted Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            print()

if __name__ == "__main__":
    main()