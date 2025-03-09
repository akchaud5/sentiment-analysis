#!/usr/bin/env python
"""
Advanced prediction script for sentiment analysis with confidence calibration and explanation
"""
import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

# Ensure NLTK resources are downloaded
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize NLTK components
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Add sentiment-specific keywords with higher weight (same as in train_improved.py)
sentiment_keywords = {
    'positive': ['excellent', 'amazing', 'wonderful', 'great', 'good', 'best', 'fantastic', 'perfect', 'love', 'awesome', 'outstanding', 'exceptional', 'superb', 'brilliant', 'happy', 'delighted', 'satisfied', 'impressed', 'recommend', 'worth'],
    'negative': ['terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'disappointing', 'disappointment', 'hate', 'dislike', 'useless', 'waste', 'broken', 'defective', 'failure', 'problem', 'issue', 'complaint', 'refund', 'return', 'never', 'avoid']
}

def preprocess_text(text):
    """Clean and preprocess text data with advanced features"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using TreebankWordTokenizer
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    # Add sentiment-specific handling
    processed_tokens = []
    for token in tokens:
        # Emphasize sentiment keywords by repeating them
        if token in sentiment_keywords['positive'] or token in sentiment_keywords['negative']:
            processed_tokens.append(token)
            processed_tokens.append(token)  # Repeat for emphasis
        else:
            processed_tokens.append(token)
    
    return " ".join(processed_tokens)

def extract_text_features(text):
    """Extract additional text features beyond TF-IDF"""
    if not isinstance(text, str) or not text.strip():
        return [0, 0, 0, 0, 0]
    
    # Text length features
    word_count = len(text.split())
    char_count = len(text)
    avg_word_length = char_count / max(word_count, 1)
    
    # Sentiment keyword counts
    positive_count = sum(1 for word in text.lower().split() if word in sentiment_keywords['positive'])
    negative_count = sum(1 for word in text.lower().split() if word in sentiment_keywords['negative'])
    
    return [word_count, char_count, avg_word_length, positive_count, negative_count]

def explain_prediction(text, prediction, confidence, vectorizer, model):
    """Provide an explanation for the sentiment prediction"""
    # Process text
    processed_text = preprocess_text(text)
    tokens = processed_text.split()
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get feature weights
    if hasattr(model, 'estimators_'):
        # For ensemble models, use the first estimator (usually Random Forest)
        if hasattr(model.estimators_[0], 'feature_importances_'):
            feature_weights = model.estimators_[0].feature_importances_
        else:
            # For other models like LogisticRegression
            try:
                feature_weights = np.abs(model.estimators_[0].coef_[0])
            except:
                feature_weights = np.ones(len(feature_names))
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        feature_weights = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        feature_weights = np.abs(model.coef_[0])
    else:
        # Default case
        feature_weights = np.ones(len(feature_names))
    
    # Get the vector representation of the text
    text_vector = vectorizer.transform([processed_text])
    
    # Get the non-zero features
    non_zero = text_vector.nonzero()[1]
    
    # Calculate the weighted score for each token
    token_scores = []
    for token in tokens:
        # Find all features containing this token
        token_features = [i for i, name in enumerate(feature_names) if token in name.split()]
        
        # Calculate the weighted score
        score = 0
        for feat_idx in token_features:
            if feat_idx in non_zero:
                score += feature_weights[feat_idx] * text_vector[0, feat_idx]
        
        if token_features:
            token_scores.append((token, score))
    
    # Sort by score
    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top positive and negative contributing words
    top_n = min(5, len(token_scores))
    top_tokens = token_scores[:top_n]
    
    # Calculate sentiment indicators
    positive_indicators = [token for token in tokens if token in sentiment_keywords['positive']]
    negative_indicators = [token for token in tokens if token in sentiment_keywords['negative']]
    
    # Create explanation
    explanation = {
        'sentiment': 'positive' if prediction == 1 or prediction == 'positive' else 'negative',
        'confidence': confidence,
        'top_tokens': top_tokens,
        'positive_indicators': positive_indicators,
        'negative_indicators': negative_indicators
    }
    
    return explanation

def format_explanation(explanation):
    """Format the explanation for display"""
    sentiment = explanation['sentiment']
    confidence = explanation['confidence']
    top_tokens = explanation['top_tokens']
    positive_indicators = explanation['positive_indicators']
    negative_indicators = explanation['negative_indicators']
    
    # Create formatted explanation
    formatted = f"Sentiment: {sentiment.upper()} (Confidence: {confidence:.2f})\n\n"
    
    if top_tokens:
        formatted += "Top contributing words:\n"
        for token, score in top_tokens:
            formatted += f"  - {token}: {score:.4f}\n"
    
    if positive_indicators:
        formatted += "\nPositive indicators: " + ", ".join(positive_indicators)
    
    if negative_indicators:
        formatted += "\nNegative indicators: " + ", ".join(negative_indicators)
    
    strength = "strong" if confidence > 0.8 else "moderate" if confidence > 0.6 else "weak"
    formatted += f"\n\nOverall, this text expresses {strength} {sentiment} sentiment."
    
    return formatted

def load_model_and_vectorizer():
    """Load the trained model and feature extractor"""
    # First try to load the ensemble model
    ensemble_model_path = 'models/ensemble_model.pkl'
    advanced_vectorizer_path = 'models/advanced_vectorizer.pkl'
    
    if os.path.exists(ensemble_model_path) and os.path.exists(advanced_vectorizer_path):
        model = joblib.load(ensemble_model_path)
        with open(advanced_vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Loaded advanced ensemble model")
        return model, vectorizer
    
    # Fall back to the standard model
    model_path = 'models/random_forest_model.pkl'
    vectorizer_path = 'models/feature_extractor.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or vectorizer not found. Please run train.py or train_improved.py first.")
        sys.exit(1)
    
    # Load model and vectorizer
    model = joblib.load(model_path)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("Loaded standard model")
    return model, vectorizer

def predict_sentiment(text, model, vectorizer, explain=True):
    """Predict sentiment for given text with explanation"""
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    if not processed_text:
        return "neutral", 0.5, None  # Default for empty text
    
    # Extract features
    features = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Get probability
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    # Convert prediction to string if needed
    sentiment = "positive" if prediction == 1 or prediction == "positive" else "negative"
    
    # Generate explanation if requested
    explanation = None
    if explain:
        explanation = explain_prediction(text, prediction, confidence, vectorizer, model)
    
    return sentiment, confidence, explanation

def predict_from_input():
    """Predict sentiment from user input with explanation"""
    model, vectorizer = load_model_and_vectorizer()
    
    print("\n===== Advanced Sentiment Analysis Predictor =====")
    print("Enter text to analyze (or 'q' to quit):")
    
    while True:
        text = input("> ")
        
        if text.lower() in ['q', 'quit', 'exit']:
            break
        
        sentiment, confidence, explanation = predict_sentiment(text, model, vectorizer)
        
        if explanation:
            print("\n" + format_explanation(explanation) + "\n")
        else:
            print(f"\nSentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}\n")

def predict_from_file(file_path):
    """Predict sentiment for texts in a CSV file with explanations"""
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
    
    for i, row in df.iterrows():
        text = row['text']
        sentiment, confidence, explanation = predict_sentiment(text, model, vectorizer)
        
        if i < 5:  # Show explanations for the first 5 entries
            print(f"\nText {i+1}: {text[:100]}...")
            print(format_explanation(explanation))
        
        # Extract top tokens if explanation is available
        top_tokens = []
        if explanation and 'top_tokens' in explanation:
            top_tokens = [token for token, _ in explanation['top_tokens']]
        
        results.append({
            'text': text,
            'predicted_sentiment': sentiment,
            'confidence': confidence,
            'top_tokens': ','.join(top_tokens[:3]) if top_tokens else '',
            'positive_indicators': ','.join(explanation['positive_indicators'][:3]) if explanation and 'positive_indicators' in explanation else '',
            'negative_indicators': ','.join(explanation['negative_indicators'][:3]) if explanation and 'negative_indicators' in explanation else ''
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save results
    output_path = f"{os.path.splitext(file_path)[0]}_predictions_advanced.csv"
    output_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    
    # Create confidence distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=output_df, x='confidence', hue='predicted_sentiment', bins=20, kde=True)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plot_path = f"{os.path.splitext(file_path)[0]}_confidence_plot.png"
    plt.savefig(plot_path)
    print(f"Confidence distribution plot saved to {plot_path}")

def main():
    if len(sys.argv) > 1:
        # Predict from file
        file_path = sys.argv[1]
        predict_from_file(file_path)
    else:
        # Example texts
        model, vectorizer = load_model_and_vectorizer()
        
        test_examples = [
            "This is the best product I've ever purchased. Absolutely love it!",
            "Terrible experience. The product broke after one use and customer service was unhelpful.",
            "Average product, nothing special but gets the job done.",
            "I'm quite satisfied with my purchase, though there's room for improvement.",
            "Don't waste your money on this. Complete disappointment."
        ]
        
        print("\n===== Advanced Sentiment Analysis Examples =====")
        
        for text in test_examples:
            sentiment, confidence, explanation = predict_sentiment(text, model, vectorizer)
            
            print("\nText:", text)
            print(format_explanation(explanation))
            print("-" * 50)
        
        # Interactive mode
        choice = input("\nWould you like to analyze your own text? (y/n): ")
        if choice.lower() in ['y', 'yes']:
            predict_from_input()

if __name__ == "__main__":
    main()