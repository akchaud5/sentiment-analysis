import argparse
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report

from data_loader import load_data, preprocess_data, split_data
from feature_engineering import TextFeatureExtractor
from models import ModelTrainer

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Sentiment Analysis Model')
    parser.add_argument('--data_path', type=str, default='data/processed/reviews.csv',
                       help='Path to the data file')
    parser.add_argument('--model_type', type=str, default='logistic',
                       choices=['logistic', 'rf', 'svm', 'lstm'],
                       help='Type of model to train')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save the model')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict', 'evaluate'],
                       help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model (for predict and evaluate modes)')
    
    return parser.parse_args()

def train_model(args):
    """
    Train a sentiment analysis model.
    
    Args:
        args: Command line arguments
    """
    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    df = preprocess_data(df)
    
    print("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    if args.model_type != 'lstm':
        print("Extracting features...")
        feature_extractor = TextFeatureExtractor()
        X_train_features = feature_extractor.fit_transform(X_train)
        X_val_features = feature_extractor.transform(X_val)
        X_test_features = feature_extractor.transform(X_test)
        
        print(f"Training {args.model_type} model...")
        model = ModelTrainer(model_type=args.model_type)
        model.train(X_train_features, y_train)
        
        print("Evaluating model on validation set...")
        metrics = model.evaluate(X_val_features, y_val)
        print(f"Validation metrics: {metrics}")
        
        # Save the model and feature extractor
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
        feature_extractor_path = os.path.join(args.output_dir, "feature_extractor.pkl")
        
        model.save_model(model_path)
        joblib.dump(feature_extractor, feature_extractor_path)
        
        print(f"Model saved to {model_path}")
        print(f"Feature extractor saved to {feature_extractor_path}")
        
        # Final evaluation on test set
        print("Evaluating model on test set...")
        test_metrics = model.evaluate(X_test_features, y_test)
        print(f"Test metrics: {test_metrics}")
        
    else:  # LSTM model
        print(f"Training {args.model_type} model...")
        model = ModelTrainer(model_type=args.model_type)
        model.train(X_train, y_train, X_val, y_val)
        
        print("Evaluating model on validation set...")
        metrics = model.evaluate(X_val, y_val)
        print(f"Validation metrics: {metrics}")
        
        # Save the model
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"{args.model_type}_model.h5")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Final evaluation on test set
        print("Evaluating model on test set...")
        test_metrics = model.evaluate(X_test, y_test)
        print(f"Test metrics: {test_metrics}")

def evaluate_model(args):
    """
    Evaluate a trained sentiment analysis model.
    
    Args:
        args: Command line arguments
    """
    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    df = preprocess_data(df)
    
    print("Splitting data into train, validation, and test sets...")
    _, _, X_test, _, _, y_test = split_data(df)
    
    if args.model_type != 'lstm':
        # Load the feature extractor
        feature_extractor_path = os.path.join(args.output_dir, "feature_extractor.pkl")
        feature_extractor = joblib.load(feature_extractor_path)
        
        # Transform test data
        X_test_features = feature_extractor.transform(X_test)
        
        # Load the model
        model = ModelTrainer(model_type=args.model_type)
        model_path = args.model_path or os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
        model.load_model(model_path)
        
        # Evaluate the model
        metrics = model.evaluate(X_test_features, y_test)
        print(f"Test metrics: {metrics}")
        
        y_pred = model.predict(X_test_features)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    else:  # LSTM model
        # Load the model
        model = ModelTrainer(model_type=args.model_type)
        model_path = args.model_path or os.path.join(args.output_dir, f"{args.model_type}_model.h5")
        model.load_model(model_path)
        
        # Evaluate the model
        metrics = model.evaluate(X_test, y_test)
        print(f"Test metrics: {metrics}")
        
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

def predict(args):
    """
    Make predictions using a trained sentiment analysis model.
    
    Args:
        args: Command line arguments
    """
    # This would typically take input from a different source
    # For this example, we'll just use a few example texts
    texts = [
        "I absolutely loved this product! It exceeded all my expectations.",
        "The customer service was terrible and the product arrived damaged.",
        "It was okay, nothing special but did the job.",
        "I would not recommend this to anyone. Complete waste of money."
    ]
    
    if args.model_type != 'lstm':
        # Load the feature extractor
        feature_extractor_path = os.path.join(args.output_dir, "feature_extractor.pkl")
        feature_extractor = joblib.load(feature_extractor_path)
        
        # Transform texts
        X_features = feature_extractor.transform(texts)
        
        # Load the model
        model = ModelTrainer(model_type=args.model_type)
        model_path = args.model_path or os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
        model.load_model(model_path)
        
        # Make predictions
        predictions = model.predict(X_features)
        probas = model.predict_proba(X_features)
        
    else:  # LSTM model
        # Load the model
        model = ModelTrainer(model_type=args.model_type)
        model_path = args.model_path or os.path.join(args.output_dir, f"{args.model_type}_model.h5")
        model.load_model(model_path)
        
        # Make predictions
        predictions = model.predict(texts)
        probas = model.predict_proba(texts)
    
    # Print predictions
    for i, text in enumerate(texts):
        sentiment = "Positive" if predictions[i] == 1 else "Negative"
        if args.model_type != 'lstm':
            confidence = probas[i][predictions[i]]
        else:
            confidence = probas[i][0] if predictions[i] == 0 else 1 - probas[i][0]
            
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print()

def main():
    """
    Main function.
    """
    args = parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'predict':
        predict(args)

if __name__ == '__main__':
    main()