#!/usr/bin/env python
"""
Improved training script for sentiment analysis model with advanced features and optimization
"""
import os
import pickle
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK components
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Add sentiment-specific stopwords to remove (these words don't add much value)
sentiment_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'of', 'in', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}

# Add sentiment-specific keywords with higher weight
sentiment_keywords = {
    'positive': ['excellent', 'amazing', 'wonderful', 'great', 'good', 'best', 'fantastic', 'perfect', 'love', 'awesome', 'outstanding', 'exceptional', 'superb', 'brilliant', 'happy', 'delighted', 'satisfied', 'impressed', 'recommend', 'worth'],
    'negative': ['terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'disappointing', 'disappointment', 'hate', 'dislike', 'useless', 'waste', 'broken', 'defective', 'failure', 'problem', 'issue', 'complaint', 'refund', 'return', 'never', 'avoid']
}

# Combine standard stopwords with sentiment-specific ones
extended_stop_words = stop_words.union(sentiment_stopwords)

def preprocess_text(text):
    """Clean and preprocess text data with advanced features"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize using TreebankWordTokenizer
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in extended_stop_words]
    
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

def load_and_prepare_data(use_extracted_features=True):
    """Load the combined dataset and prepare for training with advanced features"""
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
    
    # Convert sentiment to binary if needed
    if df['sentiment'].dtype == object:
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    y = df['sentiment']
    
    # Add additional features if requested
    if use_extracted_features:
        print("Extracting additional text features...")
        additional_features = df['text'].apply(extract_text_features)
        additional_features_df = pd.DataFrame(additional_features.tolist(), 
                                            columns=['word_count', 'char_count', 'avg_word_length', 
                                                    'positive_count', 'negative_count'])
        
        # Visualize additional features
        plt.figure(figsize=(12, 8))
        for i, column in enumerate(additional_features_df.columns, 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=additional_features_df, x=column, hue=df['sentiment'], kde=True)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.savefig('models/feature_distributions.png')
        
        # Save additional features for later use with the testing set
        additional_features_df.to_csv('data/processed/additional_features.csv', index=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def create_advanced_pipeline():
    """Create an advanced pipeline with TF-IDF and classifiers"""
    # Create a pipeline with TF-IDF vectorizer and classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,       # Increased from 10000
            ngram_range=(1, 3),       # Include up to trigrams
            min_df=3,                 # Minimum document frequency
            max_df=0.9,               # Maximum document frequency
            use_idf=True,             # Use inverse document frequency
            sublinear_tf=True,        # Apply sublinear scaling to term frequencies
            binary=False,             # Use term frequency instead of binary
            norm='l2',                # Use L2 normalization
            analyzer='word',          # Analyze words
            token_pattern=r'\w{1,}',  # Allow single character words
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=100,      # Default
            random_state=42
        ))
    ])
    
    return pipeline

def optimize_hyperparameters(X_train, y_train):
    """Optimize hyperparameters using grid search and cross-validation"""
    print("Optimizing hyperparameters...")
    start_time = time.time()
    
    # Create the pipeline
    pipeline = create_advanced_pipeline()
    
    # Define the parameter grid
    param_grid = {
        'tfidf__max_features': [10000, 15000],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 20, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Use GridSearchCV with 3-fold cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Print the results
    print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
    print(f"Best cross-validation F1 score: {best_score:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Build the best pipeline
    best_pipeline = grid_search.best_estimator_
    
    return best_pipeline

def build_ensemble_model(X_train, y_train):
    """Build an ensemble model using multiple classifiers"""
    print("Building ensemble model...")
    
    # Define individual classifiers
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        use_idf=True,
        sublinear_tf=True
    )
    
    # Transform the training data
    X_train_features = vectorizer.fit_transform(X_train)
    
    # Train each classifier
    print("Training Random Forest...")
    rf.fit(X_train_features, y_train)
    
    print("Training Gradient Boosting...")
    gb.fit(X_train_features, y_train)
    
    print("Training Logistic Regression...")
    lr.fit(X_train_features, y_train)
    
    # Create and train voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft'
    )
    
    # Train the ensemble
    print("Training ensemble model...")
    ensemble.fit(X_train_features, y_train)
    
    return ensemble, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate model performance with comprehensive metrics"""
    print("Evaluating model...")
    
    # Transform test data
    X_test_features = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_features)
    y_prob = model.predict_proba(X_test_features)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    
    # Create ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('models/roc_curve.png')
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'auc_score': auc_score
    }
    
    return metrics

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(use_extracted_features=True)
    
    # Choose one of the following approaches:
    # 1. Ensemble model (recommended)
    print("\nBuilding ensemble model...")
    model, vectorizer = build_ensemble_model(X_train, y_train)
    
    # 2. Hyperparameter optimization (uncomment to use)
    # print("\nOptimizing hyperparameters...")
    # pipeline = optimize_hyperparameters(X_train, y_train)
    # model = pipeline.named_steps['classifier']
    # vectorizer = pipeline.named_steps['tfidf']
    
    # Evaluate model
    metrics = evaluate_model(model, vectorizer, X_test, y_test)
    
    # Save the model and vectorizer
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'models/ensemble_model.pkl')
    with open('models/advanced_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metrics
    with open('models/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nModel training and evaluation completed!")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC Score: {metrics['auc_score']:.4f}")
    print("\nModel and vectorizer saved to models/ directory")
    print("Visualizations saved to models/ directory")

if __name__ == "__main__":
    main()