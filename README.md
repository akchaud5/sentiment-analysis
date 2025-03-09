# Sentiment Analysis ML Project

A complete machine learning project for sentiment analysis using large-scale datasets and classical ML algorithms.

## Overview

This project implements a sentiment analysis pipeline for classifying text as positive or negative. The system:
- Downloads and processes large sentiment analysis datasets (IMDB, Twitter, Stanford Sentiment Treebank)
- Preprocesses text data with NLTK (tokenization, stopword removal, lemmatization)
- Extracts features using TF-IDF vectorization with n-gram features
- Trains a Random Forest classifier for sentiment prediction
- Achieves approximately 74% accuracy on test data
- Makes predictions on new text with confidence scores

## Project Structure

```
├── data
│   ├── raw                 # Raw data directories for different datasets
│   │   ├── imdb            # IMDB movie reviews
│   │   ├── twitter         # Twitter sentiment dataset
│   │   ├── sst             # Stanford Sentiment Treebank
│   │   └── yelp            # Yelp reviews (optional)
│   └── processed           # Processed and combined datasets
│       ├── combined_sentiment.csv  # Balanced dataset for training
│       ├── imdb_reviews.csv        # Processed IMDB reviews
│       └── twitter_sentiment.csv   # Processed Twitter data
├── models                  # Saved trained models
│   ├── feature_extractor.pkl  # TF-IDF vectorizer
│   └── random_forest_model.pkl # Trained classifier
├── notebooks               # Jupyter notebooks for exploration
├── src                     # Source code
│   ├── data_loader.py      # Data loading utilities
│   ├── data_processing.py  # Data processing utilities
│   ├── download_datasets.py # Script to download large datasets
│   ├── feature_engineering.py # Text feature extraction
│   ├── models.py           # Model definitions
│   └── main.py             # Command-line interface
├── tests                   # Unit tests
├── train.py                # Script to train models
├── predict.py              # Script to make predictions
└── requirements.txt        # Project dependencies
```

## Quick Start

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Download and process datasets (will take some time for large datasets):
```bash
python src/download_datasets.py
```

3. Train the model:
```bash
python train.py
```

4. Make predictions with the trained model:
```bash
python predict.py
```

## Features

- Comprehensive text preprocessing with NLTK:
  - Tokenization using TreebankWordTokenizer
  - Stopword removal
  - Lemmatization to reduce word forms
- Combined training on multiple large-scale datasets (over 100,000 samples)
- TF-IDF vectorization with up to 10,000 features and bigram support
- Random Forest classifier with 100 trees
- Model performance:
  - Accuracy: 74.4%
  - Balanced precision and recall (~0.74-0.75)
  - Well-calibrated confidence scores
- Interactive and batch prediction modes

## Making Custom Predictions

### Basic Prediction

To use the system for simple sentiment predictions:

1. Run predict.py interactively:
```bash
python predict.py
```

2. Run batch predictions on a CSV file:
```bash
python predict.py /path/to/your/file.csv
```
Note: The CSV file must have a column named 'text'

3. Or import and use the prediction function in your own code:
```python
from predict import predict_sentiment, load_model_and_vectorizer

model, vectorizer = load_model_and_vectorizer()
sentiment, confidence = predict_sentiment("Your text here", model, vectorizer)
```

### Advanced Prediction with Explanations

For advanced predictions with detailed explanations of why a text was classified as positive or negative:

1. Run the improved prediction script:
```bash
python predict_improved.py
```

2. Run batch predictions with explanations:
```bash
python predict_improved.py /path/to/your/file.csv
```
This creates both a CSV file with predictions and a visualization of confidence distribution.

3. Import in your code for advanced predictions:
```python
from predict_improved import predict_sentiment, load_model_and_vectorizer

model, vectorizer = load_model_and_vectorizer()
sentiment, confidence, explanation = predict_sentiment("Your text here", model, vectorizer)
```

The advanced prediction system provides:
- Detailed explanations of which words contributed most to the prediction
- Identification of positive and negative sentiment indicators
- Confidence calibration and visualization
- Assessment of sentiment strength (strong, moderate, weak)

## Datasets

This project uses several large-scale sentiment analysis datasets:

1. **IMDB Movie Reviews** - 50,000 movie reviews labeled as positive or negative
   - Size: ~80MB uncompressed
   - Used for: Training and evaluation
   - Source: https://ai.stanford.edu/~amaas/data/sentiment/

2. **Twitter Sentiment140** - 1.6 million tweets labeled for sentiment
   - Size: ~135MB processed
   - Used for: Increasing training diversity
   - Source: https://www.kaggle.com/datasets/kazanova/sentiment140

3. **Stanford Sentiment Treebank** - Detailed sentiment labels for sentences
   - Size: <1MB
   - Used for: Fine-grained sentiment examples
   - Source: https://nlp.stanford.edu/sentiment/

4. **Yelp Reviews** (Optional) - Must be downloaded manually from Yelp
   - Source: https://www.yelp.com/dataset

## Model Performance

We provide two different model implementations with varying levels of performance:

### Standard Model (Random Forest)

The basic Random Forest model achieves the following performance metrics:

```
Accuracy: 0.7437
Confusion Matrix:
[[7358 2648]
 [2469 7488]]
Classification Report:
              precision    recall  f1-score   support
    negative       0.75      0.74      0.74     10006
    positive       0.74      0.75      0.75      9957
    accuracy                           0.74     19963
   macro avg       0.74      0.74      0.74     19963
weighted avg       0.74      0.74      0.74     19963
```

### Advanced Model (Ensemble with Optimizations)

The improved ensemble model combines Random Forest, Gradient Boosting, and Logistic Regression with advanced feature engineering to achieve better performance:

```
Accuracy: 0.7588
AUC Score: 0.8384
Confusion Matrix:
[[7279 2700]
 [2116 7868]]
Classification Report:
              precision    recall  f1-score   support
    negative       0.77      0.73      0.75      9979
    positive       0.74      0.79      0.77      9984
    accuracy                           0.76     19963
   macro avg       0.76      0.76      0.76     19963
weighted avg       0.76      0.76      0.76     19963
```

The ensemble model provides a significant improvement (+1.5% accuracy) and better interpretability with prediction explanations.

## Advanced Usage

### Running Tests

```bash
python -m unittest discover -s tests
```

### Exploring with Jupyter Notebook

To understand the model development process:

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

## Improvements and Advanced Features

We've implemented several improvements to boost model performance and explainability:

1. **Enhanced Preprocessing**
   - Custom stopword removal with domain-specific lists
   - Emphasis of sentiment keywords for stronger signals
   - Improved tokenization with TreebankWordTokenizer

2. **Advanced Feature Engineering**
   - N-gram features up to trigrams (1-3 word phrases)
   - Text statistics features (length, word count)
   - Sentiment indicator counts

3. **Model Improvements**
   - Ensemble learning combining multiple algorithms
   - Advanced hyperparameter optimization
   - Better balanced class weighting

4. **Explainability**
   - Word contribution analysis for predictions
   - Sentiment strength assessment
   - Visual confidence distribution

## Future Directions

Additional improvements to explore:

- Deep learning models (LSTM, BERT, transformers)
- Transfer learning from pre-trained language models
- Web API for serving predictions
- Multi-class sentiment classification (positive, neutral, negative)
- Aspect-based sentiment analysis
- Domain adaptation for specific industries

## License

This project is licensed under the MIT License

## Acknowledgments

- Dataset creators for providing valuable public datasets
- NLTK for natural language processing capabilities
- Scikit-learn for machine learning implementation
- Project structure follows ML engineering best practices