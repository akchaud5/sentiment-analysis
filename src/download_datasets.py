#!/usr/bin/env python
"""
Script to download and prepare sentiment analysis datasets
"""
import os
import sys
import pandas as pd
import numpy as np
import nltk
from urllib.request import urlretrieve
import zipfile
import gzip
import tarfile
import json
import kaggle
from tqdm import tqdm

# Create directories
os.makedirs('data/raw/imdb', exist_ok=True)
os.makedirs('data/raw/amazon', exist_ok=True)
os.makedirs('data/raw/twitter', exist_ok=True)
os.makedirs('data/raw/yelp', exist_ok=True)
os.makedirs('data/raw/sst', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def download_progress(block_num, block_size, total_size):
    """Display download progress"""
    progress = block_num * block_size
    if total_size > 0:
        percent = progress * 100 / total_size
        sys.stdout.write(f"\rDownloading: {percent:.2f}% ({progress} / {total_size})")
        sys.stdout.flush()

def download_imdb():
    """Download and extract IMDB dataset"""
    print("\nDownloading IMDB dataset...")
    imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = "data/raw/imdb/aclImdb_v1.tar.gz"
    
    if not os.path.exists(tar_path):
        urlretrieve(imdb_url, tar_path, download_progress)
        
    if not os.path.exists("data/raw/imdb/aclImdb"):
        print("\nExtracting IMDB dataset...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path="data/raw/imdb")
    
    print("IMDB dataset ready.")

def download_sentiment140():
    """Download Sentiment140 dataset using Kaggle API"""
    print("\nDownloading Sentiment140 dataset...")
    
    if not os.path.exists("data/raw/twitter/training.1600000.processed.noemoticon.csv"):
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('kazanova/sentiment140', 
                                              path='data/raw/twitter', 
                                              unzip=True)
        except Exception as e:
            print(f"Error with Kaggle API: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/kazanova/sentiment140")
            print("And place the files in data/raw/twitter/")
    
    print("Sentiment140 dataset ready.")

def download_yelp():
    """Provide instructions for Yelp dataset"""
    print("\nYelp dataset:")
    print("Due to size and terms, please download manually from: https://www.yelp.com/dataset")
    print("Extract and place the review.json file in data/raw/yelp/")

def download_sst():
    """Download Stanford Sentiment Treebank"""
    print("\nDownloading Stanford Sentiment Treebank...")
    sst_url = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
    zip_path = "data/raw/sst/sst.zip"
    
    if not os.path.exists(zip_path):
        urlretrieve(sst_url, zip_path, download_progress)
        
    if not os.path.exists("data/raw/sst/trees"):
        print("\nExtracting SST dataset...")
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall("data/raw/sst")
    
    print("Stanford Sentiment Treebank ready.")

def process_imdb():
    """Process IMDB dataset into a single CSV file"""
    print("\nProcessing IMDB dataset...")
    
    output_file = "data/processed/imdb_reviews.csv"
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping processing.")
        return
    
    reviews = []
    
    # Process positive reviews
    pos_dir = "data/raw/imdb/aclImdb/train/pos"
    if os.path.exists(pos_dir):
        for filename in tqdm(os.listdir(pos_dir), desc="Processing positive reviews"):
            if filename.endswith(".txt"):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    reviews.append({
                        'text': f.read().strip(),
                        'sentiment': 'positive'
                    })
    
    # Process negative reviews
    neg_dir = "data/raw/imdb/aclImdb/train/neg"
    if os.path.exists(neg_dir):
        for filename in tqdm(os.listdir(neg_dir), desc="Processing negative reviews"):
            if filename.endswith(".txt"):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    reviews.append({
                        'text': f.read().strip(),
                        'sentiment': 'negative'
                    })
    
    # Save as CSV
    df = pd.DataFrame(reviews)
    df.to_csv(output_file, index=False)
    print(f"Processed IMDB dataset saved to {output_file}")

def process_sentiment140():
    """Process Sentiment140 dataset"""
    print("\nProcessing Sentiment140 dataset...")
    
    output_file = "data/processed/twitter_sentiment.csv"
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping processing.")
        return
    
    input_file = "data/raw/twitter/training.1600000.processed.noemoticon.csv"
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Skipping processing.")
        return
    
    # The file has no header, so we add one
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(input_file, encoding='latin-1', names=columns)
    
    # Map sentiment: 0 = negative, 4 = positive
    sentiment_map = {0: 'negative', 4: 'positive'}
    df['sentiment'] = df['target'].map(sentiment_map)
    
    # Select relevant columns
    result_df = df[['text', 'sentiment']]
    
    # Save as CSV
    result_df.to_csv(output_file, index=False)
    print(f"Processed Sentiment140 dataset saved to {output_file}")

def process_combined_dataset(sample_size=100000, small_sample_size=5000):
    """Create a combined dataset from all sources"""
    print("\nCreating combined dataset...")
    
    output_file = "data/processed/combined_sentiment.csv"
    small_output_file = "data/processed/combined_sentiment_small.csv"
    
    if os.path.exists(output_file) and os.path.exists(small_output_file):
        print(f"Files {output_file} and {small_output_file} already exist. Skipping processing.")
        return
    
    datasets = []
    
    # Load IMDB
    imdb_file = "data/processed/imdb_reviews.csv"
    if os.path.exists(imdb_file):
        imdb_df = pd.read_csv(imdb_file)
        imdb_df['source'] = 'imdb'
        datasets.append(imdb_df)
    
    # Load Twitter
    twitter_file = "data/processed/twitter_sentiment.csv"
    if os.path.exists(twitter_file):
        twitter_df = pd.read_csv(twitter_file)
        twitter_df['source'] = 'twitter'
        datasets.append(twitter_df)
    
    # Combine datasets
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Balance the full dataset
        positive = combined_df[combined_df['sentiment'] == 'positive'].sample(n=min(sample_size//2, combined_df[combined_df['sentiment'] == 'positive'].shape[0]), random_state=42)
        negative = combined_df[combined_df['sentiment'] == 'negative'].sample(n=min(sample_size//2, combined_df[combined_df['sentiment'] == 'negative'].shape[0]), random_state=42)
        
        balanced_df = pd.concat([positive, negative], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        # Save full combined dataset
        balanced_df.to_csv(output_file, index=False)
        print(f"Combined dataset created with {balanced_df.shape[0]} samples, saved to {output_file}")
        
        # Create smaller sample for GitHub (under 100MB)
        small_positive = positive.sample(n=min(small_sample_size//2, positive.shape[0]), random_state=42)
        small_negative = negative.sample(n=min(small_sample_size//2, negative.shape[0]), random_state=42)
        small_balanced_df = pd.concat([small_positive, small_negative], ignore_index=True)
        small_balanced_df = small_balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        # Save small combined dataset
        small_balanced_df.to_csv(small_output_file, index=False)
        print(f"Small combined dataset created with {small_balanced_df.shape[0]} samples, saved to {small_output_file}")
    else:
        print("No processed datasets found to combine.")

def main():
    # Download datasets
    download_imdb()
    download_sentiment140()
    download_yelp()  # This just provides instructions
    download_sst()
    
    # Process datasets
    process_imdb()
    process_sentiment140()
    
    # Create combined dataset
    process_combined_dataset()
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()