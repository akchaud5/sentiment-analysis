import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """
    def __init__(self, model_type='logistic'):
        """
        Initialize the model trainer.
        
        Args:
            model_type (str): Type of model to train ('logistic', 'rf', 'svm')
        """
        self.model_type = model_type
        self.model = None
        
        if model_type == 'logistic':
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(C=1.0, kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for traditional ML models)
            y_val: Validation labels (not used for traditional ML models)
            **kwargs: Additional model-specific parameters
            
        Returns:
            self: The trained model
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary of metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = joblib.load(filepath)