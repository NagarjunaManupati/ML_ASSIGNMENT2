"""
Machine Learning Assignment 2 - Model Training
Implements 6 classification models with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

class MLClassificationPipeline:
    """
    Complete ML Pipeline for training and evaluating 6 classification models
    """
    
    def __init__(self, dataset_path):
        """Initialize with dataset path"""
        self.dataset_path = dataset_path
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        
    def load_and_preprocess_data(self, target_column):
        """
        Load dataset and perform preprocessing
        
        Args:
            target_column: Name of the target column
        """
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Instances: {df.shape[0]}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical features in X
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Encode target variable if categorical
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self
    
    def train_all_models(self):
        """Train all 6 classification models"""
        
        print("\n" + "="*50)
        print("Training Classification Models")
        print("="*50)
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # 2. Decision Tree
        print("2. Training Decision Tree...")
        dt = DecisionTreeClassifier(random_state=42, max_depth=10)
        dt.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt
        
        # 3. K-Nearest Neighbors
        print("3. Training K-Nearest Neighbors...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        self.models['kNN'] = knn
        
        # 4. Naive Bayes
        print("4. Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = nb
        
        # 5. Random Forest
        print("5. Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # 6. XGBoost
        print("6. Training XGBoost...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, 
                           eval_metric='logloss')
        xgb.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb
        
        print("\nAll models trained successfully!")
        return self
    
    def evaluate_model(self, model_name, model):
        """
        Evaluate a single model with all required metrics
        
        Args:
            model_name: Name of the model
            model: Trained model object
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # For AUC, we need probability predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(self.X_test)
            # Handle binary and multi-class cases
            if len(np.unique(self.y_test)) == 2:
                auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
        else:
            auc = None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # For precision, recall, F1 - handle multi-class
        avg_method = 'binary' if len(np.unique(self.y_test)) == 2 else 'weighted'
        precision = precision_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        
        # MCC
        mcc = matthews_corrcoef(self.y_test, y_pred)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification Report
        report = classification_report(self.y_test, y_pred)
        
        return {
            'Accuracy': accuracy,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc,
            'Confusion Matrix': cm,
            'Classification Report': report
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        
        print("\n" + "="*50)
        print("Evaluating All Models")
        print("="*50)
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            self.results[model_name] = self.evaluate_model(model_name, model)
        
        return self
    
    def print_results_table(self):
        """Print results in a formatted table"""
        
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Create results dataframe
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}" if metrics['AUC'] else "N/A",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1': f"{metrics['F1']:.4f}",
                'MCC': f"{metrics['MCC']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def save_models(self, output_dir='model'):
        """Save all trained models"""
        
        print(f"\nSaving models to {output_dir}/...")
        
        # Save individual models
        for model_name, model in self.models.items():
            filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler and label encoder
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        if self.label_encoder:
            with open(f"{output_dir}/label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        # Save results
        results_for_save = {}
        for model_name, metrics in self.results.items():
            results_for_save[model_name] = {
                'Accuracy': float(metrics['Accuracy']),
                'AUC': float(metrics['AUC']) if metrics['AUC'] else None,
                'Precision': float(metrics['Precision']),
                'Recall': float(metrics['Recall']),
                'F1': float(metrics['F1']),
                'MCC': float(metrics['MCC']),
            }
        
        with open(f"{output_dir}/results.pkl", 'wb') as f:
            pickle.dump(results_for_save, f)
        
        print("All models and results saved successfully!")
        
        return self


def main():
    """
    Main execution function
    
    Instructions:
    1. Replace 'your_dataset.csv' with your actual dataset path
    2. Replace 'target' with your actual target column name
    3. Run this script to train all models
    """
    
    # Configuration - MODIFY THESE
    DATASET_PATH = 'data/heart.csv'  # Change to your dataset
    TARGET_COLUMN = 'target'  # Change to your target column name
    
    # Create pipeline
    pipeline = MLClassificationPipeline(DATASET_PATH)
    
    # Execute pipeline
    pipeline.load_and_preprocess_data(TARGET_COLUMN)
    pipeline.train_all_models()
    pipeline.evaluate_all_models()
    results_df = pipeline.print_results_table()
    pipeline.save_models()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return pipeline


if __name__ == "__main__":
    main()
