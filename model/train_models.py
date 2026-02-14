# Machine Learning Classification Models Training Notebook

"""
This notebook implements 6 classification models and evaluates them with comprehensive metrics.
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MLClassificationPipeline:
    """
    Complete ML Classification Pipeline with 6 models
    """
    
    def __init__(self, dataset_path, target_column, test_size=0.2, random_state=42):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        dataset_path : str
            Path to the CSV dataset
        target_column : str
            Name of the target column to predict
        test_size : float
            Proportion of dataset to include in test split
        random_state : int
            Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.scaler = None
        self.is_multiclass = False
        self.results = {}
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Display basic info
        print(f"\nDataset Info:")
        print(f"- Total rows: {self.df.shape[0]}")
        print(f"- Total columns: {self.df.shape[1]}")
        print(f"- Missing values: {self.df.isnull().sum().sum()}")
        print(f"- Duplicate rows: {self.df.duplicated().sum()}")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Check if multiclass
        self.is_multiclass = len(np.unique(y_encoded)) > 2
        print(f"Classification type: {'Multi-class' if self.is_multiclass else 'Binary'}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {len(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Filling missing values...")
            X = X.fillna(X.mean())
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        
        metrics = {}
        
        # Accuracy
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        
        # AUC Score
        try:
            if self.is_multiclass:
                metrics['AUC'] = roc_auc_score(
                    y_true, y_pred_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            print(f"Warning: Could not calculate AUC - {e}")
            metrics['AUC'] = 0.0
        
        # Precision, Recall, F1
        metrics['Precision'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['Recall'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['F1'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Matthews Correlation Coefficient
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        return metrics
    
    def train_model(self, model_name, model):
        """Train a single model and calculate metrics"""
        
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred)
        }
        
        # Print metrics
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        
        return metrics
    
    def train_all_models(self):
        """Train all 6 required models"""
        
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'XGBoost': XGBClassifier(
                eval_metric='logloss', 
                random_state=self.random_state, 
                use_label_encoder=False
            )
        }
        
        for model_name, model in models.items():
            self.train_model(model_name, model)
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*60)
    
    def display_results(self):
        """Display comprehensive results"""
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Create comparison table
        metrics_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1': f"{metrics['F1']:.4f}",
                'MCC': f"{metrics['MCC']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        print("\n", metrics_df.to_string(index=False))
        
        # Find best models
        print("\n" + "="*80)
        print("BEST PERFORMING MODELS")
        print("="*80)
        
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['metrics']['Accuracy'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['metrics']['F1'])
        best_mcc = max(self.results.items(), key=lambda x: x[1]['metrics']['MCC'])
        
        print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['metrics']['Accuracy']:.4f})")
        print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['metrics']['F1']:.4f})")
        print(f"Best MCC Score: {best_mcc[0]} ({best_mcc[1]['metrics']['MCC']:.4f})")
        
        return metrics_df
    
    def plot_comparison(self):
        """Plot comparative visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            model_names = list(self.results.keys())
            values = [self.results[m]['metrics'][metric_name] for m in model_names]
            
            bars = ax.bar(range(len(model_names)), values, color='steelblue', alpha=0.7)
            
            # Highlight best model
            best_idx = values.index(max(values))
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(1.0)
            
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title(f'{metric_name} Comparison', fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved as 'model_comparison.png'")
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved as 'confusion_matrices.png'")
        plt.show()
    
    def save_results(self, output_file='model_results.csv'):
        """Save results to CSV"""
        
        metrics_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'AUC': metrics['AUC'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1': metrics['F1'],
                'MCC': metrics['MCC']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")
        
        return metrics_df


# Example usage
if __name__ == "__main__":
    
    # Instructions for use
    print("""
    ================================================================
    ML Classification Pipeline
    ================================================================
    
    To use this script:
    
    1. Replace 'your_dataset.csv' with your actual dataset path
    2. Replace 'target_column_name' with your actual target column name
    3. Run the script
    
    Example:
        pipeline = MLClassificationPipeline(
            dataset_path='data.csv',
            target_column='class',
            test_size=0.2,
            random_state=42
        )
        
        pipeline.load_data()
        pipeline.preprocess_data()
        pipeline.train_all_models()
        pipeline.display_results()
        pipeline.plot_comparison()
        pipeline.plot_confusion_matrices()
        pipeline.save_results()
    
    ================================================================
    """)
    
    # Uncomment and modify the following to run:
    # pipeline = MLClassificationPipeline(
    #     dataset_path='your_dataset.csv',
    #     target_column='your_target_column',
    #     test_size=0.2,
    #     random_state=42
    # )
    # 
    # pipeline.load_data()
    # pipeline.preprocess_data()
    # pipeline.train_all_models()
    # pipeline.display_results()
    # pipeline.plot_comparison()
    # pipeline.plot_confusion_matrices()
    # pipeline.save_results()
