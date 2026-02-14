"""
Machine Learning Assignment 2 - Streamlit Web Application
Interactive ML Model Evaluation Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}

# Model names mapping
MODEL_NAMES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}


def load_model(model_path):
    """Load a pickled model"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def load_all_models():
    """Load all pre-trained models"""
    models = {}
    model_dir = 'model'
    
    for model_name, filename in MODEL_NAMES.items():
        model_path = os.path.join(model_dir, filename)
        if os.path.exists(model_path):
            models[model_name] = load_model(model_path)
        else:
            st.warning(f"Model file not found: {model_path}")
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        models['scaler'] = load_model(scaler_path)
    
    # Load label encoder if exists
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    if os.path.exists(label_encoder_path):
        models['label_encoder'] = load_model(label_encoder_path)
    
    return models


def preprocess_data(df, scaler, target_column):
    """Preprocess uploaded data"""
    # Separate features and target
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        st.error(f"Target column '{target_column}' not found in dataset!")
        return None, None
    
    # Handle categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Encode target if categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    # AUC Score
    if y_pred_proba is not None:
        if len(np.unique(y_true)) == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    else:
        metrics['AUC'] = None
    
    # Precision, Recall, F1
    avg_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
    metrics['Precision'] = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
    
    # MCC
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 ML Classification Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Control Panel")
    st.sidebar.markdown("---")
    
    # Load models button
    if st.sidebar.button("🔄 Load Pre-trained Models", use_container_width=True):
        with st.spinner("Loading models..."):
            st.session_state.models = load_all_models()
            if st.session_state.models:
                st.session_state.models_loaded = True
                st.sidebar.success(f"✅ {len([k for k in st.session_state.models.keys() if k not in ['scaler', 'label_encoder']])} models loaded!")
    
    # File upload section
    st.sidebar.markdown("### 📁 Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your test dataset (CSV format only)"
    )
    
    # Target column input
    target_column = st.sidebar.text_input(
        "Target Column Name",
        value="target",
        help="Enter the name of your target/label column"
    )
    
    # Main content area
    if not st.session_state.models_loaded:
        st.info("👈 Please load the pre-trained models from the sidebar to get started!")
        
        # Display information about the app
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Features")
            st.markdown("""
            - **6 ML Classification Models**
                - Logistic Regression
                - Decision Tree
                - K-Nearest Neighbors
                - Naive Bayes
                - Random Forest
                - XGBoost
            """)
        
        with col2:
            st.markdown("### 📈 Evaluation Metrics")
            st.markdown("""
            - Accuracy
            - AUC Score
            - Precision
            - Recall
            - F1 Score
            - Matthews Correlation Coefficient (MCC)
            """)
    
    else:
        if uploaded_file is not None:
            # Read uploaded file
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
                
                # Display dataset info
                with st.expander("📊 Dataset Preview", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", df.shape[0])
                    col2.metric("Columns", df.shape[1])
                    col3.metric("Features", df.shape[1] - 1)
                
                # Model selection
                st.markdown("### 🎯 Select Model for Prediction")
                
                available_models = [name for name in MODEL_NAMES.keys() 
                                  if name in st.session_state.models]
                
                selected_model = st.selectbox(
                    "Choose a model:",
                    options=available_models,
                    help="Select which model to use for prediction"
                )
                
                if st.button("🚀 Run Evaluation", use_container_width=True, type="primary"):
                    
                    with st.spinner(f"Running {selected_model}..."):
                        
                        # Preprocess data
                        X_test, y_test = preprocess_data(
                            df, 
                            st.session_state.models['scaler'],
                            target_column
                        )
                        
                        if X_test is not None and y_test is not None:
                            
                            # Get model
                            model = st.session_state.models[selected_model]
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Get probabilities if available
                            y_pred_proba = None
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba = model.predict_proba(X_test)
                            
                            # Calculate metrics
                            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown(f"### 📊 Evaluation Results - {selected_model}")
                            
                            # Metrics in columns
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
                                st.metric("🎲 Precision", f"{metrics['Precision']:.4f}")
                            
                            with col2:
                                auc_value = f"{metrics['AUC']:.4f}" if metrics['AUC'] else "N/A"
                                st.metric("📈 AUC Score", auc_value)
                                st.metric("🔍 Recall", f"{metrics['Recall']:.4f}")
                            
                            with col3:
                                st.metric("⚖️ F1 Score", f"{metrics['F1']:.4f}")
                                st.metric("🧮 MCC Score", f"{metrics['MCC']:.4f}")
                            
                            st.markdown("---")
                            
                            # Confusion Matrix and Classification Report
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("#### 🔲 Confusion Matrix")
                                cm = confusion_matrix(y_test, y_pred)
                                fig = plot_confusion_matrix(cm, f"{selected_model} - Confusion Matrix")
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                st.markdown("#### 📋 Classification Report")
                                report = classification_report(y_test, y_pred)
                                st.text(report)
                            
                            # Detailed metrics table
                            st.markdown("---")
                            st.markdown("#### 📊 Detailed Metrics Summary")
                            
                            metrics_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC Score'],
                                'Value': [
                                    f"{metrics['Accuracy']:.4f}",
                                    f"{metrics['AUC']:.4f}" if metrics['AUC'] else "N/A",
                                    f"{metrics['Precision']:.4f}",
                                    f"{metrics['Recall']:.4f}",
                                    f"{metrics['F1']:.4f}",
                                    f"{metrics['MCC']:.4f}"
                                ]
                            })
                            
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                            
                            st.success("✅ Evaluation completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            st.info("👈 Please upload a CSV file from the sidebar to begin evaluation!")
            
            # Show model status
            st.markdown("### ✅ Loaded Models")
            
            loaded_models = [name for name in MODEL_NAMES.keys() 
                           if name in st.session_state.models]
            
            for i, model_name in enumerate(loaded_models, 1):
                st.markdown(f"**{i}.** {model_name} ✓")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "ML Assignment 2 - Classification Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
