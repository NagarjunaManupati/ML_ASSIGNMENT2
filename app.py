import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🤖 Machine Learning Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown("#### Build, Train, and Compare 6 Classification Models on Any Dataset")
st.markdown("---")

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Sidebar
st.sidebar.title("⚙️ Configuration Panel")
st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload CSV Dataset", 
    type=['csv'],
    help="Upload a CSV file with at least 12 features and 500 instances"
)

def preprocess_data(df, target_column):
    """Preprocess the dataset for ML models"""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target variable if categorical
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_encoded, le_target

def calculate_metrics(y_true, y_pred, y_pred_proba, is_multiclass):
    """Calculate all required metrics"""
    
    metrics = {}
    
    # Accuracy
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    # AUC Score
    try:
        if is_multiclass:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    except:
        metrics['AUC'] = 0.0
    
    # Precision, Recall, F1
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # MCC
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics

def train_all_models(X_train, X_test, y_train, y_test, is_multiclass):
    """Train all 6 required ML models"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    }
    
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, is_multiclass)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        progress_bar.progress((idx + 1) / len(models))
    
    status_text.text("✅ All models trained successfully!")
    progress_bar.empty()
    status_text.empty()
    
    return results

# Main App Logic
if uploaded_file is not None:
    
    # Load dataset
    try:
        df = pd.read_csv(uploaded_file)
        
        st.sidebar.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Dataset validation
        if df.shape[0] < 500:
            st.sidebar.warning(f"⚠️ Dataset has {df.shape[0]} rows. Minimum required: 500")
        if df.shape[1] < 12:
            st.sidebar.warning(f"⚠️ Dataset has {df.shape[1]} columns. Minimum required: 12")
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    # Display dataset preview
    with st.expander("📊 Dataset Preview & Statistics", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**First 10 Rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Dataset Info:**")
            st.metric("Total Rows", df.shape[0])
            st.metric("Total Columns", df.shape[1])
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Target column selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Target Selection")
    
    target_column = st.sidebar.selectbox(
        "Select Target Column to Predict",
        options=df.columns.tolist(),
        help="Choose the column you want to predict"
    )
    
    if target_column:
        
        # Display target distribution
        with st.expander("🎯 Target Variable Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Target Distribution:**")
                target_counts = df[target_column].value_counts()
                st.dataframe(target_counts.reset_index().rename(
                    columns={'index': 'Class', target_column: 'Count'}
                ))
                
                n_classes = df[target_column].nunique()
                st.info(f"**Number of Classes:** {n_classes}")
                
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                target_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f'Distribution of {target_column}')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()
        
        # Model training parameters
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Training Parameters")
        
        test_size = st.sidebar.slider(
            "Test Size (%)", 
            min_value=10, 
            max_value=40, 
            value=20,
            help="Percentage of data to use for testing"
        ) / 100
        
        random_state = st.sidebar.number_input(
            "Random State",
            min_value=0,
            max_value=100,
            value=42,
            help="Seed for reproducibility"
        )
        
        # Train button
        st.sidebar.markdown("---")
        train_button = st.sidebar.button("🚀 Train All Models", type="primary", use_container_width=True)
        
        if train_button:
            
            with st.spinner("🔄 Preprocessing data and training models..."):
                
                try:
                    # Preprocess data
                    X, y, label_encoder = preprocess_data(df, target_column)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Check if multiclass
                    is_multiclass = len(np.unique(y)) > 2
                    
                    st.info(f"📊 Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
                    
                    # Train all models
                    results = train_all_models(X_train, X_test, y_train, y_test, is_multiclass)
                    
                    # Store in session state
                    st.session_state.results = results
                    st.session_state.models_trained = True
                    st.session_state.y_test = y_test
                    st.session_state.label_encoder = label_encoder
                    
                    st.success("✅ All 6 models trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.stop()
        
        # Display results if models are trained
        if st.session_state.models_trained:
            
            st.markdown("---")
            st.header("📈 Model Performance Comparison")
            
            # Create metrics comparison table
            metrics_data = []
            for model_name, result in st.session_state.results.items():
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
            
            # Display table
            st.dataframe(
                metrics_df.style.set_properties(**{
                    'background-color': '#f0f2f6',
                    'color': 'black',
                    'border-color': 'white'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Best model highlight
            st.markdown("### 🏆 Best Performing Models")
            col1, col2, col3 = st.columns(3)
            
            # Find best models
            best_accuracy = max(st.session_state.results.items(), 
                              key=lambda x: x[1]['metrics']['Accuracy'])
            best_f1 = max(st.session_state.results.items(), 
                         key=lambda x: x[1]['metrics']['F1'])
            best_mcc = max(st.session_state.results.items(), 
                          key=lambda x: x[1]['metrics']['MCC'])
            
            with col1:
                st.metric(
                    "Best Accuracy",
                    f"{best_accuracy[1]['metrics']['Accuracy']:.4f}",
                    delta=best_accuracy[0]
                )
            
            with col2:
                st.metric(
                    "Best F1 Score",
                    f"{best_f1[1]['metrics']['F1']:.4f}",
                    delta=best_f1[0]
                )
            
            with col3:
                st.metric(
                    "Best MCC Score",
                    f"{best_mcc[1]['metrics']['MCC']:.4f}",
                    delta=best_mcc[0]
                )
            
            # Model selection for detailed analysis
            st.markdown("---")
            st.header("🔍 Detailed Model Analysis")
            
            selected_model = st.selectbox(
                "Select a model for detailed analysis:",
                options=list(st.session_state.results.keys())
            )
            
            if selected_model:
                result = st.session_state.results[selected_model]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        result['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='Blues',
                        ax=ax,
                        cbar_kws={'label': 'Count'}
                    )
                    ax.set_title(f'Confusion Matrix - {selected_model}')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    # Metrics Bar Chart
                    st.subheader("Performance Metrics")
                    metrics_to_plot = {k: v for k, v in result['metrics'].items()}
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.barh(list(metrics_to_plot.keys()), list(metrics_to_plot.values()))
                    
                    # Color bars
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    ax.set_xlabel('Score')
                    ax.set_title(f'Metrics - {selected_model}')
                    ax.set_xlim(0, 1)
                    
                    # Add value labels
                    for i, (metric, value) in enumerate(metrics_to_plot.items()):
                        ax.text(value, i, f' {value:.4f}', va='center')
                    
                    st.pyplot(fig)
                    plt.close()
                
                # Classification Report
                st.subheader("📋 Classification Report")
                st.text(result['classification_report'])
            
            # Comparative visualizations
            st.markdown("---")
            st.header("📊 Comparative Analysis")
            
            # Metrics comparison chart
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.ravel()
            
            metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            
            for idx, metric_name in enumerate(metric_names):
                ax = axes[idx]
                
                model_names = list(st.session_state.results.keys())
                values = [st.session_state.results[m]['metrics'][metric_name] for m in model_names]
                
                bars = ax.bar(range(len(model_names)), values, color='steelblue', alpha=0.7)
                
                # Highlight best model
                best_idx = values.index(max(values))
                bars[best_idx].set_color('green')
                bars[best_idx].set_alpha(1.0)
                
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.set_ylabel('Score')
                ax.set_title(f'{metric_name} Comparison')
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, v in enumerate(values):
                    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Download results
            st.markdown("---")
            st.header("💾 Export Results")
            
            csv_data = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics as CSV",
                data=csv_data,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )

else:
    # Landing page
    st.info("👈 Please upload a CSV dataset from the sidebar to begin")
    
    st.markdown("""
    ### 🎯 How to Use This App
    
    1. **Upload Dataset**: Click on the sidebar and upload your CSV file
    2. **Select Target**: Choose the column you want to predict
    3. **Configure**: Adjust training parameters if needed
    4. **Train**: Click "Train All Models" button
    5. **Analyze**: Review performance metrics and comparisons
    
    ### 📋 Requirements
    - Minimum 12 features
    - Minimum 500 instances
    - Classification problem (binary or multi-class)
    
    ### 🤖 Models Implemented
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)
    
    ### 📊 Evaluation Metrics
    - Accuracy
    - AUC Score
    - Precision
    - Recall
    - F1 Score
    - Matthews Correlation Coefficient (MCC)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>ML Classification Dashboard | Built with Streamlit</div>",
    unsafe_allow_html=True
)
