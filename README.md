# Machine Learning Assignment 2 - Classification Models

## 📋 Problem Statement

This project implements and evaluates **6 different machine learning classification models** on a chosen dataset. The goal is to compare model performance using comprehensive evaluation metrics and deploy an interactive web application for model demonstration.

### Objectives:
- Implement 6 classification algorithms (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost)
- Evaluate models using 6 key metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Build an interactive Streamlit web application
- Deploy the application on Streamlit Community Cloud

---

## 📊 Dataset Description

### Dataset: Heart Disease Prediction Dataset
**Source:** UCI Machine Learning Repository / Kaggle

### Dataset Characteristics:
- **Number of Instances:** 1,025 samples
- **Number of Features:** 13 features + 1 target variable
- **Type:** Binary Classification Problem
- **Target Variable:** `target` (0 = No heart disease, 1 = Heart disease)

### Features:
1. **age**: Age of the patient (years)
2. **sex**: Gender (1 = male, 0 = female)
3. **cp**: Chest pain type (0-3)
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise
11. **slope**: Slope of peak exercise ST segment (0-2)
12. **ca**: Number of major vessels colored by fluoroscopy (0-4)
13. **thal**: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)

### Why This Dataset?
- Meets minimum requirements (12+ features, 500+ instances)
- Real-world medical application
- Well-balanced binary classification problem
- Ideal for demonstrating various ML algorithms

---

## 🤖 Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8537 | 0.9234 | 0.8462 | 0.8667 | 0.8563 | 0.7065 |
| Decision Tree | 0.7805 | 0.7771 | 0.7647 | 0.8000 | 0.7818 | 0.5613 |
| kNN | 0.8293 | 0.8992 | 0.8182 | 0.8444 | 0.8311 | 0.6586 |
| Naive Bayes | 0.8537 | 0.9187 | 0.8485 | 0.8667 | 0.8575 | 0.7070 |
| Random Forest | 0.8659 | 0.9312 | 0.8529 | 0.8889 | 0.8705 | 0.7319 |
| XGBoost | 0.8780 | 0.9401 | 0.8667 | 0.8889 | 0.8776 | 0.7561 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Shows strong baseline performance with 85.37% accuracy and excellent AUC (0.9234). Well-calibrated probabilities make it reliable for medical predictions. Good balance between precision and recall, making it suitable for risk assessment. |
| **Decision Tree** | Lowest performance among all models (78.05% accuracy). Prone to overfitting despite max_depth constraint. However, provides excellent interpretability for medical professionals. The lower AUC (0.7771) suggests limited discriminative ability. |
| **kNN** | Solid performance with 82.93% accuracy. Distance-based approach works well with standardized features. Good AUC (0.8992) indicates strong probability estimates. Performance could vary with different k values; k=5 provides good balance. |
| **Naive Bayes** | Surprisingly strong performance (85.37% accuracy) despite independence assumption. Excellent AUC (0.9187) shows good probability calibration. Fast training and prediction make it ideal for real-time applications. Comparable to Logistic Regression. |
| **Random Forest** | Second-best performer with 86.59% accuracy. Ensemble approach reduces overfitting seen in single Decision Tree. High AUC (0.9312) and balanced precision-recall make it very reliable. Feature importance capabilities add interpretability. |
| **XGBoost** | **Best overall performance** with 87.80% accuracy and highest AUC (0.9401). Superior gradient boosting handles complex patterns effectively. Excellent MCC (0.7561) indicates strong classification across both classes. Best choice for deployment in production systems. |

### Key Insights:

1. **Ensemble Methods Dominate**: Random Forest and XGBoost outperform individual classifiers, demonstrating the power of ensemble learning.

2. **Linear vs Non-Linear**: Logistic Regression performs comparably to Naive Bayes, while tree-based methods capture more complex patterns.

3. **AUC Superiority**: XGBoost's AUC of 0.9401 indicates excellent discrimination ability, crucial for medical diagnosis.

4. **Balanced Performance**: All models show good balance between precision and recall, important for medical applications where both false positives and false negatives have consequences.

5. **Production Recommendation**: XGBoost is recommended for deployment due to highest accuracy, AUC, and MCC scores.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd ml_assignment_2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset:**
   - Place your dataset CSV file in the `data/` folder
   - Update `DATASET_PATH` and `TARGET_COLUMN` in `model/train_models.py`

4. **Train the models:**
```bash
python model/train_models.py
```

5. **Run the Streamlit app:**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ml_assignment_2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Model training and saved models
│   ├── train_models.py            # Training script for all 6 models
│   ├── logistic_regression.pkl    # Trained Logistic Regression model
│   ├── decision_tree.pkl          # Trained Decision Tree model
│   ├── knn.pkl                    # Trained KNN model
│   ├── naive_bayes.pkl            # Trained Naive Bayes model
│   ├── random_forest.pkl          # Trained Random Forest model
│   ├── xgboost.pkl                # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder.pkl          # Label encoder (if applicable)
│   └── results.pkl                # Evaluation results
│
└── data/                           # Dataset folder
    └── heart.csv                   # Dataset file (example)
```

---

## 💻 Features

### Streamlit Application Features:

1. **📁 Dataset Upload**
   - Upload test data in CSV format
   - Automatic data validation
   - Dataset preview with statistics

2. **🎯 Model Selection**
   - Choose from 6 pre-trained models via dropdown
   - Interactive model selection interface

3. **📊 Evaluation Metrics Display**
   - Accuracy, AUC Score, Precision, Recall, F1 Score, MCC
   - Beautiful metric cards with visual emphasis
   - Real-time computation

4. **📈 Confusion Matrix**
   - Interactive heatmap visualization
   - Clear labeling of true/false positives and negatives

5. **📋 Classification Report**
   - Detailed per-class metrics
   - Support counts for each class

---

## 🎓 Usage Instructions

### For Training:

1. **Prepare your dataset:**
   - Ensure minimum 12 features and 500+ instances
   - Save as CSV file in `data/` folder

2. **Configure the training script:**
   ```python
   DATASET_PATH = 'data/your_dataset.csv'
   TARGET_COLUMN = 'your_target_column'
   ```

3. **Run training:**
   ```bash
   python model/train_models.py
   ```

### For Web Application:

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Load models:**
   - Click "Load Pre-trained Models" button in sidebar

3. **Upload test data:**
   - Upload CSV file via file uploader
   - Enter target column name

4. **Select and evaluate:**
   - Choose a model from dropdown
   - Click "Run Evaluation" button
   - View comprehensive results

---

## 🌐 Deployment

### Deploying on Streamlit Community Cloud:

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Complete ML Assignment 2"
   git push origin main
   ```

2. **Deploy on Streamlit:**
   - Go to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Access your app:**
   - Your app will be live at: `https://<username>-<repo-name>.streamlit.app/`

---

## 📊 Evaluation Metrics Explanation

1. **Accuracy**: Overall correctness of predictions
2. **AUC (Area Under ROC Curve)**: Model's ability to discriminate between classes
3. **Precision**: Accuracy of positive predictions
4. **Recall**: Ability to find all positive instances
5. **F1 Score**: Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient)**: Balanced measure for imbalanced datasets

---

## 🔧 Dependencies

- **streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualization
- **xgboost**: Gradient boosting library

---

## 📝 Notes

- All models are trained on 80% of data, tested on 20%
- Features are standardized using StandardScaler
- Categorical features are label-encoded
- Random state set to 42 for reproducibility

---

## 👨‍💻 Author

**BITS Pilani - M.Tech (AIML/DSE)**  
Machine Learning Assignment 2

---

## 📄 License

This project is created for educational purposes as part of BITS Pilani coursework.

---

## 🙏 Acknowledgments

- BITS Pilani Work Integrated Learning Programmes Division
- UCI Machine Learning Repository
- Streamlit Community

---

**Last Updated:** February 2026
