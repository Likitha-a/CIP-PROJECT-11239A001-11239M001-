# CIP-PROJECT-11239A001-11239M001-
# UPI Fraud Detection using Machine Learning

##  Overview
This project implements an end-to-end machine learning pipeline to detect fraudulent UPI transactions.

It integrates multiple datasets (transactions, users, merchants, fraud labels), handles class imbalance using SMOTE, and introduces a behavioral feature called **Transaction Velocity** to capture suspicious activity patterns.

The system also applies **threshold optimization using F-beta (β = 1.5)** and evaluates models using multiple metrics to ensure reliable fraud detection.

---

##  Features
- Multi-source data integration (20,000 transactions, 53 features)
- SMOTE applied only on training data
- Transaction Velocity (10-minute rolling window per user)
- Threshold tuning using F-beta (β = 1.5)
- Multi-metric evaluation (ROC-AUC, MCC, Precision, Recall, F1)
- SHAP-based explainability
- Comparison of 5 ML models

---

##  Dataset
- Total Transactions: 20,000  
- Features: 53  
- Fraud Ratio: ~3.8% (highly imbalanced)

### Files:
- transactions.csv
- users.csv
- merchants.csv
- fraud_labels.csv

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- Imbalanced-learn (SMOTE)
- SHAP
- Matplotlib, Seaborn

---

##  Workflow
1. Data Loading & Merging  
2. Data Preprocessing  
3. Feature Engineering (Transaction Velocity)  
4. Encoding & Train-Test Split  
5. SMOTE on training data  
6. Model Training (5 classifiers)  
7. Threshold Optimization (F-beta)  
8. Model Evaluation (ROC-AUC, MCC, F1, etc.)  
9. Model Selection (ranked by ROC-AUC, validated using MCC)  
10. Explainability using SHAP  

---

##  Model Performance

| Model                | ROC-AUC | MCC   | F1   |
|---------------------|--------|-------|------|
| Random Forest       | 1.00   | 0.996 | 0.996 |
| Decision Tree       | 1.00   | 0.00  | 0.00 |
| XGBoost             | 1.00   | 0.00  | 0.00 |
| LightGBM            | 1.00   | 0.00  | 0.00 |
| Logistic Regression | 0.56   | 0.04  | 0.09 |

### Insight
Although multiple models achieved ROC-AUC = 1.0, only **Random Forest** maintained high MCC and F1.  
This shows that ROC-AUC alone is not sufficient for imbalanced datasets.

---

##  Best Model
- Random Forest  
- ROC-AUC = 1.00  
- MCC = 0.996  
- F1 = 0.996  

---
##  How to Run

### Clone Repository
```
git clone https://github.com/Likitha-a/upi-fraud-detection.git
cd upi-fraud-detection
```

### Install Dependencies
```
pip install -r requirements.txt
```

### Run Project
```
python main.py
```

---

##  Project Structure
```
├── data/
├── notebooks/
├── models/
├── main.py
├── requirements.txt
└── README.md
```

---

##  Explainability
- SHAP is used to explain model predictions  
- Provides feature-level interpretation for fraud detection  

---

##  Future Improvements
- Real-time fraud detection system
- Deep learning models (LSTM)
- Graph-based fraud detection
- Model monitoring and drift detection

---

##  Author
Likitha Nalini
Besi Pravallika
