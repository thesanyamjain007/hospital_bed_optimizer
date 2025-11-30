# Hospital Bed Capacity Optimization 🏥

## Overview
A Machine Learning-based Decision Support System that predicts patient stay length (Long/Short) to optimize hospital bed allocation, staffing, and resource management.

## Features

### 1. Automated Data Pipeline
- **Ingestion**: CSV and Excel file uploads
- **Preprocessing**: Missing value handling, outlier detection, and feature encoding
- **Scaling**: StandardScaler normalization for algorithms

### 2. Interactive EDA
- Correlation heatmaps
- Target class distribution analysis
- Demographic visualizations

### 3. Multi-Algorithm Training
Trains 5 algorithms simultaneously:
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine

### 4. Model Deployment
- Export/import models as `.joblib` files
- Manual single-patient prediction
- Batch prediction via CSV upload

## Performance Results
### Updated Results on LengthOfStay.csv (no LOS leakage)

| Algorithm           | Train Accuracy | Test Accuracy | ROC AUC | Precision (class 1) | Recall (class 1) | F1-score (class 1) |
|--------------------|----------------|---------------|---------|----------------------|-------------------|---------------------|
| Logistic Regression| 0.8279         | 0.7350        | 0.8008  | 0.7855               | 0.7006            | 0.7406              |
| Naive Bayes        | 0.7900         | 0.7083        | 0.6972  | 0.7899               | 0.6265            | 0.6988              |
| KNN                | 0.6608         | 0.4767        | 0.5074  | 0.5217               | 0.3704            | 0.4332              |
| Decision Tree      | 0.8325         | 0.7867        | 0.8100  | 0.8792               | 0.7099            | 0.8014              |
| SVM                | 0.8371         | 0.7517        | 0.8182  | 0.8405               | 0.6667            | 0.7435              |



## Installation

### Requirements
- Python 3.8+
- pip

### Setup
```bash
git clone https://github.com/thesanyamjain007/hospital_bed_optimizer.git
cd hospital-bed-optimization
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Upload dataset via "Upload & Inspect"
2. Clean data in "Processing + EDA"
3. Train models in "Train All Models"
4. Review results in "Comparison Table"
5. Make predictions in "Predict (Saved Model)"

## Project Structure
```
hospital-bed-optimization/
├── app.py
├── backend_models.py
├── dummy_hospital_beds_200.csv
├── requirements.txt
└── README.md
```

## Future Enhancements
- Live Hospital Information System integration
- Real-time bed availability dashboard
- Deep Learning models
- Explainable AI (SHAP values)

## License
Educational and research purposes (POC)
