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
| Algorithm | Accuracy | ROC AUC | Precision | Recall | F1-Score |
|-----------|----------|---------|-----------|--------|----------|
| Decision Tree | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Logistic Regression | 0.983 | 1.000 | 0.972 | 1.000 | 0.986 |
| Naive Bayes | 0.983 | 0.999 | 0.972 | 1.000 | 0.986 |
| SVM | 0.900 | 0.976 | 0.892 | 0.943 | 0.917 |
| KNN | 0.750 | 0.839 | 0.813 | 0.743 | 0.776 |

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
