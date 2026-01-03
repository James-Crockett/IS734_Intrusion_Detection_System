# Network Intrusion Detection System (NIDS) on NSL-KDD

This repository contains a Jupyter Notebook (is734_pre_processing.ipynb) that demonstrates the end-to-end process of building a Network Intrusion Detection System. Using the NSL-KDD dataset, the project focuses on preprocessing complex network traffic data and training machine learning models to classify activities as either "Normal" or "Attack."

## Project Overview

This project aims to automate the detection of malicious network activity by leveraging supervised machine learning techniques. The workflow includes:
1.  **Data Ingestion**: Loading training and testing sets from the NSL-KDD archive.
2.  **Exploratory Data Analysis (EDA)**: Visualizing protocol distributions and attack classes.
3.  **Preprocessing**: Handling categorical data, scaling numerical features, and encoding labels.
4.  **Feature Engineering**: Applying Principal Component Analysis (PCA) to reduce dimensionality while retaining variance.
5.  **Model Training & Evaluation**: Benchmarking various algorithms (Logistic Regression, XGBoost, SVM, etc.) using metrics like Accuracy, Precision, Recall, and Confusion Matrices.

## Dataset

The project uses the NSL-KDD dataset, an improved version of the original KDD'99 dataset, designed to solve some of the inherent problems of the original (like redundant records).

* **Input Features**: 41 features per record (e.g., duration, protocol_type, src_bytes, flag).
* **Target**: outcome (categorized into 'normal' or specific attack types like 'neptune', 'satan', etc.).
* **Transformation**: For this specific implementation, the target variable is converted into a Binary Classification problem (0 for Normal, 1 for Attack).

## Technologies & Libraries

The project is implemented in Python using the following libraries:

* **Data Manipulation**: pandas, numpy
* **Visualization**: matplotlib, seaborn
* **Machine Learning**: scikit-learn
* **Advanced Boosting**: xgboost
* **Deep Learning Framework**: tensorflow (imported for potential future extensions)

## Preprocessing Pipeline

The notebook implements a robust preprocessing pipeline:
1.  **Label Encoding**: Converting the multi-class outcome variable into binary (Normal vs. Attack).
2.  **One-Hot Encoding**: Transforming categorical features such as protocol_type, service, and flag into numeric vectors.
3.  **Scaling**: Using RobustScaler to normalize numerical features, making the models resilient to outliers.
4.  **PCA (Principal Component Analysis)**: Reducing the feature space from 122+ processed features down to 20 principal components to improve training speed and reduce overfitting.

## Models Implemented and Performance

The notebook is set up to train and compare multiple classifiers:
* Logistic Regression
* Random Forest Classifier
* Support Vector Machine (SVM)
* Decision Tree
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* XGBoost

```
------------------------------------------------------------
FINAL RESULTS (Sorted by Accuracy)
                 Model  Accuracy  Precision    Recall  F1 Score
3        Decision Tree  0.790853   0.965375  0.656121  0.781257
1        Random Forest  0.781183   0.968676  0.636172  0.767979
5            KNN (n=5)  0.755855   0.972534  0.587704  0.732660
0  Logistic Regression  0.751419   0.913606  0.622146  0.740219
2           Linear SVM  0.745564   0.914787  0.609834  0.731812
4          Naive Bayes  0.724627   0.905397  0.576483  0.704437
------------------------------------------------------------
```

## Results & Evaluation

The models are evaluated based on:
* **Accuracy**: Overall correctness of the model.
* **Precision & Recall**: Critical for security contexts to minimize false negatives (missed attacks) and false positives (false alarms).
* **Confusion Matrix**: Visual representation of prediction performance.


## Getting Started

### Prerequisites
Ensure you have Python installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost