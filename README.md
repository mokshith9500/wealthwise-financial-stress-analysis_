# WealthWise  
## Financial Stress & Banking System Analysis

---

## Overview

This project analyzes U.S. banking system indicators and financial stress conditions using machine learning techniques.

The objective is to examine structural relationships among macroeconomic and banking indicators and identify recurring financial regimes within the data.

---

## Methods Implemented

### Principal Component Analysis (PCA)

- Dimensionality reduction  
- Explained variance analysis  
- Identification of dominant components  

---

### Clustering

- K-Means Clustering  
- Hierarchical Clustering (Dendrogram)  
- DBSCAN  
- Silhouette score evaluation for cluster validation  

---

### Association Rule Mining (Apriori Algorithm)

- Quantile-based discretization  
- Support, Confidence, and Lift metrics  
- Identification of recurring financial indicator combinations  

---

## Module 3: Supervised Learning

### Naïve Bayes

- Gaussian, Multinomial, and Bernoulli Naïve Bayes models were implemented  
- Used for classification of financial stress levels (Low, Medium, High)  
- Model performance evaluated using accuracy and confusion matrices  
- Comparison across variants to determine best fit for continuous financial data  

---

### Decision Trees

- Multiple Decision Tree models built with varying depth and criteria  
- Gini Index and Entropy used for splitting  
- Tree visualization used for interpretability  
- Model performance evaluated using accuracy, confusion matrices, and cross-validation  
- Analysis of overfitting using training vs testing performance  

---

### Logistic Regression

- Used for classification of financial stress levels  
- Probability-based model using sigmoid function  
- Evaluated using accuracy and confusion matrix  
- Compared with other models for performance and generalization  

---

## Data Source

All economic indicators were collected from the Federal Reserve Economic Data (FRED) database using the official FRED API.

Indicators used include:

- Consumer Delinquency Rate  
- Business Delinquency Rate  
- Charge-Off Rate  
- Total Bank Credit  
- Federal Funds Rate  
- Treasury Yields (2-Year and 10-Year)  
- Financial Stress Index  
- Unemployment Rate  

---

## Data Preparation

The dataset was prepared using the following steps:

- Time-series datasets merged on the date column  
- Date column converted to datetime format  
- Missing values handled using forward-fill  
- Yield spread feature engineered (10-Year Treasury minus 2-Year Treasury)  
- StandardScaler used prior to PCA and clustering  
- Discretization performed for Association Rule Mining  

The final dataset contains no missing values.

---

## Repository Structure

- The notebook contains complete data collection, preprocessing, modeling, and visualization code  
- Raw dataset before cleaning is included  
- Cleaned dataset used for modeling is included  

---

## Reproducibility

To reproduce the analysis:

1. Install required Python libraries (pandas, numpy, scikit-learn, mlxtend, matplotlib, requests)  
2. Insert a valid FRED API key in the notebook  
3. Run the notebook from top to bottom  

All results can be regenerated using the provided files.

---

## Project Scope

This project focuses on structural discovery and classification within financial data using machine learning methods.

It includes both unsupervised and supervised learning techniques to understand financial patterns and classify stress levels.

---

## Requirements

The following Python libraries are required:

- pandas  
- numpy  
- scikit-learn  
- mlxtend  
- matplotlib  
- requests  

---
