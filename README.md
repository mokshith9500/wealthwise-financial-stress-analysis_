# WealthWise  
## Financial Stress & Banking System Analysis

---

## Overview

This project analyzes U.S. banking system indicators and financial stress conditions using unsupervised machine learning techniques.

The objective is to examine structural relationships among macroeconomic and banking indicators and identify recurring financial regimes within the data.

---

## Methods Implemented

### Principal Component Analysis (PCA)

- Dimensionality reduction  
- Explained variance analysis  
- Identification of dominant components  

### Clustering

- K-Means Clustering  
- Hierarchical Clustering (Dendrogram)  
- DBSCAN  
- Silhouette score evaluation for cluster validation  

### Association Rule Mining (Apriori Algorithm)

- Quantile-based discretization  
- Support, Confidence, and Lift metrics  
- Identification of recurring financial indicator combinations  

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
- StandardScaler applied prior to PCA and clustering  
- Discretization performed for Association Rule Mining  

The final dataset contains no missing values.

---

## Repository Structure
- The notebook contains complete data collection, preprocessing, modeling, and visualization code.  
- Raw and cleaned datasets are included for reproducibility.

---

## Reproducibility

To reproduce the analysis:

1. Install required Python libraries (pandas, numpy, scikit-learn, mlxtend, matplotlib).  
2. Insert a valid FRED API key in the notebook.  
3. Run the notebook from top to bottom.  

All results can be regenerated using the provided files.

---

## Project Scope

This project focuses on structural discovery within financial data using unsupervised learning methods. It does not attempt predictive modeling or forecasting.
