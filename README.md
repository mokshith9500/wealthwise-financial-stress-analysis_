# 📊 WealthWise  
### Financial Stress & Banking System Analysis

---

## 🔍 Overview

WealthWise is a data-driven project that analyzes U.S. banking system conditions and financial stress using machine learning techniques.

The goal of this project is to understand how financial indicators interact, identify underlying structures in economic data, and uncover recurring financial regimes that reflect different states of economic stability.

---

## 🎯 Objectives

- Analyze relationships between macroeconomic and banking indicators  
- Identify hidden patterns in financial systems  
- Understand how financial stress evolves over time  
- Compare multiple machine learning approaches  

---

## 🧠 Methods Implemented

### 🔹 Principal Component Analysis (PCA)

- Dimensionality reduction  
- Explained variance analysis  
- Identification of dominant economic drivers  
- 2D and 3D visualization  

---

### 🔹 Clustering (Unsupervised Learning)

- K-Means Clustering (Silhouette-based selection of k)  
- Hierarchical Clustering (Dendrogram analysis)  
- DBSCAN (density-based clustering)  

📌 Purpose: Identify distinct financial regimes  

---

### 🔹 Association Rule Mining (Apriori)

- Quantile-based discretization (Low / Medium / High)  
- Frequent itemsets  
- Association rules using:
  - Support  
  - Confidence  
  - Lift  

📌 Purpose: Discover recurring financial patterns  

---

### 🔹 Supervised Learning (Module 3)

#### ✅ Naïve Bayes
- Gaussian, Multinomial, Bernoulli variants  
- Classification of financial stress levels  
- Performance comparison using confusion matrices  

#### ✅ Decision Trees
- Multiple tree configurations (depth tuning)  
- Gini vs Entropy comparison  
- Tree visualization  
- Cross-validation and overfitting analysis  

#### ✅ Logistic Regression
- Probability-based classification  
- Sigmoid function for decision boundaries  
- Baseline model for comparison  

📌 Purpose: Classify financial stress levels (Low / Medium / High)

---

## 📊 Data Source

All data was collected using the **Federal Reserve Economic Data (FRED) API**.

Indicators include:

- Consumer Delinquency Rate  
- Business Delinquency Rate  
- Charge-Off Rate  
- Total Bank Credit  
- Federal Funds Rate  
- Treasury Yields (2-Year & 10-Year)  
- Financial Stress Index  
- Unemployment Rate  

---

## 🧹 Data Preparation

The dataset was processed using:

- Time-series merging on date  
- Datetime conversion  
- Missing values handled using forward-fill  
- Feature engineering:
  - Yield Spread = (10Y − 2Y Treasury)  
- Standardization using **StandardScaler**  
- Discretization for Association Rule Mining  

✅ Final dataset contains no missing values  

---

## 📁 Repository Structure
