# 📊 WealthWise  
## 💼 Financial Stress & Banking System Analysis

---

## 🔍 Overview

This project analyzes U.S. banking system indicators and financial stress conditions using machine learning techniques. The stability of the banking system plays a critical role in the overall health of an economy, influencing credit availability, investment activity, and economic growth.

The objective of this project is to examine structural relationships among macroeconomic and banking indicators and identify recurring financial regimes within the data. By analyzing patterns across multiple indicators, the project aims to provide a deeper understanding of how financial stress develops, evolves, and impacts the broader economic environment.

This analysis combines both unsupervised and supervised learning approaches to explore hidden structures in the data and to classify financial stress levels. The goal is not only to identify patterns but also to interpret them in a meaningful way that reflects real-world financial behavior.

---

## 🧠 Methods Implemented

### 📉 Principal Component Analysis (PCA)

Principal Component Analysis is used to reduce the dimensionality of the dataset while retaining most of the important information. Financial datasets often contain highly correlated variables, and PCA helps in transforming these variables into a smaller set of uncorrelated components.

- Dimensionality reduction to simplify complex data  
- Explained variance analysis to understand information retention  
- Identification of dominant components driving financial behavior  
- Visualization of data in reduced dimensions (2D and 3D)  

📌 This method helps uncover the underlying structure of financial indicators.

---

### 📊 Clustering

Clustering techniques are applied to group similar financial conditions together, allowing the identification of distinct financial regimes such as stable, moderate, and high-stress environments.

- K-Means Clustering for partition-based grouping  
- Hierarchical Clustering (Dendrogram) for understanding nested relationships  
- DBSCAN for detecting density-based clusters and outliers  
- Silhouette score used to evaluate clustering quality and determine optimal clusters  

📌 Clustering reveals how financial states evolve and group over time.

---

### 🔗 Association Rule Mining (Apriori Algorithm)

Association Rule Mining is used to discover relationships between different financial indicators by identifying frequent patterns and combinations.

- Quantile-based discretization to convert continuous variables into categories  
- Frequent itemset generation using the Apriori algorithm  
- Rule evaluation using:
  - Support (frequency of occurrence)  
  - Confidence (likelihood of association)  
  - Lift (strength of relationship beyond randomness)  

📌 This method helps identify recurring combinations of financial conditions.

---

## 🆕 Module 3: Supervised Learning

To complement pattern discovery, supervised learning models are used to classify financial stress levels into categories such as Low, Medium, and High.

---

### 🟦 Naïve Bayes

Naïve Bayes is a probabilistic classification model based on Bayes’ Theorem. It assumes independence between features and is efficient for classification tasks.

- Gaussian, Multinomial, and Bernoulli variants implemented  
- Used to classify financial stress levels  
- Performance evaluated using accuracy and confusion matrices  
- Gaussian Naïve Bayes performs best for continuous financial data  

📌 Provides a simple and fast baseline for classification.

---

### 🌳 Decision Trees

Decision Trees classify data by recursively splitting it based on feature values, creating a tree-like structure of decisions.

- Multiple models built with varying depth and parameters  
- Gini Index and Entropy used as splitting criteria  
- Tree visualization for interpretability of decision rules  
- Performance evaluated using accuracy, confusion matrices, and cross-validation  
- Analysis of overfitting using training vs testing performance  

📌 Decision Trees provide strong performance and interpretability.

---

### 📈 Logistic Regression

Logistic Regression is a classification model that predicts probabilities and assigns data points to categories based on those probabilities.

- Uses sigmoid function to map outputs between 0 and 1  
- Suitable for classification of financial stress levels  
- Evaluated using accuracy and confusion matrix  
- Provides good generalization with lower risk of overfitting  

📌 Acts as a reliable and interpretable baseline model.

---

## 📊 Data Source

All economic indicators were collected from the **Federal Reserve Economic Data (FRED)** database using the official API.

These indicators represent different aspects of financial system health:

- Consumer Delinquency Rate  
- Business Delinquency Rate  
- Charge-Off Rate  
- Total Bank Credit  
- Federal Funds Rate  
- Treasury Yields (2-Year and 10-Year)  
- Financial Stress Index  
- Unemployment Rate  

📌 These variables collectively capture credit conditions, interest rates, and macroeconomic trends.

---

## 🧹 Data Preparation

The dataset was prepared using a structured data cleaning and preprocessing pipeline:

- Time-series datasets merged on the date column  
- Date column converted to datetime format  
- Missing values handled using forward-fill to maintain continuity  
- Feature engineering:
  - Yield Spread = (10-Year Treasury − 2-Year Treasury)  
- StandardScaler applied before PCA and clustering to normalize data  
- Discretization performed for Association Rule Mining  

✅ The final dataset contains no missing values and is suitable for modeling.

---

## 📁 Repository Structure

- 📓 Notebook: Contains complete data collection, preprocessing, modeling, and visualization  
- 📂 Raw Data: Original datasets obtained from FRED  
- 📂 Cleaned Data: Processed dataset used for analysis  
- 📊 Outputs: Visualizations, plots, and model results  

---

## 🔁 Reproducibility

To reproduce this project:

1. Install required Python libraries (pandas, numpy, scikit-learn, mlxtend, matplotlib, requests)  
2. Obtain a FRED API key  
3. Insert the API key into the notebook  
4. Run the notebook from top to bottom  

🔄 All results, models, and visualizations will be generated automatically.

---

## 🎯 Project Scope

This project focuses on understanding financial systems through:

- Pattern discovery  
- Structural analysis  
- Classification of financial stress  

It combines multiple machine learning techniques to gain insights into financial behavior.

🚫 This project does **not** perform stock prediction or forecasting.

---

## ⚙️ Requirements

The following Python libraries are required:

- pandas  
- numpy  
- scikit-learn  
- mlxtend  
- matplotlib  
- requests  

---

## 📌 Key Insights

- Financial indicators are interconnected and influenced by common economic forces  
- Financial stress follows structured and recurring patterns rather than randomness  
- Different machine learning methods provide complementary insights  
- Model selection depends on data characteristics and problem type  
- Understanding these relationships can improve financial monitoring and decision-making  

---
