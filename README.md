# 📊 WealthWise  
## 💼 Financial Stress & Banking System Analysis

---

## 🔍 Overview

This project analyzes U.S. banking system indicators and financial stress conditions using machine learning techniques. The stability of the banking system plays a critical role in the overall health of an economy, influencing credit availability, investment activity, and economic growth.

The objective of this project is to examine structural relationships among macroeconomic and banking indicators and identify recurring financial regimes within the data. By analyzing patterns across multiple indicators, the project aims to provide a deeper understanding of how financial stress develops, evolves, and impacts the broader economic environment.

The project follows the data science lifecycle, beginning with data collection and exploration, followed by pattern discovery using unsupervised learning, and finally classification using supervised learning models.

---

# 🟦 Module 1: Data Preparation & Exploration (EDA)

Module 1 focuses on data collection, cleaning, and exploratory analysis.

---

## 📊 Data Source

All economic indicators were collected from the **Federal Reserve Economic Data (FRED)** database using the official API.

The dataset includes key financial and macroeconomic indicators:

- Consumer Delinquency Rate  
- Business Delinquency Rate  
- Charge-Off Rate  
- Total Bank Credit  
- Federal Funds Rate  
- Treasury Yields (2-Year and 10-Year)  
- Financial Stress Index  
- Unemployment Rate  

---

## 🧹 Data Preparation

The dataset was prepared using the following steps:

- Time-series datasets merged on the date column  
- Date column converted to datetime format  
- Missing values handled using forward-fill  
- Yield spread feature engineered (10-Year Treasury − 2-Year Treasury)  
- Data cleaned to remove inconsistencies and ensure completeness  

---

## 📈 Exploratory Data Analysis

- Visualization of trends across financial indicators  
- Correlation analysis to understand relationships  
- Identification of patterns across time  
- Understanding data distributions and variability  

📌 This stage builds foundational understanding of the dataset before applying models.

---

# 🟦 Module 2: Unsupervised Learning

Module 2 focuses on discovering patterns and structure within the data without using labels.

---

## 📉 Principal Component Analysis (PCA)

- Dimensionality reduction  
- Explained variance analysis  
- Identification of dominant components  
- 2D and 3D visualization of financial structure  

📌 PCA reveals underlying structure and relationships in financial indicators.

---

## 📊 Clustering

- K-Means Clustering  
- Hierarchical Clustering (Dendrogram)  
- DBSCAN  
- Silhouette score used for validation  

📌 Clustering identifies distinct financial regimes such as stable and high-stress periods.

---

## 🔗 Association Rule Mining (Apriori Algorithm)

- Quantile-based discretization  
- Frequent itemset generation  
- Rule evaluation using Support, Confidence, and Lift  

📌 Reveals recurring combinations of financial conditions.

---

# 🟦 Module 3: Supervised Learning

Module 3 focuses on classification of financial stress levels.

---

## 🟦 Naïve Bayes

- Gaussian, Multinomial, and Bernoulli variants implemented  
- Classification of financial stress levels (Low, Medium, High)  
- Performance evaluated using accuracy and confusion matrices  

📌 Provides a simple and efficient baseline model.

---

## 🌳 Decision Trees

- Multiple models with varying depth and parameters  
- Gini Index and Entropy used for splitting  
- Tree visualization for interpretability  
- Evaluation using accuracy, confusion matrices, and cross-validation  
- Analysis of overfitting  

📌 Offers strong performance and interpretability.

---

## 📈 Logistic Regression

- Probability-based classification model  
- Uses sigmoid function for decision making  
- Evaluated using accuracy and confusion matrix  
- Provides stable and generalizable performance  

📌 Serves as a reliable baseline classifier.

---

# 📁 Repository Structure

- 📓 Notebook: Complete data collection, preprocessing, modeling, and visualization  
- 📂 Raw Data: Original datasets  
- 📂 Cleaned Data: Processed dataset  
- 📊 Outputs: Visualizations and model results  

---

# 🔁 Reproducibility

To reproduce the analysis:

1. Install required libraries (pandas, numpy, scikit-learn, mlxtend, matplotlib, requests)  
2. Obtain a FRED API key  
3. Insert the API key into the notebook  
4. Run the notebook from top to bottom  

---

# 🎯 Project Scope

This project focuses on:

- Data exploration and visualization  
- Pattern discovery using unsupervised learning  
- Classification using supervised learning  

🚫 This project does not perform stock prediction or forecasting.

---

# ⚙️ Requirements

- pandas  
- numpy  
- scikit-learn  
- mlxtend  
- matplotlib  
- requests  

---

# 📌 Key Insights

- Financial indicators are interconnected and influenced by common economic forces  
- Financial stress follows structured and recurring patterns  
- Different machine learning models provide complementary insights  
- Model performance depends on data characteristics and assumptions  
- Understanding these relationships helps improve financial decision-making  

---
