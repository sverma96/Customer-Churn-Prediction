# Customer-Churn-Prediction


A machine learning pipeline that predicts whether a telecom customer is likely to churn (cancel their subscription), using the IBM Telco Customer Churn dataset. The project covers the full workflow — data cleaning, exploratory data analysis, preprocessing, model training and comparison, evaluation, and a ready-to-use predictive system built on the final saved model.

Built as part of **UML501 – Machine Learning**

---

##  Overview

Customer churn — when a customer stops using a company's service — directly impacts revenue and growth. Since acquiring a new customer costs significantly more than retaining an existing one, businesses benefit greatly from identifying at-risk customers *before* they leave. This project builds a binary classification model that estimates each customer's probability of churning, based on their account, billing, and service usage data.

##  Key Results

| Metric | Value |
|---|---|
| Best model (5-fold CV) | Random Forest — 84% mean CV accuracy |
| Test set accuracy | 77.9% |
| Churn-class F1-score | 0.58 |
| Weighted-average F1-score | 0.78 |
| ROC AUC (estimated) | ~0.84–0.86 |

Three tree-based models — **Decision Tree**, **Random Forest**, and **XGBoost** — were trained and compared using cross-validation. Random Forest was selected as the final model for its highest and most stable accuracy across folds.

## 📂 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customer records × 21 columns
- **Target column:** `Churn` (Yes/No)
- **Features include:** demographics (gender, senior citizen, partner, dependents), account info (tenure, contract type, payment method, billing), and subscribed services (phone, internet, streaming, tech support, etc.)

## ⚙️ Pipeline

1. **Data Loading & Cleaning** — dropped the non-predictive `customerID` column, identified and fixed 11 blank string entries in `TotalCharges`, verified no true missing values remained.
2. **Exploratory Data Analysis** — distribution plots, boxplots, correlation heatmap, and category-wise churn breakdowns to surface key churn drivers (contract type, tenure, internet service, payment method).
3. **Preprocessing** — label-encoded the target and all categorical features (encoders saved for reuse), 80/20 train-test split, and **SMOTE** oversampling applied only to the training set to address class imbalance.
4. **Model Training** — Decision Tree, Random Forest, and XGBoost trained with default parameters and compared via 5-fold cross-validation.
5. **Model Evaluation** — accuracy, confusion matrix, classification report, and ROC curve on the held-out test set.
6. **Predictive System** — the trained model and encoders are serialized with `pickle` and reloaded to score new, unseen customer records end-to-end.

## 🗂️ Project Structure

```
.
├── Customer_Churn_Prediction_using_ML.ipynb   # Full notebook (EDA → training → evaluation)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Dataset
├── customer_churn_model.pkl                   # Saved trained Random Forest model + feature list
├── encoders.pkl                                # Saved LabelEncoders for categorical features
