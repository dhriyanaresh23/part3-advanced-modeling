# Part 3: Advanced Modeling | House Price Prediction

## Project Overview
This project develops and evaluates machine learning regression models to predict residential house sale prices using the Kaggle House Prices dataset. It covers data preprocessing, model comparison, cross-validation, and feature-importance analysis.

## Objectives
- Build regression models to predict house sale prices.
- Handle missing numerical and categorical values.
- Apply feature scaling and one-hot encoding.
- Use scikit-learn pipelines to reduce data leakage during preprocessing.
- Compare models using five-fold cross-validation and held-out test metrics.
- Identify influential features using Random Forest feature importance.

## Dataset
- **Source:** [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- **Training records:** 1,460
- **Original columns:** 81
- **Target variable:** `SalePrice`
- **Input features:** 79, excluding `Id` and `SalePrice`

Download the dataset from Kaggle and configure the file path in the notebook before running it.

## Models Evaluated
- Linear Regression
- Ridge Regression
- Random Forest Regressor

## Model Performance

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| Cross-validation mean R² | 0.7700 | 0.8400 |
| Cross-validation R² standard deviation | 0.0835 | 0.0440 |
| Test R² | 0.8867 | 0.8915 |
| Test MAE | $18,287.70 | $17,524.44 |
| Test RMSE | $29,473.87 | $28,848.70 |

**Best held-out test performance:** Random Forest, with an R² of 0.8915.

MAE and RMSE are expressed in the dataset's sale-price units (USD).

## Preprocessing Pipeline
- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standard scaling for numerical features.
- One-hot encoding for categorical features.
- Combined preprocessing and modeling using scikit-learn pipelines.

## Feature Importance
The Random Forest model ranked these among its most influential features:
1. `OverallQual`: Overall material and finish quality.
2. `GrLivArea`: Above-ground living area.
3. `TotalBsmtSF`: Total basement area.

Feature importance describes how the model uses features; it does not establish causation.

## Repository Contents
- `part3_advanced_modeling.ipynb`: Original notebook.
- `part3_advanced_modeling_revised.ipynb`: Revised house-price regression workflow.
- `house_price_model_results.csv`: Saved model comparison results.
- `best_model.pkl`: Previously saved model artifact. Its compatibility with the revised workflow has not yet been verified.

## Technologies
Python, pandas, NumPy, scikit-learn, and Google Colab/Jupyter Notebook.

## Limitations
Results depend on the dataset and evaluation split. Performance on unseen properties may differ. The saved model artifact should be checked before using it with the revised preprocessing pipeline.

## Conclusion
Random Forest performed best among the evaluated models on the held-out test set, with an R² of 0.8915 and an MAE of approximately $17,524. The project demonstrates a reproducible regression workflow using preprocessing pipelines, cross-validation, and feature-importance analysis.
