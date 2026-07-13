# Part 3 – Advanced Modeling – Ensembles, Tuning, and Full ML Pipeline

## Project Overview

This project focuses on advanced machine learning techniques to improve classification performance using ensemble models, hyperparameter tuning, model evaluation, and a complete machine learning pipeline. Multiple classification models are trained, compared, optimized, and finally serialized for deployment.

## Objectives

- Train and evaluate Decision Tree classifiers.
- Compare unconstrained and controlled Decision Tree models.
- Compare Gini and Entropy splitting criteria.
- Train Random Forest and Gradient Boosting classifiers.
- Perform feature importance analysis and feature ablation.
- Compare models using 5-fold cross-validation.
- Optimize Random Forest using GridSearchCV.
- Analyze the learning curve of the best model.
- Serialize the best-performing model for future use.
- Recommend the most suitable model based on overall performance.

## Dataset Information

- Dataset: `cleaned_data.csv`
- Source: Part 1 Data Analysis
- Target Variable: `y_clf`
- Feature Matrix: `X`
- Total Samples: 1460
- Data Type: Structured tabular housing dataset

## Repository Structure

```
part3-advanced-modeling/
│
├── outputs/
├── best_model.pkl
├── cleaned_data.csv
├── part3_advanced_modeling.ipynb
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

## Dependencies

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/dhriyanaresh23/part3-advanced-modeling.git
```

Navigate to the project folder:

```bash
cd part3-advanced-modeling
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

Run the notebook:

```bash
jupyter notebook part3_advanced_modeling.ipynb
```

Execute all notebook cells from top to bottom.

## Environment Variables

This project does not require any API keys or environment variables.

If API keys or sensitive credentials are required in the future, they should be stored in a `.env` file and excluded from version control using `.gitignore`.

## Outputs Generated

The notebook generates the following outputs:

- Decision Tree performance comparison
- Gini vs Entropy comparison
- Random Forest evaluation metrics
- Top 5 feature importance table
- Gradient Boosting evaluation metrics
- Feature ablation comparison
- Cross-validation comparison table
- GridSearchCV best parameters and best score
- Manual learning curve table
- Serialized model (`best_model.pkl`)
- Model recommendation summary table

## Task 1 – Decision Tree Baseline

### Objective

The objective of this task is to train a baseline Decision Tree Classifier using the default hyperparameters and evaluate its performance on both the training and testing datasets. The model is used as a baseline to determine whether overfitting occurs.

### Implementation

A `DecisionTreeClassifier` from `sklearn.tree` was trained using the scaled training dataset with the default hyperparameters (`max_depth=None`). The model was evaluated by calculating the training accuracy, testing accuracy, and the train-test accuracy gap.

### Results

| Metric | Value |
|---------|------:|
| Training Accuracy | **1.0000** |
| Test Accuracy | **0.8973** |
| Train-Test Accuracy Gap | **0.1027** |

### Interpretation

The Decision Tree achieved **100% training accuracy** and **89.73% testing accuracy**, resulting in a **train-test accuracy gap of approximately 10.27%**. This indicates that the model has **overfitted** the training data.

Decision Trees are considered **high-variance models** because they greedily split the training data at each node without revisiting previous decisions. As the tree grows without any depth restriction, it can memorize the training data, leading to excellent training performance but reduced generalization on unseen data.

### Result

The baseline Decision Tree showed clear signs of **overfitting**, with perfect training accuracy but lower testing accuracy. This baseline model will be compared with a controlled Decision Tree in the next task to improve generalization performance.

## Task 2 – Controlled Decision Tree

### Objective

The objective of this task is to train a controlled Decision Tree Classifier using `max_depth=5` and `min_samples_split=20` to reduce overfitting and improve the model's ability to generalize to unseen data.

### Implementation

A second `DecisionTreeClassifier` was trained using the scaled training dataset with the following hyperparameters:

- `max_depth = 5`
- `min_samples_split = 20`

The model was evaluated using the training accuracy and testing accuracy, and its performance was compared with the baseline Decision Tree from Task 1.

### Results

| Metric | Value |
|---------|------:|
| Training Accuracy | **0.9272** |
| Test Accuracy | **0.8767** |

### Interpretation

The controlled Decision Tree achieved **92.72% training accuracy** and **87.67% testing accuracy**.

The `max_depth` parameter limits how deep the tree can grow, reducing model variance at the cost of a small increase in bias. The `min_samples_split` parameter prevents splitting a node if fewer than the specified number of samples are present, avoiding splits that respond to noise in small subsets.

Compared with the unconstrained Decision Tree, the controlled model reduced the **train-test accuracy gap from 0.1027 to 0.0505**, indicating that the model is less prone to overfitting and generalizes better to unseen data.

### Result

The controlled Decision Tree successfully reduced model complexity by limiting tree depth and restricting unnecessary splits. Although the training accuracy decreased compared with the baseline model, the reduced train-test accuracy gap demonstrates improved generalization and a lower risk of overfitting.

## Task 3 – Gini vs Entropy Comparison

### Objective

The objective of this task is to compare the performance of Decision Tree Classifiers using the **Gini** and **Entropy** impurity criteria. The comparison helps determine which splitting criterion provides better classification performance on the test dataset.

### Implementation

Two `DecisionTreeClassifier` models were trained using the same hyperparameters (`max_depth=5`) but different splitting criteria:

- `criterion='gini'`
- `criterion='entropy'`

The test accuracy of both models was computed and compared.

### Results

| Criterion | Test Accuracy |
|------------|--------------:|
| Gini | **0.8938** |
| Entropy | **0.9007** |

### Interpretation

The Decision Tree using the **Entropy** criterion achieved a slightly higher **test accuracy (90.07%)** than the **Gini** criterion (**89.38%**), indicating that Entropy performed marginally better on this dataset.

**Gini Impurity Formula:**

\[
\text{Gini} = 1 - \sum p_i^2
\]

**Entropy Formula:**

\[
\text{Entropy} = -\sum p_i \log_2(p_i)
\]

A node with **Gini = 0** means that all samples in the node belong to a single class, making it a perfectly pure node with no class impurity.

### Result

Both splitting criteria produced similar classification performance, with the **Entropy** criterion slightly outperforming the **Gini** criterion on the test dataset. The results indicate that either criterion is suitable for this problem, although Entropy achieved the highest test accuracy.

## Task 4 – Random Forest

### Objective

The objective of this task is to train a Random Forest Classifier and evaluate its performance using training accuracy, testing accuracy, and ROC-AUC. The model's feature importance scores are also analyzed to identify the most influential features contributing to the prediction.

### Implementation

A `RandomForestClassifier` was trained using the following hyperparameters:

- `n_estimators = 100`
- `max_depth = 10`
- `random_state = 42`

The model was evaluated using training accuracy, testing accuracy, and ROC-AUC. The top five most important features were obtained using the `feature_importances_` attribute.

### Results

#### Model Performance

| Metric | Value |
|---------|------:|
| Training Accuracy | **0.9957** |
| Test Accuracy | **0.9452** |
| ROC-AUC | **0.9844** |

#### Top 5 Feature Importance Scores

| Feature | Importance Score |
|---------|-----------------:|
| GrLivArea | **0.083904** |
| YearBuilt | **0.060458** |
| OverallQual | **0.059282** |
| FullBath | **0.051729** |
| TotalBsmtSF | **0.037707** |

### Interpretation

The Random Forest model achieved a **training accuracy of 99.57%**, a **testing accuracy of 94.52%**, and a **ROC-AUC score of 0.9844**, indicating excellent classification performance and strong generalization ability.

Random Forest computes feature importance by measuring the **average reduction in Gini impurity** contributed by each feature across all decision trees in the ensemble. Features that reduce impurity more frequently and by larger amounts receive higher importance scores.

Unlike Linear Regression coefficients, feature importance scores do not indicate the direction (positive or negative) of a relationship. Instead, they measure how much each feature contributes to improving the model's predictive performance.

Random Forest uses the **bagging (Bootstrap Aggregating)** technique to improve model performance and reduce overfitting. During training, each decision tree is built using a **bootstrap sample**, which is a random sample of the training data drawn **with replacement**, so the same observation may appear multiple times while others may not appear at all. At every split in a tree, only a **random subset of √(number of features)** is considered when selecting the best split. This randomness makes the individual trees different from one another. The final prediction is obtained by combining the predictions of all trees through majority voting. Since the errors made by individual trees tend to cancel each other out, the ensemble model has **lower variance**, is less prone to overfitting, and generalizes better than a single deep Decision Tree.

### Result

The Random Forest model significantly improved predictive performance compared with the Decision Tree models. The high ROC-AUC score demonstrates excellent class separation, while the feature importance analysis identified **GrLivArea, YearBuilt, OverallQual, FullBath, and TotalBsmtSF** as the most influential features. The ensemble approach successfully reduced variance and improved overall model generalization.

## Task 4(a) – Gradient Boosting

### Objective

The objective of this task is to train a Gradient Boosting Classifier and evaluate its performance using training accuracy, testing accuracy, and ROC-AUC. The model is then compared with the Random Forest model and included in the cross-validation analysis in Task 5.

### Implementation

A `GradientBoostingClassifier` from `sklearn.ensemble` was trained using the following hyperparameters:

- `n_estimators = 100`
- `learning_rate = 0.1`
- `max_depth = 3`
- `random_state = 42`

The model was evaluated using training accuracy, testing accuracy, and ROC-AUC.

### Results

| Metric | Value |
|---------|------:|
| Training Accuracy | **0.9889** |
| Test Accuracy | **0.9452** |
| ROC-AUC | **0.9857** |

### Interpretation

The Gradient Boosting model achieved a **training accuracy of 98.89%**, a **testing accuracy of 94.52%**, and a **ROC-AUC score of 0.9857**. These results indicate excellent classification performance and strong generalization on unseen data.

Unlike Random Forest, which builds many decision trees independently and combines their predictions using bagging, Gradient Boosting builds trees **sequentially**. Each new tree attempts to correct the prediction errors made by the previous trees, gradually improving the overall model performance. This boosting strategy allows the model to learn complex patterns while maintaining high predictive accuracy.

### Result

The Gradient Boosting Classifier achieved excellent predictive performance with a high ROC-AUC score and strong classification accuracy. The model will be included in the cross-validation comparison in Task 5 to determine its overall robustness relative to the other classification models.

## Task 4(b) – Feature Ablation Study

### Objective

The objective of this task is to evaluate the impact of the least important features identified by the Random Forest model. A second Random Forest model is trained after removing the five lowest-importance features to determine whether they contribute meaningfully to the model's predictive performance.

### Implementation

The five features with the lowest importance scores obtained from the Random Forest model in Task 4 were removed from both the training and testing datasets.

A second `RandomForestClassifier` with the same hyperparameters (`n_estimators=100`, `max_depth=10`, `random_state=42`) was trained using the reduced feature set. The ROC-AUC scores of both the full model and the reduced model were then compared.

### Results

#### Five Lowest-Importance Features

| Feature | Importance Score |
|---------|-----------------:|
| LotConfig_FR3 | **0.0000** |
| Street_Pave | **0.0000** |
| Condition2_RRNn | **0.0000** |
| Exterior1st_CBlock | **0.0000** |
| Condition2_RRAe | **0.0000** |

#### ROC-AUC Comparison

| Model | Test ROC-AUC |
|--------|-------------:|
| Full Model | **0.9844** |
| Reduced Model | **0.9852** |

### Interpretation

The reduced Random Forest model achieved a **ROC-AUC of 0.9852**, which is slightly higher than the **0.9844** obtained by the full model. This indicates that the five removed features were **uninformative** and contributed very little to the model's predictive performance.

Removing these features simplified the model without reducing classification performance. A lower-dimensional model requires **less computation during inference**, reduces **memory usage**, and lowers the **maintenance burden** in production. Since the ROC-AUC slightly improved after removing these features, the reduction in model complexity is acceptable because there is **no AUC degradation**. In general, deploying a simpler model is beneficial **only if any decrease in AUC remains below a tolerable threshold**, ensuring that prediction quality is not sacrificed for efficiency.

### Result

The feature ablation study demonstrated that removing the five least important features did not negatively affect model performance. The reduced model achieved a slightly higher ROC-AUC score, suggesting that these features mainly contributed noise rather than useful predictive information. This supports using a simpler and more efficient model in production.

## Task 5 – Cross-Validated Comparison

### Objective

The objective of this task is to compare the performance of multiple classification models using **5-fold Stratified Cross-Validation** with **ROC-AUC** as the evaluation metric. Cross-validation provides a more reliable estimate of model performance than a single train-test split.

### Implementation

The following models were evaluated using `cross_val_score()` with `cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and `scoring='roc_auc'`:

- Logistic Regression
- Controlled Decision Tree
- Random Forest
- Gradient Boosting

The mean ROC-AUC and standard deviation of the five folds were computed for each model.

### Results

| Model | Mean ROC-AUC | Standard Deviation |
|--------------------------|--------------:|------------------:|
| Logistic Regression | **0.9561** | **0.0082** |
| Controlled Decision Tree | **0.9243** | **0.0143** |
| Random Forest | **0.9771** | **0.0046** |
| Gradient Boosting | **0.9734** | **0.0084** |

### Interpretation

The **Random Forest** model achieved the highest **mean ROC-AUC (0.9771)** while also having the **lowest standard deviation (0.0046)**, indicating both excellent predictive performance and consistent results across different folds.

Cross-validation provides a more reliable estimate of model generalization than a single train-test split because every sample is used for both training and validation across different folds. This reduces the effect of random data partitioning and provides a more stable estimate of how the model is expected to perform on unseen data.

### Result

Among all evaluated models, **Random Forest** demonstrated the best overall performance, achieving the highest mean ROC-AUC with the lowest variability across folds. These results indicate that it is the most robust and reliable model among the models evaluated in this study.

## Task 6 – Hyperparameter Tuning with GridSearchCV

### Objective

The objective of this task is to optimize the Random Forest Classifier by searching for the best combination of hyperparameters using **GridSearchCV**. A machine learning pipeline was built to automate preprocessing and model training while identifying the best-performing model based on ROC-AUC.

### Implementation

A machine learning pipeline was created using:

- `SimpleImputer(strategy='median')`
- `StandardScaler()`
- `RandomForestClassifier(random_state=42)`

The following parameter grid was evaluated:

| Hyperparameter | Values |
|---------------|--------|
| n_estimators | 50, 100, 200 |
| max_depth | 5, 10, None |
| min_samples_leaf | 1, 5 |

`GridSearchCV` was performed using **5-fold Stratified Cross-Validation** with **ROC-AUC** as the scoring metric.

### Results

#### Best Hyperparameters

| Hyperparameter | Best Value |
|----------------|-----------|
| max_depth | **10** |
| min_samples_leaf | **1** |
| n_estimators | **200** |

#### Best Cross-Validation ROC-AUC

| Metric | Value |
|---------|------:|
| Best ROC-AUC | **0.9783** |

#### Grid Search Statistics

| Metric | Value |
|---------|------:|
| Total Model Configurations | **18** |
| Total Fits (5-Fold CV) | **90** |

### Interpretation

The parameter grid contained **18 unique model configurations** (3 values for `n_estimators` × 3 values for `max_depth` × 2 values for `min_samples_leaf`). Using **5-fold cross-validation**, GridSearchCV evaluated a total of **90 model fits** (18 × 5).

The best-performing model used **200 decision trees**, a **maximum tree depth of 10**, and **1 minimum sample per leaf**, achieving a **cross-validation ROC-AUC of 0.9783**.

GridSearchCV performs an **exhaustive search**, evaluating **every possible hyperparameter combination**, which guarantees that the best combination within the specified search space is found. However, this approach requires more computational time and resources. In contrast, **Randomized Search** evaluates only a random subset of hyperparameter combinations, making it much faster and more computationally efficient, but it does not guarantee finding the optimal combination.

### Result

Hyperparameter tuning successfully identified the optimal Random Forest configuration, resulting in the highest cross-validation ROC-AUC score. The tuned model provides a robust and well-optimized classifier that will be used in the remaining tasks.

## Task 7 – Manual Learning Curve

### Objective

The objective of this task is to analyze how the performance of the best model changes as the amount of training data increases. The learning curve helps determine whether the model is limited by the amount of training data or by its learning capacity.

### Implementation

The best pipeline obtained from **GridSearchCV** in Task 6 was trained using progressively larger subsets of the training data:

- 20%
- 40%
- 60%
- 80%
- 100%

For each training fraction, the **Training ROC-AUC** and **Test ROC-AUC** were calculated and compared.

### Results

| Training Fraction | Training ROC-AUC | Test ROC-AUC |
|------------------:|-----------------:|-------------:|
| 20% | **1.000000** | **0.979778** |
| 40% | **1.000000** | **0.982457** |
| 60% | **0.999992** | **0.984354** |
| 80% | **0.999986** | **0.984164** |
| 100% | **0.999941** | **0.984448** |

### Interpretation

The **Training ROC-AUC** decreased slightly from **1.0000** to **0.9999** as the training set size increased. This is expected because larger training datasets are more difficult to fit perfectly, reducing the likelihood of overfitting.

The **Test ROC-AUC** generally increased as more training data was used, improving from **0.9798** at 20% of the training data to **0.9844** at 100%. This indicates that the model benefits from additional training data, suggesting that collecting more high-quality data would likely improve predictive performance.

Although there is a very small fluctuation between the **60% (0.984354)** and **80% (0.984164)** training fractions, the **Test ROC-AUC reaches its highest value at the 100% training fraction (0.984448)**. Since the test performance is **still improving at 100%** rather than plateauing, the model is **currently limited by data quantity rather than model capacity**. Therefore, additional training data is likely to provide further performance improvements.

### Result

The manual learning curve demonstrates that the tuned Random Forest model maintains consistently high training performance while achieving gradual improvements in test performance as the training dataset grows. The results indicate good generalization, with minor potential gains achievable through additional training data.

## Task 8 – Serialize the Best Model

### Objective

The objective of this task is to save the best-performing machine learning pipeline to disk and verify that it can be successfully reloaded for future predictions without retraining.

### Implementation

The best pipeline obtained from **GridSearchCV** was serialized using the `joblib` library and saved as **`best_model.pkl`**.

The saved model was then reloaded using `joblib.load()`, and predictions were generated on two sample test rows to verify that the model was successfully restored and functioned correctly.

### Results

| Task | Status |
|------|--------|
| Model Saved | ✅ `best_model.pkl` |
| Model Reloaded | ✅ Successfully |
| Sample Predictions | **[0, 1]** |

### Interpretation

The model was successfully serialized and saved as **`best_model.pkl`**, allowing it to be reused without retraining. After reloading the saved model, predictions were successfully generated for two sample test instances, confirming that the serialization and deserialization process preserved the model correctly.

Model serialization enables faster deployment, reduces computational cost by avoiding repeated training, and ensures that the same trained model can be consistently used in production environments.

### Result

The best-performing machine learning pipeline was successfully saved, reloaded, and validated through sample predictions. This confirms that the serialized model is ready for deployment and future inference without requiring retraining.

## Task 9 – Summary Comparison Table

### Objective

The objective of this task is to compare the performance of all classification models developed in Parts 2 and 3 using their **5-fold Cross-Validation Mean ROC-AUC**, **5-fold Cross-Validation Standard Deviation**, and **Test-Set ROC-AUC**. Based on the comparison, the most suitable model for deployment is recommended.

### Implementation

The performance metrics obtained from the previous tasks were combined into a single comparison table. The models were evaluated based on:

- 5-Fold Cross-Validation Mean ROC-AUC
- 5-Fold Cross-Validation Standard Deviation
- Test-Set ROC-AUC

The model with the best balance of predictive performance and generalization capability was selected as the final recommendation.

### Results

| Model | 5-Fold CV Mean ROC-AUC | 5-Fold CV Std AUC | Test-Set ROC-AUC |
|---------------------------|----------------:|----------------:|---------------:|
| Logistic Regression | **0.9561** | **0.0082** | **0.9663** |
| Controlled Decision Tree | **0.9243** | **0.0143** | **0.9384** |
| Random Forest | **0.9771** | **0.0046** | **0.9844** |
| Gradient Boosting | **0.9734** | **0.0084** | **0.9857** |

### Interpretation

The **Random Forest** model achieved the **highest mean cross-validation ROC-AUC (0.9771)** and the **lowest standard deviation (0.0046)**, indicating the most stable and consistent performance across different folds. Although the **Gradient Boosting** model achieved a slightly higher **test-set ROC-AUC (0.9857)** than Random Forest (**0.9844**), its cross-validation mean ROC-AUC was lower (**0.9734**) and its variability across folds was higher (**0.0084**).

Considering both predictive performance and generalization ability, the **Random Forest** model provides the best balance between accuracy, robustness, and consistency.

### Client Recommendation

I recommend the **Random Forest** model for deployment because it achieved the **highest 5-fold cross-validation mean ROC-AUC (0.9771)** and the **lowest standard deviation (0.0046)** among all evaluated models, demonstrating excellent stability and consistency. Although the **Gradient Boosting** model achieved a slightly higher **test-set ROC-AUC (0.9857)**, the Random Forest model showed better overall generalization across multiple validation folds. It also provides feature importance scores, making the model easier to interpret and explain. Furthermore, the Random Forest model is robust against overfitting due to its bagging approach and random feature selection. Therefore, Random Forest is the most reliable and suitable model for deployment in a real-world production environment.

# Part 3 Summary

In this part, multiple advanced machine learning models were trained, evaluated, and compared to identify the most reliable classifier. Decision Tree, Random Forest, and Gradient Boosting models were analyzed using cross-validation, feature importance, feature ablation, hyperparameter tuning, and learning curve analysis.

The Random Forest model consistently demonstrated the best balance between predictive performance and generalization, achieving the highest mean cross-validation ROC-AUC with the lowest variability across folds. Hyperparameter tuning further optimized its performance, and the trained pipeline was successfully serialized as `best_model.pkl` for future deployment.

Overall, the advanced modeling pipeline produced a robust, well-optimized, and deployment-ready machine learning model capable of delivering reliable predictions on unseen data.
