# Practical_Application_20.1
This repository uses EDA to see what data can reveal beyond the formal modeling, hypothesis testing task, and data training to provide a better understanding of dataset variables and the relationships between them.
# Tax Lien Redemption Probability – Initial Report and EDA

## Research Question
**What is the probability that this tax lien will be redeemed within X months?**

## Project Summary
For this module, I focused on building the initial exploratory data analysis and baseline model for my capstone project. My long-term goal is to predict whether a tax lien will be redeemed, because redemption probability is one of the most important drivers of expected return, capital lock-up, and portfolio risk in a tax lien investing strategy.

Since publicly available tax lien redemption datasets are limited, I used the **Home Credit Default Risk** dataset from Kaggle as a **proxy dataset** for this stage of the project. I treated **loan repayment** as a proxy for **tax lien redemption** and **loan default** as a proxy for **non-redemption**. This is not a perfect substitute for real lien data, but it gives me a practical way to test the data science workflow and evaluate whether this type of problem is learnable.

## Dataset
- **Dataset used:** Home Credit Default Risk
- **File used in this module:** `application_train.zip` / `application_train.csv`
- **Source:** Kaggle Home Credit Default Risk competition - https://www.kaggle.com/c/home-credit-default-risk
- The dataset contains borrower information such as income, loan amount, credit history, and demographic characteristics.
- In this repository the dataset is stored in a compressed **7z archive** (`application_train.7z`).
- To reduce repository size and improve download speed, the dataset will need to be extracted before loading the CSV file into pandas.

## Methodology Update
Compared with the earlier project concept, this module uses a **proxy repayment dataset** rather than true county tax lien redemption data. I made this change because:

1. the Kaggle dataset is easier to work with in a structured ML workflow,
2. it includes a large number of observations and relevant financial variables,
3. it supports the main goal of this module, which is EDA plus a baseline model.

Later modules can move closer to real tax lien data and property-specific features.

## What I did in the notebook
The notebook includes:

- initial inspection of dataset size, types, and target balance
- duplicate checking
- missing value analysis
- visualizations for both numeric and categorical variables
- outlier analysis using the IQR rule
- feature engineering for repayment stress indicators
- baseline modeling with logistic regression
- evaluation using ROC-AUC

## Key EDA Findings
Some of the main patterns I found were:

- The dataset contains **307,511 rows** and **122 columns**, so it is large enough for a meaningful baseline analysis.
- The target is **imbalanced**: about **8.07%** of observations are defaults and about **91.93%** are non-defaults.
- There were **no duplicate rows**.
- A large number of columns had substantial missingness, so I removed features with **more than 40% missing values** for this first-pass baseline workflow.
- Core financial variables such as `AMT_INCOME_TOTAL`, `AMT_CREDIT`, and `AMT_ANNUITY` are strongly skewed and contain outliers, which I documented and handled carefully in visualizations.
- Default rates were generally higher for more financially vulnerable groups, such as applicants in rented housing or lower education categories.

## Feature Engineering
I created several derived features to better capture repayment stress:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `GOODS_CREDIT_RATIO`
- `EXT_SOURCE_MEAN`
- `YEARS_BIRTH`
- `YEARS_EMPLOYED`
- `DAYS_EMPLOYED_ANOM`

These features were designed to make the repayment problem more interpretable and move the project closer to the type of engineered variables I would want in a real tax lien model.

## Baseline Model
I used **logistic regression** as the baseline model.

### Why logistic regression?
I selected logistic regression because it is:
- a standard baseline for binary classification,
- reasonably interpretable,
- efficient for a first model,
- useful as a benchmark before testing more advanced models later.

## Evaluation Metric
The main evaluation metric was **ROC-AUC**.

### Why ROC-AUC?
I chose ROC-AUC because:
- the classes are imbalanced,
- the project is focused on ranking risk rather than relying on one arbitrary threshold,
- it is a strong baseline metric for binary classification.

## Initial Results
This initial EDA and baseline modeling output showed:
**- ROC-AUC: 0.6678**
**- PR-AUC: 0.1519**
**- Positive-class rate: 8.07%**

My evaluation is that 0.6678 is modest, not poor, but not strong. It is clearly better than random guessing at 0.50, so the model is learning signal, but it is not yet competitive for a high-quality credit-risk style classifier. On an imbalanced problem like this one, the PR-AUC of 0.1519 matters a lot too; since the baseline event rate is 0.0807, the model is doing better than random ranking, but only by about 1.9× on precision-recall.

The ROC-AUC score of 0.6678 indicates **the model learned a meaningful signal but still has substantial room for improvement.** The main limitation is likely feature exclusion, especially the omission of the EXT_SOURCE_* and EXT_SOURCE_MEAN variables, which are known to be highly informative in this dataset.

To improve the ROC-AUC results, my plan for the next iteration to ensure the scores are actually reliable (and not due to chance), I will retain the omitted variables and use a proper imputation strategy to handle the missing values.

- Switch to Stratified Cross-Validation: This will give a much more honest look at how the model handles the class imbalance.
- Benchmark different architectures: I want to see how a tuned Logistic Regression holds up against more complex boosting models like XGBoost or LightGBM.

## Interpretation
The baseline result supports the broader capstone idea that **probability-based risk scoring is feasible**. Even though this dataset is only a proxy, the workflow already shows that financial behavior can be modeled with enough signal to distinguish lower-risk from higher-risk cases.

That gives me a reasonable foundation for the later stages of the project, where I plan to move closer to the true research question using more tax-lien-specific variables such as:

- lien amount
- property value
- lien-to-value ratio
- delinquency duration
- owner occupancy
- neighborhood conditions

## Future Work
The next stage of the project will include:

- stronger models such as **XGBoost**
- **SHAP** explainability for local and global interpretation
- more advanced feature engineering
- additional proxy or property-level datasets
- eventual transition toward a more tax-lien-specific dataset design

## Notebook Link
[Open the notebook](./notebooks/tax_lien_redemption_eda_baseline.ipynb)
