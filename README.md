# Credit Risk Modeling & Interactive Prediction App

An end-to-end data analytics and machine learning project that analyzes historical credit applicant data to identify default risk drivers, benchmarks four tree-based classifiers, and deploys a real-time credit risk scoring web app.

🔗 **Live Interactive App:** [Launch Credit Risk Prediction App](https://xswaraj-credit-risk-modeling-app-uiqy5m.streamlit.app/)

---

## Business Problem
Lenders must balance loan approval volume against default risk. Manual credit evaluation is time-consuming and can miss multi-variable interactions between loan size, repayment term, and applicant liquidity. This project analyzes the German Credit Risk dataset to uncover key risk patterns and provides a self-service web tool that classifies loan applications as **Good (Lower Risk)** or **Bad (Higher Risk)**.

---

## Dataset & Preprocessing
* **Source:** German Credit Risk dataset (`German_Credit_Data.csv`) — `1,000` raw records, `11` columns (`0` duplicates).
* **Missing Value Handling:** Dropped records with missing `Saving accounts` or `Checking account` history, yielding **`522` complete records** (`291` Good / `55.75%`, `231` Bad / `44.25%`).
* **Feature Selection & Encoding:** Selected `8` predictive features (`Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`). Categorical features and target labels (`1` = Good, `0` = Bad) were encoded using `LabelEncoder` and serialized with `Joblib`.
* **Train-Test Split:** 80/20 stratified split (`417` training rows, `105` test rows) with balanced class weights to handle target imbalance.

---

## Key Analytical Insights
* **Loan Duration & Size Drive Risk:** `Credit amount` and `Duration` show the strongest positive correlation (`0.61`). High-risk (`Bad`) loans average **3,881** over **25.45 months**, compared to **2,801** over **18.08 months** for low-risk (`Good`) loans.
* **Long-Term Loan Exposure:** Every loan in the cleaned dataset with a duration of **60+ months** (`7` of `7` records) was classified as **Bad** risk.
* **Liquidity & Employment Segments:** Applicants with `rich` or `moderate` checking and savings balances show noticeably lower default rates than those with `little` liquidity. Average requested credit scales steadily with job tier (`Job 0`: **1,768** → `Job 3`: **5,649**).

---

## Model Benchmarking & Results
Four tree-based models were evaluated using 5-fold `GridSearchCV` hyperparameter tuning (scoring on accuracy):

| Model | Test Accuracy | Best Hyperparameters |
| :--- | :---: | :--- |
| Decision Tree | 58.10% | `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=1` |
| **Extra Trees (Deployed)** | **64.76%** | `n_estimators=100`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2` |
| Random Forest | 66.67% | `n_estimators=200`, `max_depth=7`, `min_samples_split=5`, `min_samples_leaf=2` |
| XGBoost | 66.67% | `n_estimators=100`, `learning_rate=0.2`, `subsample=0.7`, `colsample_bytree=0.7` |

*(Note: Extra Trees is currently deployed to the Streamlit app for fast inference and low variance; future iterations can swap in the tuned Random Forest model and optimize for Recall/ROC-AUC on high-risk defaults.)*

---

## Project Structure
```text
Credit_Risk_Modeling/
├── analysis_model.ipynb            # Full EDA, feature engineering, and model tuning
├── app.py                          # Streamlit web application for live predictions
├── German_Credit_Data.csv          # Raw dataset
├── extra_trees_credit_model.pkl    # Serialized Extra Trees classifier
├── *_encoder.pkl                   # Saved LabelEncoders for categorical inputs
├── requirements.txt                # Python dependencies for Streamlit Cloud
└── README.md                       # Project documentation
```
---

## Tech Stack
- **Data Analysis & Visualization:** Python, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning & Serialization:** Scikit-learn, XGBoost, Joblib
- **Deployment:** Streamlit & Streamlit Community Cloud