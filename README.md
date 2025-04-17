# 🏦 Credit Line Eligibility
---

## 🚀 Project Highlights

- ✅ Predicts whether a loan will be **Fully Paid** or **Charged Off**
- ✅ Handles **heavily imbalanced data** with targeted class balancing techniques
- ✅ Uses **multiple ML models** including RandomForest, XGBoost, LightGBM, and CatBoost
- ✅ Integrates **Optuna** for hyperparameter tuning
- ✅ Tracks experiments using **MLflow**
- ✅ Built with a clean **modular architecture** and orchestrated via **ZenML pipeline**

---
 ## 📊 Dataset Overview

- Rows: **395,000+**
- Features: **27 columns** (anonymized)
- Target Variable: `loan_status` → values: `Fully Paid` (majority), `Charged Off` (minority)
- Class Distribution: ~**77% majority** class, ~**23% minority** class  
- Includes customer attributes like `annual income`, `loan amount`, `purpose`, `interest rate`, `revolving balance`, and many more.

---

## 🧰 Tech Stack

| Tool            | Purpose                               |
|-----------------|----------------------------------------|
| **Pandas / NumPy** | Data manipulation and preprocessing |
| **Scikit-learn** | ML models, metrics, preprocessing     |
| **XGBoost / CatBoost / LightGBM** | Advanced ML algorithms |
| **Optuna**       | Hyperparameter tuning (automated)     |
| **MLflow**       | Experiment tracking and model logging |
| **ZenML**        | ML pipeline orchestration             |

---

## 📁 Project Structure

```
credit_line_eligibility
│
├───data
│       cleaned_data.csv
│       credit_eligibility.csv
│
├───notebooks
│       01_Data-cleaning.ipynb
│       02_Exploratory-Data-Analysis.ipynb
│       03_Feature-Engineering.ipynb
│       04_Model-Training.ipynb
│       
│
├───pipelines
│       training_pipeline.py
│
├───src
│   │   config.yaml
│   │   data_cleaning.py
│   │   model_evaluation.py
│   │
│   ├───core_utils
│   │       config_loader.py
│   │       __init__.py
│   │
│   └───models
│           base_trainer.py
│           catboost_model.py
│           lgbm_model.py
│           xgb_model.py
│
├───steps
│       clean_data.py
│       evaluation.py
│       ingest_data.py
│       train_model.py
│       config.py
│       __init__.py
│
├───requirements.txt
└───run_pipeline.py        
        
```
---

## 📈 Model Performance

| Classifier Models            | Data Balancing | Recall (Class 0) | Recall (Class 1) | ROC AUC Score |
|------------------|----------------|------------------|------------------|---------------|
| XGBoost| Non-balanced   | 0.66            | 0.66            | 0.658         |
| LightBGM| Non-balanced           | 0.66             | 0.65             | 0.657          |    
| CatBoost | Non-balanced           | 0.66            | 0.66             | 0.656          |
| XGBoost | SMOTE Balanced  | 0.74             | 0.54             | 0.640         |
| LightBoost | SMOTE Balanced            | 0.32           | 0.88       | 0.600       |
| CatBoost| SMOTE Balanced            | 0.72            | 0.60        | 0.655        |
| XGBoost| CV* SMOTE          | 0.71             | 0.56             | 0.637         |
| LightGBM | CV* SMOTE          | 0.76            | 0.52            | 0.641         |
| CatBoost | CV* SMOTE           | 0.69            | 0.61             | 0.637         |

- CV* : SMOTE was applied on training data passed through Cross Validation folds individually
> ⚖️ Focus was on improving **recall for minority class (Class 0)** while keeping overall performance balanced.

---

## 🧠 Key Learnings

- Handling severe class imbalance using SMOTE, class weights, and calibrated classifiers
- Building modular, reusable ML code using OOP principles
- Hyperparameter tuning with Optuna and visualizing metrics with MLflow
- Creating a fully operational ML pipeline using ZenML

---

## 🚀 Future Work 

- Add model deployment using Streamlit or Flask
- Set up Docker + CI/CD pipeline for full reproducibility

--- 
## 🙌 Acknowledgements

Thanks to the open-source tools and the data providers who made this analysis possible! As well as countless articles and research papers on Data Imbalance Handling, and Stack Exchange!

--- 
## 🔗 Connect With Me

Feel free to reach out if you'd like to chat about ML, this project, or anything tech-related!
💼 linkedin.com/in/nidhipawar810 
📫 nidhipawar.810@gmail.com


























