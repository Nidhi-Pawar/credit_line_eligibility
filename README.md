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
- Includes customer attributes like `annual income`, `debt-to-income ratio`, `purpose`, `employment length`, `revolving balance`, and more.

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
├───run_pipeline.py        
        
        
        
        
        

```




























