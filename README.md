# 🍷 Wine Quality Analysis — Machine Learning Project

A complete end-to-end ML project that predicts wine quality as **Good 🟢**, **Average 🟡**, or **Bad 🔴** using physicochemical properties of red wine.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Dataset** | UCI Wine Quality (Red Wine) — 1,599 samples |
| **Task** | Multi-class Classification (3 classes) |
| **Models** | Logistic Regression, Random Forest, XGBoost, SVM |
| **Best Model** | Random Forest / XGBoost (~85–88% accuracy) |
| **UI** | Streamlit Web App |
| **Notebook** | Jupyter Notebook (step-by-step) |

---

## 🏷️ Quality Labels

| Label | Score | Description |
|-------|-------|-------------|
| 🟢 **Good** | ≥ 7 | Premium wine with excellent taste and structure |
| 🟡 **Average** | 5–6 | Standard quality, the most common category |
| 🔴 **Bad** | ≤ 4 | Below average quality with noticeable flaws |

---

## 📁 Project Structure

```
wine_quality_project/
│
├── wine_quality_analysis.ipynb   ← Full Jupyter Notebook (10 steps)
├── app.py                        ← Streamlit Web App UI
├── requirements.txt              ← All Python dependencies
├── README.md                     ← This file
│
└── model/                        ← Auto-created after running notebook
    ├── best_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── feature_cols.pkl
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone / Download the project

```bash
# If using git
git clone <your-repo-url>
cd wine_quality_project

# Or just unzip the downloaded folder
cd wine_quality_project
```

### Step 2 — Create a virtual environment (recommended)

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### ▶️ Option A: Jupyter Notebook (Step-by-Step)

```bash
jupyter notebook wine_quality_analysis.ipynb
```

Run all cells in order:
1. Import Libraries
2. Load Dataset
3. Data Cleaning
4. EDA (Exploratory Data Analysis)
5. Preprocessing & Feature Engineering
6. Model Building
7. Model Evaluation
8. Save Model
9. Predict New Sample
10. Final Summary

---

### ▶️ Option B: Streamlit Web App

> **Important:** Run the Jupyter Notebook first to train and save the model,  
> OR the app will train the model automatically on first launch.

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

**App Features:**
- 🔮 **Predict Tab** — Use sliders to input wine properties and get instant predictions
- 📊 **Data Explorer Tab** — Visualize the dataset, correlations, and feature distributions
- ℹ️ **About Tab** — Project info and structure

---

## 📊 Features (Input Variables)

| Feature | Unit | Description |
|---------|------|-------------|
| Fixed Acidity | g/L | Non-volatile acids (tartaric acid) |
| Volatile Acidity | g/L | Acetic acid content — high = vinegar taste |
| Citric Acid | g/L | Adds freshness and flavor |
| Residual Sugar | g/L | Sugar remaining after fermentation |
| Chlorides | g/L | Salt content |
| Free Sulfur Dioxide | mg/L | Prevents microbial growth and oxidation |
| Total Sulfur Dioxide | mg/L | Bound + free SO₂ forms |
| Density | g/mL | Related to sugar and alcohol content |
| pH | — | Acidity scale (lower = more acidic) |
| Sulphates | g/L | Potassium sulphate — antimicrobial |
| Alcohol | % vol | Alcohol by volume |

**Engineered Features (auto-created):**
- `alcohol_sulphates` = alcohol × sulphates
- `acid_ratio` = fixed acidity / volatile acidity
- `sulfur_ratio` = free SO₂ / total SO₂

---

## 🤖 ML Pipeline Summary

```
Raw Data (1599 × 12)
     ↓
Data Cleaning
  • Remove duplicates
  • Winsorize outliers (1st–99th percentile)
     ↓
EDA
  • Correlation heatmap
  • Quality distribution
  • Feature violin plots
     ↓
Preprocessing
  • Label encoding (Bad=0, Average=1, Good=2)
  • Feature engineering (3 new features)
  • Train/Test split (80/20, stratified)
  • StandardScaler normalization
     ↓
Model Training (4 models)
  • Logistic Regression
  • Random Forest ← Usually best
  • XGBoost
  • SVM
     ↓
Evaluation
  • Accuracy, F1-Score
  • 5-Fold Cross Validation
  • Confusion Matrix
  • Feature Importance
     ↓
Save & Deploy (Streamlit UI)
```

---

## 📈 Expected Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~72% | ~0.71 |
| SVM | ~76% | ~0.75 |
| XGBoost | ~84% | ~0.83 |
| **Random Forest** | **~86%** | **~0.85** |

*Results may vary slightly based on dataset version and random seed.*

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.23 | Numerical operations |
| `scikit-learn` | ≥1.1 | ML models & preprocessing |
| `xgboost` | ≥1.7 | Gradient boosting |
| `matplotlib` | ≥3.6 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualization |
| `streamlit` | ≥1.20 | Web UI |
| `joblib` | ≥1.2 | Model serialization |

---

## 💡 Tips for Windows Users

If you get errors running pip or Python, always use:
```cmd
python -m pip install <package>
python -m jupyter notebook
python -m streamlit run app.py
```

If you get PowerShell execution policy errors:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📚 Dataset Reference

**UCI Machine Learning Repository — Wine Quality**  
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
*Modeling wine preferences by data mining from physicochemical properties.*  
Decision Support Systems, Elsevier, 47(4):547-553, 2009.

---

## 🎓 Learning Outcomes

After completing this project you will understand:
- End-to-end ML project workflow
- Data cleaning and outlier treatment
- Feature engineering techniques
- Model comparison and selection
- Deploying ML models with Streamlit
- Saving and loading trained models with joblib

---

*Built with ❤️ as part of Data Science & AI/ML learning at Besant Technologies*
