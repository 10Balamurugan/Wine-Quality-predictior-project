# 🍷 Wine Quality Analyzer — ML Project

A complete Machine Learning project that predicts wine quality as **Good**, **Average**, or **Bad** using physicochemical properties. Built with Scikit-learn and a polished Streamlit UI.

---

## 📸 Features

- **3-Class Classification**: Bad (score ≤5) / Average (score 6) / Good (score ≥7)
- **Ensemble Model**: Random Forest + Gradient Boosting (Voting Classifier)
- **Interactive UI**: Adjust 12 wine parameters via sliders
- **Confidence Chart**: Probability bar chart for all 3 classes
- **Feature Importance**: Visual breakdown of what matters most
- **Chemistry Guide**: Explainer for every wine parameter

---

## 📁 Project Structure

```
wine_quality/
│
├── app.py               # Streamlit UI app
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
└── models/              # Auto-created after training
    ├── wine_model.pkl   # Trained Voting Ensemble model
    ├── scaler.pkl       # StandardScaler
    ├── feature_cols.pkl # Feature column names
    └── meta.pkl         # Accuracy & metadata
```

---

## ⚙️ Step-by-Step Setup Instructions

### Step 1 — Prerequisites

Make sure you have **Python 3.10+** installed.

```bash
python --version
```

### Step 2 — Create Virtual Environment

```bash
# Create venv
python -m venv wine_env

# Activate (Windows)
wine_env\Scripts\activate

# Activate (Mac/Linux)
source wine_env/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model

```bash
python train_model.py
```

**What this does:**
- Downloads the UCI Wine Quality dataset (red + white)
- Maps quality scores to 3 classes (Bad / Average / Good)
- Trains a Voting Ensemble (Random Forest + Gradient Boosting)
- Runs 5-Fold Cross-Validation
- Saves model artifacts to the `models/` folder

**Expected output:**
```
✅ Loaded 6497 rows from UCI (red=1599, white=4898)

📊 Class distribution:
   Bad (0<=5)  :  2384 (36.7%)
   Average (6) :  2836 (43.6%)
   Good (>=7)  :  1277 (19.7%)

🔧 Training ensemble model …

✅ Test Accuracy : 72.xx%

📋 Classification Report:
              precision    recall  f1-score   support
         Bad       0.72      0.71      0.71       477
     Average       0.73      0.78      0.75       567
        Good       0.72      0.63      0.67       256

🎉 Training complete!
```

> **Note:** If the UCI server is unreachable, the script automatically generates realistic synthetic data for training.

### Step 5 — Launch the App

```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## 🧪 How to Use the App

1. **Set Wine Type** (Red or White) in the sidebar
2. **Adjust the 11 chemical sliders** to describe your wine
3. Click **"🍷 Analyze Wine Quality"**
4. View the predicted quality label + confidence chart
5. Explore **Feature Insights** tab for importance rankings
6. Check **About Dataset** tab for chemistry reference

---

## 🤖 ML Pipeline

| Step | Detail |
|------|--------|
| Dataset | UCI Wine Quality (red + white, ~6500 samples) |
| Target | Quality score → 3 classes |
| Preprocessing | StandardScaler (z-score normalization) |
| Model | VotingClassifier (RF + GB, soft voting) |
| RF Config | 300 estimators, max_depth=20 |
| GB Config | 200 estimators, lr=0.08, max_depth=6 |
| Validation | Stratified 5-Fold CV |
| Metrics | Accuracy, Precision, Recall, F1-Score |

---

## 🍷 Quality Class Definition

| Class | Original Score | Label | Description |
|-------|---------------|-------|-------------|
| 0 | ≤ 5 | ❌ Bad Quality | Noticeable flaws; high acidity or poor balance |
| 1 | 6 | ⚠️ Average Quality | Acceptable; typical commercial wine |
| 2 | ≥ 7 | ✅ Good Quality | Well-balanced; recommended |

---

## 📊 Key Features & Their Importance

| Feature | Impact | Ideal Range |
|---------|--------|------------|
| Alcohol % | ⭐⭐⭐ High | 10–13% |
| Volatile Acidity | ⭐⭐⭐ High | < 0.6 g/L |
| Sulphates | ⭐⭐ Medium | 0.4–0.8 g/L |
| Citric Acid | ⭐⭐ Medium | 0.25–0.5 g/L |
| Total SO₂ | ⭐ Medium | < 150 mg/L |
| Residual Sugar | ⭐ Low | Varies by style |
| Chlorides | ⭐ Low | < 0.1 g/L |
| pH | ⭐ Low | 3.0–3.5 |

---

## 📚 Dataset Reference

> P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis.
> *Modeling wine preferences by data mining from physicochemical properties.*
> In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

Dataset URL: https://archive.ics.uci.edu/ml/datasets/wine+quality

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Web UI
- **Scikit-learn** — ML models
- **Pandas / NumPy** — Data processing
- **Matplotlib** — Visualization
- **Joblib** — Model serialization

---

## 🚀 Future Improvements

- [ ] Add SHAP explainability (per-prediction feature contributions)
- [ ] Upload a CSV file to batch-predict multiple wines
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Deploy to Streamlit Cloud / Hugging Face Spaces
- [ ] Add XGBoost for comparison

---

*Built as part of Data Science curriculum at Besant Technologies, Bangalore 🇮🇳*
