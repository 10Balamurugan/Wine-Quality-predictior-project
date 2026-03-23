"""
Wine Quality ML Model Trainer
Trains Random Forest + Gradient Boosting models on UCI Wine Quality dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    """Load wine quality dataset (red + white combined)"""
    red_url   = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

    try:
        red   = pd.read_csv(red_url,   sep=';')
        white = pd.read_csv(white_url, sep=';')
        red['wine_type']   = 0   # red
        white['wine_type'] = 1   # white
        df = pd.concat([red, white], ignore_index=True)
        print(f"✅ Loaded {len(df)} rows from UCI (red={len(red)}, white={len(white)})")
    except Exception as e:
        print(f"⚠️  Could not fetch from UCI ({e}). Generating synthetic data …")
        df = generate_synthetic_data()

    return df


def generate_synthetic_data(n=6000):
    """Fallback: generate realistic synthetic wine data"""
    np.random.seed(42)
    data = {
        'fixed acidity':        np.random.normal(7.5,  1.5,  n),
        'volatile acidity':     np.random.normal(0.35, 0.15, n),
        'citric acid':          np.random.normal(0.30, 0.15, n),
        'residual sugar':       np.random.normal(5.5,  4.0,  n),
        'chlorides':            np.random.normal(0.055,0.02, n),
        'free sulfur dioxide':  np.random.normal(30,   15,   n),
        'total sulfur dioxide': np.random.normal(115,  55,   n),
        'density':              np.random.normal(0.995,0.003,n),
        'pH':                   np.random.normal(3.22, 0.15, n),
        'sulphates':            np.random.normal(0.53, 0.15, n),
        'alcohol':              np.random.normal(10.5, 1.1,  n),
        'wine_type':            np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    quality = (
        0.3 * df['alcohol']
        - 0.5 * df['volatile acidity']
        + 0.2 * df['sulphates']
        + 0.1 * df['citric acid']
        + np.random.normal(0, 0.5, n)
    )
    quality_scaled = ((quality - quality.min()) / (quality.max() - quality.min()) * 9 + 1).round().astype(int)
    df['quality'] = quality_scaled.clip(3, 9)
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df):
    """Map raw quality scores → 3-class labels"""
    def label(q):
        if q <= 5:   return 0   # Bad
        elif q <= 6: return 1   # Average
        else:        return 2   # Good

    df = df.copy()
    df['quality_label'] = df['quality'].apply(label)

    feature_cols = [
        'fixed acidity','volatile acidity','citric acid','residual sugar',
        'chlorides','free sulfur dioxide','total sulfur dioxide',
        'density','pH','sulphates','alcohol','wine_type'
    ]
    # keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df['quality_label']

    print(f"\n📊 Class distribution:\n"
          f"   Bad (0<=5)  : {(y==0).sum():>5} ({(y==0).mean()*100:.1f}%)\n"
          f"   Average (6) : {(y==1).sum():>5} ({(y==1).mean()*100:.1f}%)\n"
          f"   Good (>=7)  : {(y==2).sum():>5} ({(y==2).mean()*100:.1f}%)")

    return X, y, feature_cols


# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  min_samples_split=4, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                      max_depth=5, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)],
                                 voting='soft')

    print("\n🔧 Training ensemble model …")
    ensemble.fit(X_train_s, y_train)

    y_pred = ensemble.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy : {acc*100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=['Bad','Average','Good']))

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train_s, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"🔁 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return ensemble, scaler, X_test_s, y_test, y_pred, acc


# ─────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, feature_cols, acc):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model,        'models/wine_model.pkl')
    joblib.dump(scaler,       'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    joblib.dump({'accuracy': acc, 'classes': ['Bad','Average','Good'],
                 'version': '1.0'}, 'models/meta.pkl')
    print("\n💾 Saved: models/wine_model.pkl  |  models/scaler.pkl  |  models/meta.pkl")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    X, y, feature_cols = preprocess(df)
    model, scaler, X_test_s, y_test, y_pred, acc = train(X, y)
    save_artifacts(model, scaler, feature_cols, acc)
    print("\n🎉 Training complete!")
