"""
🍷 Wine Quality Prediction — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.6rem; font-weight:800; color:#8B0000; text-align:center; margin-bottom:0.2rem; }
    .sub-title   { font-size:1.1rem; color:#666; text-align:center; margin-bottom:1.5rem; }
    .metric-card { background:#f8f4f0; border-radius:12px; padding:1.2rem 1rem;
                   border-left:5px solid #8B0000; margin:0.4rem 0; }
    .good-badge  { background:#d4edda; color:#155724; padding:0.4rem 1rem;
                   border-radius:20px; font-weight:700; font-size:1.1rem; }
    .avg-badge   { background:#fff3cd; color:#856404; padding:0.4rem 1rem;
                   border-radius:20px; font-weight:700; font-size:1.1rem; }
    .bad-badge   { background:#f8d7da; color:#721c24; padding:0.4rem 1rem;
                   border-radius:20px; font-weight:700; font-size:1.1rem; }
    .stSlider > label { font-weight: 600; }
    div[data-testid="metric-container"] { background:#fff8f5; border-radius:10px; padding:0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load / Train Model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🍷 Loading model...")
def load_or_train_model():
    if os.path.exists("model/best_model.pkl"):
        model   = joblib.load("model/best_model.pkl")
        scaler  = joblib.load("model/scaler.pkl")
        le      = joblib.load("model/label_encoder.pkl")
        f_cols  = joblib.load("model/feature_cols.pkl")
        return model, scaler, le, f_cols, "Loaded from disk"

    # Train on-the-fly
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split

    try:
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            sep=";"
        )
    except Exception:
        np.random.seed(42)
        n = 1599
        df = pd.DataFrame({
            "fixed acidity": np.round(np.random.normal(8.32, 1.74, n), 2),
            "volatile acidity": np.round(np.random.normal(0.53, 0.18, n), 2),
            "citric acid": np.round(np.random.normal(0.27, 0.19, n), 2),
            "residual sugar": np.round(np.random.normal(2.54, 1.41, n), 2),
            "chlorides": np.round(np.random.normal(0.087, 0.047, n), 3),
            "free sulfur dioxide": np.round(np.random.normal(15.87, 10.46, n), 1),
            "total sulfur dioxide": np.round(np.random.normal(46.47, 32.9, n), 1),
            "density": np.round(np.random.normal(0.9967, 0.0019, n), 4),
            "pH": np.round(np.random.normal(3.31, 0.154, n), 2),
            "sulphates": np.round(np.random.normal(0.658, 0.17, n), 2),
            "alcohol": np.round(np.random.normal(10.42, 1.07, n), 1),
            "quality": np.random.choice([3,4,5,6,7,8], n, p=[0.006,0.033,0.426,0.399,0.124,0.012])
        })

    def label_quality(s):
        if s >= 7: return "Good"
        elif s >= 5: return "Average"
        else: return "Bad"

    df["quality_label"] = df["quality"].apply(label_quality)
    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
    df["acid_ratio"]        = df["fixed acidity"] / (df["volatile acidity"] + 0.001)
    df["sulfur_ratio"]      = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 0.001)

    base_cols = [c for c in df.columns if c not in ("quality", "quality_label")]
    f_cols    = base_cols

    le = LabelEncoder()
    le.fit(["Bad", "Average", "Good"])
    y  = le.transform(df["quality_label"])
    X  = df[f_cols]

    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model,  "model/best_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le,     "model/label_encoder.pkl")
    joblib.dump(f_cols, "model/feature_cols.pkl")

    return model, scaler, le, f_cols, "Trained fresh"


# ── Load Dataset for EDA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            sep=";"
        )
    except Exception:
        np.random.seed(42)
        n = 1599
        df = pd.DataFrame({
            "fixed acidity": np.round(np.random.normal(8.32, 1.74, n), 2),
            "volatile acidity": np.round(np.random.normal(0.53, 0.18, n), 2),
            "citric acid": np.round(np.random.normal(0.27, 0.19, n), 2),
            "residual sugar": np.round(np.random.normal(2.54, 1.41, n), 2),
            "chlorides": np.round(np.random.normal(0.087, 0.047, n), 3),
            "free sulfur dioxide": np.round(np.random.normal(15.87, 10.46, n), 1),
            "total sulfur dioxide": np.round(np.random.normal(46.47, 32.9, n), 1),
            "density": np.round(np.random.normal(0.9967, 0.0019, n), 4),
            "pH": np.round(np.random.normal(3.31, 0.154, n), 2),
            "sulphates": np.round(np.random.normal(0.658, 0.17, n), 2),
            "alcohol": np.round(np.random.normal(10.42, 1.07, n), 1),
            "quality": np.random.choice([3,4,5,6,7,8], n, p=[0.006,0.033,0.426,0.399,0.124,0.012])
        })

    def label_quality(s):
        if s >= 7: return "Good"
        elif s >= 5: return "Average"
        else: return "Bad"

    df["quality_label"] = df["quality"].apply(label_quality)
    return df


# ── Prediction Helper ─────────────────────────────────────────────────────────
def predict_quality(sample_dict, model, scaler, le, f_cols):
    row = pd.DataFrame([sample_dict])
    row["alcohol_sulphates"] = row["alcohol"] * row["sulphates"]
    row["acid_ratio"]        = row["fixed acidity"] / (row["volatile acidity"] + 0.001)
    row["sulfur_ratio"]      = row["free sulfur dioxide"] / (row["total sulfur dioxide"] + 0.001)
    X_in   = scaler.transform(row[f_cols])
    pred   = model.predict(X_in)[0]
    proba  = model.predict_proba(X_in)[0]
    label  = le.inverse_transform([pred])[0]
    probs  = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    return label, probs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
model, scaler, le, f_cols, model_status = load_or_train_model()
df = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🍷 Wine Quality Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Classify wine as Good 🟢 · Average 🟡 · Bad 🔴 using Machine Learning</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Navigation ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Quality", "📊 Data Explorer", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🧪 Enter Wine Chemical Properties")
    st.caption("Adjust the sliders to match your wine's lab measurements, then click Predict.")

    col_left, col_right, col_result = st.columns([1, 1, 1])

    with col_left:
        st.markdown("**🧪 Acidity & Acids**")
        fixed_acidity     = st.slider("Fixed Acidity",     4.0, 16.0, 8.32, 0.1,
                                       help="g/L tartaric acid equivalent")
        volatile_acidity  = st.slider("Volatile Acidity",  0.1, 1.6,  0.53, 0.01,
                                       help="g/L acetic acid — too high = vinegar taste")
        citric_acid       = st.slider("Citric Acid",        0.0, 1.0,  0.27, 0.01,
                                       help="g/L — adds freshness and flavor")
        pH                = st.slider("pH",                  2.7, 4.0,  3.31, 0.01,
                                       help="Acidity scale — lower = more acidic")

    with col_right:
        st.markdown("**🧂 Sulfur & Other**")
        residual_sugar    = st.slider("Residual Sugar",     0.5, 16.0, 2.54, 0.1,
                                       help="g/L — remaining sugar after fermentation")
        chlorides         = st.slider("Chlorides",          0.01, 0.6,  0.087, 0.001,
                                       help="g/L sodium chloride")
        free_sulfur       = st.slider("Free Sulfur Dioxide",1.0, 70.0, 15.87, 0.5,
                                       help="mg/L — prevents microbial growth & oxidation")
        total_sulfur      = st.slider("Total Sulfur Dioxide",5.0, 300.0,46.47, 1.0,
                                       help="mg/L SO₂ — bound + free forms")
        density           = st.slider("Density",            0.990, 1.005, 0.9967, 0.0001,
                                       help="g/mL — related to sugar & alcohol content")
        sulphates         = st.slider("Sulphates",          0.2, 2.0,  0.658, 0.01,
                                       help="g/L potassium sulphate — antimicrobial agent")
        alcohol           = st.slider("Alcohol (%)",        8.0, 15.0, 10.42, 0.1,
                                       help="% vol — alcohol by volume")

    # ── Predict Button ───────────────────────────────────────────────────────
    with col_result:
        st.markdown("**🏷️ Prediction Result**")

        sample = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_sulfur,
            "total sulfur dioxide": total_sulfur,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol,
        }

        if st.button("🔮 Predict Wine Quality", use_container_width=True, type="primary"):
            label, probs = predict_quality(sample, model, scaler, le, f_cols)

            # Badge
            emoji_map = {"Good": ("🟢", "good-badge"), "Average": ("🟡", "avg-badge"), "Bad": ("🔴", "bad-badge")}
            emoji, badge = emoji_map[label]
            st.markdown(f'<div style="text-align:center; margin:1rem 0;">'
                        f'<span class="{badge}">{emoji} {label} Quality Wine</span></div>',
                        unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Confidence Breakdown:**")
            order = ["Good", "Average", "Bad"]
            colors_map = {"Good": "#27ae60", "Average": "#f39c12", "Bad": "#e74c3c"}
            for cls in order:
                p = probs.get(cls, 0)
                st.markdown(f"{emoji_map[cls][0]} **{cls}**")
                st.progress(p, text=f"{p*100:.1f}%")

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 3.5))
            cls_names = [f"{emoji_map[c][0]} {c}" for c in order]
            values    = [probs.get(c, 0) * 100 for c in order]
            bar_colors = [colors_map[c] for c in order]
            bars = ax.barh(cls_names, values, color=bar_colors, height=0.5, edgecolor="white")
            for bar, v in zip(bars, values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{v:.1f}%", va="center", fontweight="bold", fontsize=11)
            ax.set_xlim(0, 115)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Classification Confidence", fontweight="bold")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.info("👆 Adjust sliders and click **Predict Wine Quality**")

            # Show input summary
            st.markdown("**Current Values:**")
            for k, v in sample.items():
                st.markdown(f"- `{k}`: **{v}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Dataset Explorer")

    # Dataset metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Samples", df.shape[0])
    with c2: st.metric("Features",      df.shape[1] - 2)
    with c3: st.metric("🟢 Good Wines",  (df["quality_label"]=="Good").sum())
    with c4: st.metric("🔴 Bad Wines",   (df["quality_label"]=="Bad").sum())

    st.markdown("---")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("**Quality Score Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["quality"].value_counts().sort_index()
        color_map = {3:"#c0392b",4:"#e74c3c",5:"#f39c12",6:"#f1c40f",7:"#27ae60",8:"#1e8449"}
        bars = ax.bar(counts.index, counts.values,
                      color=[color_map.get(i,"#3498db") for i in counts.index],
                      edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                    str(v), ha="center", fontweight="bold", fontsize=9)
        ax.set_xlabel("Quality Score"); ax.set_ylabel("Count")
        ax.set_title("Wine Quality Distribution", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with plot_col2:
        st.markdown("**Category Breakdown**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cats = df["quality_label"].value_counts()
        colors_pie = {"Good":"#27ae60","Average":"#f39c12","Bad":"#e74c3c"}
        ax.pie(cats.values, labels=cats.index,
               colors=[colors_pie[c] for c in cats.index],
               autopct="%1.1f%%", startangle=140,
               wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title("Good / Average / Bad Split", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Correlation
    st.markdown("**Feature Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(11, 7))
    corr = df.drop(columns=["quality_label"]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, annot_kws={"size":8}, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Feature selector
    st.markdown("**Feature vs Quality**")
    sel_feat = st.selectbox("Select feature to explore",
                             [c for c in df.columns if c not in ("quality","quality_label")])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = {"Bad":"#e74c3c","Average":"#f39c12","Good":"#27ae60"}
    sns.boxplot(data=df, x="quality_label", y=sel_feat, ax=axes[0],
                palette=palette, order=["Bad","Average","Good"])
    axes[0].set_title(f"{sel_feat} — Boxplot", fontweight="bold")
    sns.violinplot(data=df, x="quality_label", y=sel_feat, ax=axes[1],
                   palette=palette, order=["Bad","Average","Good"], inner="box", cut=0)
    axes[1].set_title(f"{sel_feat} — Violin Plot", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Raw data
    if st.checkbox("Show raw dataset"):
        st.dataframe(df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### 🍷 Wine Quality Analysis — ML Project

    This app classifies red wine into **three quality categories** based on its physicochemical properties
    using a machine learning model trained on the UCI Wine Quality dataset.

    ---
    #### 📁 Project Structure
    ```
    wine_quality_project/
    ├── wine_quality_analysis.ipynb   ← Full step-by-step notebook
    ├── app.py                        ← This Streamlit UI
    ├── model/
    │   ├── best_model.pkl            ← Trained classifier
    │   ├── scaler.pkl                ← StandardScaler
    │   ├── label_encoder.pkl         ← Label encoder
    │   └── feature_cols.pkl          ← Feature column names
    └── README.md                     ← Project documentation
    ```

    ---
    #### 🎯 Quality Labels

    | Label | Quality Score | Description |
    |-------|--------------|-------------|
    | 🟢 Good | ≥ 7 | Premium wine with excellent characteristics |
    | 🟡 Average | 5–6 | Standard quality, most common category |
    | 🔴 Bad | ≤ 4 | Below average, significant flaws |

    ---
    #### 🛠️ Tech Stack
    - **Python 3.9+** · **Pandas** · **NumPy**
    - **Scikit-learn** · **XGBoost**
    - **Matplotlib** · **Seaborn**
    - **Streamlit** · **Joblib**

    ---
    #### 📊 Dataset
    UCI Wine Quality Dataset (Red Wine) — 1,599 samples, 11 physicochemical features.
    """)

    st.markdown("---")
    st.caption(f"Model status: `{model_status}` | Model file: `{type(model).__name__}`")
