
# Campus Shuttle On-Time Classifier - Single File Streamlit App
# Save as shuttle_full_app.py and run with:
#    streamlit run shuttle_full_app.py
#
# The app expects the dataset at: /mnt/data/projectAI_improved.csv

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("projectAI_improved.csv")
MODEL_DIR = Path("./models")
PLOTS_DIR = Path("./plots")
for d in (MODEL_DIR, PLOTS_DIR):
    d.mkdir(exist_ok=True)

# -----------------------------
# Data Handler
# -----------------------------
class DataHandler:
    REQUIRED_COLS = ["route", "stop_distance", "day_of_week", "time_block",
                     "traffic", "is_rain", "temp_c", "event_nearby", "on_time"]

    @staticmethod
    def load_csv(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df

    @staticmethod
    def validate(df):
        missing = set(DataHandler.REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    @staticmethod
    def prepare_features(df):
        df = df.copy()
        df = df.drop_duplicates()
        # ensure on_time numeric 0/1
        df['on_time'] = df['on_time'].astype(int)
        X = df.drop(columns=['on_time'])
        y = df['on_time']
        return X, y

# -----------------------------
# Model Trainer
# -----------------------------
class ModelTrainer:
    CATEGORICAL = ["route", "day_of_week", "time_block", "traffic", "event_nearby", "is_rain"]
    NUMERICAL = ["stop_distance", "temp_c"]

    MODELS = {
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier
    }

    def __init__(self, random_state=42):
        self.pipeline = None
        self.model_name = None
        self.metrics = {}
        self.random_state = random_state
        self.X_test = None
        self.y_test = None

    def build_pipeline(self, model_name, model_params=None):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        model_cls = self.MODELS[model_name]
        params = model_params or {}
        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), self.CATEGORICAL),
            ("num", StandardScaler(), self.NUMERICAL)
        ], remainder="passthrough")
        clf = model_cls(**params)
        self.pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])
        self.model_name = model_name
        return self.pipeline

    def train(self, X, y, test_size=0.2, model_name="Logistic Regression", model_params=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.build_pipeline(model_name, model_params)
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        y_proba = None
        if hasattr(self.pipeline.named_steps['clf'], "predict_proba"):
            y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        # store test set
        self.X_test = X_test
        self.y_test = y_test
        # metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        # cross val (safe)
        try:
            cv = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="f1")
            self.metrics['cv_mean'] = float(np.mean(cv))
            self.metrics['cv_std'] = float(np.std(cv))
        except Exception as e:
            self.metrics['cv_error'] = str(e)
        return self.metrics

    def save(self, filename=None):
        if self.pipeline is None:
            raise RuntimeError("No trained pipeline to save")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"{self.model_name.replace(' ', '_')}_{ts}.joblib"
        path = MODEL_DIR / filename
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics}, path)
        return str(path)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.pipeline = data.get("pipeline")
        self.metrics = data.get("metrics", {})
        return self.pipeline

    def predict(self, df_row):
        if self.pipeline is None:
            raise RuntimeError("Model not trained/loaded")
        proba = None
        if hasattr(self.pipeline.named_steps['clf'], "predict_proba"):
            proba = float(self.pipeline.predict_proba(df_row)[:, 1][0])
        pred = int(self.pipeline.predict(df_row)[0])
        return pred, proba

    def get_feature_importance(self):
        """Return (feature_names, importances)"""
        if self.pipeline is None:
            return None, None
        pre = self.pipeline.named_steps['pre']
        clf = self.pipeline.named_steps['clf']
        # get feature names from preprocessor
        feature_names = []
        # OneHotEncoder names
        try:
            cat_transformer = pre.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_names = list(cat_transformer.get_feature_names_out(self.CATEGORICAL))
            else:
                cat_names = []
                for i, col in enumerate(self.CATEGORICAL):
                    vals = cat_transformer.categories_[i]
                    # include all categories (skipping none) to be safe
                    cat_names += [f"{col}_" + str(v) for v in vals]
        except Exception:
            cat_names = []
        num_names = list(self.NUMERICAL)
        feature_names = cat_names + num_names
        # Now get importances
        importances = None
        if hasattr(clf, "coef_"):
            coef = clf.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importances = np.array(coef)
        elif hasattr(clf, "feature_importances_"):
            importances = np.array(clf.feature_importances_)
        else:
            importances = None
        return feature_names, importances

# -----------------------------
# Plot helpers
# -----------------------------
def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    return fig

def plot_roc(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(5,4))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_pr(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(5,4))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
    plt.tight_layout()
    return fig

def plot_threshold_tuner(y_test, y_proba):
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(thresholds, precisions, label="Precision", color="#1f77b4")
    ax.plot(thresholds, recalls, label="Recall", color="#ff7f0e")
    ax.plot(thresholds, f1s, label="F1", color="#2ca02c")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuner")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, thresholds, precisions, recalls, f1s

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Campus Shuttle On-Time Classifier", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0b3d91;'>🚌 Campus Shuttle On-Time Classifier</h1>", unsafe_allow_html=True)
st.write("Predict whether shuttle arrival is On-time (≤5 min late) or Late (>5 min).")

# Load dataset
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    return df

try:
    df = load_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("Mode", ["Overview", "Train", "Threshold Tuner", "Predict", "Analysis"])
st.sidebar.markdown("---")

# Show sample and basic stats
if mode == "Overview":
    st.header("Dataset Overview")
    st.write(f"Records: {len(df)}")
    st.write("Columns:", list(df.columns))
    st.subheader("Class distribution (on_time)")
    st.bar_chart(df['on_time'].value_counts())
    st.subheader("Sample rows")
    st.dataframe(df.head(10))

# TRAINING
if mode == "Train":
    st.header("Train a Model")
    st.info("Choose algorithm, set test size, and train. Results, confusion matrix, ROC/PR will appear.")
    algo = st.selectbox("Algorithm", ["Logistic Regression", "Decision Tree", "Random Forest"])
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    if algo == "Logistic Regression":
        c = st.number_input("C (inverse regularization)", 0.01, 10.0, 1.0)
        params = {"C": c, "max_iter": 2000, "random_state": 42}
    elif algo == "Decision Tree":
        max_depth = st.slider("Max depth", 2, 30, 6)
        params = {"max_depth": int(max_depth), "random_state": 42}
    else:
        n_est = st.slider("n_estimators", 10, 300, 100)
        params = {"n_estimators": int(n_est), "random_state": 42}

    if st.button("Train model"):
        try:
            DataHandler.validate(df)
        except Exception as e:
            st.error(f"Data validation error: {e}")
            st.stop()
        X, y = DataHandler.prepare_features(df)
        trainer = ModelTrainer()
        metrics = trainer.train(X, y, test_size=test_size, model_name=algo, model_params=params)
        st.success("Training completed")
        # display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("F1", f"{metrics['f1']:.4f}")
        if 'cv_mean' in metrics:
            st.info(f"CV F1 mean: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
        # confusion
        labels = ["Late (0)", "On-time (1)"]
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            fig_cm = plot_confusion(cm, labels)
            st.subheader("Confusion Matrix")
            st.pyplot(fig_cm)
        # ROC / PR
        if metrics.get("y_proba") is not None:
            fig_roc = plot_roc(metrics['y_test'], metrics['y_proba'])
            fig_pr = plot_pr(metrics['y_test'], metrics['y_proba'])
            st.subheader("ROC Curve")
            st.pyplot(fig_roc)
            st.subheader("Precision-Recall Curve")
            st.pyplot(fig_pr)
            # store threshold data in session for tuner
            st.session_state['trainer'] = trainer
        else:
            st.warning("Model does not provide probabilities; ROC/PR/Threshold tuner disabled.")
            st.session_state['trainer'] = trainer

        # feature importance
        feature_names, importances = trainer.get_feature_importance()
        if feature_names and importances is not None:
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_df['abs'] = fi_df['importance'].abs()
            fi_df = fi_df.sort_values('abs', ascending=False).head(20)
            st.subheader("Top features")
            st.table(fi_df[['feature','importance']].set_index('feature'))
        else:
            st.info("No feature importance available for this model.")

        # save option
        if st.button("Save trained model"):
            path = trainer.save()
            st.success(f"Model saved to: {path}")
            st.session_state['trainer'] = trainer

# THRESHOLD TUNER
if mode == "Threshold Tuner":
    st.header("Threshold Tuner (Precision / Recall / F1 vs threshold)")
    if 'trainer' not in st.session_state:
        st.warning("Train a model first in Train mode. Trainer must be in session_state['trainer'].")
        st.stop()
    trainer = st.session_state['trainer']
    if trainer.metrics.get("y_proba") is None:
        st.error("The trained model did not produce probabilities; cannot tune threshold.")
        st.stop()
    y_test = np.array(trainer.metrics['y_test'])
    y_proba = np.array(trainer.metrics['y_proba'])
    fig, thresholds, precisions, recalls, f1s = plot_threshold_tuner(y_test, y_proba)
    st.pyplot(fig)
    # allow user to select threshold and see metrics
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    preds = (y_proba >= thr).astype(int)
    st.write("Metrics at threshold = ", thr)
    st.write("Precision:", precision_score(y_test, preds, zero_division=0))
    st.write("Recall:", recall_score(y_test, preds, zero_division=0))
    st.write("F1:", f1_score(y_test, preds, zero_division=0))
    # show confusion matrix at threshold
    cm = confusion_matrix(y_test, preds)
    st.subheader("Confusion Matrix at selected threshold")
    st.pyplot(plot_confusion(cm, ["Late (0)", "On-time (1)"]))

# PREDICTION / WHAT-IF
if mode == "Predict":
    st.header("What-If Prediction")
    st.info("Enter scenario and get predicted class and probability (if available).")

    # inputs
    route = st.selectbox("Route", sorted(df['route'].unique()))
    day = st.selectbox("Day", sorted(df['day_of_week'].unique()))
    time_block = st.selectbox("Time Block", sorted(df['time_block'].unique()))
    traffic = st.selectbox("Traffic", sorted(df['traffic'].unique()))
    is_rain = st.selectbox("Is Rain", sorted(df['is_rain'].unique()))
    event = st.selectbox("Event Nearby", sorted(df['event_nearby'].unique()))
    temp = st.number_input("Temperature (C)", int(df['temp_c'].min()), int(df['temp_c'].max()), int(df['temp_c'].median()))
    stop_dist = st.number_input("Stop Distance (km)", int(df['stop_distance'].min()), int(df['stop_distance'].max()), int(df['stop_distance'].median()))
    threshold = st.slider("Decision threshold for classifying On-time", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict"):
        if 'trainer' not in st.session_state:
            st.warning("Train a model first in Train mode.")
            st.stop()
        trainer = st.session_state['trainer']
        row = pd.DataFrame([{
            "route": route, "stop_distance": stop_dist, "day_of_week": day,
            "time_block": time_block, "traffic": traffic, "is_rain": is_rain,
            "temp_c": temp, "event_nearby": event
        }])
        pred, proba = trainer.predict(row)
        # apply threshold if probability exists
        if proba is not None:
            pred_thr = 1 if proba >= threshold else 0
            label = "On-time (1)" if pred_thr == 1 else "Late (0)"
            st.markdown(f"### Predicted class (threshold applied = {threshold}): **{label}**")
            st.markdown(f"### Predicted probability of On-time: **{proba:.2%}**")
        else:
            label = "On-time (1)" if pred == 1 else "Late (0)"
            st.markdown(f"### Predicted class: **{label}** (no probability available)")

        # show probability breakdown if possible
        if hasattr(trainer.pipeline.named_steps['clf'], 'predict_proba'):
            st.write("Model supports probabilities.")

        # save to session history
        hist = st.session_state.get("history", [])
        hist.append({"timestamp": datetime.now().isoformat(), "input": row.to_dict('records')[0], "pred": int(pred), "proba": float(proba) if proba is not None else None})
        st.session_state['history'] = hist
        st.success("Prediction saved to session history.")

# ANALYSIS / HISTORY
if mode == "Analysis":
    st.header("Analysis & History")
    st.subheader("Feature distributions")
    col = st.selectbox("Feature to plot", df.columns)
    fig, ax = plt.subplots()
    if df[col].dtype == 'object' or len(df[col].unique()) < 20:
        df[col].value_counts().plot(kind='bar', ax=ax)
    else:
        df[col].hist(ax=ax, bins=20)
    st.pyplot(fig)
    plt.close()
    st.subheader("Numeric correlation")
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(num.corr(), annot=True, ax=ax2, cmap="coolwarm")
        st.pyplot(fig2)
        plt.close(fig2)
    st.subheader("Prediction history (session)")
    history = st.session_state.get("history", [])
    if history:
        st.dataframe(pd.DataFrame(history))
    else:
        st.info("No prediction history in this session.")
