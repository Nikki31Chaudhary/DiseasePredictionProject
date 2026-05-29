"""
Disease Prediction from Medical Data
CodeAlpha ML Internship - Task 4

Dataset: Pima Indians Diabetes Dataset (UCI ML Repository)
Models: Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data():
    """
    Downloads Pima Indians Diabetes dataset from URL.
    If offline, place 'diabetes.csv' in the same folder.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    try:
        df = pd.read_csv(url, names=columns)
        print("✅ Dataset downloaded successfully!")
    except Exception:
        # Fallback: load from local file
        df = pd.read_csv("diabetes.csv")
        print("✅ Dataset loaded from local file!")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def explore_data(df):
    print("\n📊 Dataset Shape:", df.shape)
    print("\n📋 First 5 Rows:\n", df.head())
    print("\n📈 Basic Statistics:\n", df.describe())
    print("\n🔍 Missing Values:\n", df.isnull().sum())
    print("\n🎯 Class Distribution:\n", df['Outcome'].value_counts())


# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_data(df):
    # Replace 0s with NaN for columns where 0 is biologically impossible
    zero_not_valid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_not_valid] = df[zero_not_valid].replace(0, np.nan)

    # Fill NaN with median (robust to outliers)
    df.fillna(df.median(numeric_only=True), inplace=True)

    print("\n✅ Preprocessing done. No missing values:", df.isnull().sum().sum())
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(df):
    # BMI categories
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    ).astype(str)

    # Age groups
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=['Young', 'Middle', 'Senior', 'Elderly']
    ).astype(str)

    # Glucose-to-Insulin ratio
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

    # Drop string categorical columns (encode them if needed)
    df = df.drop(columns=['BMI_Category', 'Age_Group'])

    print("\n✅ Feature engineering complete. Columns:", df.columns.tolist())
    return df


# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────
def prepare_splits(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler and feature names for Flask app
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print("\n✅ Train/Test split done. Train:", X_train_sc.shape, "| Test:", X_test_sc.shape)

    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM":                 SVC(kernel='rbf', probability=True, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  🔁 {name} — 5-Fold CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

    return trained


# ─────────────────────────────────────────────
# 7. EVALUATE MODELS
# ─────────────────────────────────────────────
def evaluate_models(trained_models, X_test, y_test):
    print("\n" + "="*65)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*65)

    results = {}
    best_model_name = None
    best_f1 = 0

    for name, model in trained_models.items():
        y_pred     = model.predict(X_test)
        y_proba    = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)

        results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

        print(f"\n🔸 {name}")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall   : {rec:.4f}")
        print(f"   F1-Score : {f1:.4f}")
        print(f"   ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} (F1={best_f1:.4f})")
    return results, best_model_name


# ─────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────
def save_plots(df, trained_models, X_test, y_test, feature_names):
    os.makedirs('static', exist_ok=True)

    # --- (a) Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap='RdYlGn', linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig('static/heatmap.png', dpi=120)
    plt.close()
    print("✅ Saved: static/heatmap.png")

    # --- (b) ROC Curves ---
    plt.figure(figsize=(8, 6))
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('static/roc_curves.png', dpi=120)
    plt.close()
    print("✅ Saved: static/roc_curves.png")

    # --- (c) Feature Importance (Random Forest) ---
    rf = trained_models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title("Feature Importance — Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('static/feature_importance.png', dpi=120)
    plt.close()
    print("✅ Saved: static/feature_importance.png")

    # --- (d) Confusion Matrix (Best model = XGBoost) ---
    model = trained_models["XGBoost"]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title("Confusion Matrix — XGBoost")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=120)
    plt.close()
    print("✅ Saved: static/confusion_matrix.png")


# ─────────────────────────────────────────────
# 9. SAVE BEST MODEL
# ─────────────────────────────────────────────
def save_best_model(trained_models, best_model_name):
    best = trained_models[best_model_name]
    joblib.dump(best, 'models/best_model.pkl')
    print(f"\n✅ Best model saved: models/best_model.pkl ({best_model_name})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Disease Prediction Pipeline...\n")

    df = load_data()
    explore_data(df)
    df = preprocess_data(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, feature_names = prepare_splits(df)

    print("\n🤖 Training Models...")
    trained_models = train_models(X_train, y_train)

    results, best_model_name = evaluate_models(trained_models, X_test, y_test)

    print("\n🖼 Saving Plots...")
    save_plots(df, trained_models, X_test, y_test, feature_names)

    save_best_model(trained_models, best_model_name)

    print("\n🎉 Training Complete! Run 'python app.py' to start the web app.")
