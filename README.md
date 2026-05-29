# 🏥 Disease Prediction from Medical Data
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.56%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey?style=flat-square)


---

## 📌 Project Overview

This project predicts the likelihood of **Diabetes** based on patient medical data using multiple Machine Learning classification algorithms.

| Item | Detail |
|------|--------|
| **Dataset** | Pima Indians Diabetes (UCI ML Repository) |
| **Problem Type** | Binary Classification |
| **Models Used** | Logistic Regression, Random Forest, XGBoost, SVM |
| **Tech Stack** | Python, Scikit-learn, XGBoost, Flask, Pandas, Matplotlib |

---

## 📁 Project Structure

```
disease_prediction/
│
├── train_model.py          ← Training script (run this FIRST)
├── app.py                  ← Flask web app (run this SECOND)
├── requirements.txt        ← All dependencies
├── README.md               ← This file
│
├── templates/
│   └── index.html          ← Web UI (auto-served by Flask)
│
├── static/                 ← Auto-created after training
│   ├── heatmap.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
└── models/                 ← Auto-created after training
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

---

## 🚀 How to Run (Step by Step)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model
```bash
python train_model.py
```
This will:
- ✅ Download the dataset automatically
- ✅ Preprocess & feature engineer the data
- ✅ Train 4 ML models with 5-fold cross-validation
- ✅ Print evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Save plots to `static/` folder
- ✅ Save best model to `models/` folder

### Step 3 — Run the Web App
```bash
python app.py
```
Then open your browser: **http://127.0.0.1:5000**

---

## 📊 Dataset Details

**Pima Indians Diabetes Dataset**
- Source: [UCI ML Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 samples, 8 features + 1 target
- Target: 0 = No Diabetes, 1 = Diabetes

### Features:
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Family history score |
| Age | Age in years |

---

## 🤖 Models & Metrics

After training, you'll see a comparison like:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~77% | ~0.68 | ~0.83 |
| Random Forest | ~78% | ~0.70 | ~0.84 |
| **XGBoost** | **~80%** | **~0.72** | **~0.85** |
| SVM | ~78% | ~0.70 | ~0.83 |

---

## 🔧 Feature Engineering Done
- Replaced biologically invalid 0s with median values
- Added **Glucose-to-Insulin Ratio** feature
- StandardScaler normalization

---


# 👤 Author

## Nikki Chaudhary
CodeAlpha Machine Learning Intern

### 🔗 Connect With Me

- GitHub: https://github.com/Nikki31Chaudhary
- LinkedIn: https://www.linkedin.com/in/nikki-chaudhary-57b976332
