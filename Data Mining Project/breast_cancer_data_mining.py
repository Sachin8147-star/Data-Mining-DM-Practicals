"""
DATA MINING PROJECT: Breast Cancer Diagnosis (Wisconsin Diagnostic Dataset)
===========================================================================
AIM: Predict whether a tumour is Malignant or Benign from cell-nucleus
     measurements, using data mining techniques.

TECHNIQUES USED:
  1. Data Understanding / EDA
  2. Data Preprocessing (scaling, train-test split)
  3. Feature Selection (correlation, SelectKBest, Random Forest importance)
  4. Dimensionality Reduction (PCA)
  5. Classification (LR, KNN, Decision Tree, Random Forest, SVM, Naive Bayes)
  6. Clustering (K-Means) - unsupervised
  7. Evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, CV)

DATASET: UCI Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 features.
         Loaded via sklearn.datasets.load_breast_cancer (verified benchmark data).

Run:  pip install pandas numpy matplotlib seaborn scikit-learn
      python breast_cancer_data_mining.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # remove this line if you want pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             silhouette_score, adjusted_rand_score)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
os.makedirs("figures", exist_ok=True)
RANDOM_STATE = 42

# Wider console tables (easier to screenshot for the report)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 8)


# Save everything printed on screen to output_log.txt as well
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)

    def flush(self):
        for st in self.streams:
            st.flush()


_log = open("output_log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
section("1. LOADING THE DATASET")
raw = load_breast_cancer()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target                       # 0 = malignant, 1 = benign
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ----------------------------------------------------------------------
# 2. DATA UNDERSTANDING / EDA
# ----------------------------------------------------------------------
section("2. EXPLORATORY DATA ANALYSIS")
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("\nClass distribution:")
print(df["diagnosis"].value_counts())

print("\nStatistical summary (first 6 features):")
print(df.iloc[:, :6].describe().round(3))

# Graph 1: class distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="diagnosis", palette=["#e74c3c", "#2ecc71"])
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("figures/01_class_distribution.png", dpi=150)
plt.close()

# Graph 2: histograms of key features by class
key_feats = ["mean radius", "mean texture", "mean perimeter", "mean area"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, f in zip(axes.ravel(), key_feats):
    sns.histplot(data=df, x=f, hue="diagnosis", kde=True, ax=ax,
                 palette={"Malignant": "#e74c3c", "Benign": "#2ecc71"})
    ax.set_title(f)
plt.tight_layout()
plt.savefig("figures/02_feature_histograms.png", dpi=150)
plt.close()

# Graph 3: correlation heatmap
corr = df[raw.feature_names].corr()
plt.figure(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.7})
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.savefig("figures/03_correlation_heatmap.png", dpi=150)
plt.close()

# Correlation of each feature with the target
target_corr = df[raw.feature_names].corrwith(df["target"]).sort_values()
print("\nTop 5 features most correlated with diagnosis (negative = malignant):")
print(target_corr.head(5).round(3))

# ----------------------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------------------
section("3. DATA PREPROCESSING")
X = df[raw.feature_names]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)       # fit ONLY on train data
X_test_s = scaler.transform(X_test)
X_all_s = scaler.fit_transform(X)               # used for PCA / clustering

print("Train size:", X_train.shape, "| Test size:", X_test.shape)
print("Features standardised (mean=0, std=1).")

# ----------------------------------------------------------------------
# 4. FEATURE SELECTION
# ----------------------------------------------------------------------
section("4. FEATURE SELECTION")

# 4a. ANOVA F-test
skb = SelectKBest(score_func=f_classif, k=10).fit(X_train_s, y_train)
top_f = pd.Series(skb.scores_, index=raw.feature_names).sort_values(ascending=False)
print("Top 10 features by ANOVA F-score:")
print(top_f.head(10).round(2))

# 4b. Random Forest importance
rf_sel = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
rf_sel.fit(X_train_s, y_train)
imp = pd.Series(rf_sel.feature_importances_, index=raw.feature_names)
imp = imp.sort_values(ascending=False)
print("\nTop 10 features by Random Forest importance:")
print(imp.head(10).round(4))

plt.figure(figsize=(8, 6))
imp.head(10).sort_values().plot(kind="barh", color="#3498db")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("figures/04_feature_importance.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5. PCA (DIMENSIONALITY REDUCTION)
# ----------------------------------------------------------------------
section("5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
pca_full = PCA().fit(X_all_s)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n95 = int(np.argmax(cum_var >= 0.95) + 1)
print(f"PC1 explains {pca_full.explained_variance_ratio_[0]*100:.2f}% variance")
print(f"PC2 explains {pca_full.explained_variance_ratio_[1]*100:.2f}% variance")
print(f"Components needed for 95% variance: {n95} (out of 30)")

pca2 = PCA(n_components=2).fit(X_all_s)
X_pca = pca2.transform(X_all_s)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(range(1, 31), cum_var, marker="o")
ax[0].axhline(0.95, color="r", linestyle="--", label="95%")
ax[0].set_xlabel("Number of components")
ax[0].set_ylabel("Cumulative explained variance")
ax[0].set_title("PCA - Explained Variance")
ax[0].legend()
for lab, col in [("Malignant", "#e74c3c"), ("Benign", "#2ecc71")]:
    m = df["diagnosis"] == lab
    ax[1].scatter(X_pca[m, 0], X_pca[m, 1], c=col, label=lab, alpha=0.6, s=25)
ax[1].set_xlabel("PC1")
ax[1].set_ylabel("PC2")
ax[1].set_title("PCA - 2D Projection")
ax[1].legend()
plt.tight_layout()
plt.savefig("figures/05_pca.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. CLASSIFICATION
# ----------------------------------------------------------------------
section("6. CLASSIFICATION MODELS")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, fitted, probas = [], {}, {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    cv_acc = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "CV Mean Acc": cv_acc.mean(),
        "CV Std": cv_acc.std(),
    })
    fitted[name] = model
    probas[name] = prob

res = pd.DataFrame(results).set_index("Model").round(4)
res = res.sort_values("Accuracy", ascending=False)
print(res)
res.to_csv("model_comparison.csv")

# Graph: model comparison
res[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
    kind="bar", figsize=(11, 5), ylim=(0.85, 1.01), colormap="viridis")
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/06_model_comparison.png", dpi=150)
plt.close()

# Graph: ROC curves
plt.figure(figsize=(7, 6))
for name, prob in probas.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/07_roc_curves.png", dpi=150)
plt.close()

# Best model details
best_name = res.index[0]
best_model = fitted[best_name]
best_pred = best_model.predict(X_test_s)
print(f"\nBEST MODEL: {best_name}")
print("\nClassification report:")
print(classification_report(y_test, best_pred,
                            target_names=["Malignant", "Benign"]))

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("figures/08_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix:\n", cm)

# Effect of feature selection: train on top-10 features only
top10 = imp.head(10).index.tolist()
idx10 = [list(raw.feature_names).index(f) for f in top10]
rf10 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf10.fit(X_train_s[:, idx10], y_train)
acc10 = accuracy_score(y_test, rf10.predict(X_test_s[:, idx10]))
print(f"\nRandom Forest with only top-10 features: accuracy = {acc10:.4f}")

# ----------------------------------------------------------------------
# 7. CLUSTERING (UNSUPERVISED)
# ----------------------------------------------------------------------
section("7. CLUSTERING - K-MEANS")
inertias, sils = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_all_s)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_all_s, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(K_range), inertias, marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(list(K_range), sils, marker="o", color="green")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.savefig("figures/09_kmeans_elbow.png", dpi=150)
plt.close()

km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(X_all_s)
clusters = km.labels_
# align cluster ids with labels (cluster ids are arbitrary)
if accuracy_score(y, clusters) < 0.5:
    clusters = 1 - clusters
print(f"Silhouette score (k=2): {silhouette_score(X_all_s, clusters):.4f}")
print(f"Adjusted Rand Index vs true labels: {adjusted_rand_score(y, clusters):.4f}")
print(f"Cluster-vs-diagnosis agreement: {accuracy_score(y, clusters)*100:.2f}%")
print("\nCluster vs actual diagnosis:")
print(pd.crosstab(df["diagnosis"], clusters, rownames=["Actual"],
                  colnames=["Cluster"]))

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="coolwarm", alpha=0.6, s=25)
plt.title("K-Means Clusters (k=2) on PCA axes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("figures/10_kmeans_clusters.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 8. CONCLUSION
# ----------------------------------------------------------------------
section("8. SUMMARY")
print(f"Best classifier        : {best_name}")
print(f"Best test accuracy     : {res.iloc[0]['Accuracy']*100:.2f}%")
print(f"Best ROC-AUC           : {res['ROC-AUC'].max():.4f}")
print(f"PCA components for 95% : {n95} of 30")
print("All graphs saved in the 'figures/' folder; table in model_comparison.csv")

sys.stdout = sys.__stdout__
_log.close()
