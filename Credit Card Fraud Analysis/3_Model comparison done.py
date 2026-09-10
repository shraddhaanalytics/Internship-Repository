import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, roc_curve, auc,
                              precision_recall_curve)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/HP/Desktop/Internship Final Submission/Credit Card Fraud Analysis/1_credit_card_fraud_ Original Dataset.csv")
df = df.drop(columns=["transaction_id"])
df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
num_cols = ["amount", "transaction_hour", "device_trust_score",
            "velocity_last_24h", "cardholder_age"]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("Original train class counts:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_res.value_counts().to_dict())
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                             random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}
results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    results[name] = (pred, prob)
    print(f"===== {name} =====")
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
    print()
summary = []
for name, (pred, prob) in results.items():
    report = classification_report(y_test, pred, output_dict=True)
    summary.append({
        "Model": name,
        "Precision (fraud)": round(report["1"]["precision"], 3),
        "Recall (fraud)": round(report["1"]["recall"], 3),
        "F1 (fraud)": round(f1_score(y_test, pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, prob), 3),
    })
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv("credit_card_fraud_10k.csv", index=False)
plt.figure(figsize=(7, 6))
for name, (pred, prob) in results.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - Credit Card Fraud Detection")
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=150)
plt.close()
plt.figure(figsize=(7, 6))
for name, (pred, prob) in results.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    plt.plot(rec, prec, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("pr_comparison.png", dpi=150)
plt.close()
print("\nSaved: model_comparison.csv, roc_comparison.png, pr_comparison.png")