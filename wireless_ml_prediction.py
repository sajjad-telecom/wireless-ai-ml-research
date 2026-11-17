
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Generate simulated wireless channel data
np.random.seed(42)

snr = np.random.uniform(0, 30, 1000)
interference = np.random.uniform(0, 15, 1000)

# 2. Generate channel quality labels
quality = ((snr - interference) > 15).astype(int)

X = np.column_stack((snr, interference))
y = quality

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Define machine learning models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42)
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

# 5. Train and evaluate models
os.makedirs("results", exist_ok=True)

results = []
trained_models = {}

print("Wireless Channel ML Model Comparison")
print("=" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    report = classification_report(
        y_test,
        predictions,
        target_names=["Poor", "Good"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        f"results/{name.lower().replace(' ', '_')}_report.csv"
    )

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Poor", "Good"]
    )

    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(
        f"results/{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# 6. Save comparison results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

# 7. Plot model performance
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("Wireless Channel ML Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 8. Save feature importance
rf_model = trained_models["Random Forest"]

feature_importance = pd.DataFrame({
    "Feature": ["SNR", "Interference"],
    "Importance": rf_model.feature_importances_
})

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

# 9. Predict a new wireless channel
new_channel = np.array([[25, 4]])

print("\nPrediction for New Wireless Channel:")

for name, model in trained_models.items():
    prediction = model.predict(new_channel)
    quality_label = "Good" if prediction[0] == 1 else "Poor"
    print(f"{name}: {quality_label}")

print("\nAll evaluation results saved successfully!")
