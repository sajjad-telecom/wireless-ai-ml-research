
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
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
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Wireless Channel Prediction")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# 6. Test a new wireless channel
new_channel = np.array([[25, 4]])
result = model.predict(new_channel)

print(
    "New Channel Quality:",
    "Good" if result[0] == 1 else "Poor"
)

# 7. Create results directory
os.makedirs("results", exist_ok=True)

# 8. Save evaluation metrics
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    "results/evaluation_metrics.csv"
)

# 9. Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Poor", "Good"]
)

disp.plot()
plt.title("Wireless Channel Prediction")
plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 10. Save feature importance
feature_importance = pd.DataFrame({
    "Feature": ["SNR", "Interference"],
    "Importance": model.feature_importances_
})

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nAll evaluation results saved successfully!")
