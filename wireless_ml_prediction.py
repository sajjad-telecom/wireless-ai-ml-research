
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate simulated wireless channel data
np.random.seed(42)

snr = np.random.uniform(0, 30, 1000)
interference = np.random.uniform(0, 15, 1000)

# Simulated channel quality labels
quality = ((snr - interference) > 15).astype(int)

X = np.column_stack((snr, interference))
y = quality

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Wireless Channel Prediction")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Test a new wireless channel
new_channel = np.array([[25, 4]])
result = model.predict(new_channel)

print("New Channel Quality:",
      "Good" if result[0] == 1 else "Poor")
