
# Wireless AI/ML - Experimental Results

## 1. Experiment Overview

Four machine learning algorithms were evaluated for wireless channel quality classification using a simulated dataset.

- Dataset size: 1000 samples
- Input features: SNR and Interference
- Training set: 80%
- Testing set: 20%
- Random state: 42

## 2. Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM | 0.9900 | 0.9792 | 0.9792 | 0.9792 |
| Random Forest | 0.9750 | 0.9574 | 0.9375 | 0.9474 |
| Gradient Boosting | 0.9700 | 0.9375 | 0.9375 | 0.9375 |

## 3. Best Performing Model

Logistic Regression achieved the highest scores across all four evaluation metrics in this experiment.

## 4. Channel Prediction Test

Test input:
- SNR: 25
- Interference: 4

All four models predicted: Good

## 5. Important Note

The dataset is synthetically generated, and channel quality labels are based on a predefined threshold rule.

Therefore, these results demonstrate an educational machine learning experiment and should not be interpreted as validated performance on real-world wireless network measurements.
