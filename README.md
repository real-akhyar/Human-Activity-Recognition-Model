---

# Human Activity Recognition (HAR) @ Scale
### High-Performance Time-Series Classification with XGBoost

This repository contains an end-to-end Machine Learning pipeline for classifying human physical activities using tri-axial accelerometer data. The project emphasizes **high-speed inference** and **data scaling**, processing over **1 million raw samples** with **GPU acceleration**.

## Key Performance Metrics
* **Accuracy:** 91.42%
* **Dataset Size:** 1,073,623 raw entries
* **Inference Speed:** 0.84s (Entire test set)
* **Training Time:** 64.46s (Nvidia T4 GPU)

---

## Technical Architecture

### 1. Data Engineering & Preprocessing
To capture the nuances of human motion, the raw X, Y, Z coordinates were transformed into a **23-feature vector** space.
* **Feature Engineering:** Generated temporal features including **Jerk metrics** and **Correlation coefficients** between axes.
* **Class Balancing:** Addressed significant class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique), expanding the training set to **1.64M+ samples** to ensure high recall across all activity classes.

### 2. The Model Stack
* **Algorithm:** XGBoost (Extreme Gradient Boosting)
* **Hardware:** Utilized **CUDA-accelerated tree construction** to handle the high-dimensional data efficiently.
* **Frameworks:** Scikit-Learn, Pandas, NumPy, XGBoost.

---

## Results & Insights

### Classification Report
The model demonstrates exceptional reliability in distinguishing between static and dynamic states.

| Activity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Sitting/Standing** | 1.00 | 1.00 | **1.00** |
| **Jogging** | 0.95 | 0.91 | **0.93** |
| **Walking** | 0.94 | 0.90 | **0.92** |

**Key Finding:** The feature `sq_acc_mean` (Square acceleration mean) was identified as the most critical predictor, followed closely by Y-axis variance, highlighting the importance of vertical displacement in activity recognition.

---

## Repository Structure
```text
├── data/               # Dataset documentation
├── notebook/          # Exploratory Data Analysis & Model Training
├── screenshots/
├── models/             # Saved XGBoost model (.json/.bin)
└── README.md
```

---
