# Human Activity Recognition

### High Performance Time Series Classification with XGBoost

This repository contains an end to end machine learning pipeline for classifying human physical activities using tri axial accelerometer data. The system is designed to handle large scale data and fast inference, processing over 1 million samples with GPU acceleration.

## Key Performance Metrics

Accuracy  
91.42 percent  

Dataset Size  
1,073,623 raw entries  

Inference Speed  
0.84 seconds for entire test set  

Training Time  
64.46 seconds using Nvidia T4 GPU  

---

## Technical Architecture

### Data Engineering and Preprocessing

Raw X Y Z accelerometer data is transformed into a 23 feature vector to capture motion patterns.

Feature Engineering  
Includes jerk metrics and correlation between axes  

Class Balancing  
SMOTE is applied to handle imbalance, increasing the dataset to more than 1.64 million samples and improving recall across all classes  

---

### Model Stack

Algorithm  
XGBoost  

Hardware  
CUDA based GPU acceleration for faster training  

Frameworks  
Scikit Learn  
Pandas  
NumPy  
XGBoost  

---

## Results and Insights

The model performs well across both static and dynamic activities.

Sitting and Standing  
Precision 1.00  
Recall 1.00  
F1 Score 1.00  

Jogging  
Precision 0.95  
Recall 0.91  
F1 Score 0.93  

Walking  
Precision 0.94  
Recall 0.90  
F1 Score 0.92  

Key Finding  
Square acceleration mean is the most important feature, followed by variance along the Y axis, showing that vertical movement plays a major role in activity detection  

---

## Repository Structure

data  
Dataset documentation  

notebook  
Exploratory analysis and training  

screenshots  
Project visuals  

models  
Saved XGBoost model  

README.md  
Project overview  
