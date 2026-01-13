# NeuroApp ML Internship — Week 3  

## Introduction to Machine Learning

This repository contains the complete implementation of Week 3 internship tasks focused on core Machine Learning concepts including regression modeling, optimization using gradient descent, model evaluation, overfitting analysis, and model persistence.

The objective of this week is to build a strong mathematical and practical foundation in regression-based machine learning models.

---

## Topics Covered

- Linear Regression from scratch using Gradient Descent  
- Multiple Linear Regression using Scikit-Learn  
- Polynomial Regression and Overfitting Analysis  
- Model Persistence using Pickle, Joblib, and JSON  

---

## Repository Structure

```
NeuroApp-ML-Internship-Week-3/
│
├── linear_regression_scratch.py
├── multiple_regression.py
├── polynomial_regression.py
├── model_persistence.py
├── load_and_predict.py
├── README.md
└── plots/
    └── Contains all generated graphs including regression lines, cost convergence plots, residual plots, and polynomial regression visualizations
```


## Task 3.1 — Linear Regression from Scratch

This task implements simple linear regression without using any machine learning libraries.

### Model Equation

\[
y = wx + b
\]

Where:

- `w` is the weight (slope)  
- `b` is the bias (intercept)  

### Optimization Method

Gradient Descent is used to minimize the **Mean Squared Error (MSE)** in linear regression.

#### Mean Squared Error (MSE)

The MSE measures the average squared difference between predicted and actual values:

MSE = (1/m) * Σ (y_i - y_pred_i)^2

markdown
Copy code

Where:  
- `m` = number of training examples  
- `y_i` = actual value of the i-th sample  
- `y_pred_i` = predicted value of the i-th sample  

#### Gradient Descent Update Rules

The weight `w` and bias `b` are updated iteratively to minimize MSE:

w = w - α * (∂MSE / ∂w)
b = b - α * (∂MSE / ∂b)

markdown
Copy code

Where:  
- `w` = weight (slope)  
- `b` = bias (intercept)  
- `α` = learning rate  
- `∂MSE/∂w` and `∂MSE/∂b` = gradients of MSE with respect to `w` and `b`  

> These updates are repeated until the MSE converges or a maximum number of iterations is reached.


### Features Implemented

- Synthetic dataset generation  
- Manual gradient descent optimization  
- Manual MSE loss computation  
- Manual R² score calculation  
- Regression line visualization  
- Cost function convergence plot  

---

## Task 3.2 — Multiple Linear Regression using Scikit-Learn

This task implements a real-world regression pipeline using the California Housing dataset.

### Steps

- Dataset loading using `sklearn.datasets`  
- Train-test split  
- Model training using `LinearRegression`  
- Prediction on test data  
- Evaluation using:  
  - MAE  
  - MSE  
  - RMSE  
  - R² Score  
- Visualization of:  
  - Actual vs Predicted values  
  - Residual distribution  
- Printing model coefficients and intercept  

---

## Task 3.3 — Polynomial Regression and Overfitting

This task demonstrates how model complexity affects performance.

Polynomial regression models were trained using degrees:

- 1 (Linear)  
- 2 (Quadratic)  
- 3 (Cubic)  
- 5  
- 10  

### Analysis Performed

- Training and testing error comparison  
- Visualization of polynomial curves  
- Identification of underfitting and overfitting  
- Error trend analysis across model complexity  

This task highlights the bias-variance tradeoff in machine learning.

---

## Task 3.4 — Model Persistence

This task demonstrates saving and loading machine learning models using different serialization formats.

### Formats Implemented

- Pickle (`.pkl`)  
- Joblib (`.joblib`)  
- JSON (weights only)  

A trained regression model is saved in all three formats and then loaded for prediction.

---

## Performance Comparison

| Format  | File Size (bytes) | Load Time (seconds) | Prediction Supported |
|--------|------------------|--------------------|----------------------|
| Pickle | Measured using `os.path.getsize()` | Measured using `time` module | Yes |
| Joblib | Measured using `os.path.getsize()` | Measured using `time` module | Yes |
| JSON   | Smallest (weights only) | Fastest | Manual prediction |

### Observations

- Joblib is optimized for large NumPy objects  
- Pickle is flexible but slower for large models  
- JSON is lightweight but stores only model parameters  

---

## How to Run

### Install Dependencies

```bash
pip install numpy matplotlib scikit-learn joblib

## Run Each Task

python linear_regression_scratch.py
python multiple_regression.py
python polynomial_regression.py
python model_persistence.py
python load_and_predict.py


### Learning Outcomes

By completing this week, the following skills were developed:

Mathematical understanding of regression

Gradient descent optimization

Model evaluation metrics

Overfitting analysis

Real-world ML pipeline implementation

Model serialization and deployment concepts

Internship Program

This project is part of the NeuroApp Machine Learning Internship Program
Week 3: Introduction to Machine Learning

##  Author

Hamza Ahmad
NeuroApp ML Intern