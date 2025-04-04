# DS/ML/AI Portfolio

This repository contains a collection of machine learning projects covering various datasets and techniques. Below is a summary of each notebook:

---

## 📁 Project List

### 1. **CNN & MNIST Classification**  
- **Objective**: Implement neural networks for classification tasks.  
- **Key Steps**:  
  - Logistic regression from scratch using PyTorch.  
  - Fully Connected Neural Network (FCNN) and Convolutional Neural Network (LeNet) on MNIST.  
  - Activation function comparison (ELU, ReLU, LeakyReLU).  
- **Results**:  
  - LeNet achieved **98.59% accuracy** on MNIST.  
  - ELU activation performed best among tested options.

---

### 2. **Customer Churn Prediction**
- **Objective**: Predict customer churn for a telecom company.  
- **Key Steps**:  
  - Data preprocessing (handling missing values, categorical encoding).  
  - Logistic Regression and CatBoost with hyperparameter tuning.  
- **Results**:  
  - CatBoost achieved **ROC-AUC = 0.847** with minimal effort.  
  - Logistic Regression baseline: **ROC-AUC = 0.845**.

---

### 3. **Linear Models & Gradient Descent**
- **Objective**: Implement optimization algorithms and linear models.  
- **Key Steps**:  
  - Gradient Descent for a 2D function.  
  - Logistic Regression and Elastic Net from scratch.  
  - Evaluation on synthetic data and MNIST.  
- **Results**:  
  - Elastic Net achieved **99.38% accuracy** on binary MNIST classification.

---

### 4. **Game of Thrones Survival Prediction**
- **Objective**: Predict character survival in *Game of Thrones*.  
- **Key Steps**:  
  - Feature engineering (e.g., `alive_age_date`).  
  - Model comparison (Logistic Regression, AdaBoost, Random Forest).  
- **Results**:  
  - AdaBoost achieved the highest accuracy (**83.97%**).

---

## 🛠️ Technologies  
- **Frameworks**: PyTorch, scikit-learn, CatBoost.  
- **Libraries**: NumPy, pandas, Matplotlib, Seaborn.  
- **Techniques**: Gradient Descent, Grid Search, Cross-Validation, Regularization.
