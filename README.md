# 🧠 DS/ML/AI Portfolio

This repository contains a collection of machine learning and data analysis projects, ranging from classic supervised learning to deep learning and exploratory analysis. Each notebook is designed to showcase specific concepts, techniques, or domains.

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

### 2. **Customer Churn Prediction**
- **Objective**: Predict customer churn for a telecom company.
- **Key Steps**:
  - Data preprocessing (handling missing values, categorical encoding).
  - Logistic Regression and CatBoost with hyperparameter tuning.
- **Results**:
  - CatBoost achieved **ROC-AUC = 0.847** with minimal effort.
  - Logistic Regression baseline: **ROC-AUC = 0.845**.

### 3. **Linear Models & Gradient Descent**
- **Objective**: Implement optimization algorithms and linear models.
- **Key Steps**:
  - Gradient Descent for a 2D function.
  - Logistic Regression and Elastic Net from scratch.
  - Evaluation on synthetic data and MNIST.
- **Results**:
  - Elastic Net achieved **99.38% accuracy** on binary MNIST classification.

### 4. **Game of Thrones Survival Prediction**
- **Objective**: Predict character survival in *Game of Thrones*.
- **Key Steps**:
  - Feature engineering (e.g., `alive_age_date`).
  - Model comparison (Logistic Regression, AdaBoost, Random Forest).
- **Results**:
  - AdaBoost achieved the highest accuracy (**83.97%**).

### 5. **Wolt Delivery Analysis** 🛵
- **Objective**: Explore Wolt delivery data to uncover performance trends and operational insights.
- **Key Steps**:
  - Data preprocessing and time-based feature extraction.
  - Exploratory Data Analysis (EDA) using pandas and Seaborn.
  - Delivery performance trends across dates, hours, and weekdays.
- **Results**:
  - Revealed delivery time peaks and operational efficiency patterns across timeframes.

### 6. **AKI Development in ICU Patients** 🏥
- **Objective**: Analyze factors contributing to Acute Kidney Injury (AKI) development among ICU patients.
- **Key Steps**:
  - Clinical data analysis and grouping by AKI stages.
  - Time-series visualization of creatinine levels and lab results.
  - Mortality and comorbidity rate comparison.
- **Results**:
  - Found correlations between AKI stages, time to onset, and increased mortality rates.

### 7. **Image Classification — CNN & Transfer Learning** (new)
- **Objective**: Build robust image classifiers using CNNs and transfer learning.
- **Key Steps**:
  - Implemented end-to-end CNN training pipelines (data loaders, augmentation, training loops).
  - Experimented with transfer learning (pretrained backbones) and custom architectures for small datasets.
  - Evaluated using standard metrics (accuracy, confusion matrices) and visualized feature maps and misclassifications.
- **Results**:
  - Reproducible training pipelines and comparative experiments demonstrating the effect of augmentation and backbone choice on generalization.

### 8. **Generative Adversarial Networks (GANs) for Image Synthesis** (new)
- **Objective**: Implement and experiment with GANs to generate realistic images.
- **Key Steps**:
  - Implemented vanilla GAN and architectural improvements (BatchNorm, LeakyReLU, learning-rate schedules).
  - Monitored training stability using losses, inception/IS-like qualitative checks and visual samples.
  - Explored conditioning mechanisms and simple augmentation to improve sample diversity.
- **Results**:
  - Working GAN models with checkpointed sample grids illustrating progressive improvement and qualitative diversity.

---

## 🛠️ Technologies & Tools
- **Frameworks**: PyTorch, scikit-learn, CatBoost
- **Libraries**: NumPy, pandas, Matplotlib, Seaborn, SciPy
- **Techniques**: Gradient Descent, EDA, Grid Search, Cross-Validation, Regularization, Time-Series Analysis, Multi-Output Linear Models, Transfer Learning, GANs

