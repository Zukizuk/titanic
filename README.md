# Titanic - Machine Learning from Disaster

## Introduction

This project marks my first foray into machine learning. The objective is to predict passenger survival based on various features such as age, gender, and ticket class. Throughout this journey, I have found the field of machine learning to be both fascinating and rewarding. This initial experience has sparked a deep interest in machine learning, and I am eager to continue exploring and expanding my knowledge in this exciting domain.

---

## Objective

The primary objective of this project is to:

1. Build a predictive model to determine survival probabilities.
2. Strengthen understanding of the machine learning pipeline, including data preprocessing, feature engineering, model selection, evaluation, and optimization.

---

## Features

The datasets includes the following features:

- **Survival**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Refer to `/Week 9 - Titanic.pdf` for more information on this work.

---

## Workflow

### 1. **Data Exploration**

- Examined the dataset for missing values, data types, and relationships with survival.
- Visualized survival rates by gender, passenger class, age groups and Title.

### 2. **Data Preprocessing**

- Handled missing values for.
- Encoded categorical variables.
- Normalized numerical features to ensure balanced contributions to the models.

### 3. **Feature Engineering**

- **Generated New Features**:

  - `FamilySize`: Calculated as the total number of family members aboard (SibSp + Parch + 1). This feature helps capture the impact of family presence on survival rates.
  - `IsAlone`: A binary feature indicating whether the passenger was traveling alone (1 if FamilySize is 1, else 0). This feature helps understand the effect of isolation on survival.
  - `AgeGroup`: Classified passengers into age groups: `['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']`. This feature helps in analyzing survival rates across different age demographics.
  - `Title`: Extracted and grouped titles from the `Name` column to reflect social status. This feature helps in understanding the influence of social status on survival.

- **Dropped Features**:
  - `Parch`: Incorporated into `FamilySize`.
  - `SibSp`: Incorporated into `FamilySize`.

### 4. **Modeling**

- Trained three models: Logistic Regression, Random Forest, and Support Vector Machines.
- Used cross-validation to evaluate performance with metrics like Accuracy, Precision, Recall, F1-score, and ROC-AUC.

---

## Results Before Optimization

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Score |
| ------------------- | -------- | --------- | ------ | -------- | ------- | -------- |
| Logistic Regression | 0.842    | 0.757     | 0.767  | 0.757    | 0.810   | 0.791    |
| Random Forest       | 0.807    | 0.692     | 0.730  | 0.711    | 0.835   | 0.789    |
| SVM                 | 0.789    | 0.659     | 0.730  | 0.692    | 0.833   | 0.802    |

---

### 5. **Model Optimization**

- Performed hyperparameter tuning using `RandomizedSearchCV`.

---

## Results After Optimization

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.816    | 0.735     | 0.676  | 0.704    | 0.779   |
| Random Forest       | 0.842    | 0.806     | 0.676  | 0.735    | 0.799   |
| SVM                 | 0.789    | 0.659     | 0.730  | 0.692    | 0.773   |

---

## Key Learnings

- Feature engineering plays a pivotal role in improving model performance.
- Hyperparameter tuning can significantly optimize model results and sometimes not.
- Machine learning is both a challenging and highly rewarding field that I am excited to explore further.

---

## Tools Used

- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment**: Jupyter Notebook

---

## How to Run

1. Clone this repository.
2. Install required libraries Pandas, Scikit-Learn, Matplotlip and Seaborn.
3. Open the Jupyter Notebook and execute cells sequentially.
4. Generate predictions for the test dataset.
