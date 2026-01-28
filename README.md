# Employee Salary Prediction Using Linear Regression

This project implements a **Machine Learning model using Linear Regression** to predict employee salaries based on their years of experience. It demonstrates the complete ML workflow — from data preprocessing and visualization to model training, evaluation, and prediction.

## 📌 Problem Statement

Salary prediction is a common real-world regression problem in Human Resource analytics.  
The objective of this project is to **predict an employee’s salary** given their **years of experience**, using a supervised learning approach.

## 🎯 Project Objectives

- Understand the relationship between experience and salary
- Apply **Linear Regression** to a real dataset
- Visualize trends and model performance
- Evaluate the model using standard regression metrics
- Build a reusable prediction system

## 🧠 Machine Learning Approach

- **Type:** Supervised Learning  
- **Algorithm:** Linear Regression  
- **Target Variable:** Salary  
- **Input Feature:** Years of Experience

  Linear Regression was chosen because:
- The relationship between experience and salary is linear
- It is simple, interpretable, and effective for beginners
- It provides strong baseline performance for regression problems

 ## 🗂️ Project Structure

Employee_Salary_Prediction-linear-regression-
│
├── Employee-salary-prediction.csv # Dataset
├── employee_salary.ipynb # Jupyter Notebook (EDA + Model)
├── salary_predictor.py # Python script for prediction
├── README.md # Project documentation
└── requirements.txt # Project dependencies

## 📊 Dataset Description

The dataset contains employee salary information with the following columns:

| Column Name        | Description                          |
|-------------------|--------------------------------------|
| YearsExperience   | Number of years of work experience   |
| Salary            | Annual salary (target variable)      |

### Dataset Characteristics:
- Small and clean dataset
- No missing values
- Ideal for regression analysis
- Suitable for visualization and interpretation

## 🔍 Exploratory Data Analysis (EDA)

During EDA, the following steps were performed:

- Checked dataset shape and data types
- Verified missing values
- Visualized salary vs experience using scatter plots
- Observed a strong positive linear relationship

📈 **Insight:**  
As years of experience increase, salary increases linearly — making Linear Regression a suitable model.

## 🛠️ Technologies & Tools Used

- **Programming Language:** Python
- **Libraries:**
  - `pandas` – Data manipulation
  - `numpy` – Numerical computations
  - `matplotlib` – Data visualization
  - `seaborn` – Statistical plots
  - `scikit-learn` – ML model implementation
- **Environment:** Jupyter Notebook

## ⚙️ Model Development Steps

1. **Data Loading**
   
   - Read CSV file using pandas

2. **Feature Selection**

   - Independent variable: YearsExperience

   - Dependent variable: Salary

3. **Train-Test Split**

   - 80% training data

   - 20% testing data

5. **Model Training**

   - Used `LinearRegression()` from scikit-learn

7. **Prediction**

   - Predicted salaries on test data

9. **Evaluation**

   - Compared predicted vs actual values

   - Visualized regression line
  
<img width="1440" height="811" alt="Screenshot 2025-07-23 at 6 07 51 PM" src="https://github.com/user-attachments/assets/02cb451a-d781-43b5-be2f-8936dca2c882" />
