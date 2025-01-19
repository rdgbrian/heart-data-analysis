# Heart Disease Prediction Dataset Analysis

## Overview
This repository contains the analysis and findings based on the Heart Failure Prediction Dataset. The dataset, sourced from Kaggle, includes clinical information on 918 participants. It features 11 attributes and a binary label, `HeartDisease`, indicating the presence or absence of heart disease.

The project aimed to:
1. Preprocess and clean the dataset for accurate analysis.
2. Perform exploratory data analysis (EDA) to understand the features and distributions.
3. Apply statistical methods, including hypothesis testing.
4. Build and evaluate logistic regression models for heart disease prediction.

### Main Results
- **Data Cleaning**: Missing values in critical features like cholesterol were imputed using KNN imputation, and original values were preserved for analysis.
- **Feature Engineering**:
  - One-hot encoding and ordinal encoding were applied to categorical variables.
  - Principal Component Analysis (PCA) addressed multicollinearity issues.
- **Statistical Analysis**:
  - ANOVA showed no significant differences in cholesterol levels across chest pain types.
  - Pairwise proportion tests revealed significant associations between binary features and heart disease.
- **Model Development**:
  - Logistic regression models were trained on various feature sets.
  - The best-performing model achieved improved metrics after removing influential outliers.

Key features for heart disease prediction:
- Sex, FastingBS, ExerciseAngina, Oldpeak, ST\_Slope, ChestPainType, MissingCholesterolNum.

**Performance Metrics**:
- High accuracy across all models.
- Receiver Operating Characteristic (ROC) curves showed superior threshold performance for the pruned model.

## Getting Started

### Prerequisites
(To be completed: This section will include details about dependencies, installation steps, and dataset preparation.)

### Installation
(To be completed: Provide steps to clone and set up the repository for analysis.)

### Running the Analysis
(To be completed: Instructions on executing the data preprocessing, analysis scripts, and models.)

## Dataset
The dataset is available [here](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data).

### Features
| Feature         | Description                                      |
|-----------------|--------------------------------------------------|
| Age             | Age of the patient [years]                      |
| Sex             | Gender of the patient [M: Male, F: Female]      |
| ChestPainType   | Chest pain type [TA, ATA, NAP, ASY]             |
| RestingBP       | Resting blood pressure [mm Hg]                  |
| Cholesterol     | Serum cholesterol [mg/dl]                       |
| FastingBS       | Fasting blood sugar [1: > 120 mg/dl, 0: ≤ 120] |
| RestingECG      | Resting ECG results [Normal, ST, LVH]           |
| MaxHR           | Maximum heart rate achieved                     |
| ExerciseAngina  | Exercise-induced angina [Y: Yes, N: No]         |
| Oldpeak         | ST depression induced by exercise               |
| ST\_Slope        | Slope of the peak exercise ST segment [Up, Flat, Down] |

## Contributions
Contributions to improve the code, analysis, or extend the findings are welcome. Please follow standard practices for submitting pull requests.

## License
This project is licensed under the MIT License.

---

For questions or further clarifications, contact Juliana Garcia or Brian Rodriguez.

