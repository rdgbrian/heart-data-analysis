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

Here’s the updated **Getting Started** section of your README with clear instructions:

## Getting Started

### Prerequisites
Before you begin, ensure that you have the following installed on your system:
- Python (version 3.7 or higher is recommended). We suggest using [Anaconda](https://www.anaconda.com/) for easier environment and package management.
- Basic knowledge of working with Python virtual environments.

### Installation
Follow these steps to set up the repository for analysis:

1. **Clone the Repository**:
   Open a terminal and run:
   ```bash
   git clone https://github.com/rdgbrian/heart-data-analysis.git
   cd heart-data-analysis
   ```

2. **Create a Python Environment**:
   If you are using Anaconda, create and activate a virtual environment:
   ```bash
   conda create --name <env_name> python=3.9
   conda activate <env_name>
   ```

   Alternatively, using `venv`:
   ```bash
   python -m venv <env_name>
   source <env_name>/bin/activate   # On MacOS/Linux
   <env_name>\Scripts\activate      # On Windows
   ```

3. **Install Dependencies**:
   Use the `requirements.txt` file to install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the Setup**:
   To ensure all dependencies are installed correctly, you can test the installation by running:
   ```bash
   python -m pip check
   ```

Now you’re ready to begin analysis with the repository!

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

