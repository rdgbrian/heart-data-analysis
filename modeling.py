
#%% 
# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.api import Logit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def show_metrics(result,X_test,y_test):
    # Predict probabilities for the positive class
    y_pred_prob = result.predict(X_test)
    # Predict binary outcomes (0 or 1) based on a threshold of 0.5
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    # Calculate accuracy directly
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy:.4f}")

    # Generate a classification report
    print(classification_report(y_test, y_pred_class, target_names=['Negative', 'Positive']))

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred_class)

    # Display the confusion matrix nicely
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')

    # Customize the plot for professional quality
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.grid(False)  # Remove gridlines for cleaner appearance
    plt.show()

def show_plots(result,X_test,y_test):
    # Predict probabilities for the positive class
    y_pred_prob = result.predict(X_test)

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Calculate precision-recall curve
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

# Read the cleaned dataset
df = pd.read_csv("clean_heart_pca.csv")


# Define the feature matrix (X) and target variable (y)
X = df.drop('HeartDisease', axis=1)  # Exclude target variable
y = df['HeartDisease']

# Add a constant for intercept
X['Constant'] = 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 40% reserved for val+test
# # Split temp into validation and test
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp for test

# Show the shapes of the resulting datasets
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

#%% 
# Calculate VIF
df_temp = X
vif_data = pd.DataFrame()
vif_data["Feature"] = df_temp.columns
vif_data["VIF"] = [variance_inflation_factor(df_temp.values, i) for i in range(df_temp.shape[1])]

print(vif_data)

#%%
# Fit the logistic regression model with all features
logit_model = Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary())
show_metrics(result,X_test,y_test)

#%%
# Extract p-values and filter columns with p-value < 0.05
significant_vars = result.pvalues[result.pvalues < 0.05].index
# Create a new DataFrame with only the relevant variables
X_relevant = X[significant_vars]
# X_relevant = X_relevant.drop("MissingCholesterolNum",axis=1)
# Display the relevant DataFrame
X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train_relevant, X_temp, y_train_relevant, y_temp = train_test_split(X_relevant, y, test_size=0.4, random_state=42)  # 40% reserved for val+test
# # Split temp into validation and test
# X_val_relevant, X_test_relevant, y_val_relevant, y_test_relevant = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp for test

# X_train_relevant, X_temp, y_train_relevant, y_temp = train_test_split(X_relevant, y, test_size=0.4, random_state=42)  # 40% reserved for val+test
# # Split temp into validation and test
# X_val_relevant, X_test_relevant, y_val_relevant, y_test_relevant = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 50% of temp for test



logit_model = Logit(y_train_relevant, X_train_relevant)
# logit_model = Logit(X_train, y_train_relevant)

result_relevant = logit_model.fit()
print(result_relevant.summary())

show_metrics(result_relevant,X_test_relevant,y_test_relevant)

#%%
# Calculate Outliers That May be effecting model
influence = result_relevant.get_influence()
cooks_d = influence.cooks_distance[0]

# Plot Cook's distance
plt.figure(figsize=(10, 6))
plt.scatter(range(len(cooks_d)), cooks_d, color='blue', label='Cook\'s Distance')
plt.axhline(y=4/len(cooks_d), color='red', linestyle='--', label='Threshold (4/n)')
plt.xlabel('Index of Data Points')
plt.ylabel('Cook\'s Distance')
plt.title('Cook\'s Distance for Each Data Point')
plt.legend()
plt.tight_layout()
plt.show()

# Identify outliers
outliers = np.where(cooks_d > 4/len(cooks_d))[0]
print(f"Outliers (indices with Cook's distance > 4/n): {outliers}")



#%%
# Remove outliers from the dataset
X_train_cleaned = np.delete(X_train_relevant, outliers, axis=0)
y_train_cleaned = np.delete(y_train_relevant, outliers, axis=0)

#%% 
# Train a new logistic regression model on the pruned dataset
logit_model_cleaned = Logit(y_train_cleaned, X_train_cleaned)
result_outlier = logit_model_cleaned.fit()
print(result_outlier.summary())
show_metrics(result_outlier,X_test_relevant,y_test_relevant)


# %%
# Create ROC Curve for three models

# Plot the ROC curve
plt.figure(figsize=(8, 6))

y_pred_prob = result.predict(X_test)
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Entire Features (AUC = {roc_auc:.4f})')


y_pred_prob = result_relevant.predict(X_test_relevant)
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_relevant, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='green', lw=2, label=f'Pruned Features (AUC = {roc_auc:.4f})')


y_pred_prob = result_outlier.predict(X_test_relevant)
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_relevant, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='red', lw=2, label=f'Pruned Outlies & Features (AUC = {roc_auc:.4f})')


plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



# %%
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Calculate the precision-recall curve for the entire features model
y_pred_prob = result.predict(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='blue', lw=2, label=f'Entire Features (AUC = {pr_auc:.4f})')

# Calculate the precision-recall curve for the pruned features model
y_pred_prob = result_relevant.predict(X_test_relevant)
precision, recall, thresholds = precision_recall_curve(y_test_relevant, y_pred_prob)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='green', lw=2, label=f'Pruned Features (AUC = {pr_auc:.4f})')

# Calculate the precision-recall curve for the pruned outliers & features model
y_pred_prob = result_outlier.predict(X_test_relevant)
precision, recall, thresholds = precision_recall_curve(y_test_relevant, y_pred_prob)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='red', lw=2, label=f'Pruned Outliers & Features (AUC = {pr_auc:.4f})')

# Plot random classifier line
plt.plot([0, 1], [0.5, 0.5], color='gray', linestyle='--', label='Random Classifier')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid()
plt.show()

# %%
