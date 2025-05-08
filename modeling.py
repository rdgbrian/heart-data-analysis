#%%
# Imports
import numpy as np
import pandas as pd
from statsmodels.api import Logit
from sklearn.model_selection import train_test_split

from model_utils import (
    show_metrics,
    calculate_vif,
    plot_cooks_distance,
    plot_multiple_roc,
    plot_multiple_pr,
)

#%%
# Load data
df = pd.read_csv("data/clean_heart_pca.csv")

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X['Constant'] = 1  # Add intercept

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# VIF analysis
vif_data = calculate_vif(X)
print(vif_data)

#%%
# Model 1: Full feature set
model_full = Logit(y_train, X_train).fit()
print(model_full.summary())
show_metrics(model_full, X_test, y_test)

#%%
# Model 2: Pruned by p-values
significant_vars = model_full.pvalues[model_full.pvalues < 0.05].index
X_relevant = X[significant_vars]

X_train_relevant, X_test_relevant, y_train_relevant, y_test_relevant = train_test_split(
    X_relevant, y, test_size=0.2, random_state=42
)

model_relevant = Logit(y_train_relevant, X_train_relevant).fit()
print(model_relevant.summary())
show_metrics(model_relevant, X_test_relevant, y_test_relevant)

#%%
# Outlier detection (Cook’s Distance)
outliers = plot_cooks_distance(model_relevant)
print(f"Outliers (indices with Cook's distance > 4/n): {outliers}")

#%%
# Model 3: Remove outliers
X_train_cleaned = np.delete(X_train_relevant.values, outliers, axis=0)
y_train_cleaned = np.delete(y_train_relevant.values, outliers, axis=0)

model_cleaned = Logit(y_train_cleaned, X_train_cleaned).fit()
print(model_cleaned.summary())
show_metrics(model_cleaned, X_test_relevant, y_test_relevant)

#%%
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Helper function to evaluate and return metrics
def get_metrics(model, X_test, y_true):
    y_pred_prob = model.predict(X_test)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred_class),
        "Precision": precision_score(y_true, y_pred_class),
        "Recall": recall_score(y_true, y_pred_class),
    }

# Apply get_metrics with correct test inputs
metrics_table = pd.DataFrame([
    get_metrics(model_full, X_test, y_test),
    get_metrics(model_relevant, X_test_relevant, y_test_relevant),
    get_metrics(model_cleaned, X_test_relevant, y_test_relevant),
], index=["Entire Features", "Pruned Features", "Pruned + No Outliers"])

# Round and display
print(metrics_table.round(4))

#%%
# Compare ROC curves
roc_models = [
    ("Entire Features", y_test, model_full.predict(X_test), 'blue'),
    ("Pruned Features", y_test_relevant, model_relevant.predict(X_test_relevant), 'green'),
    ("Pruned + No Outliers", y_test_relevant, model_cleaned.predict(X_test_relevant), 'red'),
]
plot_multiple_roc(roc_models)

#%%
# Compare PR curves
pr_models = [
    ("Entire Features", y_test, model_full.predict(X_test), 'blue'),
    ("Pruned Features", y_test_relevant, model_relevant.predict(X_test_relevant), 'green'),
    ("Pruned + No Outliers", y_test_relevant, model_cleaned.predict(X_test_relevant), 'red'),
]
plot_multiple_pr(pr_models)
