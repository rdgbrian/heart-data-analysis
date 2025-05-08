import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, precision_recall_curve
)

def show_metrics(result, X_test, y_test):
    y_pred_prob = result.predict(X_test)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred_class, target_names=['Negative', 'Positive']))

    cm = confusion_matrix(y_test, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.grid(False)
    plt.show()

def show_plots(result, X_test, y_test):
    y_pred_prob = result.predict(X_test)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def plot_cooks_distance(result):
    influence = result.get_influence()
    cooks_d = influence.cooks_distance[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(cooks_d)), cooks_d, color='blue', label='Cook\'s Distance')
    plt.axhline(y=4/len(cooks_d), color='red', linestyle='--', label='Threshold (4/n)')
    plt.xlabel('Index')
    plt.ylabel('Cook\'s Distance')
    plt.title('Cook\'s Distance')
    plt.legend()
    plt.tight_layout()
    plt.show()

    outliers = np.where(cooks_d > 4/len(cooks_d))[0]
    return outliers

def plot_multiple_roc(models_info):
    plt.figure(figsize=(8, 6))
    for label, y_true, y_prob, color in models_info:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{label} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_multiple_pr(models_info):
    plt.figure(figsize=(8, 6))
    for label, y_true, y_prob, color in models_info:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, color=color, label=f'{label} (AUC = {pr_auc:.4f})')

    plt.plot([0, 1], [0.5, 0.5], color='gray', linestyle='--', label='Random Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()
