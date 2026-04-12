
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

# def plot_feature_importance(model, x):
#     """
#     Plot a bar chart showing the feature importances.
    
#     Args:
#         feature_names (list): List of feature names.
#         feature_importances (list): List of feature importance values.
#     """
#     fig, ax = plt.subplots()
#     ax = sns.barplot(x=model.feature_importances_, y=x.columns)
#     plt.title("Feature importance chart")
#     plt.xlabel("Importance")
#     plt.ylabel("Feature")
#     plt.tight_layout()
#     fig.savefig("feature_importance.png")

def plot_feature_importance(model, x):
    import pandas as pd

    coef_df = pd.DataFrame({
        "Feature": x.columns,
        "Importance": model.coef_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=coef_df, ax=ax)

    plt.title("Feature Importance (Linear Regression)")
    plt.tight_layout()

    fig.savefig("feature_importance.png")
   
 
#Plot actual vs predicted    
def plot_actual_vs_predicted(y_true, y_pred):
    

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred)
    
    # Perfect prediction line
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             linestyle='--')

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")

    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix for the given true and predicted labels.
    
    Args:
        y_true (numpy.ndarray): Array of true labels.
        y_pred (numpy.ndarray): Array of predicted labels.
        classes (list): List of class labels.
        normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
        title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=16)
    plt.show()