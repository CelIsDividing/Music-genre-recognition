import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")
    # plt.show()  # Commented out so it does not block the terminal

def print_classification_report(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)
    return report
