import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

def get_best_threshold(model, val_X, val_y):
    """
    Finds the optimal threshold that maximizes the F1 score based on predicted probabilities.

    Args:
        model: A trained classifier that has a `predict_proba` method.
        val_X: Features of the validation set (numpy array or pandas DataFrame).
        val_y: True labels of the validation set (numpy array or pandas Series).

    Returns:
        best_threshold (float): The threshold that yields the best F1 score.
    """
    # Get predicted probabilities for the positive class
    y_probs = model.predict_proba(val_X)[:, 1]
    f1s = []
    
    # Create a range of thresholds between 0.0 and 1.0, with a step of 0.025
    thresholds = np.arange(0.0, 1.0, 0.025)
    
    # Iterate through thresholds, calculate F1 score, and store the results
    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        f1s.append(f1_score(val_y, y_pred_thresh))
    
    # Find the threshold that maximizes the F1 score
    best_threshold = thresholds[np.argmax(f1s)]
    return best_threshold

def compute_metrics(model, inputs, targets, name='', threshold=0.5):
    """
    Computes classification metrics (precision, recall, F1 score) and displays a confusion matrix.

    Args:
        model: A trained classifier with a `predict_proba` method.
        inputs: Feature data for predictions (numpy array or pandas DataFrame).
        targets: True target labels (numpy array or pandas Series).
        name (str): Optional name to use for labeling the confusion matrix plot (default: '').
        threshold (float): The decision threshold to use for classifying predictions (default: 0.5).

    Returns:
        None
    """
    # Get predicted probabilities for the positive class
    preds_probas = model.predict_proba(inputs)[:, 1]
    # Apply threshold to convert probabilities to binary predictions
    preds = (preds_probas >= threshold).astype(int)
    
    # Compute precision, recall, and F1 score
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    
    # Display the computed metrics
    print(f"Precision score: {precision:.2f}")
    print(f"Recall score: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    # Build absolute and percentage confusion matrix
    cm_abs = confusion_matrix(targets, preds)
    cm_percentage = confusion_matrix(targets, preds, normalize='true')
    
    # Plot confusion matrix
    plt.figure()
    ax = sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues')
    
    # Add absolute values to confusion matrix
    threshold = cm_abs.max() / 2
    for i in range(cm_abs.shape[0]):
        for j in range(cm_abs.shape[1]):
            ax.text(j + 0.5, i + 0.6, f'\n({cm_abs[i, j]})',
                    ha='center', va='center', fontsize=10,
                    color="white" if cm_abs[i, j] < threshold else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

def compute_auroc_and_build_roc(model, inputs, targets, name=''):
    """
    Computes AUROC and plots the ROC curve.

    Args:
        model: A trained classifier with a `predict_proba` method.
        inputs: Feature data for predictions (numpy array or pandas DataFrame).
        targets: True target labels (numpy array or pandas Series).
        name (str): Optional name to label the ROC curve plot (default: '').

    Returns:
        None
    """
    # Get predicted probabilities for the positive class
    y_pred_proba = model.predict_proba(inputs)[:, 1]
    
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(targets, y_pred_proba, pos_label=1)
    
    # Compute Area Under the ROC Curve (AUROC)
    roc_auc = auc(fpr, tpr)
    print(f'AUROC for {name}: {roc_auc:.2f}')
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()