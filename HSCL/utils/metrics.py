from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def compute_metrics(y_true, y_pro):
    y_pred = np.argmax(y_pro, axis=1)
    y_score = y_pro[:, 1]
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    f_score = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_score)

    return accuracy*100, sensitivity*100, specificity*100, f_score*100, auc_score*100


