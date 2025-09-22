
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_print(y_true, y_pred, target_names=None):
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
