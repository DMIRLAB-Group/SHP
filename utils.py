import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def get_performance(fitted, real, threshold=0, name="NPHC", drop_diag=True):
    fitted = np.abs(fitted)
    if drop_diag:
        fitted = fitted - np.diag(np.diag(fitted))
        real = real - np.diag(np.diag(real))

    f1 = f1_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    precision = precision_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    recall = recall_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    temp_result = np.array((f1, precision, recall, threshold))
    result = pd.DataFrame(columns=['F1', "Precision", "Recall", "threshold"])
    result.loc[0] = temp_result
    result["method"] = name
    return result