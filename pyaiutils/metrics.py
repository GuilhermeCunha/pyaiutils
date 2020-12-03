import pandas as pd
import numpy as np
import sklearn
import os
from . import utils

def recall(tp, p):
    return tp / p

def specificity(tn, n):
    return tn / n

def accuracy(tn, tp, p, n):
    return (tn + tp) / (p + n)

def precision(tp, fp):
    return tp / (fp + tp)

def f1_score(y_true, y_pred):
    assert len(np.shape(y_true)) == 1, "y_true must be a 1-dimension array of integers"
    assert len(np.shape(y_pred)) == 1, "y_pred must be a 1-dimension array of integers"

    return sklearn.metrics.f1_score(y_true, y_pred, average=None)

def prc_auc(y_true, y_pred, class_names):
    n_classes = len(class_names)
    assert (len(np.shape(y_pred)) > 1 and np.shape(y_pred)[-1] == n_classes), f"y_pred must be a n-dimension array like (n_samples, {n_classes})"
    assert (len(np.shape(y_true)) > 1 and np.shape(y_true)[-1] == n_classes), f"y_true must be a n-dimension array like (n_samples, {n_classes})"

    precision = dict()
    recall = dict()
    average_precision = []
    for i in range(n_classes):
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_true[:, i],
                                                                    y_pred[:, i])
        average_precision.append(
            sklearn.metrics.average_precision_score(y_true[:, i], y_pred[:, i]))
    return average_precision


def roc_auc(y_true, y_pred, class_names):
    n_classes = len(class_names)
    assert (len(np.shape(y_pred)) > 1 and np.shape(y_pred)[-1] == n_classes), f"y_pred must be a n-dimension array like (n_samples, {n_classes})"
    assert (len(np.shape(y_true)) > 1 and np.shape(y_true)[-1] == n_classes), f"y_true must be a n-dimension array like (n_samples, {n_classes})"

    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc.append(sklearn.metrics.auc(fpr[i], tpr[i]))
    return roc_auc


def get_metrics(y_test, y_pred, class_names, save_path=None):
    """Returns and saves a dataframe containing the `['F1', 'ROC AUC', 'PRC AUC', 'Precision', 'Recall', 'Specificity', 'Accuracy']` metrics.
    
    Parameters
    ----------
    y_true : 1-dimension array of integers
        Ground truth (correct) target values.

    y_pred : 1-dimension array of integers
        Estimated targets as returned by a classifier.

    class_names : 1-dimension array of strings
        List of labels containing each class of the dataset

    save_path : string, default=None
        Path to the folder where the metrics are to be saved
    """


    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    n_classes = len(class_names)

    assert len(np.shape(y_test)) == 1, "y_test must be a 1-dimension array of integers"
    assert (len(np.shape(y_pred)) > 1 and np.shape(y_pred)[-1] == n_classes), f"y_pred must be a n-dimension array like (n_samples, {n_classes})"
    assert len(y_test) == len(y_pred), "y_test and y_pred must have the same size"
    assert (isinstance(class_names, list) and isinstance(class_names[0], str)), "class_names must be a 1-dimension array of strings"

    y_pred_1d = utils.to_1d(y_pred)

    y_test_categorical = utils.to_categorical(y_test, n_classes=n_classes)

    matrix = sklearn.metrics.confusion_matrix(y_test, y_pred_1d, labels=np.arange(n_classes))


    TP = np.diag(matrix)
    FP = matrix.sum(axis=0) - TP
    FN = matrix.sum(axis=1) - TP
    TN = matrix.sum() - (FP + FN + TP)

    P = TP+FN
    N = TN+FP

    
    metrics_ = pd.DataFrame()
    rows = list(class_names).copy()
    rows.append('Média')
    metrics_['Classes'] = rows


    _f1 = np.around(f1_score(y_test, y_pred_1d), decimals=2)
    _f1 = np.append(_f1, np.around(np.mean(_f1), decimals=2))

    _roc_auc = np.around(roc_auc(y_test_categorical, y_pred, class_names), decimals=2)
    _roc_auc = np.append(_roc_auc, np.around(np.mean(_roc_auc), decimals=2))

    _prc_auc = np.around(prc_auc(y_test_categorical, y_pred, class_names), decimals=2)
    _prc_auc = np.append(_prc_auc, np.around(np.mean(_prc_auc), decimals=2))

    _precision = np.around(precision(TP, FP), decimals=2)
    _precision = np.append(_precision, np.around(
        np.mean(_precision), decimals=2))

    _recall = np.around(recall(TP, P), decimals=2)
    _recall = np.append(_recall, np.around(np.mean(_recall), decimals=2))
    _specificity = np.around(specificity(TN, N), decimals=2)
    _specificity = np.append(_specificity, np.around(
        np.mean(_specificity), decimals=2))

    _accuracy = np.around(accuracy(TN, TP, P, N), decimals=2)
    _accuracy = np.append(_accuracy, np.around(np.mean(_accuracy), decimals=2))

    metrics_["F1"] = _f1
    metrics_["ROC AUC"] = _roc_auc
    metrics_["PRC AUC"] = _prc_auc
    metrics_["Precision"] = _precision
    metrics_["Recall"] = _recall
    metrics_["Specificity"] = _specificity
    metrics_["Accuracy"] = _accuracy

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        metrics_.to_csv(os.path.join(save_path, 'metrics.csv'),
                        index=False, header=True)
    return metrics_