import pandas as pd
import numpy as np
import sklearn
import os
from . import utils

def recall(tp, p):
    result = tp / p
    return [0 if (x != x) else x for x in result]

def specificity(tn, n):
    result = tn / n
    return [0 if (x != x) else x for x in result]

def accuracy(tn, tp, p, n):
    result = (tn + tp) / (p + n)
    return [0 if (x != x) else x for x in result]

def precision(tp, fp):
    result = tp / (fp + tp)

    return [0 if (x != x) else x for x in result]

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
    labels = np.arange(n_classes)

    assert len(np.shape(y_test)) == 1, "y_test must be a 1-dimension array of integers"
    assert (len(np.shape(y_pred)) > 1 and np.shape(y_pred)[-1] == n_classes), f"y_pred must be a n-dimension array like (n_samples, {n_classes})"
    
    assert (isinstance(class_names, list) and isinstance(class_names[0], str)), "class_names must be a 1-dimension array of strings"

    y_pred_1d = utils.to_1d(y_pred)

    assert len(y_test) == len(y_pred_1d), "y_test and y_pred must have the same size"

    y_test_categorical = utils.to_categorical(y_test, n_classes=n_classes)

    matrix = sklearn.metrics.confusion_matrix(y_test, y_pred_1d, labels=labels)


    model_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_1d)
    model_prc_auc = sklearn.metrics.average_precision_score(y_test_categorical, y_pred)
    model_roc_auc= sklearn.metrics.roc_auc_score(y_test_categorical, y_pred)

    
    

    TP = np.diag(matrix)
    FP = matrix.sum(axis=0) - TP
    FN = matrix.sum(axis=1) - TP
    TN = matrix.sum() - (FP + FN + TP)

    P = TP+FN
    N = TN+FP

    metrics_ = pd.DataFrame()
    rows = list(class_names).copy()
    rows.append('Mean')
    rows.append('Min')
    rows.append('Max')
    metrics_['Classes'] = rows

    

    _f1 = np.around(f1_score(y_test, y_pred_1d), decimals=2)

    _roc_auc = np.around(roc_auc(y_test_categorical, y_pred, class_names), decimals=2)

    _prc_auc = np.around(prc_auc(y_test_categorical, y_pred, class_names), decimals=2)

    _precision = np.around(precision(TP, FP), decimals=2)

    _recall = np.around(recall(TP, P), decimals=2)

    _specificity = np.around(specificity(TN, N), decimals=2)

    _accuracy = np.around(accuracy(TN, TP, P, N), decimals=2)

    def get_statistics(value, class_names):

        mean = np.around(np.mean(value), decimals=2)
        argmin = np.argmin(value)
        argmax = np.argmax(value)

        return mean, class_names[argmin], class_names[argmax]


    metrics_["F1"] = np.append(_f1, get_statistics(_f1, class_names))
    metrics_["ROC AUC"] = np.append(_roc_auc, get_statistics(_roc_auc, class_names))
    metrics_["PRC AUC"] =  np.append(_prc_auc, get_statistics(_prc_auc, class_names))
    metrics_["Precision"] =  np.append(_precision, get_statistics(_precision, class_names))
    metrics_["Recall"] =  np.append(_recall, get_statistics(_recall, class_names))
    metrics_["Specificity"] =  np.append(_specificity, get_statistics(_specificity, class_names))
    metrics_["Accuracy"] =  np.append(_accuracy, get_statistics(_accuracy, class_names))

    model_metrics = [["accuracy" , model_accuracy], ["prc_auc", model_prc_auc], ["roc_auc", model_roc_auc]]

    model_metrics_df = pd.DataFrame(model_metrics, columns=['metric', 'value'])

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        metrics_.to_csv(os.path.join(save_path, 'metrics.csv'),
                        index=False, header=True)
        model_metrics_df.to_csv(os.path.join(save_path, 'model_metrics.csv'),
                        index=False, header=True)
                        
    return metrics_ , model_metrics_df