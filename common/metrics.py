from numpy import sqrt, where, argmax
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, recall_score, r2_score, mean_squared_error


#######################################################
#### Metrics made available throughout SLM package ####
#######################################################
def calculate_root_mean_squared_error(y_true, y_pred, sample_weight=None):
    """Root mean squared error

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape = (n_samples), optional
        Sample weights to compute the weighted root mean squared error.

    Returns
    -------
    loss : float or ndarray of floats
    """
    
    if y_true.ndim != y_pred.ndim:
        print('y_true.ndim != y_pred.ndim')
    
    if sample_weight is not None:
        return sqrt(mean_squared_error(y_true, y_pred, sample_weight))
    else:
        return sqrt(mean_squared_error(y_true, y_pred))


def calculate_r2(y_true, y_pred, sample_weight=None):
    """R^2

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape = (n_samples), optional
        Sample weights to compute the weighted root mean squared error.

    Returns
    -------
    loss : float or ndarray of floats
    """
    if sample_weight is not None:
        return r2_score(y_true, y_pred, sample_weight)
    else:
        return r2_score(y_true, y_pred)


def calculate_accuracy(y_true, y_pred, sample_weight=None):
    """Accuracy score

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples)
        Sample weights.

    Returns
    -------
    score : float
        The number of correctly classified samples.
    """
    
    if y_true.ndim != y_pred.ndim:
        print('y_true.ndim != y_pred.ndim')

    #===========================================================================
    # Both predictions and targets need to be converted into bool values before being sent to accuracy_score function:
    #===========================================================================
    #===========================================================================
    # if y_true.ndim == 1:
    #     y_pred = where(y_pred >= 0.5, 1, 0)
    #===========================================================================
    #===========================================================================
    # else:
    #     if y_true.shape[1] > 1:
    #         y_true = argmax(y_true, axis=1)
    #         y_pred = argmax(y_pred, axis=1)
    #     else:
    #         y_true = y_true.ravel()
    #         y_pred = y_pred.ravel()
    #         y_pred = where(y_pred >= 0.5, 1, 0)
    #===========================================================================
    
    return accuracy_score(y_true, y_pred, normalize=True, sample_weight=sample_weight)


def calculate_AUROC(y_true, y_pred):
    """Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    bound: bool
        Indicates if the estimated target values should be bounded, i.e.,
        their values should range between 0 and 1.

    Returns
    -------
    auc : float
    """
    return roc_auc_score(y_true, y_pred)


def calculate_cross_entropy_loss(y_true, y_pred, sample_weight=None):
    """Binary cross-entropy loss

    Note: The log loss is only defined for two or more labels.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's predict_proba method.

    Returns
    -------
    loss : float
    """
    
    if y_true.ndim != y_pred.ndim:
        print('y_true.ndim != y_pred.ndim')

    import sklearn.neural_network._base as base
    return base.log_loss(y_true, y_pred)


def calculate_recall(y_true, y_pred):
    """Recall score

    Warning: it can only be used for binary classification.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    score : float
        The ratio of samples classified as positive.
    """
    # Both predictions and targets need to be converted into bool values before being sent to accuracy_score function:
    return recall_score(where(y_true >= 0.5, True, False), where(y_pred >= 0.5, True, False))


def calculate_f1_score(y_true, y_pred):
    """F1 score

    Warning: it can only be used for binary classification.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    score : float
        The a weighted average of the precision and recall scores (F1 score
        reaches its best value at 1 and worst score at 0).
    """
    # Both predictions and targets need to be converted into bool values before being sent to accuracy_score function:
    return f1_score(where(y_true >= 0.5, True, False), where(y_pred >= 0.5, True, False))

########################################
#### Dictionary of metric functions ####
########################################


METRICS_DICT = {
    'accuracy': calculate_accuracy,
    'recall': calculate_recall,
    'f1_score': calculate_f1_score,
    'auroc': calculate_AUROC,
    'cross_entropy_loss': calculate_cross_entropy_loss,
    'rmse': calculate_root_mean_squared_error,
    'r2': calculate_r2
}
