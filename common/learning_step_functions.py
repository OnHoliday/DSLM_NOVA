from numpy.linalg import pinv as np_pinv, lstsq as np_lstsq

from numpy import dot
from scipy.linalg import pinv as sp_pinv, pinv2 as sp_pinv2, lstsq as sp_lstsq


#################################################################################
#### Functions used to calculate optimized learning step for neural networks ####
#################################################################################
def scipy_pinv(partial_semantics, delta_target):
    """Computes Moore-Penrose pseudo-inverse of partial_semantics matrix by using
    a least-squares solver (scipy library) and returns the dot product between the
    inverted matrix and the delta_target vector.

    Parameters
    ----------
    partial_semantics : {array-like, sparse matrix}, shape (n_samples, num_neurons)
        Semantics retrieved from the last neurons of the last hidden layer of a
        neural network.

    delta_target : {array-like, sparse matrix}, shape (n_samples,)
        Distance between predictions from a given neural network and the target
        vector.

    Returns
    -------
    optimal_weights : {array-like, sparse matrix}, shape (num_neurons,)
        Value of learning steps.
    """
    partial_semantics_inverse = sp_pinv(partial_semantics)
    optimal_weights = dot(partial_semantics_inverse, delta_target)
    return optimal_weights


def scipy_pinv2(partial_semantics, delta_target):
    """Computes Moore-Penrose pseudo-inverse of partial_semantics matrix by using
    singular-value decomposition (scipy library) and returns the dot product between
    the inverted matrix and the delta_target vector.

    Parameters
    ----------
    partial_semantics : {array-like, sparse matrix}, shape (n_samples, num_neurons)
        Semantics retrieved from the last neurons of the last hidden layer of a
        neural network.

    delta_target : {array-like, sparse matrix}, shape (n_samples,)
        Distance between predictions from a given neural network and the target
        vector.

    Returns
    -------
    optimal_weights : {array-like, sparse matrix}, shape (num_neurons,)
        Value of learning steps.
    """
    partial_semantics_inverse = sp_pinv2(partial_semantics)
    optimal_weights = dot(partial_semantics_inverse, delta_target)
    return optimal_weights


def scipy_lstsq(partial_semantics, delta_target):
    """Computes least-squares of partial_semantics matrix in order to solve the
    equation Ax = b (scipy library), and returns the dot product between the
    inverted matrix and the delta_target vector.

    Parameters
    ----------
    partial_semantics : {array-like, sparse matrix}, shape (n_samples, num_neurons)
        Semantics retrieved from the last neurons of the last hidden layer of a
        neural network.

    delta_target : {array-like, sparse matrix}, shape (n_samples,)
        Distance between predictions from a given neural network and the target
        vector.

    Returns
    -------
    optimal_weights : {array-like, sparse matrix}, shape (num_neurons,)
        Value of learning steps.
    """
    optimal_weights, residuals, rank, singular_values = sp_lstsq(partial_semantics, delta_target)
    return optimal_weights


def numpy_pinv(partial_semantics, delta_target):
    """Computes Moore-Penrose pseudo-inverse of partial_semantics matrix by using
    singular-value decomposition (numpy library) and returns the dot product between
    the inverted matrix and the delta_target vector.

    Parameters
    ----------
    partial_semantics : {array-like, sparse matrix}, shape (n_samples, num_neurons)
        Semantics retrieved from the last neurons of the last hidden layer of a
        neural network.

    delta_target : {array-like, sparse matrix}, shape (n_samples,)
        Distance between predictions from a given neural network and the target
        vector.

    Returns
    -------
    optimal_weights : {array-like, sparse matrix}, shape (num_neurons,)
        Value of learning steps.
    """
    partial_semantics_inverse = np_pinv(partial_semantics)
    #===========================================================================
    # partial_semantics_inverse = np_pinv(partial_semantics, rcond=1e-10)
    #===========================================================================
    optimal_weights = dot(partial_semantics_inverse, delta_target)
    return optimal_weights


def numpy_lstsq(partial_semantics, delta_target):
    """Computes least-squares of partial_semantics matrix in order to solve the
    equation Ax = b (numpy library), and returns the dot product between the
    inverted matrix and the delta_target vector.

    Parameters
    ----------
    partial_semantics : {array-like, sparse matrix}, shape (n_samples, num_neurons)
        Semantics retrieved from the last neurons of the last hidden layer of a
        neural network.

    delta_target : {array-like, sparse matrix}, shape (n_samples,)
        Distance between predictions from a given neural network and the target
        vector.

    Returns
    -------
    optimal_weights : {array-like, sparse matrix}, shape (num_neurons,)
        Value of learning steps.
    """
    optimal_weights, residuals, rank, singular_values = np_lstsq(partial_semantics, delta_target, rcond=None)
    return optimal_weights

###########################################################
#### Dictionaries of optimized learning step functions ####
###########################################################


LEARNING_STEP_FUNCTIONS_DICT = {  # numpy_lstsq, scipy_lstsq, scipy_pinv, scipy_pinv2
    'scipy_pinv': scipy_pinv,
    'scipy_pinv2': scipy_pinv2,
    'scipy_lstsq': scipy_lstsq,
    'numpy_pinv': numpy_pinv,
    'numpy_lstsq': numpy_lstsq
}
