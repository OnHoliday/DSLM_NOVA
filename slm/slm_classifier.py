from numpy import vstack, log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, column_or_1d

from common.metrics import METRICS_DICT
from slm.base_slm import BaseSLM


class SLMClassifier(BaseSLM):
    """Semantic Learning Machine (SLM) classifier.

    SLM is a neural network construction algorithm. It generates a sample
    of neural networks and searches on the best performer neural network's neighborhood new topologies
    with better loss value. The search is conducted by a mutation operator that acts upon the hidden
    layers of the neural networks.

    Parameters
    ----------
    sample_size : int, optional, default 5
        Number of neural networks that should be randomly generated at the beginning of the algorithm.
        Must be a positive number.

    neighborhood_size : int, optional, default 5
        Number of neighbors that should be generated during the search at each iteration. Must be a
        positive number.

    max_iter : int, optional, default 100
        Maximum number of iterations. SLM iterates until a given stopping criterion (if exists) is met
        or this number of iterations. Must be a positive number.

    learning_step : {'optimized'} or float, optional, default 0.000001
        Weight for connection in the last hidden layer to output neuron.

    learning_step_solver : {'numpy_pinv', 'numpy_lstsq', 'scipy_pinv', 'scipy_pinv2', 'scipy_lstsq'}, optional, default 'numpy_pinv'
        Function to use when calculating the inverse of a matrix or the least-squares solution
        for equation Ax = b.

        numpy_pinv - Function from numpy library that computes Moore-Penrose pseudo-inverse
        of a matrix (used to invert partial semantics' matrix). It uses singular-value
        decomposition (SVD).

        numpy_lstsq - Function from numpy library that computes least-squares solution for
        Ax = b.

        scipy_pinv . Function from scipy library that computes Moore-Penrose pseudo-inverse
        of a matrix (used to invert partial semantics' matrix). It uses a least-square solver.

        scipy_pinv2 - Function from scipy library that computes Moore-Penrose pseudo-inverse
        of a matrix (used to invert partial semantics' matrix). It uses singular-value
        decomposition (SVD).

        scipy_lstsq - Function from scipy library that computes least-squares solution for
        Ax = b.

    activation_function : {'identity', 'logistic', 'tanh', 'relu'}, optional, default 'relu'
        Activation function for the output layer.

    activation_function_for_hidden_layers : {'identity', 'logistic', 'tanh', 'relu', 'random'} or None, optional, default None
        Activation function for the hidden layer(s).

    prob_activation_hidden_layers : float, optional, default 0.1,
        Probability of the neurons on hidden layers have an activation function associated.
        Must be between 0 and 1. Only used when activation_function_for_hidden_layers is not None.

    mutation_operator : {'all_hidden_layers'}, optional, default 'all_hidden_layers'
        Type of mutation operator used to generate the neighborhood of a neural network.

    init_minimum_layers : int, optional, default 1
        Minimum number of hidden layers neural networks should have at the moment of their
        creation.

    init_max_layers : int, optional, default 4
        Maximum number of hidden layers neural networks should have at the moment of their
        creation.

    init_maximum_neurons_per_layer : int, optional, default 5
        Maximum number of neurons that each hidden layer should have at the moment of generating
        neural networks.

    maximum_new_neurons_per_layer : int, optional, default 3
        Maximum number of neurons that each hidden layer should receive at the moment of generating
        a neural network's neighborhood.

    maximum_neuron_connection_weight : float, optional, default 1.0
        Maximum value a weight connection between two neurons can have.

    maximum_bias_weight : float, optional, default 1.0
        Maximum value a bias of a neuron can have.

    random_state : int, RandomState instance or None, optional, default None
        Works in the same fashion as scikit-learn random_state parameter.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    stopping_criterion : {'edv', 'tie'} or None, optional, default None
        Additional criterion that can be used to stop training process and avoid
        overfitting. The possible criteria are:

        Error Deviation Variation (edv) - measures the percentage of neural
        networks that reduce the error deviation in comparison with the error
        deviation of the current best model.

        Training Improvement Effectiveness (tie) - measures the effectiveness of the
        semantic variation operator used to perform the sampling, i.e., the percentage
        of times that the operator is able to produce a neural network that is superior
        to the current model.

        References: GonÃ§alves, I., Fonseca, C. M., Silva, S., & Castelli, M. (2017).
            Unsure when to stop? : Ask your semantic neighbors. In GECCO 2017 - Proceedings
            of the 2017 Genetic and Evolutionary Computation Conference (pp. 929-936).

    edv_threshold : float, optional, default 0.25
        Minimum proportion of neural networks capable of reducing the error deviation.
        Must be a positive number. Only used if stopping_criterion is 'edv'.

    tie_threshold : float, optional, default 0.25
        Minimum proportion of neural networks that improved loss or score values from one
        iteration to the other. Must be a positive number. Only used if stopping criterion
        is 'tie'.

    sparse : bool, optional, default True
        Whether the network is fully-connected or not.

    minimum_sparseness : float, optional, default 0.01
        Minimum proportion of not established connections on the neural networks.
        Must be between [0.0, 1.0[. Only used if sparse is True.

    maximum_sparseness : float, optional, default 1.0
        Maximum proportion of not established connections on the neural networks.
        Must be between [0, 1[ and should be higher than minimum_sparseness.
        Only used if sparse is True.

    early_stopping : bool, optional, default False
        Works in a similar fashion as scikit-learn early_stopping parameter. It determines
        if the training should terminate when validation score is not improving.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for early stopping.
        Must be between ]0, 1[. Only used if early_stopping is True.

    tol : float, optional, default 0.0001
        Tolerance for the optimization, i.e., when the loss or score is not improving by at
        least "tol" for "n_iter_no_change" consecutive iterations, convergence is considered
        to be reached and training stops. Only used if early_stopping is True.

    n_iter_no_change : int, optional, default 10
        Maximum number of epochs to not meet "tol" improvement.
        Must be a positive number. Only used if early_stopping is True.

    metric : {'rmse', 'cross_entropy_loss'},
             optional, default 'rmse'
        Evaluation metric to evaluate neural networks' performance during training.

    prob_skip_connection : float, optional, default 0.0,
        Probability of the neurons on hidden layers establish a connection with another neuron of
        a layer other than the previous one.
        Must be between 0 and 1.

    Attributes
    ----------
    # to be decided which attributes are shown to the user.

    References
    ----------
    Jagusch, J-B., Gonçalves, I. & Castelli, M.
        "Neuroevolution under unimodal error landscapes: An exploration of the semantic learning
        machine algorithm" In: Proceedings of the 2018 Genetic and Evolutionary Computation Conference
        Companion, pp. 159-160 2

    I. Gonçalves, S. Silva, and C. M. Fonseca.
        “Semantic Learning Machine: A Feedforward Neural Network Construction Algorithm Inspired by
        Geometric Semantic Genetic Programming.” In: Progress in Artiﬁcial Intelligence. Vol. 9273.
        pp. 280–285
    """

    def __init__(self,
                 sample_size=5,
                 neighborhood_size=5,
                 max_iter=100,
                 learning_step=0.000001,
                 learning_step_solver='numpy_pinv',
                 activation_function='relu',
                 activation_function_for_hidden_layers=None,
                 prob_activation_hidden_layers=0.1,
                 mutation_operator='all_hidden_layers',
                 init_minimum_layers=1,
                 init_max_layers=4,
                 init_maximum_neurons_per_layer=5,
                 maximum_new_neurons_per_layer=3,
                 maximum_neuron_connection_weight=1.0,
                 maximum_bias_weight=1.0,
                 random_state=None,
                 verbose=False,
                 stopping_criterion=None,
                 edv_threshold=0.25,
                 tie_threshold=0.25,
                 sparse=True,
                 minimum_sparseness=0.01,
                 maximum_sparseness=1.0,
                 early_stopping=False,
                 validation_fraction=0.1,
                 tol=0.0001,
                 n_iter_no_change=10,
                 metric='rmse',
                 prob_skip_connection=0.0
                 ):

        self._estimator_type = "classifier"

        # Validate hyperparameters dependent on estimator_type:
        self._validate_metric_hyperparameter(metric)

        super().__init__(sample_size,
                         neighborhood_size,
                         max_iter,
                         learning_step,
                         learning_step_solver,
                         activation_function,
                         activation_function_for_hidden_layers,
                         prob_activation_hidden_layers,
                         mutation_operator,
                         init_minimum_layers,
                         init_max_layers,
                         init_maximum_neurons_per_layer,
                         maximum_new_neurons_per_layer,
                         maximum_neuron_connection_weight,
                         maximum_bias_weight,
                         random_state,
                         verbose,
                         stopping_criterion,
                         edv_threshold,
                         tie_threshold,
                         sparse,
                         minimum_sparseness,
                         maximum_sparseness,
                         early_stopping,
                         validation_fraction,
                         tol,
                         n_iter_no_change,
                         metric,
                         prob_skip_connection)

        self.classes_ = None
        self._label_binarizer = None

    def fit(self, X, y):
        """Fit the model to a data matrix X and a target matrix y.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels).

        Returns
        -------
        self : returns a trained SLM model.
        """

        # Validate X and y arrays:
        X, y = self._validate_input_and_target(X, y)
        # Get number of neurons for output layer (overwrites super class' attributes number_output_neurons):
        self._number_output_neurons = self._get_number_output_neurons(y)

        # Set validation sets for early_stopping:
        if self.early_stopping:
            # Should not stratify in multilabel classification:
            if self._number_output_neurons == 1:
                X, X_val, y, y_val = train_test_split(X, y, random_state=self._random_state, test_size=self.validation_fraction, stratify=y)
            else:
                X, X_val, y, y_val = train_test_split(X, y, random_state=self._random_state, test_size=self.validation_fraction, stratify=None)

            y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val, y_val = None, None

        # Fit data:
        return super()._fit(X, y, X_val, y_val)

    def predict(self, X, clip=False):
        """Predict labels for input data.

        Note: the code is heavily inspired on scikit-learn's code repository.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array of shape (n_samples, n_classes)
            The predicted classes.
        """
        if self.estimator_ is not None:  # The estimator is fitted
            # Check if input_data is in a valid format:
            X = self._validate_input(X)
            y_pred = super().predict(X)

            if self._number_output_neurons == 1:
                y_pred = y_pred.ravel()
            
            if clip:
                y_pred = y_pred.clip(0, 1)
                
            return y_pred
            #===================================================================
            # # The inverse_transform turns binary labels back to multi-class labels:
            # return self._label_binarizer.inverse_transform(y_pred)
            #===================================================================
        else:
            pass  # TODO: throw error/warning

    def predict_proba(self, X):
        """Probability estimates.

        Note: the code is heavily inspired on scikit-learn's code repository.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_prob : array of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        if self.estimator_ and self._validate_input(X):  # The estimator is fitted and input_data is in a valid format:
            y_pred = super().predict(X)

            if self._number_output_neurons == 1:
                y_pred = y_pred.ravel()

            if y_pred.ndim == 1:
                return vstack([1 - y_pred, y_pred]).T
            else:
                return y_pred
        else:
            pass  # TODO: error handling

    def predict_log_proba(self, X):
        """Return the log of probability estimates.

        Note: the code is heavily inspired on scikit-learn's code repository.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        log_y_prob : array of shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model, where classes are ordered as they are in
            `self.classes_`. Equivalent to log(predict_proba(X))
        """
        y_prob = self.predict_proba(X)
        return log(y_prob, out=y_prob)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Test data.

        y : array of shape (n_samples,)
            True labels for X.

        sample_weight : array of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of predictions.
        """
        predictions = self.predict(X)
        metric = METRICS_DICT.get('accuracy')
        return metric(y, predictions, sample_weight)

    def _validate_metric_hyperparameter(self, metric):
        """Verifies if the metric set by the user is suitable for SLMClassifier.

        Parameters
        ----------
        metric : string
            Metric defined by the user when initializing SLMClassifier.
        """
        supported_metrics = ('rmse', 'cross_entropy_loss')
        if metric not in supported_metrics:
            raise ValueError("metric '%s' is not supported. Supported metrics are %s." % (metric, supported_metrics))

    def _validate_input_and_target(self, X, y):
        """Input validation for standard estimators. Among other tasks, ensures
        X and y have the same length, enforces X to be 2D and y to be 1D, and X
        is also checked to be non-empty and containing only finite values.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data requiring validation.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (real values).

        Returns
        -------
        X_converted : object
           Input data converted and validated.

        y_converted : object
           Target values converted and validated.
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], multi_output=True)

        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        return X, self._label_binarizer.transform(y)

    def _get_number_output_neurons(self, y):
        """Returns the number of neurons for the output layer of neural networks."""
        return y.shape[1]
