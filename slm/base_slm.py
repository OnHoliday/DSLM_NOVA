from collections import defaultdict
from inspect import signature
from operator import attrgetter
import timeit

from sklearn.utils.validation import check_random_state, check_array

from common.activation_functions import ACTIVATION_FUNCTIONS_DICT
from common.learning_step_functions import LEARNING_STEP_FUNCTIONS_DICT
from common.metrics import METRICS_DICT
from common.metrics import calculate_accuracy
from common.neural_network_builder import NeuralNetworkBuilder
from common.stopping_criteria import STOPPING_CRITERION_FUNCTIONS_DICT
from common.utilities import is_better
from slm.initialization import init_standard

from .mutation import mutate_hidden_layers


class BaseSLM():
    """Base class for Semantic Learning Machine (SLM) classification and regression.

    Note: the code of methods _get_param_names(), get_params(), and set_params() was
          copied from the class BaseEstimator from scikit-learn package's code. These
          methods are needed in order to make BaseSLM estimators
          compatible with some of scikit-learn algorithms, namely GridSearchCV and
          RandomizedSearchCV.

    Warning: This class should not be used directly. Use derived classes instead.

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

        self.sample_size = sample_size
        self.neighborhood_size = neighborhood_size
        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        self.edv_threshold = edv_threshold
        self.tie_threshold = tie_threshold
        self.learning_step = learning_step
        self.learning_step_solver = learning_step_solver
        self.activation_function = activation_function
        self.activation_function_for_hidden_layers = activation_function_for_hidden_layers
        self.prob_activation_hidden_layers = prob_activation_hidden_layers
        self.mutation_operator = mutation_operator
        self.init_minimum_layers = init_minimum_layers
        self.init_max_layers = init_max_layers
        self.init_maximum_neurons_per_layer = init_maximum_neurons_per_layer
        self.maximum_new_neurons_per_layer = maximum_new_neurons_per_layer
        self.maximum_neuron_connection_weight = maximum_neuron_connection_weight
        self.maximum_bias_weight = maximum_bias_weight
        self.random_state = random_state
        self.verbose = verbose
        self.sparse = sparse
        self.minimum_sparseness = minimum_sparseness
        self.maximum_sparseness = maximum_sparseness
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.metric = metric
        self.prob_skip_connection = prob_skip_connection

        # Validate parameters defined by the user:
        if not self._validate_hyperparameters():
            self._validate_hyperparameters()

        # Use scikit-learn's function check_random_state to turn the seed into a np.random.RandomState instance:
        self._random_state = check_random_state(self.random_state)

        # Create remaining auxiliary variables after hyperparameter validation:
        self._sample = list()
        self._current_iteration = 0
        self._current_best = None
        self._next_best = None
        self._greater_is_better = None

        self._number_output_neurons = None
        self._input_data = None
        self._target_vector = None
        self._metric = METRICS_DICT.get(self.metric)

        # Dictionary encapsulating the details related to neural networks' connections and will be used in NeuralNetworkBuilder:
        self._sparseness = {
            'sparse': self.sparse,
            'minimum_sparseness': self.minimum_sparseness,
            'maximum_sparseness': self.maximum_sparseness,
            'prob_skip_connection': self.prob_skip_connection
        }

        if self.mutation_operator == 'all_hidden_layers':
            self._mutation_operator = mutate_hidden_layers

        if self.activation_function_for_hidden_layers:
            self._hidden_activation_functions_ids = self._get_activation_functions_ids(self.activation_function_for_hidden_layers)
        else:
            self.prob_activation_hidden_layers = None
            self._hidden_activation_functions_ids = None

        if self.learning_step == 'optimized':
            self._learning_step_function = LEARNING_STEP_FUNCTIONS_DICT.get(self.learning_step_solver)
        else:
            self._learning_step_function = None

        if self.stopping_criterion == 'edv':
            self._stopping_criterion_function = STOPPING_CRITERION_FUNCTIONS_DICT.get('edv')

        elif self.stopping_criterion == 'tie':
            self._stopping_criterion_function = STOPPING_CRITERION_FUNCTIONS_DICT.get('tie')

        if metric in ['accuracy', 'recall', 'f1_score', 'auroc', 'r2']:
            self._greater_is_better = True
        else:
            self._greater_is_better = False

        if self.early_stopping:
            self._best_validation_score = None
            self._no_improvement_count = None
            self._last_best_solution = None
            self._X_validation = None
            self._y_validation = None

        # Attributes to expose to users:
        self.estimator_ = None

    def __repr__(self):
        return "BaseSLM"

    def _fit(self, X, y, X_val=None, y_val=None, time_print=False):
        """Fit the model to a data matrix X and a target matrix y.

        Note: when _fit method of the super class is called, X and y have been
        already validated.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained SLM model.
        """

        # Update input_data and target_vector attributes:
        self._input_data = X
        self._target_vector = y.astype(float)
        # Check validation datasets for early_stopping:
        if X_val is not None and y_val is not None:
            self._X_validation = X_val
            self._y_validation = y_val
        
        # Create list with input neurons and passes the input data as semantics:
        input_layer = NeuralNetworkBuilder.create_input_neurons(self._input_data)

        # Generate N initial random NNs:
        for _ in range(self.sample_size):
            # Note: when neural networks are added to the sample, their semantics and predictions have been already calculated
            self._sample.append(init_standard(self._input_data, self._target_vector, input_layer, self._random_state)) #todo tu sie zacinam wejsc głebiej !

        # Get current best NN:
        self._current_best = self._evaluate_sample(self._target_vector)
        
        self._free_components()

        if self.verbose:
            self._print_epoch()

        # Initialize early_stopping process:
        if self.early_stopping:
            self._update_no_improvement_count()

        # Main loop (run N-1 epochs, with N = self.max_iter). The first epoch is considered to be the initialization previously done:
        for self._current_iteration in range(1, self.max_iter):
            
            iteration_start_time = timeit.default_timer()

            # Empty sample:
            self._sample.clear()

            # Apply GSM-NN mutation on the solutions sample:
            mutation_on_sample_start_time = timeit.default_timer()
            self._sample = self._apply_mutation_on_sample()
            mutation_on_sample_time = timeit.default_timer() - mutation_on_sample_start_time
            if time_print:
                print('\n\tmutation on sample = %.3f seconds' % (mutation_on_sample_time))

            start_time = timeit.default_timer()
            # Get new candidate to best NN:
            self._next_best = self._evaluate_sample(self._target_vector)
            time = timeit.default_timer() - start_time
            if time_print:
                print('\n\t_evaluate_sample = %.3f seconds' % (time))
            
            # Compare offspring with parent solution (which is stored in self.current_best).
            # This step is only required if 'EDV' is used as a stopping criterion:
            if self.stopping_criterion == 'edv':
                for offspring in self._sample:
                    if is_better(offspring.get_loss(), self._current_best.get_loss(), greater_is_better=self._greater_is_better):
                        offspring.update_parent(is_better_than_parent=True)
                    else:
                        offspring.update_parent(is_better_than_parent=False)

            # Check stopping criterion:
            if self.stopping_criterion:
                stop_training = self._apply_stopping_criterion()
                if stop_training:
                    print('Training process stopped earlier due to', self.stopping_criterion.upper(), 'criterion')
                    self._print_epoch()
                    # Training process stops (exit from main loop):
                    break

            # Check if there is a new best solution:
            self._get_best_solution()
            self._free_components_2()

            # Check early stopping:
            if self.early_stopping:
                stop_training = self._update_no_improvement_count()
                if stop_training:
                    print('Early stopping of training process.')
                    self._current_best = self._last_best_solution
                    self._print_epoch()
                    # Training process stops (exit from main loop):
                    break

            iteration_time = timeit.default_timer() - iteration_start_time
            if time_print:
                print('\n\titeration time = %.3f seconds\n' % (iteration_time))

            if self.verbose:  # and (self._current_iteration == 0 or self._current_iteration % 10 == 0 or self._current_iteration == self.max_iter - 1):
                # Print only every 10 generations or in the last iteration:
                self._print_epoch()

            self._current_iteration += 1
        
        # Store best solution as main estimator and clear sample:
        self.estimator_ = self._current_best
        self._sample.clear()
        self._current_best = None
        self._next_best = None

        if self.verbose:
            print(self.estimator_.get_topology())

        return self.estimator_

    def _free_components(self):
        for nn in self._sample:
            if nn != self._current_best:
                nn.free()
    
    def _free_components_2(self):
        for nn in self._sample:
            if nn != self._current_best:
                [[hn.free() for hn in hl] for hl in nn.new_neurons]
                [on.free() for on in nn.output_layer]

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, returns the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators, as well as on nested objects
        (such as pipelines).

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' % 
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def predict(self, X):
        """Predict using the SLM model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes or real values for input data.
        """

        self._input_data = X
        return self.estimator_.generate_predictions(X)

    def score(self, X, y):
        pass

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # Fetch the constructor or the original constructor before deprecation wrapping if any:
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect:
            return []

        # Introspect the constructor arguments to find the model parameters to represent:
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self':
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("Estimators to use with scikit-learn resources should always"
                                   " specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self':
        return sorted([p.name for p in parameters])

    def _print_epoch(self):
        """Print in the console loss value of the best estimator so far at current
        iteration of the training process."""
        print('Training loss at iteration {}: \t\t{:.5f}'.format(self._current_iteration, self._current_best.get_loss()))
        
        
        print('\t\t\tAccuracy\t%.5f%%' % (calculate_accuracy(self._target_vector.argmax(axis=1), self._current_best.get_predictions().argmax(axis=1)) * 100))

    def _get_best_solution(self):
        """Returns the best solution considering the current best and a candidate
        solution to that position.

        Returns
        -------
        current_best_estimator : NeuralNetwork
        """
        
        [on.free() for on in self._current_best.output_layer]
        self._current_best.predictions = None
        self._current_best = self._next_best
    
        return self._current_best

    def _evaluate_sample(self, y):
        """Returns best solution in the sample.

        Parameters
        ----------
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression) used to compute loss value.

        Returns
        -------
        best_estimator : NeuralNetwork
        """
        for neural_network in self._sample:
            neural_network.update_loss(self._metric(y, neural_network.get_predictions().clip(0, 1)))
            # [neural_network.update_loss(self._metric(y, neural_network.get_predictions())) for neural_network in self._sample]
            #===================================================================
            # if self.verbose:
            #     print('\t[Debug] loss = %.8f' % neural_network.get_loss())
            #===================================================================

        best_solution = max(self._sample, key=attrgetter('loss')) if self._greater_is_better else min(self._sample, key=attrgetter('loss'))
        return best_solution

    def _update_no_improvement_count(self):
        """Function called in the scope of early_stopping mechanism. It determines if training
        process should be stopped or not. The training process receives order to stop if validation
        score of the best estimator so far did not improved for more than 'n_iter_no_change'
        iterations.

        Returns
        -------
        stop_training_process : bool
            True if training process should be stopped, and False otherwise.
        """
        # Compute predictions for self._X_validation using current best solution:
        predictions = self._current_best.generate_predictions(self._X_validation)
        # Get validation score for current iteration and append result to scores tracker array:
        validation_score = self._metric(self._y_validation, predictions)

        # Restore previous predictions and loss value:
        self._current_best.generate_predictions(self._input_data)
        self._current_best.update_loss(self._metric(self._target_vector, self._current_best.get_predictions()))

        if self._current_iteration == 0:
            # Initialize parameters:
            self._best_validation_score = validation_score
            self._no_improvement_count = 0
            self._last_best_solution = self._current_best

            if self.verbose:
                print("Validation loss: %.8f" % validation_score)

        else:
            # Compare last validation score against best validation score and considering tol parameter:
            if is_better(self._best_validation_score + self.tol, validation_score, greater_is_better=self._greater_is_better):
                self._no_improvement_count += 1
            else:  # It has improved:
                self._no_improvement_count = 0
                self._best_validation_score = validation_score
                self._last_best_solution = self._current_best

            if self.verbose and (self._current_iteration % 10 == 0 or self._current_iteration == self.max_iter - 2):
                # Print only every 10 generations or in the last iteration:
                print("Validation loss: %.8f" % validation_score)

        # Check if training process should be stopped:
        return False if self._no_improvement_count <= self.n_iter_no_change else True

    def _get_delta_target(self):
        """Calculates distance of current best solution's predictions to target vector. In case
        of not existing a neural network marked as being the current best, it returns a copy of
        the target vector.

        Returns
        -------
        delta_target : array of shape (num_samples,)
            Distances to target vector.
        """
        delta_target = self._target_vector.copy()
        
        if self._current_best:
            delta_target -= self._current_best.get_predictions()

        return delta_target

    def _apply_mutation_on_sample(self):
        """Fills self._sample with N offspring generated from the current best solution, using
        a mutation operator, and with N being equal to self.neighborhood_size.

        Returns
        -------
        sample : array of shape (num_neighbors,)
            Sample containing N mutated (and fully functional) neural networks.
        """

        delta_target = self._get_delta_target() if self.learning_step == 'optimized' else None
        
        # Make N copies/clones of current best solution and apply mutation on each copy to obtain the offspring:
        for _ in range(self.neighborhood_size):
            parent_neural_network = NeuralNetworkBuilder.clone_neural_network(self._current_best)
             
            # Generate one child using the mutation operator:
            child = self._mutation_operator(parent_neural_network, self._input_data, self._random_state, self.learning_step, self._sparseness, self.maximum_new_neurons_per_layer,
                                                self.maximum_neuron_connection_weight, self.maximum_bias_weight, self._target_vector, delta_target, self._learning_step_function,
                                                self._hidden_activation_functions_ids, self.prob_activation_hidden_layers)
 
            self._sample.append(child)

        return self._sample

    def _apply_stopping_criterion(self):
        """Evaluate if training process should be stopped or not according to a
        given stopping criterion.

        Returns
        -------
        stop_training_process : bool
            True if training process should be stopped, and False otherwise.
        """
        if self.stopping_criterion == 'edv':
            return self._stopping_criterion_function(self._current_best, self._sample, self.edv_threshold, self._target_vector)
        else:  # self.stopping_criterion == 'tie':
            return self._stopping_criterion_function(self._current_best, self._sample, self.tie_threshold, self._greater_is_better)

    def _get_activation_functions_ids(self, activation_function_for_hidden_layers):
        """Extracts activation function (python function)."""
        if activation_function_for_hidden_layers is 'random':
            functions = list(ACTIVATION_FUNCTIONS_DICT.keys())
            functions.remove('identity')
            return functions
        else:
            # Parameter activation_function_for_hidden_layers contains only 1 function id:
            functions = list()
            functions.append(activation_function_for_hidden_layers)
            return functions

    def _validate_hyperparameters(self):
        """Applies validation rules over hyperparameters inserted by user."""
        if self.sample_size <= 0 or not isinstance(self.sample_size, int):
            raise ValueError("sample_size must be an integer > 0, got %s." % self.sample_size)

        if self.neighborhood_size <= 0 or not isinstance(self.neighborhood_size, int):
            raise ValueError("neighborhood_size must be an integer > 0, got %s." % self.neighborhood_size)

        if self.max_iter <= 0 or not isinstance(self.max_iter, int):
            raise ValueError("max_iter must be an integer > 0, got %s." % self.max_iter)

        if not isinstance(self.learning_step, float) and self.learning_step != 'optimized':
            raise ValueError("learning_step must be a float > 0 or 'optimized', got %s." % self.learning_step)
        elif isinstance(self.learning_step_solver, float) and self.learning_step <= 0:
            raise ValueError("learning_step must be a float > 0 or 'optimized', got %s." % self.learning_step)

        if self.learning_step_solver not in LEARNING_STEP_FUNCTIONS_DICT.keys():
            raise ValueError("learning_step_solver '%s' is not supported. Supported solvers are %s." % (self.learning_step_solver, LEARNING_STEP_FUNCTIONS_DICT.keys()))

        if self.activation_function not in ACTIVATION_FUNCTIONS_DICT.keys():
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s." % (self.activation_function, ACTIVATION_FUNCTIONS_DICT.keys()))

        if self.activation_function_for_hidden_layers is not None and self.activation_function_for_hidden_layers not in ACTIVATION_FUNCTIONS_DICT.keys() and self.activation_function_for_hidden_layers != 'random':
            raise ValueError("The activation '%s' is not supported. Supported "
                             "activations are %s." % (self.activation_function_for_hidden_layers, ACTIVATION_FUNCTIONS_DICT.keys()))

        if self.prob_activation_hidden_layers > 1.0 or self.prob_activation_hidden_layers < 0.0:
            raise ValueError("prob_activation_hidden_layers must be a float between 0.0 and 1.0, got %s." % self.prob_activation_hidden_layers)

        # supported_mutation_operators = ('all_hidden_layers')
        # if self.mutation_operator not in supported_mutation_operators:
        #     raise ValueError("mutation_operator '%s' is not supported. Supported operators are %s." % (self.mutation_operator, supported_mutation_operators))

        if not isinstance(self.init_minimum_layers, int) or self.init_minimum_layers <= 0:
            raise ValueError("init_minimum_layers must be an integer > 0, got %s." % self.init_minimum_layers)

        if not isinstance(self.init_max_layers, int) or self.init_max_layers <= 0:
            raise ValueError("init_max_layers must be an integer > 0, got %s." % self.init_max_layers)
        elif self.init_max_layers <= self.init_minimum_layers:
            raise ValueError("init_max_layers must higher than init_minimum_layers.")

        if not isinstance(self.init_maximum_neurons_per_layer, int) or self.init_maximum_neurons_per_layer <= 0:
            raise ValueError("init_maximum_neurons_per_layer must be an integer > 0, got %s." % self.init_maximum_neurons_per_layer)

        if not isinstance(self.maximum_new_neurons_per_layer, int) or self.maximum_new_neurons_per_layer <= 0:
            raise ValueError("maximum_new_neurons_per_layer must be an integer > 0, got %s." % self.maximum_new_neurons_per_layer)

        if not isinstance(self.maximum_neuron_connection_weight, float) or self.maximum_neuron_connection_weight <= 0:
            raise ValueError("maximum_neuron_connection_weight must be a float > 0, got %s." % self.maximum_neuron_connection_weight)

        if not isinstance(self.maximum_bias_weight, float) or self.maximum_bias_weight <= 0:
            raise ValueError("maximum_bias_weight must be a float > 0, got %s." % self.maximum_bias_weight)

        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be either True or False, got %s." % self.verbose)

        supported_stopping_criteria = ('tie', 'edv')
        if self.stopping_criterion is not None and self.stopping_criterion not in supported_stopping_criteria:
            raise ValueError("stopping_criterion '%s' is not supported. Supported criteria are None or %s." % (self.mutation_operator, supported_stopping_criteria))

        if not isinstance(self.edv_threshold, float) or self.edv_threshold > 1.0 or self.edv_threshold <= 0.0:
            raise ValueError("edv_threshold must be a float between 0.0 and 1.0, got %s." % self.edv_threshold)

        if not isinstance(self.tie_threshold, float) or self.tie_threshold > 1.0 or self.tie_threshold <= 0.0:
            raise ValueError("edv_threshold must be a float between 0.0 and 1.0, got %s." % self.tie_threshold)

        if not isinstance(self.sparse, bool):
            raise ValueError("sparse must be either True or False, got %s." % self.sparse)

        if not isinstance(self.minimum_sparseness, float) or self.minimum_sparseness > 1.0 or self.minimum_sparseness < 0.0:
            raise ValueError("minimum_sparseness must be a float between 0.0 and 1.0, got %s." % self.minimum_sparseness)

        if not isinstance(self.maximum_sparseness, float) or self.maximum_sparseness > 1.0 or self.maximum_sparseness <= 0.0:
            raise ValueError("maximum_sparseness must be a float between 0.0 and 1.0, got %s." % self.maximum_sparseness)
        elif self.maximum_sparseness < self.minimum_sparseness:
            raise ValueError("maximum_sparseness must be equal or higher than minimum_sparseness.")

        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be either True or False, got %s." % self.early_stopping)

        if self.validation_fraction < 0 or self.validation_fraction >= 1:
            raise ValueError("validation_fraction must be >= 0 and < 1, got %s" % self.validation_fraction)

        if not isinstance(self.n_iter_no_change, int) or self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be an integer > 0, got %s." % self.n_iter_no_change)

        if self.tol < 0:
            raise ValueError("tol must be > 0, got %s." % self.tol)

        if self.prob_skip_connection > 1.0 or self.prob_skip_connection < 0.0:
            raise ValueError("prob_skip_connection must be a float between 0.0 and 1.0, got %s." % self.prob_skip_connection)

    def _validate_input_and_target(self, X, y):
        pass

    def _validate_input(self, X):
        """Input validation for standard estimators.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data requiring validation.

        Returns
        -------
        X_converted : object
            The converted and validated X.
        """
        return check_array(X)

    def _get_number_output_neurons(self, y):
        """ """
        pass
