import sys
sys.path.append('../../')
from slm.slm_classifier import SLMClassifier

from numpy import vstack, log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, column_or_1d

from common.metrics import METRICS_DICT
from slm.base_slm import BaseSLM

from collections import defaultdict
from inspect import signature
from operator import attrgetter
import timeit

from sklearn.utils.validation import check_random_state, check_array

from common.activation_functions import ACTIVATION_FUNCTIONS_DICT
from common.learning_step_functions import LEARNING_STEP_FUNCTIONS_DICT
from common.metrics import METRICS_DICT
from common.metrics import calculate_accuracy
from common.conv_neural_network_builder import ConvNeuralNetworkBuilder
from common.stopping_criteria import STOPPING_CRITERION_FUNCTIONS_DICT
from common.utilities import is_better
from dslm_new.algorithm.deep_semantic_learning_machine.initialiation import init_conv_standard
from dslm_new.algorithm.deep_semantic_learning_machine.mutation import simple_conv_mutation


class DSLMApp(SLMClassifier):
    def __init__(self, sample_size, neighborhood_size, max_iter, learning_step, learning_step_solver, activation_function, activation_function_for_hidden_layers, activation_function_for_conv_layers,prob_activation_conv_layers,
                 prob_activation_hidden_layers, mutation_operator, init_minimum_layers, init_max_layers, init_maximum_neurons_per_layer, maximum_new_neurons_per_layer,
                 maximum_neuron_connection_weight, maximum_bias_weight, random_state, verbose, stopping_criterion, edv_threshold, tie_threshold, sparse,
                 minimum_sparseness, maximum_sparseness, early_stopping, validation_fraction, tol, n_iter_no_change, metric, prob_skip_connection):

        self.sample_size=sample_size
        self.neighborhood_size=neighborhood_size
        self.max_iter=max_iter
        self.learning_step=learning_step
        self.learning_step_solver=learning_step_solver
        self.activation_function=activation_function
        self.activation_function_for_hidden_layers=activation_function_for_hidden_layers
        self.activation_function_for_conv_layers=activation_function_for_conv_layers
        self.prob_activation_hidden_layers=prob_activation_hidden_layers
        self.prob_activation_conv_layers=prob_activation_conv_layers
        self.mutation_operator=mutation_operator
        self.init_minimum_layers=init_minimum_layers
        self.init_max_layers=init_max_layers
        self.init_maximum_neurons_per_layer=init_maximum_neurons_per_layer
        self.maximum_new_neurons_per_layer=maximum_new_neurons_per_layer
        self.maximum_neuron_connection_weight=maximum_neuron_connection_weight
        self.maximum_bias_weight=maximum_bias_weight
        self.random_state=random_state
        self.verbose=verbose
        self.stopping_criterion=stopping_criterion
        self.edv_threshold=edv_threshold
        self.tie_threshold=tie_threshold
        self.sparse=sparse
        self.minimum_sparseness=minimum_sparseness
        self.maximum_sparseness=maximum_sparseness
        self.early_stopping=early_stopping
        self.validation_fraction=validation_fraction
        self.tol=tol
        self.n_iter_no_change=n_iter_no_change
        self.metric=metric
        self.prob_skip_connection=prob_skip_connection

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

        if self.mutation_operator == 'simple_conv_mutation':
            self._mutation_operator = simple_conv_mutation

        if self.activation_function_for_conv_layers:
            self.cnn_activation_functions_ids = self._get_activation_functions_ids(self.activation_function_for_conv_layers)
        else:
            self.prob_activation_cnn_layers = None
            self.cnn_activation_functions_ids = None

    def fit(self, X, y, time_print=False):
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

        # Update input_data and target_vector attributes:
        self._input_data = X
        self._target_vector = y.astype(float)
        # Check validation datasets for early_stopping:
        if X_val is not None and y_val is not None:
            self._X_validation = X_val
            self._y_validation = y_val

        # Create list with input neurons and passes the input data as semantics:
        input_layer = ConvNeuralNetworkBuilder.create_input_neurons(self._input_data)

        # Generate N initial random NNs:
        for _ in range(self.sample_size):
            # Note: when neural networks are added to the sample, their semantics and predictions have been already calculated
            self._sample.append(init_conv_standard(self._input_data, self._target_vector, input_layer, self._random_state)) #todo add here much more parameters from main file ...

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
                    if is_better(offspring.get_loss(), self._current_best.get_loss(),
                                 greater_is_better=self._greater_is_better):
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
            parent_neural_network = ConvNeuralNetworkBuilder.clone_neural_network(self._current_best)

            # Generate one child using the mutation operator:
            child = self._mutation_operator(parent_neural_network, self._input_data, self._random_state,
                                            self.learning_step, self._sparseness, self.maximum_new_neurons_per_layer,
                                            self.maximum_neuron_connection_weight, self.maximum_bias_weight,
                                            self._target_vector, delta_target, self._learning_step_function,
                                            self._hidden_activation_functions_ids, self.cnn_activation_functions_ids,
                                            self.prob_activation_hidden_layers, self.prob_activation_conv_layers)

            self._sample.append(child)

        return self._sample





