from copy import copy
from random import uniform, sample, randint
from statistics import median, mean
import pickle

from numpy import array, dot, resize, shape, concatenate, zeros, empty, empty_like, ones
from numpy import std
from numpy.core.multiarray import arange
from numpy.linalg import pinv
from numpy.random import choice as np_choice
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

from algorithms.common.algorithm import EvolutionaryAlgorithm
from algorithms.common.neural_network.connection import Connection
from algorithms.common.neural_network.neural_network import ConvNeuralNetwork, create_neuron, \
    create_output_neuron, create_cnn_neuron
from algorithms.common.neural_network.node import Sensor
from algorithms.semantic_learning_machine.lbfgs import LBFGS
from algorithms.semantic_learning_machine.mutation_operator import Mutation_CNN_1
from algorithms.semantic_learning_machine.solution import Solution


class DeepSemanticLearningMachine(EvolutionaryAlgorithm):
    """
    Class represents Semantic Learning Machine (SLM) algorithms:
    https://www.researchgate.net/publication/300543369_Semantic_Learning_Machine_
    A_Feedforward_Neural_Network_Construction_Algorithm_Inspired_by_Geometric_Semantic_Genetic_Programming
    Attributes:
        layer: Number of layers for base topology.
        learning_step: Weight for connection to output neuron.
        max_connections: Maximum connections for neuron.
        mutation_operator: Operator that augments neural network.
        next_champion: Solution that will replace champion.
    Notes:
        learning_step can be positive numerical value of 'optimized' for optimized learning step.
    """

    def __init__(self, population_size, stopping_criterion, min_cp_layers, max_cp_layers, cnn_neurons_per_layer, min_ncp_layers, max_ncp_layers, init_last_hidden_layer_neurons, mutation_last_hidden_layer_neurons, conv_prob, learning_step=None,
                max_connections=None, mutation_operator=Mutation_CNN_1(), init_minimum_layers=2, init_maximum_neurons_per_layer=5, maximum_neuron_connection_weight=0.5, maximum_bias_connection_weight=1.0, subset_ratio=1, weight_range=1.0,
                random_sampling_technique=False, random_weighting_technique=False, protected_ols=False, bootstrap_ols=False, bootstrap_ols_samples=10, bootstrap_ols_criterion='median', high_absolute_ls_difference=1, store_ls_history=False):
        super().__init__(population_size, stopping_criterion)
        
        self.min_cp_layers = min_cp_layers
        self.max_cp_layers = max_cp_layers
        self.cnn_neurons_per_layer = cnn_neurons_per_layer
        self.min_ncp_layers = min_ncp_layers
        self.max_ncp_layers = max_ncp_layers
        """ TODO: To Konrad: added the parameters init_last_hidden_layer_neurons and mutation_last_hidden_layer_neurons to control this number of neurons as this is very influential in the learning step computation """
        self.init_last_hidden_layer_neurons = init_last_hidden_layer_neurons
        self.mutation_last_hidden_layer_neurons = mutation_last_hidden_layer_neurons
        self.conv_prob = conv_prob
        
        

        self.learning_step = learning_step
        self.max_connections = max_connections
        self.mutation_operator = mutation_operator
        self.init_maximum_neurons_per_layer = init_maximum_neurons_per_layer
        self.maximum_neuron_connection_weight = maximum_neuron_connection_weight
        self.maximum_bias_connection_weight = maximum_bias_connection_weight
        self.next_champion = None
        self.random_sampling_technique = random_sampling_technique
        self.random_weighting_technique = random_weighting_technique
        self.subset_ratio = subset_ratio
        self.weight_range = weight_range
        self.protected_ols = protected_ols
        self.bootstrap_ols = bootstrap_ols
        self.bootstrap_ols_samples = bootstrap_ols_samples
        self.bootstrap_ols_criterion = bootstrap_ols_criterion
        if self.bootstrap_ols:
            self.high_absolute_ls_difference = high_absolute_ls_difference
            self.high_absolute_differences_history = []
        self.store_ls_history = store_ls_history
        if self.store_ls_history:
            self.ls_history = []
        self.zero_ls_by_activation_function = {}
        self.zero_ls_history = []
        self.lr_intercept = None


    def _get_learning_step(self, partial_semantics):
        """Returns learning step."""
        
        if self.learning_step == 'lr-ls':
            ls = self._get_linear_regression_learning_step(partial_semantics)
        # If learning step is 'optimized', calculate optimized learning step.
        elif self.learning_step == 'optimized':
            ls = self._get_optimized_learning_step(partial_semantics)
        # Else, return numerical learning step.
        else:
            ls = uniform(-self.learning_step, self.learning_step)
            # ls = self.learning_step
        
        if self.store_ls_history:
            self.ls_history += [ls]
        
        return ls

    def _get_linear_regression_learning_step(self, partial_semantics):
        
        delta_target = copy(self.target_vector).astype(float)
        if self.champion:
            delta_target -= self.champion.neural_network.get_predictions()
        X = partial_semantics.reshape(-1, 1)
        y = delta_target.reshape(-1, 1)
        lr = LinearRegression().fit(X, y)
        ls = lr.coef_[0][0]
        self.lr_intercept = lr.intercept_[0]
        #=======================================================================
        # print('Score:', lr.score(X, y))
        # print('lr.coef_:', lr.coef_)
        # print('lr.intercept_:', lr.intercept_)
        #=======================================================================
        return ls
    
    def _get_optimized_learning_step(self, partial_semantics):
        """Calculates optimized learning step."""
        
        """ bootstrap samples; compute OLS for each; use desired criterion to select the final LS """
        if self.bootstrap_ols:
            
            weights = []
            size = self.target_vector.shape[0]
            
            for sample in range(self.bootstrap_ols_samples):
                
                idx = np_choice(arange(size), size, replace=True)
                
                bootstrap_delta_target = copy(self.target_vector[idx]).astype(float)
                if self.champion:
                    full_predictions = self.champion.neural_network.get_predictions()
                    bootstrap_delta_target -= full_predictions[idx]
                
                bootstrap_partial_semantics = partial_semantics[idx]
                inverse = array(pinv(resize(bootstrap_partial_semantics, (1, bootstrap_partial_semantics.size))))
                ols = dot(inverse.transpose(), bootstrap_delta_target)[0]
                
                weights += [ols]
            
            ols_median = median(weights)
            ols_mean = mean(weights)
            ols = self._compute_ols(partial_semantics)
            abs_dif = abs(ols_median - ols_mean)
            
            if abs_dif >= self.high_absolute_ls_difference:
                self.high_absolute_differences_history.append([abs_dif, ols_median, ols_mean, ols])
                #===============================================================
                # print('Absolute difference: %.3f, median vs. mean: %.3f vs. %.3f' % (abs_dif, ols_median, ols_mean))
                #===============================================================
                #===============================================================
                # print('Absolute difference: %.3f, median vs. mean vs. OLS: %.3f vs. %.3f vs. %.3f' % (abs_dif, ols_median, ols_mean, ols))
                # print()
                #===============================================================
            
            if self.bootstrap_ols_criterion == 'median':
                return median(weights)
            else:
                return mean(weights)
        
        else:
            return self._compute_ols(partial_semantics)

    def _compute_ols(self, partial_semantics):
            # Calculates distance to target vector.
            delta_target = copy(self.target_vector).astype(float)
            if self.champion:
                """ version to use when no memory issues exist """
                delta_target -= self.champion.neural_network.get_predictions()
                """ version attempting to circumvent the memory issues """
                #===============================================================
                # predictions = self.champion.neural_network.get_predictions()
                # for i in range(delta_target.shape[0]):
                #     delta_target[i] -= predictions[i]
                #===============================================================
            # Calculates pseudo-inverse of partial_semantics.
            inverse = array(pinv(resize(partial_semantics, (1, partial_semantics.size))))
            # inverse = array(pinv(matrix(partial_semantics)))
            # Returns dot product between inverse and delta.

            ols = dot(inverse.transpose(), delta_target.transpose())[0]
            # ols = dot(inverse.transpose(), delta_target)[0]

            if ols == 0:
                self.zero_ls_history.append([partial_semantics, delta_target, None])
            
            if self.protected_ols:
                if self._valid_ols(delta_target, partial_semantics, ols) == False:
                    ols = 0
            
            return ols
    
    def _valid_ols(self, delta_target, partial_semantics, ols):
        size = delta_target.shape[0]
        absolute_ideal_weights = []
        for i in range(size):
            if partial_semantics[i] != 0:
                absolute_ideal_weight = delta_target[i] / partial_semantics[i]
            else:
                absolute_ideal_weight = 0
            
            absolute_ideal_weights += [abs(absolute_ideal_weight)]
        
        #=======================================================================
        # print(median(absolute_ideal_weights))
        # print(mean(absolute_ideal_weights))
        # print(std(absolute_ideal_weights))
        # print(abs(ols))
        #=======================================================================
        
        upper_bound = mean(absolute_ideal_weights) + 2 * std(absolute_ideal_weights)
        lower_bound = mean(absolute_ideal_weights) - 2 * std(absolute_ideal_weights)
        if abs(ols) > upper_bound or abs(ols) < lower_bound:
            #===================================================================
            # print('\tInvalid OLS')
            #===================================================================
            return False
        else:
            return True
    
    def _get_connection_weight(self, weight):
        """Returns connection weight if defined, else random value between -1 and 1."""
        
        if weight:
            return weight
        else:
            return uniform(-self.maximum_neuron_connection_weight, self.maximum_neuron_connection_weight)

    def _connect_cnn_nodes(self, from_nodes, to_nodes, full):
        """
        Connects list of from_nodes with list of to_nodes.
        Args:
            from_nodes: List of from_nodes.
            to_nodes: List of to_nodes.
            weight: Weight from connection.
            random: Flag if random number of connections.
        Notes:
            If weight is None, then weight will be chosen at random between -1 and 1.
        """
        if full:
            for to_node in to_nodes:
                Connection(from_nodes, to_node, 1)
        else:
            for to_node in to_nodes:
                # If random, create random sample of connection partners
                from_node = sample(from_nodes, 1)
                # Connect to_node to each node in from_node_sample.
                Connection(from_node, to_node, 1)

    def _connect_nodes(self, from_nodes, to_nodes, weight=None, random=False):
        """
        Connects list of from_nodes with list of to_nodes.
        Args:
            from_nodes: List of from_nodes.
            to_nodes: List of to_nodes.
            weight: Weight from connection.
            random: Flag if random number of connections.
        Notes:
            If weight is None, then weight will be chosen at random between -1 and 1.
        """

        for to_node in to_nodes:
            # If random, create random sample of connection partners
            if random:
                max_connections = len(from_nodes)
                random_connections = randint(1, max_connections)
                from_nodes_sample = sample(from_nodes, random_connections)
            else:
                from_nodes_sample = from_nodes
            # Connect to_node to each node in from_node_sample.
            # for from_node in from_nodes_sample:
            #     Connection(from_node, to_node, self._get_connection_weight(weight))
            [Connection(from_node, to_node, self._get_connection_weight(weight)) for from_node in from_nodes_sample]

    def _connect_nodes_mutation(self, hidden_layers):
        """Connects new mutation neurons to remainder of network."""

        # Sets reference to champion neural network.
        neural_network = self.champion.neural_network
        # Create hidden origin layer.
        from_layers = [copy(hidden_layer) for hidden_layer in hidden_layers]
        for hidden_layer_new, hidden_layer_old in zip(from_layers, neural_network.hidden_layers):
            hidden_layer_new.extend(hidden_layer_old)
        # Establish connections.
        self._connect_nodes(neural_network.sensors, hidden_layers[0], random=True)
        previous_neurons = from_layers[0]
        for from_layer, to_layer in zip(from_layers[1:], hidden_layers[1:]):
            self._connect_nodes(previous_neurons, to_layer, random=True)
            previous_neurons = from_layer

    def _connect_nodes_cnn_mutation(self, cnn_layers, hidden_layers):
        """Connects new mutation neurons to remainder of network."""

        # Sets reference to champion neural network.
        neural_network = self.champion.neural_network

        # Create cnn origin layer.
        from_layers = [copy(cnn_layers) for cnn_layers in cnn_layers]
        for cnn_layer_new, cnn_layer_old in zip(from_layers, neural_network.cnn_layers):
            cnn_layer_new.extend(cnn_layer_old)

        # Create hidden origin layer.
        from_layers = [copy(hidden_layer) for hidden_layer in hidden_layers]
        for hidden_layer_new, hidden_layer_old in zip(from_layers, neural_network.hidden_layers):
            hidden_layer_new.extend(hidden_layer_old)

        # Establish connections.
        self._connect_nodes(neural_network.sensors, hidden_layers[0], random=True)
        previous_neurons = from_layers[0]
        for from_layer, to_layer in zip(from_layers[1:], hidden_layers[1:]):
            self._connect_nodes(previous_neurons, to_layer, random=True)
            previous_neurons = from_layer

        # Establish connections.
        self._connect_nodes(neural_network.sensors, hidden_layers[0], random=True)
        previous_neurons = from_layers[0]
        for from_layer, to_layer in zip(from_layers[1:], hidden_layers[1:]):
            self._connect_nodes(previous_neurons, to_layer, random=True)
            previous_neurons = from_layer

    def _connect_learning_step(self, neural_network, last_hidden_layer_neurons=None):
        """Connects last hidden neuron with defined learning step."""

        """ TODO: To Konrad: new code for the learning steps computation """
        if last_hidden_layer_neurons is None:
            self.init_lbfgs(self.input_matrix, self.y, neural_network)
        else:
            self.mutation_lbfgs(self.input_matrix, self.y, neural_network, last_hidden_layer_neurons)
        
        # Get last hidden neuron.
        # last_neuron = neural_network.hidden_layers[-1][-1]
        #=======================================================================
        # last_layer = neural_network.hidden_layers[-1]
        #=======================================================================
        # Get semantics of last neuron.

#===============================================================================
#         for i in range(len(last_layer)):
# 
#             last_neuron = last_layer[i]
#             last_semantics = last_neuron.semantics
#             # Connect last neuron to output neuron.
# 
#             ls = self._get_learning_step(last_semantics)
#             if self.learning_step == 'optimized' and self.bootstrap_ols == False and ls == 0:
#                 # print('\tActivation function:', last_neuron.activation_function)
#                 if len(self.zero_ls_history) > 0 and self.zero_ls_history[-1][2] == None:
#                     self.zero_ls_history[-1][2] = last_neuron.activation_function
#                     if last_neuron.activation_function in self.zero_ls_by_activation_function:
#                         count = self.zero_ls_by_activation_function[last_neuron.activation_function]
#                         self.zero_ls_by_activation_function[last_neuron.activation_function] = count + 1
#                     else:
#                         self.zero_ls_by_activation_function[last_neuron.activation_function] = 1
# 
#                 #===================================================================
#                 # if last_neuron.activation_function != 'relu':
#                 #     print('\tActivation function:', last_neuron.activation_function)
#                 #     print(self.zero_ls_history[-1])
#                 #     print()
#                 #===================================================================
# 
#             if self.stopping_criterion.__class__ == algorithms.common.stopping_criterion.MaxGenerationsCriterion:
#                 if self.current_generation == self.stopping_criterion.max_generation:
# 
#                     #===============================================================
#                     # if self.bootstrap_ols:
#                     #     if len(self.high_absolute_differences_history) > 0:
#                     #         print(self.high_absolute_differences_history)
#                     #         print('Number of high absolute differences:', len(self.high_absolute_differences_history))
#                     #===============================================================
# 
#                     #===============================================================
#                     # if len(self.zero_ls_by_activation_function) > 0:
#                     #     print(self.zero_ls_by_activation_function)
#                     #     print()
#                     #===============================================================
# 
#                     #===================================================================
#                     # if self.store_ls_history:
#                     #     print(self.ls_history)
#                     #===================================================================
#                     pass
# 
#             if self.lr_intercept:
#                 neural_network.output_neuron.input_connections[0].weight += self.lr_intercept  # todo never used so didn't adjust for now
# 
#             self._connect_nodes([last_neuron], [neural_network.output_layer[i]], ls)
#===============================================================================

    def init_lbfgs(self, X, y, nn, random_state=None):
    
        n_samples = y.shape[0]
        n_neurons = len(nn.hidden_layers[-1])
        hidden_semantics = zeros((n_samples, n_neurons))
        for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
            hidden_semantics[:, i] = hidden_neuron.semantics
        
        layer_units = [n_neurons, y.shape[1]]
        #===========================================================================
        # activations = [X]
        #===========================================================================
        activations = []
        activations.extend([hidden_semantics])
        activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
        deltas = [empty_like(a_layer) for a_layer in activations]
        coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
        intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
        
        solver = LBFGS()
        
        coef_init = zeros((layer_units[0], layer_units[1]))
        intercept_init = zeros(layer_units[1])
        coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state=random_state, coef_init=coef_init, intercept_init=intercept_init)
    
        coefs = coefs[-1]
        intercepts = intercepts[-1]
        hidden_neurons = nn.hidden_layers[-1]
        for output_index, output_neuron in enumerate(nn.output_layer):
            for i in range(n_neurons):
                # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
                Connection(hidden_neurons[i], output_neuron, coefs[i, output_index])
            
            # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
            output_neuron.input_connections[0].weight = intercepts[output_index]

    def mutation_lbfgs(self, X, y, nn, new_neurons, random_state=None):
        n_samples = y.shape[0]
        n_new_neurons = len(new_neurons)
        hidden_semantics = zeros((n_samples, n_new_neurons))
        for i, hidden_neuron in enumerate(new_neurons):
            hidden_semantics[:, i] = hidden_neuron.semantics
        
        layer_units = [n_new_neurons, y.shape[1]]
        activations = []
        activations.extend([hidden_semantics])
        activations.extend(empty((n_samples, n_fan_out)) for n_fan_out in layer_units[1:])
        deltas = [empty_like(a_layer) for a_layer in activations]
        coef_grads = [empty((n_fan_in_, n_fan_out_)) for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])]
        intercept_grads = [empty(n_fan_out_) for n_fan_out_ in layer_units[1:]]
        
        solver = LBFGS()
        
        """ zero-weight initialization for new neurons """
        coef_init = zeros((layer_units[0], layer_units[1]))
        
        intercept_init = zeros(layer_units[1])
        for output_index, output_neuron in enumerate(nn.output_layer):
            intercept_init[output_index] = output_neuron.input_connections[0].weight
        
        fixed_weighted_input = zeros((n_samples, layer_units[1]))
        for output_index, output_neuron in enumerate(nn.output_layer):
            previous_bias = output_neuron.input_connections[0].weight
            fixed_weighted_input[:, output_index] = output_neuron.get_weighted_input() - previous_bias
            output_neuron.set_previous_bias(previous_bias)
        
        coefs, intercepts = solver.fit(X, y, activations, deltas, coef_grads, intercept_grads, layer_units, random_state=random_state, coef_init=coef_init, intercept_init=intercept_init, fixed_weighted_input=fixed_weighted_input)
        
        coefs = coefs[-1]
        intercepts = intercepts[-1]
        for output_index, output_neuron in enumerate(nn.output_layer):
            for i in range(n_new_neurons):
                # print('coefs[%d, %d] = %.5f\n' % (i, output_index, coefs[i, output_index]))
                Connection(new_neurons[i], output_neuron, coefs[i, output_index])
            
            # print('intercepts[%d] = %.5f\n' % (output_index, intercepts[output_index]))
            output_neuron.input_connections[0].weight = intercepts[output_index]
            # output_neuron.input_connections[0].weight = intercepts[output_index] - output_neuron.input_connections[0].weight

    def _create_solution(self, neural_network):
        """Creates solution for population."""

        # Creates solution object.
        solution = Solution(neural_network, None, None)
        # Calculates error.
        neural_network.get_predictions()
        #solution.value = self.metric.evaluate(neural_network.final_output, self.target_vector)
        solution.value = self.metric.evaluate(neural_network.y_pred_ce, self.y)
        solution.ce_loss = solution.value
        solution.accuracy = accuracy_score(self.y_labels, neural_network.y_pred_labels)
        #print('\tAccuracy\t\t%.8f%%' % (calculate_accuracy(self._target_vector.argmax(axis=1), self._current_best.get_predictions().argmax(axis=1)) * 100))

        
        # Checks, if solution is better than parent.
        solution.better_than_ancestor = self._is_better_solution(solution, self.champion)
        # After the output semantics are updated, we can remove the semantics from the final hidden neuron.
        # neural_network.output_neuron.input_connections[-1].from_node.semantics = None
        # for neuron in neural_network.get_output_neurons(): neuron.input_connections[-1].from_node.semantics = None ###TODO make sure if that can be commented out or not  ?!?!?!?!?!

        # Returns solution.
        return solution

    def _initialize_sensors(self):
        """Initializes sensors based on input matrix."""

        return [Sensor(input_data) for input_data in self.input_matrix.T]

    def _initialize_flat_sensors(self, data):
        """Initializes sensors based on input matrix."""

        return [Sensor(d) for d in data]

    def _initialize_bias(self, neural_network):
        """Initializes biases with same length as sensors."""

        return Sensor(ones(neural_network.sensors[0].semantics.shape))

    def _initialize_hidden_layers(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""
        
        number_of_layers = randint(self.min_ncp_layers, self.max_ncp_layers)
        neurons_per_layer = [randint(1, self.init_maximum_neurons_per_layer) for _ in range(number_of_layers - 1)]
        neurons_per_layer.append(self.init_last_hidden_layer_neurons)
        hidden_layers = [[create_neuron(None, neural_network.bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight) for _ in range(neurons_per_layer[layer])] for layer in range(number_of_layers)]
        
        # From Jan: Create hidden layers with one neuron with random activation function each.
        # hidden_layers = [[create_neuron(None, neural_network.bias)] for i in range(self.layers - 1)]
        
        # Add final hidden layer with one neuron
        # activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
        # hidden_layers.append([create_neuron(activation_function, neural_network.bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight)])
        #=======================================================================
        # activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
        # if self.nr_of_classes > 2:
        #     hidden_layers.append([create_neuron(activation_function, neural_network.bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight) for i in range(self.nr_of_classes)])
        # else:
        #     hidden_layers.append([create_neuron(activation_function, neural_network.bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight)])
        #=======================================================================

        # Returns hidden layers.
        return hidden_layers

    def _initialize_cnn_layers(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""

        number_of_layers = randint(self.min_cp_layers, self.max_cp_layers)
        neurons_per_layer = [randint(1, self.cnn_neurons_per_layer) for _ in range(number_of_layers)]
        cnn_layers = [[create_cnn_neuron(conv_prob=self.conv_prob) for _ in range(neurons_per_layer[layer])] for layer in range(number_of_layers)]
        return cnn_layers

    def _initialize_flatten_layer(self, neural_network):
        last_layer = neural_network.cnn_layers[-1]
        flatten_layer = None
        for neuron in last_layer:
            temp_arr = neuron.semantics
            temp_arr2 = temp_arr.reshape(-1, self.target_vector.shape[0])

            if flatten_layer is not None:
                flatten_layer = concatenate((flatten_layer, temp_arr2), axis=0)
            else:
                flatten_layer = temp_arr2

        # flatten_layer = flatten_layer.reshape(-1, self.target_vector.shape[0])
        return flatten_layer

    def _initialize_flatten_layer_for_mutated_part(self, mutated_layers):
        last_layer = mutated_layers[-1]
        flatten_layer = None
        for neuron in last_layer:
            temp_arr = neuron.semantics
            temp_arr2 = temp_arr.reshape(-1, self.target_vector.shape[0])

            if flatten_layer is not None:
                flatten_layer = concatenate((flatten_layer, temp_arr2), axis=0)
            else:
                flatten_layer = temp_arr2

        # flatten_layer = flatten_layer.reshape(-1, self.target_vector.shape[0])
        return flatten_layer

    def _initialize_output_layer(self, neural_network):
        """Initializes hidden layers, based on defined number of layers."""

        if self.nr_of_classes > 2:
            output_layer = [create_output_neuron('identity', neural_network.bias) for _ in range(self.nr_of_classes)]
        else:
            output_layer = [create_output_neuron('identity', neural_network.bias)]

        return output_layer

    def _initialize_topology(self):
        """Initializes topology."""

        # Create sensors.
        sensors = self._initialize_sensors()

        # Create neural network.
        neural_network = ConvNeuralNetwork(sensors, None, None, None, None, None)
        # Create bias.
        neural_network.bias = self._initialize_bias(neural_network)
        # Return neural network.
        return neural_network

    def _initialize_neural_network(self, topology):
        """Creates neural network from initial topology."""

        # Create shallow copy of topology.
        neural_network = copy(topology)
        # Create output neuron.
        neural_network.output_layer = self._initialize_output_layer(neural_network)
        
        # Create cnn layer.
        neural_network.cnn_layers = self._initialize_cnn_layers(neural_network)
        self._connect_cnn_nodes(neural_network.sensors, neural_network.cnn_layers[0], full=True)

        previous_neurons = neural_network.cnn_layers[0]
        for cnn_layer in neural_network.cnn_layers[1:]:
            """ TODO: To Konrad: temporarily changed to full=True """
            # self._connect_cnn_nodes(previous_neurons, cnn_layer, full=False)
            self._connect_cnn_nodes(previous_neurons, cnn_layer, full=True)
            previous_neurons = cnn_layer
        
        i = 0
        for layer in neural_network.cnn_layers:
            if i == 0:
                for neuron in layer:
                    neuron.calculate()
            else:
                for neuron in layer:
                    neuron.calculate2()
            i += 1

        flat_data = self._initialize_flatten_layer(neural_network)
        neural_network.flatten_layer = self._initialize_flat_sensors(flat_data)

        # Create hidden layer.
        neural_network.hidden_layers = self._initialize_hidden_layers(neural_network)
        # connect cnn layers
        #
        # Establish connections
        self._connect_nodes(neural_network.flatten_layer, neural_network.hidden_layers[0], random=False)
        
        previous_neurons = neural_network.hidden_layers[0]
        for hidden_layer in neural_network.hidden_layers[1:]:
            self._connect_nodes(previous_neurons, hidden_layer, random=True)
            previous_neurons = hidden_layer
        # Calculate hidden neurons.
        for layer in neural_network.hidden_layers:
            for neuron in layer:
                neuron.calculate()
        # Connect last neuron to output neuron with learning step.
        self._connect_learning_step(neural_network)
        # Calculate output semantics.
        for neuron in neural_network.output_layer:
            neuron.calculate()
        # Return neural network.
        return neural_network

    def _initialize_solution(self, topology):
        """Creates solution for initial population."""

        # Initialize neural network.
        neural_network = self._initialize_neural_network(topology)
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution

    def _initialize_population(self):
        """Initializes population in first generation."""
        # def time_seconds(): return default_timer()
        # start_time = time_seconds() 
        # Initializes neural network topology.
        topology = self._initialize_topology()
        # Create initial population from topology.
        for _ in range(self.population_size):
            solution = self._initialize_solution(topology)

            if not self.next_champion:
                self.next_champion = solution
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)
        # print("time to initialize population: ", time_seconds()-start_time)

    def _mutate_network(self):
        """Creates mutated offspring from champion neural network."""

        # Create shallow copy of champion neural network.
        neural_network = copy(self.champion.neural_network)
        # Create mutated hidden layers.
        mutation_layers = self.mutation_operator.mutate_network(self)
        # Connect hidden neurons to remainder of network.
        self._connect_nodes_mutation(mutation_layers)
        # Calculate mutated hidden layer.
        for mutation_layer in mutation_layers:
            for neuron in mutation_layer:
                neuron.calculate()
        # Extend hidden layers.
        for hidden_layer, mutation_layers in zip(neural_network.hidden_layers, mutation_layers):
            hidden_layer.extend(mutation_layers)
        # Connect final hidden neuron to output neuron.
        self._connect_learning_step(neural_network)
        # Get most recent connection.
        for neuron in neural_network.output_layer:
            connection = neuron.input_connections[-1]

            # Update semantics of output neuron.
            if self.lr_intercept:
                #===================================================================
                # neural_network.output_neuron.semantics2 = copy(neural_network.output_neuron.semantics)
                # neural_network.output_neuron.semantics2 += connection.from_node.semantics * connection.weight
                #===================================================================
                neuron.semantics += connection.from_node.semantics * connection.weight + self.lr_intercept
                #===================================================================
                # print(neural_network.output_neuron.semantics2 - neural_network.output_neuron.semantics)
                # print(self.lr_intercept)
                # print()
                #===================================================================
            else:
                neuron.semantics += connection.from_node.semantics * connection.weight

            # Return neural network.
        return neural_network

    def _mutate_cnn_network(self):
        """Creates mutated offspring from champion neural network."""

        # Create shallow copy of champion neural network.
        neural_network = copy(self.champion.neural_network)
        # Create mutated hidden layers.
        mutation_cnn_layers, mutation_hidden_layers = self.mutation_operator.mutate_network(self, last_hidden_layer_neurons=self.mutation_last_hidden_layer_neurons, conv_prob=self.conv_prob)
        # Connect hidden neurons to remainder of network.

        for i, cnn_layer in enumerate(mutation_cnn_layers):
            if i == 0:
                if len(cnn_layer) > 0:
                    self._connect_cnn_nodes(neural_network.sensors, mutation_cnn_layers[0], full=True)
            else:
                if len(cnn_layer) > 0:
                    if len(mutation_cnn_layers[i - 1]) == 0:
                        previous_neurons = neural_network.cnn_layers[i - 1]
                    else:
                        previous_neurons = mutation_cnn_layers[i - 1]
                    """ TODO: To Konrad: temporarily changed to full=True """
                    # self._connect_cnn_nodes(previous_neurons, cnn_layer, full=False)
                    self._connect_cnn_nodes(previous_neurons, cnn_layer, full=True)
            # previous_neurons = cnn_layer

        for i, layer in enumerate(mutation_cnn_layers):
            if i == 0:
                if len(layer) > 0:
                    for neuron in layer:
                        neuron.calculate()
            else:
                if len(layer) > 0:
                    for neuron in layer:
                        neuron.calculate2()

        flat_data = self._initialize_flatten_layer_for_mutated_part(mutation_cnn_layers)
        flatten_layer_mutated = self._initialize_flat_sensors(flat_data)

        self._connect_nodes(flatten_layer_mutated, mutation_hidden_layers[0], random=False)
        previous_neurons = mutation_hidden_layers[0]
        for hidden_layer in mutation_hidden_layers[1:]:
            self._connect_nodes(previous_neurons, hidden_layer, random=True)
            previous_neurons = hidden_layer
        # Calculate hidden neurons.
        for layer in mutation_hidden_layers:
            for neuron in layer:
                neuron.calculate()

        # Extend cnn layers.
        for cnn_layer, mutation_cnn_layer in zip(neural_network.cnn_layers, mutation_cnn_layers):
            cnn_layer.extend(mutation_cnn_layer)

        # Extend flatten layers.
        neural_network.flatten_layer.extend(flatten_layer_mutated)

        # Extend hidden layers.
        for hidden_layer, mutation_hidden_layer in zip(neural_network.hidden_layers, mutation_hidden_layers):
            hidden_layer.extend(mutation_hidden_layer)
        # Connect final hidden neuron to output neuron.
        self._connect_learning_step(neural_network, last_hidden_layer_neurons=mutation_hidden_layers[-1])
        
        for output_neuron in neural_network.output_layer:
            output_neuron.incremental_semantics_computation(len(mutation_hidden_layers[-1]))
        
#===============================================================================
#         # Get most recent connection.
#         for neuron in neural_network.output_layer:
#             connection = neuron.input_connections[-1]
# 
#             # Update semantics of output neuron.
#             if self.lr_intercept:
#                 # ===================================================================
#                 # neural_network.output_neuron.semantics2 = copy(neural_network.output_neuron.semantics)
#                 # neural_network.output_neuron.semantics2 += connection.from_node.semantics * connection.weight
#                 # ===================================================================
#                 neuron.semantics += connection.from_node.semantics * connection.weight + self.lr_intercept
#                 # ===================================================================
#                 # print(neural_network.output_neuron.semantics2 - neural_network.output_neuron.semantics)
#                 # print(self.lr_intercept)
#                 # print()
#                 # ===================================================================
#             else:
#                 neuron.semantics = connection.from_node.semantics * connection.weight
#===============================================================================

        # Return neural network.
        return neural_network

    def _mutate_solution(self):
        """Applies mutation operator to current champion solution."""

        # Created mutated offspring of champion neural network.
        neural_network = self._mutate_cnn_network()
        # neural_network = self._mutate_network()
        # Create solution.
        solution = self._create_solution(neural_network)
        # Return solution.
        return solution

    def _mutate_population(self):
        """ ... """
        if self.random_sampling_technique:
            # calculate the new predictions and update the champion's error according to the new input matrix previously generated
            champ_predictions = self.champion.neural_network.predict(self.input_matrix)
            self.champion.predictions = champ_predictions 
            self.champion.value = self.metric.evaluate(self.champion.predictions, self.target_vector)
        
        if self.random_weighting_technique:
            # calculate the new predictions and update the champion's error according to the new input matrix previously generated
            self.champion.value = self.metric.evaluate(self.champion.predictions, self.target_vector)
        
        print('\t\tMutation, champion topology:', self.champion.neural_network.get_topology())
        
        for _ in range(self.population_size):
            solution = self._mutate_solution()
            
            if not self.next_champion:
                if self._is_better_solution(solution, self.champion):
                    self.next_champion = solution
                else:
                    solution.neural_network = None
            elif self._is_better_solution(solution, self.next_champion):
                self.next_champion.neural_network = None
                self.next_champion = solution
            else:
                solution.neural_network = None
            self.population.append(solution)

    def _wipe_population(self):
        self.population = list()

    def _override_current_champion(self):
        if self.next_champion:
            self.champion = self.next_champion
            self.next_champion = None

    def _epoch(self):
        if self.current_generation == 0:
            self._initialize_population()
        else:
            self._mutate_population()
        stopping_criterion = self.stopping_criterion.evaluate(self)
        self._override_current_champion()
        self._wipe_population()
        return stopping_criterion

    def fit(self, input_matrix, target_vector, metric, verbose=True):
        """ TODO: To Konrad: added transformation for target vector using label binarizer """
        self._label_binarizer = LabelBinarizer()
        self.y = self._label_binarizer.fit_transform(target_vector)
        self.y_labels = target_vector
        # self._label_binarizer.inverse_transform(y_pred)
        
        super().fit(input_matrix, target_vector, metric, verbose)
        #self.champion.neural_network = deepcopy(self.champion.neural_network)
        
        self._label_binarizer = None
        self.y = None
        self.y_labels = None
        self.champion.neural_network.wipe_semantics()
        
        return self.champion.neural_network

    def _print_generation(self):
        """ TODO: To Konrad: new _print_generation for CE loss and accuracy """
        print('Iteration %d: CE loss %.5f, accuracy %.5f%%' % (self.current_generation, self.champion.ce_loss, self.champion.accuracy * 100))

    def predict(self, input_matrix):
        neural_network = self.champion.neural_network
        neural_network.load_sensors(input_matrix)
        neural_network.calculate()
        return neural_network.get_predictions()

    def __repr__(self):
        return 'DeepSLM'
