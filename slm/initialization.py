from numpy import zeros, ones, where, append as np_append
from sklearn.linear_model import LinearRegression

from common.neural_network_builder import NeuralNetworkBuilder


def init_ols_balanced(X, y, nn, random_state):
    
    n_samples = y.shape[0]
    n_neurons = nn.get_number_last_hidden_neurons()
    partial_semantics = zeros((n_samples, n_neurons))
    for i, hidden_neuron in enumerate(nn.hidden_layers[-1]):
        partial_semantics[:, i] = hidden_neuron.get_semantics()
    
    for output_index, output_neuron in enumerate(nn.output_layer):
        output_y = y[:, output_index]
        output_y_class_1_indices = where(output_y == 1)[0]
        output_y_class_1_count = output_y_class_1_indices.shape[0]
        output_y_class_0_count = n_samples - output_y_class_1_count
        
        sample_weights = ones(n_samples)
        class_1_weight = output_y_class_0_count / output_y_class_1_count
        sample_weights[output_y_class_1_indices] = class_1_weight
        
        reg = LinearRegression().fit(partial_semantics, output_y, sample_weights)
        optimal_weights = np_append(reg.coef_.T, reg.intercept_)

        # Update connections with the learning step value:
        for i in range(n_neurons):
            output_neuron.input_connections[-n_neurons + i].weight = optimal_weights[i]
        
        output_neuron.increment_bias(optimal_weights[-1])


def init_standard(X, y, input_layer, random_state, learning_step_function=init_ols_balanced, init_min_layers=1, init_max_layers=1, init_min_neurons_per_layer=1, init_max_neurons_per_layer=10, init_last_hl_neurons=100, max_neuron_connection_weight=0.1, max_bias_weight=0.1, hidden_activation_functions_ids='relu', prob_activation_hls=1, output_activation_function='identity'):
    """Creates a new NeuralNetwork's instance from scratch and returns a neural network
    with all connections established, as well as bias, learning_step, semantics and predictions
    already computed.

    Returns
    -------
    nn : NeuralNetwork
    """
    
    """ Generate random values for hidden layers' topology: """
    number_of_hls = random_state.randint(init_min_layers, init_max_layers + 1)
    
    number_hidden_neurons = [random_state.randint(init_min_neurons_per_layer, init_max_neurons_per_layer + 1) for _ in range(number_of_hls)]
    if init_last_hl_neurons is not None:
        number_hidden_neurons[-1] = init_last_hl_neurons
    
    """ Evoke Neural Network builder to construct a new neural network: """
    sparseness = { 'sparse': False, 'minimum_sparseness': .25, 'maximum_sparseness': .75, 'prob_skip_connection': 0 }
    nn = NeuralNetworkBuilder.generate_new_neural_network(number_of_hls, number_hidden_neurons, y.shape[1], max_neuron_connection_weight, max_bias_weight, output_activation_function, input_layer, random_state, sparseness, hidden_activation_functions_ids, prob_activation_hls)
    
    """ Calculate learning step: """
    learning_step_function(X, y, nn, random_state)
    
    """ calculate output semantics after computing learning_step: """
    nn.calculate_output_semantics()

    """ Return newly created neural network: """
    return nn
