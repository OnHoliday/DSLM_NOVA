from copy import copy

from numpy import zeros, ones, where, percentile, append as np_append
from scipy.special import xlogy
from sklearn.linear_model import LinearRegression
from sklearn.neural_network._base import softmax

from common.conv_neural_network_builder import ConvNeuralNetworkBuilder
from common.neural_network_builder import NeuralNetworkBuilder
from common.conv_neural_network_components.conv_neuron import ConvNeuron
from common.conv_neural_network_components.pool_neuron import PoolNeuron
from common.conv_neural_network_components.flat_neuron import FlatNeuron
from common.neural_network_components.hidden_neuron import HiddenNeuron
from common.neural_network_components.input_neuron import InputNeuron
from common.utilities import get_closest_positive_number_index

from slm.mutation import mutation_ols_ls_global_margin, mutation_ols_ls_local_margin

def simple_conv_mutation(neural_network, X, random_state, learning_step, sparseness, maximum_new_neurons_per_layer=3,
                         maximum_neuron_connection_weight=1.0, maximum_bias_weight=1.0, target=None,
                         delta_target=None, learning_step_function=None,
                         hidden_activation_functions_ids=None,
                        cnn_activation_functions_ids=None,
                         prob_activation_hidden_layers=None,
                         prob_activation_cnn_layers=None, time_print=False):
    """
    Function that changes a given neural network's topology by possibly
    adding neurons on each one of the hidden layers.

    Parameters
    ----------
    neural_network : NeuralNetwork
        Neural network to be changed.

    random_state : RandomState instance
        A Numpy random number generator.

    learning_step : float or {'optimized'}
        Strategy to calculate the weight for connections in the last
        hidden layer's neurons to the output layer.

    sparseness : dict
        Dictionary containing information regarding neurons' connections,
        namely sparseness and the existence of skip connections (keys:
        'sparse', 'minimum_sparseness', 'maximum_sparseness', and
        'prob_skip_connection').

    maximum_new_neurons_per_layer : int
        Maximum number of neurons that can be added to a hidden layer.

    maximum_neuron_connection_weight : float
        Maximum value a weight connection between two neurons can have.

    maximum_bias_weight : float
        Maximum value a bias of a neuron can have.

    delta_target : array of shape (num_samples,), optional
        Array containing the distance of neural network's predictions
        to target values. Required if learning_step is set as 'optimized'.

    learning_step_function : function, optional, default None
        Optional function used to calculate optimized learning step
        for the neuron. The available functions are: pinv and lstsq
        from numpy package, and pinv, pinv2 and lstsq from scipy package. ,

    hidden_activation_functions_ids : array of shape (num_functions,), optional
        Names of activation functions that can be used in hidden layers.

    prob_activation_hidden_layers : float, optional
        Probability of the neurons on hidden layers have an activation
        function associated.

    Returns
    -------
    mutated_neural_network : NeuralNetwork
    """

    import timeit
    start_time = timeit.default_timer()
    hidden_semantics_time = 0

    # Add between 0 up to 'maximum_new_neurons_per_layer' neurons in each layer:
    # ===========================================================================
    # new_neurons_per_layer = [random_state.randint(1, maximum_new_neurons_per_layer) for _ in range(neural_network.get_number_hidden_layers())]
    # ===========================================================================
    new_conv_neurons_per_layer = [random_state.randint(1, 4 + 1) for _ in range(neural_network.get_number_conv_layers())]
    new_neurons_per_layer = [random_state.randint(50, 50 + 1) for _ in range(neural_network.get_number_hidden_layers())]

    # The last hidden layer needs to receive at least 1 new neuron:
    if new_neurons_per_layer[-1] == 0:
        new_neurons_per_layer[-1] = 1

    # Auxiliary list containing neurons to be filtered in skip connections:
    if sparseness.get('prob_skip_connection') > 0:
        neurons_for_skip_connections = copy(neural_network.input_layer)
        neurons_for_skip_connections.extend(neural_network.get_hidden_neurons())
    else:
        neurons_for_skip_connections = None

    # Auxiliary list that will contain references for new hidden neurons created for each layer:
    added_new_flat_neurons = list()
    added_new_conv_neurons = list()
    added_new_neurons = list()

    for i, number_neurons in enumerate(new_conv_neurons_per_layer):

        if number_neurons > 0:
            # Get new hidden layer to extend:
            new_conv_neurons = ConvNeuralNetworkBuilder.create_cnn_neurons(number_neurons=number_neurons,
                                                                            random_state=random_state, level_layer=i,
                                                                            maximum_bias_weight=maximum_bias_weight,
                                                                            cnn_activation_functions_ids =  cnn_activation_functions_ids,
                                                                             prob_activation_cnn_layers=prob_activation_cnn_layers)
            # Establish connections with previous layer:
            if i == 0:
                # Note: Previous layer is the input layer, so there are no skipped connections (although we might have sparseness):
                ConvNeuralNetworkBuilder.connect_layers(layer_to_connect=new_conv_neurons,
                                                    previous_layer=neural_network.input_layer,
                                                    maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                    random_state=random_state,
                                                    sparseness=sparseness, neurons_for_skip_connections=None)
            else:
                # Filter neurons for skip connections:
                if neurons_for_skip_connections:
                    skip_connections_set = list(
                        filter(lambda x: isinstance(x, InputNeuron), neurons_for_skip_connections) if i == 1
                        else list(filter(lambda x: isinstance(x, InputNeuron) or (
                                    isinstance(x, HiddenNeuron) and x.level_layer < i - 1),
                                         neurons_for_skip_connections)))
                else:
                    skip_connections_set = None

                # Note: i-1 is the index of the previous layer:
                ConvNeuralNetworkBuilder.connect_layers(layer_to_connect=new_conv_neurons,
                                                    previous_layer=neural_network.conv_layers[i - 1],
                                                    maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                    random_state=random_state, sparseness=sparseness,
                                                    neurons_for_skip_connections=skip_connections_set)



                #todo how to apply that also to cnn??!?! should I do it at all?
                # Starting at second hidden layer (i > 0), all neurons must be connected with at least 1 new neuron added to the previous layer:
                for neuron in new_conv_neurons:
                    if not any(connection.is_from_previous_layer for connection in neuron.input_connections):
                        # Warning: previous_hidden_layer_index can be None if none of previous layers received new neurons during this mutation.
                        previous_hidden_layer_index = get_closest_positive_number_index(new_neurons_per_layer, i - 1)

                        if previous_hidden_layer_index:
                            ConvNeuralNetworkBuilder.connect_consecutive_mutated_layers(neuron, added_new_neurons[
                                previous_hidden_layer_index],
                                                                                    random_state,
                                                                                    maximum_neuron_connection_weight)

            # Compute semantics for cnn neurons:
            if i == 0:
                for neuron in new_conv_neurons:
                    neuron.calculate()
            else:
                for neuron in new_conv_neurons:
                    neuron.calculate2()

            # Extend new conv neurons to the respective conv layer:
            neural_network.extend_conv_layer(layer_index=i, new_neurons=new_conv_neurons)

            # Store references of new conv neurons:
            added_new_conv_neurons.append(new_conv_neurons)

    flat_data = ConvNeuralNetworkBuilder.create_mutated_flaten_layer(new_conv_neurons, target)
    mutated_flaten_layer = ConvNeuralNetworkBuilder._initialize_flat_sensors(flat_data)

    #todo isnt it redudant? I mean the mutated laten layer already has semantics inside why do I need to connect it with anythin?

    mutated_flaten_layer = ConvNeuralNetworkBuilder.connect_layers(layer_to_connect=mutated_flaten_layer, previous_layer=new_conv_neurons,
                                                       maximum_neuron_connection_weight=maximum_neuron_connection_weight, random_state=random_state,
                                                       sparseness={'sparse': False}, neurons_for_skip_connections=None)

    # Extend new hidden neurons to the respective hidden layer:
    neural_network.extend_flat_layer(mutated_flaten_layer)

    # Store references of new flat neurons:
    added_new_flat_neurons.append(mutated_flaten_layer)

    for i, number_neurons in enumerate(new_neurons_per_layer):

        if number_neurons > 0:
            # Get new hidden layer to extend:
            new_hidden_neurons = NeuralNetworkBuilder.create_hidden_neurons(number_neurons=number_neurons,
                                                                            random_state=random_state, level_layer=i,
                                                                            maximum_bias_weight=maximum_bias_weight,
                                                                            hidden_activation_functions_ids=hidden_activation_functions_ids,
                                                                            prob_activation_hidden_layers=prob_activation_hidden_layers)
            # Establish connections with previous layer:
            if i == 0:
                # Note: Previous layer is the input layer, so there are no skipped connections (although we might have sparseness):
                NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
                                                    previous_layer=mutated_flaten_layer,
                                                    maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                    random_state=random_state,
                                                    sparseness=sparseness, neurons_for_skip_connections=None)
            else:
                # Filter neurons for skip connections:
                if neurons_for_skip_connections:
                    skip_connections_set = list(
                        filter(lambda x: isinstance(x, InputNeuron), neurons_for_skip_connections) if i == 1
                        else list(filter(lambda x: isinstance(x, InputNeuron) or (
                                    isinstance(x, HiddenNeuron) and x.level_layer < i - 1),
                                         neurons_for_skip_connections)))
                else:
                    skip_connections_set = None

                # Note: i-1 is the index of the previous layer:
                NeuralNetworkBuilder.connect_layers(layer_to_connect=new_hidden_neurons,
                                                    previous_layer=neural_network.hidden_layers[i - 1],
                                                    maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                                    random_state=random_state, sparseness=sparseness,
                                                    neurons_for_skip_connections=skip_connections_set)

                # Starting at second hidden layer (i > 0), all neurons must be connected with at least 1 new neuron added to the previous layer:
                for neuron in new_hidden_neurons:
                    if not any(connection.is_from_previous_layer for connection in neuron.input_connections):
                        # Warning: previous_hidden_layer_index can be None if none of previous layers received new neurons during this mutation.
                        previous_hidden_layer_index = get_closest_positive_number_index(new_neurons_per_layer, i - 1)

                        if previous_hidden_layer_index:
                            NeuralNetworkBuilder.connect_consecutive_mutated_layers(neuron, added_new_neurons[
                                previous_hidden_layer_index],
                                                                                    random_state,
                                                                                    maximum_neuron_connection_weight)

            # Calculate semantics for new hidden neurons:
            hidden_semantics_time_start_time = timeit.default_timer()
            # ===================================================================
            # print("[Initialization] Starting hidden semantics computation ...", end='')
            # ===================================================================
            [hidden_neuron.calculate_semantics() for hidden_neuron in new_hidden_neurons]
            # ===================================================================
            # print(' done!')
            # ===================================================================
            hidden_semantics_time += timeit.default_timer() - hidden_semantics_time_start_time


            # Extend new hidden neurons to the respective hidden layer:
            neural_network.extend_hidden_layer(layer_index=i, new_neurons=new_hidden_neurons)


            # Store references of new hidden neurons:
            added_new_neurons.append(new_hidden_neurons)

        else:  # No new hidden neurons were added to this hidden layer:
            added_new_neurons.append(list())

    # Connect new hidden neurons from last hidden layer with output layer:
    NeuralNetworkBuilder.connect_layers(layer_to_connect=neural_network.output_layer,
                                        previous_layer=neural_network.get_last_n_neurons(new_neurons_per_layer[-1]),
                                        maximum_neuron_connection_weight=maximum_neuron_connection_weight,
                                        random_state=random_state,
                                        sparseness={'sparse': False}, neurons_for_skip_connections=None)

    ls_start_time = timeit.default_timer()
    # Calculate learning step for new neurons added in the last hidden layer:

    mutation = mutation_ols_ls_local_margin
    # ===========================================================================
    # mutation = mutation_ols_ls_global_margin
    # ===========================================================================
    mutation(X, target, neural_network, new_neurons_per_layer[-1], random_state)

    ls_time = timeit.default_timer() - ls_start_time

    incremental_start_time = timeit.default_timer()
    neural_network.incremental_output_semantics_update(num_last_connections=new_neurons_per_layer[-1])
    incremental_time = timeit.default_timer() - incremental_start_time

    time = timeit.default_timer() - start_time
    others_time = time - ls_time - incremental_time - hidden_semantics_time
    if time_print:
        print(
            '\n\t\tmutate_hidden_layers total time = %.3f seconds\n\t\t\tcalculate_learning_step = %.3f seconds\n\t\t\tincremental_output_semantics_update = %.3f seconds\n\t\t\thidden semantics computation = %.3f seconds\n\t\t\tothers = %.3f seconds' % (
            time, ls_time, incremental_time, hidden_semantics_time, others_time))
        print(
            '\n\t\t\tcalculate_learning_step = %% of total mutation time %.2f\n\t\t\tincremental_output_semantics_update = %% of total mutation time %.2f\n\t\t\thidden semantics computation = %% of total mutation time %.2f\n\t\t\tothers = %% of total mutation time %.2f' % (
            ls_time / time * 100, incremental_time / time * 100, hidden_semantics_time / time * 100,
            others_time / time * 100))

    neural_network.new_neurons = added_new_neurons
    neural_network.new_conv_neurons = added_new_conv_neurons
    neural_network.new_flat_neurons = added_new_flat_neurons

    # Return mutated neural_network:
    return neural_network
