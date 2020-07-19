from random import randint, choice
from algorithms.common.neural_network.activation_function import _NON_LINEAR_ACTIVATION_FUNCTIONS
from algorithms.common.neural_network.neural_network import create_neuron, create_cnn_neuron


class Mutation(object):
    """
    Class represents mutation operator for semantic learning machine.
    """

    def mutate_network(self, algorithm):
        pass

    def _create_final_hidden_neuron(self, bias, maximum_bias_connection_weight=None):
        """ Creates the final hidden neuron """
        
        activation_function = choice(list(_NON_LINEAR_ACTIVATION_FUNCTIONS.keys()))
        return create_neuron(activation_function, bias, maximum_bias_connection_weight=maximum_bias_connection_weight)


class Mutation1(Mutation):
    """Adds one neuron to the last hidden layer."""

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        # Creates empty layers
        """ -1 here """
        hidden_layers = [[] for _ in len(neural_network.hidden_layers) - 1]
        # Adds one neuron to the last hidden layer.
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers


class Mutation2(Mutation):
    """Adds one neuron the each hidden layer."""

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        """ -1 here """
        hidden_layers = [[create_neuron(activation_function=None, bias=bias)]
                         for _ in range(len(neural_network.hidden_layers) - 1)]
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers

    def __repr__(self):
        return self.__class__.__name__ 

        
class Mutation3(Mutation):
    """Adds an equal, random number of neurons to each hidden layer."""

    def __init__(self, max_neurons):
        self.max_neurons = max_neurons

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        neurons = randint(1, self.max_neurons)
        """ -1 here """
        hidden_layers = [[create_neuron() for _ in range(neurons)]
                         for _ in range(len(neural_network.hidden_layers) - 1)]
        hidden_layers.append([self._create_final_hidden_neuron(bias)])
        return hidden_layers


class Mutation4(Mutation):
    """Adds a distinct, random number of neurons to each hidden layer."""

    def __init__(self, maximum_new_neurons_per_layer=3, maximum_bias_connection_weight=1.0):
        self.maximum_new_neurons_per_layer = maximum_new_neurons_per_layer
        self.maximum_bias_connection_weight = maximum_bias_connection_weight
    
    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        number_layers = len(neural_network.hidden_layers)
        
        """ -1 here """
        new_neurons = [randint(1, self.maximum_new_neurons_per_layer) for _ in range(number_layers - 1)]
        # neurons = self.maximum_new_neurons_per_layer if self.maximum_new_neurons_per_layer >= number_layers else number_layers
        # neurons = sample(range(1, neurons), number_layers - 1)
        
        hidden_layers = [[create_neuron(activation_function=None, bias=bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight) for _ in range(neuron)] for neuron in new_neurons]
        
        hidden_layers.append([self._create_final_hidden_neuron(bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight)])
        return hidden_layers


class Mutation_CNN_1(Mutation):
    """Adds one CNN neurons to first CNN layer."""

    def __init__(self, maximum_new_cnn_neurons_per_layer =3,maximum_new_neurons_per_layer =10,  maximum_bias_connection_weight=1.0):
        self.maximum_new_cnn_neurons_per_layer = maximum_new_cnn_neurons_per_layer
        self.maximum_new_neurons_per_layer = maximum_new_neurons_per_layer
        self.maximum_bias_connection_weight = maximum_bias_connection_weight

    def mutate_network(self, algorithm, last_hidden_layer_neurons=50, conv_prob=0.9):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        number_cnn_layers = len(neural_network.cnn_layers)
        number_hidden_layers = len(neural_network.hidden_layers)

        new_cnn_neurons = [randint(1, self.maximum_new_cnn_neurons_per_layer) for _ in range(number_cnn_layers)]

        #=======================================================================
        # isAllEmpty = False
        # while isAllEmpty == False:
        #     lower_range = 0
        #     new_cnn_neurons = []
        #     for i in range(number_cnn_layers):
        #         neurons = randint(lower_range, self.maximum_new_cnn_neurons_per_layer)
        #         if neurons > 0:
        #             lower_range = 1
        #         new_cnn_neurons.append(neurons)
        #     isAllEmpty = any(i > 0 for i in new_cnn_neurons)
        #=======================================================================

        new_neurons = [randint(1, self.maximum_new_neurons_per_layer) for _ in range(number_hidden_layers - 1)]
        new_neurons.append(last_hidden_layer_neurons)
        # new_cnn_neurons = [randint(1, self.maximum_new_cnn_neurons_per_layer) for i in range(number_cnn_layers)]
        # new_hidden_neurons = [randint(1, self.maximum_new_cnn_neurons_per_layer) for i in range(number_hidden_layers - 1)]
        cnn_layers = [[create_cnn_neuron(activation_function=None, conv_prob=conv_prob) for _ in range(neuron)] for neuron in new_cnn_neurons]
        # hidden_layers = [[create_neuron(activation_function=None) for i in range(neuron)] for neuron in new_hidden_neurons]
        hidden_layers = [[create_neuron(activation_function=None, bias=bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight) for _ in range(neuron)] for neuron in new_neurons]
        return cnn_layers, hidden_layers

class Mutation_CNN_2(Mutation):
    """Adds one CNN neurons to first CNN layer."""

    def __init__(self, maximum_new_cnn_neurons_per_layer =1,maximum_new_neurons_per_layer =1,  maximum_bias_connection_weight=1.0):
        self.maximum_new_cnn_neurons_per_layer = maximum_new_cnn_neurons_per_layer
        self.maximum_new_neurons_per_layer = maximum_new_neurons_per_layer
        self.maximum_bias_connection_weight = maximum_bias_connection_weight

    def mutate_network(self, algorithm):
        neural_network = algorithm.champion.neural_network
        bias = neural_network.bias
        number_cnn_layers = len(neural_network.cnn_layers)
        number_hidden_layers = len(neural_network.hidden_layers)
        """ -1 here """
        new_neurons = [randint(1, self.maximum_new_neurons_per_layer) for _ in range(number_hidden_layers - 1)]
        new_cnn_neurons = [randint(1, self.maximum_new_cnn_neurons_per_layer) for _ in range(number_cnn_layers)]
        # new_hidden_neurons = [randint(1, self.maximum_new_cnn_neurons_per_layer) for i in range(number_hidden_layers - 1)]
        cnn_layers = [[create_cnn_neuron(activation_function=None) for _ in range(neuron)] for neuron in new_cnn_neurons]
        # hidden_layers = [[create_neuron(activation_function=None) for i in range(neuron)] for neuron in new_hidden_neurons]
        hidden_layers = [[create_neuron(activation_function=None, bias=bias, maximum_bias_connection_weight=self.maximum_bias_connection_weight) for _ in range(neuron)] for neuron in new_neurons]
        return cnn_layers, hidden_layers























