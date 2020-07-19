import numpy as np
from copy import copy, deepcopy

class Node(object):
    """
    Class represents abstract node in neural network.

    Attributes:
        semantics: Semantic vector
    """

    def __init__(self, semantics):
        self.semantics = semantics

class Neuron(Node):
    """
    Class represents neuron in neural network.

    Attributes:
        input_connections = Set of input connections
        activation_function = String for activation function id
    """

    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics)
        self.input_connections = input_connections
        self.activation_function = activation_function

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return Neuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = Neuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron


    def _calculate_weighted_input(self):
        """Calculates weighted input from input connections."""
        # for connection in self.input_connections:
        #     print(len(connection.from_node.semantics))


        # return np.sum([connection.from_node.semantics * connection.weight for connection in self.input_connections],
        #               axis=0)
        return np.sum(
            (np.multiply(connection.from_node.semantics, connection.weight) for connection in self.input_connections),
            axis=0)

        # try:
        #     return np.sum((np.multiply(connection.from_node.semantics, connection.weight) for connection in self.input_connections), axis=0)
        #     # return np.sum((connection.from_node.semantics * connection.weight for connection in self.input_connections), axis=0)
        # except ValueError:
        #     for connection in self.input_connections:
        #         if len(connection.from_node.semantics) == 0:
        #             connection.from_node.semantics = np.ones(shape=self.input_connections[0].from_node.semantics.shape[0])
        #     return np.sum((connection.from_node.semantics * connection.weight for connection in self.input_connections), axis=0)


    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        weighted_input = self._calculate_weighted_input()
        self.semantics = self._calculate_output(weighted_input)


