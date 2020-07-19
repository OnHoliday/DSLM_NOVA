"""
This file contains the class used to represent a Neural Network.
"""

from numpy import zeros
from sklearn.neural_network._base import softmax
from .neural_network import NeuralNetwork
from common.conv_neural_network_components.flat_neuron import FlatNeuron
from common.neural_network_components.connection import Connection
import numpy as np

class ConvNeuralNetwork(NeuralNetwork):
    """Class representing a neural network.

    Parameters
    ----------
    input_layer : array of shape (n_input_neurons,), optional
        Layer of neural network containing input neurons that receives the
        input data.

    hidden_layers : array of shape (n_layers, n_hidden_neurons), optional
        Set of hidden layers of neural network.

    output_layer : array of shape (n_output_neurons,), optional
        Layer containing one or more output neurons that apply an activation
        function to some weighted input.

    Attributes
    ----------
    input_layer : array of shape (n_input_neurons,)
        Layer of neural network containing input neurons that receives the
        input data. The number of neurons should be equal to the number of
        dimensions of input data.

    hidden_layers : array of shape (n_layers, n_hidden_neurons)
        Set of hidden layers of neural network. The last hidden layer should
        be fully connected with the output layer.

    output_layer : array of shape (n_output_neurons,)
        Layer containing one or more output neurons that apply an activation
        function to some weighted input.

    loss : float
        Loss value computed with the loss function.

    better_than_parent : boolean
        True if this neural network's parent has a worse loss than
        its child; otherwise is False.

    predictions : array of shape (num_samples,)
        Predictions computed by the neural network for input data.
    """

    def __init__(self, input_layer, conv_layers, flat_layer,  hidden_layers, output_layer):
        super().__init__(input_layer, hidden_layers, output_layer)
        self.conv_layers = conv_layers
        self.flat_layer = flat_layer


    def __repr__(self):
        return "ConvNeuralNetwork"

    def calculate_hidden_semantics(self):
        """Calculate semantics of the hidden layers."""

        # Compute semantics for hidden neurons:
        for hidden_layer in self.hidden_layers:
            for hidden_neuron in hidden_layer:
                hidden_neuron.calculate_semantics()

    def _get_output_shape(self):
        instances = self.input_layer[0].get_semantics().shape[0]
        outputs = len(self.output_layer)
        return instances, outputs

    def calculate_output_semantics(self, apply_soft_max=False):
        """Calculate semantics of output layer without recalculating the semantics for
        all hidden layers."""

        # Compute semantics for output neuron(s) and store predictions
        self.predictions = zeros(self._get_output_shape())
        for i, output_neuron in enumerate(self.output_layer):
            self.predictions[:, i] = output_neuron.calculate_semantics()

        if apply_soft_max:
            self.predictions = softmax(self.predictions)

    def calculate_semantics(self):
        """Calculate semantics of all hidden neurons and output neurons. At the end,
        it stores the obtained predictions in the neural network itself."""

        self.calculate_conv_semantics()
        self.calculate_hidden_semantics()
        self.calculate_output_semantics()

    def incremental_output_semantics_update(self, num_last_connections, apply_soft_max=False):
        """Sums a given partial_semantics' value to current semantics' value in the
        output layer and, consequently, updates the predictions emitted by the neural
        network. These partial_semantics result from the addition of new (usually
        hidden) neurons to the neural network.

        Parameters
        ----------
        partial_semantics : array of shape (num_samples,)
            Partial semantics to be added arising from the addition of new neurons to
            the neural network.
        """

        for i, output_neuron in enumerate(self.output_layer):
            self.predictions[:, i] = output_neuron.incremental_semantics_update(num_last_connections)

        if apply_soft_max:
            self.predictions = softmax(self.predictions)

    def load_input_neurons(self, X):
        """Loads new input data on the input layer"""
        for neuron, input_data in zip(self.input_layer, X.T):
            neuron.semantics = input_data

    def get_hidden_neurons(self):
        """Returns a list containing all hidden neurons."""
        neurons = list()
        [neurons.extend(hidden_neurons) for hidden_neurons in self.hidden_layers]
        return neurons

    def count_connections(self):
        """Determines the number of connections on the neural network.

        Returns
        -------
        number_connections : int
            The number of connections this neural network contains.
        """

        def _count_connections(layer):
            return sum([len(neuron.input_connections) for neuron in layer])

        counter = _count_connections(self.get_hidden_neurons())
        # Count connections between last hidden layer and the output layer:
        counter += _count_connections(self.output_layer)

        return counter

    def get_topology(self):
        """Creates a dictionary containing the number of hidden layers, hidden
        neurons and connections on the neural network.

        Returns
        -------
        topology : dict
            Dictionary containing information regarding the neural network's
            architecture.
        """
        return {
            "number of input neurons": len(self.input_layer),
            "number of hidden_layers": len(self.hidden_layers),
            "number of hidden_neurons": len(self.get_hidden_neurons()),
            'number of hidden_neurons per layer': [len(layer) for layer in self.hidden_layers],
            "number of output neurons": len(self.output_layer),
            "number of connections": self.count_connections()
        }

    def get_number_hidden_layers(self):
        """Determines the number of hidden layers of the neural network.

        Returns
        -------
        number_hidden_layers : int
            Number of hidden layers present in the neural network.
        """
        return len(self.hidden_layers)

    def get_number_conv_layers(self):
        """Determines the number of hidden layers of the neural network.

        Returns
        -------
        number_hidden_layers : int
            Number of hidden layers present in the neural network.
        """
        return len(self.conv_layers)

    def generate_predictions(self, X):
        """Obtains predictions for a new sample of input data.

        Returns
        -------
        predictions : array of shape (num_samples,)
            Predictions computed by the neural network for input data.
        """
        # Reset semantics across the entire neural network:
        self.clean_semantics()
        # Load data:
        self.load_input_neurons(X)
        # Calculate and store predictions:
        self.calculate_semantics()
        return self.predictions

    def clean_semantics(self):
        """Cleans semantics from entire neural network."""
        [input_neuron.clean_semantics() for input_neuron in self.input_layer]
        [conv_neuron.clean_semantics() for conv_neuron in self.get_conv_neurons()]
        [hidden_neuron.clean_semantics() for hidden_neuron in self.get_hidden_neurons()]
        [output_neuron.clean_semantics() for output_neuron in self.output_layer]

        self.predictions = None

    def free(self):
        [hidden_neuron.free() for hidden_neuron in self.get_hidden_neurons()]
        [output_neuron.free() for output_neuron in self.output_layer]

        self.predictions = None

    def get_loss(self):
        """Returns current neural network's loss value."""
        return self.loss

    def update_loss(self, loss):
        """Overrides current neural network's loss value. Usually, this occurs when new
        input data enters the neural network and semantics are recalculated.

        Parameters
        ----------
        loss : float
            New loss value for input data.
        """
        self.loss = loss

    def update_parent(self, is_better_than_parent):
        """Updates information regarding neural network's parent, i.e., if
        this neural network is a better performer or not than its parent.

        Parameters
        ----------
        is_better_than_parent : bool
            True if this neural network's parent has a worse loss than
            its child; otherwise is False.
        """
        self.is_better_than_parent = is_better_than_parent

    def override_predictions(self, predictions):
        self.predictions = predictions.copy()

    def is_better_than_parent(self):
        """Returns if this neural network is a better performer than its parent
        neural network."""
        return self.is_better_than_parent

    def get_predictions(self, clip=False):
        """Returns the predictions currently stored in the neural network."""

        if clip:
            for i in range(len(self.output_layer)):
                self.output_layer[i].print_semantics_range()
            self.predictions = self.predictions.clip(0, 1)
        return self.predictions

    def get_last_n_neurons(self, num_last_neurons):
        """Returns the last N neurons from the last hidden layer.

        Parameters
        ----------
        num_last_neurons : int
            Number of neurons to be retrieved from last hidden layer.
        """
        last_layer = self.hidden_layers[-1]
        return last_layer[-num_last_neurons:]

    def get_number_last_hidden_neurons(self):
        """Returns the number of neurons in the last hidden layer."""
        return len(self.hidden_layers[-1])

    def extend_hidden_layer(self, layer_index, new_neurons):
        """Adds a set of newly created neurons to a given hidden layer of the
        neural network.

        Parameters:
        ----------
        layer_index : int
            Index of hidden layer that will receive the new neurons.

        new_neurons : array of shape (num_neurons,)
            Neurons to be added on the specified hidden layer.
        """
        self.hidden_layers[layer_index].extend(new_neurons)

    def extend_conv_layer(self, layer_index, new_neurons):
        """Adds a set of newly created neurons to a given hidden layer of the
        neural network.

        Parameters:
        ----------
        layer_index : int
            Index of hidden layer that will receive the new neurons.

        new_neurons : array of shape (num_neurons,)
            Neurons to be added on the specified hidden layer.
        """
        self.conv_layers[layer_index].extend(new_neurons)

    def extend_flat_layer(self,  flatten_layer_mutated):
        """Adds a set of newly created neurons to a given hidden layer of the
        neural network.

        Parameters:
        ----------
        layer_index : int
            Index of hidden layer that will receive the new neurons.

        new_neurons : array of shape (num_neurons,)
            Neurons to be added on the specified hidden layer.
        """
        self.flat_layer.extend(flatten_layer_mutated)

    def calculate_conv_semantics(self):
            """Calculate semantics of the conv and flat layers."""

            for i, layer in enumerate(self.conv_layers):
                if i == 0:
                    for neuron in layer:
                        neuron.calculate()
                else:
                    for neuron in layer:
                        neuron.calculate2()

            last_layer = self.conv_layers[-1]
            flatten_layer = None
            for neuron in last_layer:
                temp_arr = neuron.semantics
                temp_arr2 = temp_arr.reshape(-1, neuron.semantics.shape[3])

                if flatten_layer is not None:
                    flatten_layer = np.concatenate((flatten_layer, temp_arr2), axis=0)
                else:
                    flatten_layer = temp_arr2

            self.flaten_layer = np.array([FlatNeuron(d, list()) for d in flatten_layer])

            self.load_semantic_after_flat_layer()

    def load_semantic_after_flat_layer(self):
        """Loads input variables of dataset into sensors. Adjusts length of bias."""

        for hidden_neuron in self.hidden_layers[0]:
            for connection, flat_data in zip(hidden_neuron.input_connections, self.flaten_layer.T):
                connection.from_neuron.semantics = flat_data.semantics
            # self.bias.semantics = np.array([1 for i in range(flat_data.semantics.shape[0])])

    def _connect_flat_hidden(self, from_nodes, to_nodes):

        for to_node in to_nodes:

            to_node.input_connections = []
            for from_node in from_nodes:
                Connection(from_node, to_node, np.random.uniform(-2, 2))

        # Auxiliary list that will contain neurons used for skip co


    def get_conv_neurons(self):
        """Returns a list containing all hidden neurons."""
        neurons = list()
        [neurons.extend(conv_neurons) for conv_neurons in self.conv_layers]
        return neurons