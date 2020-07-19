
from common.neural_network_components.neuron import Neuron

class FlatNeuron(Neuron):
    """Class representing an neuron of the input layer in a neural network.

    Parameters
    ----------
    semantics : array of shape (num_samples,), optional

    Attributes
    ----------
    semantics : array of shape (num_samples,)
        Output vector of neuron. In this case, semantics is the vector with input data.

    bias : float
        Value for neuron's bias (inherited from Neuron).

    bias : array of shape (num_samples,)
        Vector of the same dimension of semantics vector that contains the bias repeated
        across the array.
    """

    def __init__(self, semantics, input_connections):
        super().__init__(bias=None, semantics=semantics)
        self.input_connections = input_connections

    def __repr__(self):
        return "FlatNeuron"

    def add_input_connection(self, new_connection):
        """Receives a new input connection for this neuron.

        Parameters
        ----------
        new_connection : Connection
            New connection with neuron from a previous layer in the neural
            network.
        """
        self.input_connections.append(new_connection)