from copy import copy, deepcopy
import numpy as np
from .activation_function import calculate_output
from common.conv_neural_network_components.neuron import Neuron
from ..activation_functions import ACTIVATION_FUNCTIONS_DICT

class ConvNeuron(Neuron):

    def __init__(self, semantics, input_connections, level_layer, activation_function_id):
        self.activation_function = ACTIVATION_FUNCTIONS_DICT.get(activation_function_id)
        super().__init__(semantics, input_connections,  self.activation_function)
        self.level_layer = level_layer
        self._initialize_parameters()

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return ConvNeuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = ConvNeuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron

    def _initialize_parameters(self):
        self.nr_of_channel = 1
        self.kernel_size = np.random.randint(2, 4)
        self.stride = np.random.randint(1, 3)
        # prepare yourself for different number of channels not only RGB but also 1 dim black/white
        self.filter = np.random.randint(-2, 3, (self.kernel_size, self.kernel_size, 3))


    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = int(np.ceil((input_pic_array.shape[0] - self.kernel_size + 1) / self.stride))
        # self.output_width = (input_pic_array.shape[0] - self.kernel_size + 2 * (self.kernel_size - 1)) // self.stride + 1
        self.output_length = int(np.ceil((input_pic_array.shape[1] - self.kernel_size + 1) / self.stride))
        # self.output_length = (input_pic_array.shape[1] - self.kernel_size + 2 * (self.kernel_size - 1)) // self.stride + 1

    def convolv(self):
        # sensors = self.input_connections[0].from_neuron
        all_semantics = [connection.from_neuron.semantics for connection in self.input_connections]
        semantics_array = np.array(all_semantics).reshape((32, 32, 3, self.input_connections[0].from_neuron.semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        return convolv(semantics_array, self.output_width, self.output_length, self.input_width, self.input_length, self.dimenstions, self.kernel_size, self.stride, self.filter)

    def convolv2(self):
        all_semantics = [connection.from_neuron.semantics for connection in self.input_connections]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, self.input_connections[0].from_neuron.semantics.shape[-1]))
        semantics_array = np.pad(semantics_array, (self.kernel_size - 1, self.kernel_size - 1), 'constant', constant_values=(0, 0))
        semantics_array = semantics_array[ :, :, self.kernel_size - 1:-(self.kernel_size - 1), self.kernel_size - 1:-(self.kernel_size - 1)]

        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        return convolv(semantics_array, self.output_width, self.output_length, self.input_width, self.input_length, self.dimenstions, self.kernel_size, self.stride, self.filter)

    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.convolv()

    def calculate2(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.convolv2()

    def __str__(self):
            return 'ConvNeuron'

    def add_input_connection(self, new_connection):
        """Receives a new input connection for this neuron.

        Parameters
        ----------
        new_connection : Connection
            New connection with neuron from a previous layer in the neural
            network.
        """
        self.input_connections.append(new_connection)

    def clean_semantics(self):
        """Resets the semantics."""
        self.semantics = None

def convolv(semantics_array, output_width, output_length, input_width, input_length, dimenstions, kernel_size, stride, filter):
    input_pic_array = semantics_array
    dataset_whole_output = np.zeros(
        (output_width, output_length, dimenstions, semantics_array.shape[3]))

    col = 0
    for j in range(0, input_length - kernel_size + 1, stride):
        row_output = np.zeros((output_width, dimenstions, semantics_array.shape[3]))
        row = 0
        for i in range(0, input_width - kernel_size + 1, stride):
            new_array = input_pic_array[i:(i + kernel_size), j:(j + kernel_size), :, :]

            output = np.multiply(new_array, filter[:,:, :, np.newaxis])
            row_output[row, :, :] = np.sum(output, axis=(0, 1))
            row += 1
        dataset_whole_output[:, col, :, :] = row_output
        col += 1
    return dataset_whole_output

