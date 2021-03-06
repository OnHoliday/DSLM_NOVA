import numpy as np
from algorithms.common.neural_network.activation_function import calculate_output
from algorithms.common.neural_network.pooling_function import *
from copy import copy, deepcopy
from numba import jit
import random


class Node(object):
    """
    Class represents abstract node in neural network.

    Attributes:
        semantics: Semantic vector
    """

    def __init__(self, semantics):
        self.semantics = semantics
    
    def free_semantics(self):
        self.semantics = None


class Sensor(Node):
    """
    Class represents input sensor in neural network.
    """

    def __deepcopy__(self, memodict={}):
        sensor = Sensor(np.array([]))
        memodict[id(self)] = sensor
        return sensor


class Neuron(Node):
    """
    Class represents neuron in neural network.

    Attributes:
        input_connections = Set of input connections
        activation_function = String for activation function id
    """

    def __init__(self, semantics, input_connections, activation_function, cache_weighted_input=False):
        super().__init__(semantics)
        self.input_connections = input_connections
        self.activation_function = activation_function
        self.cache_weighted_input = cache_weighted_input
        self.weighted_input = None

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
        if self.cache_weighted_input:
            self.weighted_input = weighted_input
        self.semantics = self._calculate_output(weighted_input)
    
    def get_weighted_input(self):
        return self.weighted_input
    
    def set_previous_bias(self, previous_bias):
        self.previous_bias = previous_bias
    
    def incremental_semantics_computation(self, n_new_neurons):
        self.weighted_input -= self.previous_bias
        
        for connection in self.input_connections[-n_new_neurons:]:
            self.weighted_input += connection.from_node.semantics * connection.weight
        
        self.weighted_input += self.input_connections[0].weight
        self.semantics = self._calculate_output(self.weighted_input)
    
    def free_semantics(self):
        self.semantics = None
        if self.cache_weighted_input:
            self.free_weighted_input()
    
    def free_weighted_input(self):
        self.cache_weighted_input = False
        self.weighted_input = None


class ConvNeuron_3dfilter(Neuron):

    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics, input_connections, activation_function)
        self._initialize_parameters()

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return ConvNeuron_3dfilter(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = ConvNeuron_3dfilter(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron

    def _initialize_parameters(self):
        self.nr_of_channel = 1
        self.kernel_size = np.random.randint(2, 7)
        self.stride = np.random.randint(1, 3)
        self.filter = np.random.uniform(-1, 1, (self.kernel_size, self.kernel_size, self.dimenstions))

    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = int(np.ceil((input_pic_array.shape[0] - self.kernel_size + 1) / self.stride))
        self.output_length = int(np.ceil((input_pic_array.shape[1] - self.kernel_size + 1) / self.stride))

    def convolv(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        semantics_array = np.array(all_semantics).reshape((32, 32, 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            print(s)
            input_pic_array = semantics_array[:, :, :, s]
            self.get_sizes(input_pic_array)
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))

            dim_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            col = 0
            for j in range(0, self.input_length - self.kernel_size + 1, self.stride):
                row_output = np.zeros((self.output_width,))
                row = 0
                for i in range(0, self.input_length - self.kernel_size + 1, self.stride):
                    new_array = input_pic_array[i:(i + self.kernel_size), j:(j + self.kernel_size), d]
                    output = np.multiply(new_array, self.filter)
                    row_output[row] = np.sum(output)
                    row += 1
                dim_output[col, :] = row_output
                col += 1

            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output

    def convolv2(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        dataset_whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions, semantics_array.shape[3]))
        for s in range(semantics_array.shape[3]):
            print(s)
            input_pic_array = semantics_array[:, :, :, s]
            self.get_sizes(input_pic_array)
            whole_output = np.zeros((self.output_width, self.output_length, self.dimenstions))
            for d in range(self.dimenstions):
                dim_output = np.zeros((self.output_width, self.output_length))
                col = 0
                for j in range(0, self.input_length - self.kernel_size + 1, self.stride):
                    row_output = np.zeros((self.output_width,))
                    row = 0
                    for i in range(0, self.input_length - self.kernel_size + 1, self.stride):
                        new_array = input_pic_array[i:(i + self.kernel_size), j:(j + self.kernel_size), d]
                        output = np.multiply(new_array, self.filter)
                        row_output[row] = np.sum(output)
                        row += 1
                    dim_output[col, :] = row_output
                    col += 1
                whole_output[:, :, d] = dim_output
            dataset_whole_output[:, :, :, s] = whole_output
        return dataset_whole_output


class ConvNeuron(Neuron):

    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics, input_connections, activation_function)
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
        self.filter = np.random.randint(1, 2, (self.kernel_size, self.kernel_size, 3, 50000))


    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = int(np.ceil((input_pic_array.shape[0] - self.kernel_size + 1) / self.stride))
        # self.output_width = (input_pic_array.shape[0] - self.kernel_size + 2 * (self.kernel_size - 1)) // self.stride + 1
        self.output_length = int(np.ceil((input_pic_array.shape[1] - self.kernel_size + 1) / self.stride))
        # self.output_length = (input_pic_array.shape[1] - self.kernel_size + 2 * (self.kernel_size - 1)) // self.stride + 1

    def convolv(self):
        sensors = self.input_connections[0].from_node
        all_semantics = [sensor.semantics for sensor in sensors]
        semantics_array = np.array(all_semantics).reshape((32, 32, 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        return convolv(semantics_array, self.output_width, self.output_length, self.input_width, self.input_length, self.dimenstions, self.kernel_size, self.stride, self.filter)

    def convolv2(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        # print(len(sensors))
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, sensors[0].semantics.shape[-1]))
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


class PoolNeuron(Neuron):

    def __init__(self, semantics, input_connections, activation_function):
        super().__init__(semantics, input_connections, activation_function)
        self._initialize_parameters()

    def __copy__(self):
        copy_semantics = copy(self.semantics)
        copy_input_connections = copy(self.input_connections)
        copy_activation_function = self.activation_function
        return PoolNeuron(copy_semantics, copy_input_connections, copy_activation_function)

    def __deepcopy__(self, memodict={}):
        input_connections = deepcopy(self.input_connections, memodict)
        activation_function = self.activation_function
        semantics = np.array([])
        neuron = PoolNeuron(semantics, input_connections, activation_function)
        memodict[id(self)] = neuron
        return neuron

    def _initialize_parameters(self):
        self.pool_size = np.random.randint(2, 5)
        self.stride = np.random.randint(1, 3)
        self.operation = random.choice(['max', 'min', 'avg'])

    def get_sizes(self, input_pic_array):
        self.input_width = input_pic_array.shape[0]
        self.input_length = input_pic_array.shape[1]
        self.dimenstions = input_pic_array.shape[2]
        self.output_width = int(np.ceil((input_pic_array.shape[0] - self.pool_size + 1) / self.stride))
        # self.output_width = (input_pic_array.shape[0] - self.pool_size + 2 * (self.pool_size - 1)) // self.stride + 1

        self.output_length = int(np.ceil((input_pic_array.shape[1] - self.pool_size + 1) / self.stride))
        # self.output_length = (input_pic_array.shape[1] - self.pool_size + 2 * (self.pool_size - 1)) // self.stride + 1

    def pool(self):
        sensors = self.input_connections[0].from_node
        all_semantics = [sensor.semantics for sensor in sensors]
        semantics_array = np.array(all_semantics).reshape((int(np.sqrt(len(sensors) / 3)), int(np.sqrt(len(sensors) / 3)), 3, sensors[0].semantics.shape[-1]))
        input_pic_array = semantics_array[:, :, :, 0]
        self.get_sizes(input_pic_array)
        return pool(semantics_array, self.output_width, self.output_length, self.input_width, self.input_length, self.dimenstions, self.pool_size, self.stride, self.operation)

    def pool2(self):
        all_semantics = []
        sensors = self.input_connections[0].from_node
        [all_semantics.append(sensor.semantics) for sensor in sensors]
        all_semantics = all_semantics[0]
        semantics_array = np.array(all_semantics).reshape((all_semantics.shape[0], all_semantics.shape[1], 3, sensors[0].semantics.shape[-1]))
        semantics_array2 = np.pad(semantics_array, (self.pool_size - 1, self.pool_size - 1), 'constant', constant_values=(0, 0))
        semantics_array3 = semantics_array2[:, :, self.pool_size - 1 :-(self.pool_size - 1), self.pool_size - 1:-(self.pool_size - 1)]
        input_pic_array = semantics_array3[:, :, :, 0]
        self.get_sizes(input_pic_array)
        return pool(semantics_array3, self.output_width, self.output_length, self.input_width, self.input_length, self.dimenstions, self.pool_size, self.stride, self.operation)

    def _calculate_output(self, weighted_input):
        """Calculates semantics, based on weighted input."""
        return calculate_output(weighted_input, self.activation_function)

    def calculate(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.pool()

    def calculate2(self):
        """Calculates weighted input, then calculates semantics."""
        self.semantics = self.pool2()

    def __str__(self):
            return 'PoolNeuron'


@jit(nopython=True, fastmath=True)
def convolv2(semantics_array, output_width, output_length, input_width, input_length, dimenstions, kernel_size, stride, filter):
    dataset_whole_output = np.zeros(
        (output_width, output_length, dimenstions, semantics_array.shape[3]))
    for s in range(semantics_array.shape[3]):
        # print(s)
        input_pic_array = semantics_array[:, :, :, s]
        whole_dim_output = np.zeros((output_width, output_length, dimenstions))
        for dim in range(semantics_array.shape[2]):
            input_pic_array_dim = input_pic_array[:, :, dim]
            whole_output = np.zeros((output_width, output_length))
            col = 0
            for j in range(0, input_width - kernel_size + 1, stride):
                row_output = np.zeros((output_width,))
                row = 0
                for i in range(0, input_length - kernel_size + 1, stride):
                    new_array = input_pic_array_dim[i:(i + kernel_size), j:(j + kernel_size)]
                    output = np.multiply(new_array, filter)
                    row_output[row] = np.sum(output)
                    #                row_output[row, :] = np.sum(output)
                    row += 1
                whole_output[:, col] = row_output
                col += 1
            whole_dim_output[:, :, dim] = whole_output
        dataset_whole_output[:, :, :, s] = whole_dim_output
    return dataset_whole_output


# @jit(nopython=True, fastmath=True)
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

            output = np.multiply(new_array, filter)
            row_output[row, :, :] = np.sum(output, axis=(0, 1))
            row += 1
        dataset_whole_output[:, col, :, :] = row_output
        col += 1
    return dataset_whole_output


@jit(nopython=True, parallel=True)
def pool2(semantics_array, output_width, output_length, input_width, input_length, dimenstions, pool_size):
    dataset_whole_output = np.zeros((output_width, output_length, dimenstions, semantics_array.shape[3]))
    for s in range(semantics_array.shape[3]):
        # print(s)
        input_pic_array = semantics_array[:, :, :, s]
        whole_dim_output = np.zeros((output_width, output_length, dimenstions))
        for dim in range(semantics_array.shape[2]):
            input_pic_array_dim = input_pic_array[:, :, dim]
            whole_output = np.zeros((output_width, output_length))
            col = 0
            for j in range(input_length - pool_size + 1):
                row_output = np.zeros((output_width,))
                row = 0
                for i in range(input_width - pool_size + 1):
                    new_array = input_pic_array_dim[i:(i + pool_size), j:(j + pool_size)]
                    row_output[row] = np.max(new_array)
                    row += 1
                whole_output[:, col] = row_output
                col += 1
            whole_dim_output[:, :, dim] = whole_output
        dataset_whole_output[:, :, :, s] = whole_dim_output
    return dataset_whole_output


def pool(semantics_array, output_width, output_length, input_width, input_length, dimenstions, pool_size, stride, operation):
    dataset_whole_output = np.zeros((output_width, output_length, dimenstions, semantics_array.shape[3]))
    input_pic_array = semantics_array
    col = 0
    for j in range(0, input_length - pool_size + 1, stride):
        row_output = np.zeros((output_width, dimenstions, semantics_array.shape[3]))
        row = 0
        for i in range(0, input_width - pool_size + 1, stride):
            new_array = input_pic_array[i:(i + pool_size), j:(j + pool_size), :, :]
            row_output[row, :, :] = calculate_pool_output(new_array, operation)
            row += 1
        dataset_whole_output[:, col, :, :] = row_output
        col += 1
    return dataset_whole_output
