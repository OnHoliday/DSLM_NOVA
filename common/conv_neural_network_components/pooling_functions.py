from numpy import max, min, average


def calculate_max(input_array):
    """Applies sigmoid function to input."""
    return max(input_array, axis=(0, 1))


def calculate_min(input_array):
    """Applies hyperbolic tangent function to input."""
    return min(input_array, axis=(0, 1))


def calculate_avg(input_array):
    """Applies rectified linear unit function to input."""
    return average(input_array, axis=(0, 1))


def calculate_pool_output(input_array, pooling_function_id):
    """Applies activation function to input, based on determined id."""
    activation_function = _POOLING_FUNCTIONS.get(pooling_function_id)
    return activation_function(input_array)


_POOLING_FUNCTIONS = {
    'max': calculate_max,
    'min': calculate_min,
    'avg': calculate_avg,
}
