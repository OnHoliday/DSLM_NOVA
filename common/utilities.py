####################################################################
#### Comparison functions made available throughout SLM package ####
####################################################################


def is_better(value1, value2, greater_is_better):
    """Function used to compare two solutions' loss values. Returns True if the value1 is
    considered better than value2.

    Parameters
    ----------
    value1 : float
        Loss value of first solution.

    value2 : float
        Loss value of second solution.

    greater_is_better : bool
        If True, value1 will be considered better than value2 when higher; otherwise, value1
        is better than value2 when lower.

    Returns
    -------
    is_better : bool
        True if value1 is perceived as being better than value2, and False otherwise.
    """
    if greater_is_better:
        return value1 > value2
    else:
        return value1 < value2


##################
#### Snippets ####
##################

def get_closest_positive_number_index(numbers_array, start_search_index):
    """This is a custom-built snippet which looks from the end of
    the array for the first number greater than zero. At the end,
    returns its index."""
    while start_search_index >= 0:
        if numbers_array[start_search_index] > 0:
            return start_search_index
        start_search_index -= 1

    return None
