from numpy import std, abs

from .utilities import is_better


#################################################################
#### Stopping criteria functions for evolutionary algorithms ####
#################################################################
def apply_error_deviation_variation_criterion(best_solution, offspring, threshold, target_values):
    """Function that stops training process if the share of solutions with lower error deviation variation
    amongst the superior offspring (i.e., offspring that are better performers than the parent), when
    compared with best solution's error deviation variation, falls below than a certain threshold.

    Parameters
    ----------
    best_solution : NeuralNetwork
        Current best performer amongst all neural networks.

    offspring : array-like of shape (number_offspring,)
        Set of offspring generated at a given iteration (NeuralNetwork's instances).

    threshold : float
        Minimum share of offspring that are required to present a lower error deviation variation
        than the current best solution.

    target_values : array-like of shape (n_samples,)
        Target values to compare with solutions' stored predictions.

    Returns
    -------
    stop_training_process : bool
        True if training process should be stopped, and False otherwise.
    """

    # Subsets offspring that are better than the parent:
    superior_offspring = [solution for solution in offspring if solution.is_better_than_ancestor()]
    # If not superior offspring exists, then the training process shouldn't stop:
    if not superior_offspring:
        return False

    # Calculate error standard deviation for both best solution and superior_offspring:
    var_best_solution = std(abs(best_solution.get_predictions() - target_values))
    var_superior_offspring = [std(abs(solution.get_predictions() - target_values)) for solution in superior_offspring]

    # Subsets offspring with lower error deviation variation than var_best_solution:
    lower_var_offspring = [var for var in var_superior_offspring if var < var_best_solution]
    # Calculate percentage (if is lower than threshold, stop training process):
    percentage_lower_var_offspring = len(lower_var_offspring) / len(var_superior_offspring)

    return True if percentage_lower_var_offspring < threshold else False


def apply_training_improvement_effectiveness_criterion(best_solution, offspring, threshold, greater_is_better):
    """Function that stops training process if the mutation effectiveness (i.e., percentage of solutions
    better than the best solution from previous training iteration) falls below a certain threshold.

    Parameters
    ----------
    best_solution : NeuralNetwork
        Current best performer amongst all neural networks (elected in the previous training
        iteration).

    offspring : array-like of shape (number_offspring,)
        Set of offspring generated at a given iteration (NeuralNetwork's instances).

    threshold : float
        Minimum share of offspring that are required to be better than the best solution.

    greater_is_better : bool
        Flag that indicates if objective function is meant to be maximized or minimized.


    Returns
    -------
    stop_training_process : bool
        True if training process should be stopped, and False otherwise.
    """
    # Subsets offspring that are better than best solution:
    superior_offspring = [solution for solution in offspring if is_better(solution.get_loss(), best_solution.get_loss(), greater_is_better=greater_is_better)]
    # If not superior offspring exists, then the training process shouldn't stop:
    if not superior_offspring:
        return False

    # Calculate percentage (if is lower than threshold, stop training process):
    percentage_superior_offspring = len(superior_offspring) / len(offspring)

    return True if percentage_superior_offspring < threshold else False


####################################################
#### Dictionary of stopping criterion functions ####
####################################################

STOPPING_CRITERION_FUNCTIONS_DICT = {
    'edv': apply_error_deviation_variation_criterion,
    'tie': apply_training_improvement_effectiveness_criterion
}
