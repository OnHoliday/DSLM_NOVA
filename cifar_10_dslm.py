from dslm_new.algorithm.deep_semantic_learning_machine.deep_semantic_learning_machine_app import DSLMApp

from cifar_10 import load
from utils.common import fit_and_predict
from dslm_new.algorithm.deep_semantic_learning_machine.deep_semantic_learning_machine_app import DSLMApp


def get_estimator():
    iterations = 5
    min_cp_layers = 1
    max_cp_layers = 3
    
    cnn_neurons_per_layer = 4
    
    min_ncp_layers = 2
    max_ncp_layers = 4
    init_last_hidden_layer_neurons = 200
    mutation_last_hidden_layer_neurons = 50
    conv_prob = 0.75

    learning_step_parameter = 0.01
    hidden_activation_function = 'sigmoid'
    conv_activation_function = 'sigmoid'
    maximum_neuron_connection_weight = 1.0
    maximum_bias_weight = 1.0


    estimator = DSLMApp(
        sample_size=1,
        neighborhood_size=1,
        max_iter=5,
        learning_step=learning_step_parameter,
        learning_step_solver='numpy_pinv',
        activation_function='identity',
        activation_function_for_hidden_layers=hidden_activation_function,
        activation_function_for_conv_layers=conv_activation_function,
        prob_activation_hidden_layers=1.0,
        prob_activation_conv_layers=1.0,
        mutation_operator='simple_conv_mutation',
        init_minimum_layers=1,
        init_max_layers=1 + 1,
        init_maximum_neurons_per_layer=3,
        maximum_new_neurons_per_layer=3,
        maximum_neuron_connection_weight=maximum_neuron_connection_weight,
        maximum_bias_weight=maximum_bias_weight,
        random_state=None,
        verbose=True,
        stopping_criterion=None,
        edv_threshold=0.25,
        tie_threshold=0.25,
        sparse=False,
        minimum_sparseness=0.25,
        maximum_sparseness=0.75,
        early_stopping=False,
        validation_fraction=0.1,
        tol=0.0001,
        n_iter_no_change=10,
        metric='rmse',
        prob_skip_connection=0.0)
    return estimator


if __name__ == '__main__':

    scale = True
    X_train, y_train, X_test, y_test = load(scale)
    
    for run in range(1, 1 + 1):
        print('CIFAR-10: SLM run', run)
        estimator = get_estimator()
        fit_and_predict(estimator, X_train, y_train, X_test, y_test)
