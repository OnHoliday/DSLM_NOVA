from mnist import load
from slm.slm_classifier import SLMClassifier
from utils.common import fit_and_predict


def get_estimator():
    learning_step_parameter = 'optimized'
    maximum_neuron_connection_weight = 0.1
    maximum_bias_weight = 0.1
    hidden_activation_function = 'relu'
    
    estimator = SLMClassifier(sample_size=1,  # 3
        neighborhood_size=1,
        max_iter=50,
        learning_step=learning_step_parameter,
        learning_step_solver='numpy_pinv',  # numpy_pinv, numpy_lstsq, scipy_pinv, scipy_pinv2, scipy_lstsq
        activation_function='identity',
        activation_function_for_hidden_layers=hidden_activation_function,
        prob_activation_hidden_layers=1.0,
        mutation_operator='all_hidden_layers',
        #===================================================
        # init_minimum_layers=5,
        # init_max_layers=5 + 1,
        #===================================================
        init_minimum_layers=1,
        init_max_layers=1 + 1,
        init_maximum_neurons_per_layer=3,
        maximum_new_neurons_per_layer=3,
        maximum_neuron_connection_weight=maximum_neuron_connection_weight,
        maximum_bias_weight=maximum_bias_weight,
        #===================================================
        # maximum_neuron_connection_weight =1 / (784 / 2),
        # maximum_bias_weight = 1 / (784 / 2),
        #===================================================
        random_state=None,
        verbose=True,
        stopping_criterion=None,  # 'edv', 'tie'
        edv_threshold=0.25,
        tie_threshold=0.25,
        #=======================================================================
        # sparse=True,
        #=======================================================================
        sparse=False,
        minimum_sparseness=0.25,
        maximum_sparseness=0.75,
        early_stopping=False,
        validation_fraction=0.1,
        tol=0.0001,
        n_iter_no_change=10,
        metric='rmse',
        prob_skip_connection=0.0)  # 0.35
    
    return estimator


if __name__ == '__main__':
    
    scale = True
    X_train, y_train, X_test, y_test = load(scale)
    
    for run in range(1, 1 + 1):
        print('MNIST: SLM run', run)
        estimator = get_estimator()
        fit_and_predict(estimator, X_train, y_train, X_test, y_test)
