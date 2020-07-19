from mnist import load
from dslm.algorithms.semantic_learning_machine.algorithm_cnn import DeepSemanticLearningMachine
from dslm.algorithms.common.stopping_criterion import MaxGenerationsCriterion
from utils.common import fit_and_predict


def get_estimator():
    iterations = 5
    min_cp_layers = 1
    max_cp_layers = 2

    cnn_neurons_per_layer = 3
    
    min_ncp_layers = 2
    max_ncp_layers = 4
    init_last_hidden_layer_neurons = 200
    mutation_last_hidden_layer_neurons = 50
    conv_prob = 0.75

    estimator = DeepSemanticLearningMachine(
        1,
        MaxGenerationsCriterion(iterations),
        min_cp_layers,
        max_cp_layers,
        cnn_neurons_per_layer,
        min_ncp_layers,
        max_ncp_layers,
        init_last_hidden_layer_neurons,
        mutation_last_hidden_layer_neurons,
        conv_prob)


    return estimator


if __name__ == '__main__':

    scale = True
    X_train, y_train, X_test, y_test = load(scale)

    for run in range(1, 1 + 1):
        print('MNIST: SLM run', run)
        estimator = get_estimator()
        fit_and_predict(estimator, X_train, y_train, X_test, y_test)
