from algorithms.common.ensemble import Ensemble
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.common.stopping_criterion import MaxGenerationsCriterion
from algorithms.semantic_learning_machine.mutation_operator import Mutation2
from algorithms.common.metric import RootMeanSquaredError
from data.io_plm import load_samples
from data.extract import get_input_variables, get_target_variable
import unittest

class TestEnsemble(unittest.TestCase):

    def setUp(self):
        base_learner = SemanticLearningMachine(50, MaxGenerationsCriterion(10), 2, 'optimized', 10, Mutation2())
        self.ensemble_learner = Ensemble(base_learner, 50)
        self.training, self.validation, self.testing = load_samples('c_diabetes', 0)

    def test_fit(self):
        print("testing fit()...")
        self.ensemble_learner.fit(get_input_variables(self.training).values,
                                  get_target_variable(self.training).values, RootMeanSquaredError, verbose=True)
        self.assertTrue(expr=self.ensemble_learner.learners)

    def test_predict(self):
        print("testing predict()...")
        self.ensemble_learner.fit(get_input_variables(self.training).values,
                                  get_target_variable(self.training).values, RootMeanSquaredError, verbose=True)

        prediction = self.ensemble_learner.predict(get_input_variables(self.validation).values)
        self.assertTrue(expr=len(prediction) == len(get_target_variable(self.validation).values))


if __name__ == '__main__':
    unittest.main()