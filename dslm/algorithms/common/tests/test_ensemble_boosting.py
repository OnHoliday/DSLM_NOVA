from algorithms.common.ensemble import EnsembleBoosting
from algorithms.semantic_learning_machine.algorithm import SemanticLearningMachine
from algorithms.common.stopping_criterion import MaxGenerationsCriterion, ErrorDeviationVariationCriterion
from algorithms.semantic_learning_machine.mutation_operator import Mutation2
from algorithms.common.metric import RootMeanSquaredError
from numpy import mean, median
from data.io_plm import load_samples
from data.extract import get_input_variables, get_target_variable
import unittest
from timeit import default_timer

class TestEnsembleBoosting(unittest.TestCase):

    def setUp(self):
        self.training, self.validation, self.testing = load_samples('c_diabetes', 0)

    def test_slm_ols_wo_edv(self):
        print("testing fit() for SLM (OLS) without EDV ...")
        base_learner = SemanticLearningMachine(50, MaxGenerationsCriterion(20), 2, 'optimized', 10, Mutation2())
        ensemble_learner = EnsembleBoosting(base_learner, 100, meta_learner=median, learning_rate=1)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        ensemble_learner.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=ensemble_learner.learners)
        print() 

    def test_slm_fls(self):
        print("testing fit() for SLM (FLS) ...")
        base_learner = SemanticLearningMachine(50, MaxGenerationsCriterion(100), 2, 1, 10, Mutation2())
        ensemble_learner = EnsembleBoosting(base_learner, 100, meta_learner=mean, learning_rate=1)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        ensemble_learner.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=ensemble_learner.learners)
        print() 
        
    def test_predict(self): 
        print("testing predict()...")
        base_learner = SemanticLearningMachine(50, ErrorDeviationVariationCriterion(0.25), 2, 1, 10, Mutation2())
        ensemble_learner = EnsembleBoosting(base_learner, 100, meta_learner=median, learning_rate=1)
        X = get_input_variables(self.training).values
        y = get_target_variable(self.training).values
        def time_seconds(): return default_timer()
        start_time = time_seconds()
        ensemble_learner.fit(X, y, RootMeanSquaredError, verbose=False)
        print("time to train algorithm: ", (time_seconds()-start_time))
        start_time = time_seconds()
        prediction = ensemble_learner.predict(get_input_variables(self.validation).values)
        print("time to predict algorithm: ", (time_seconds()-start_time))
        self.assertTrue(expr=len(prediction) == len(get_target_variable(self.validation).values))
        print()


if __name__ == '__main__':
    unittest.main()