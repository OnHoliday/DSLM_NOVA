from algorithms.common.metric import RootMeanSquaredError, Accuracy, is_better, WeightedRootMeanSquaredError
from numpy import array
import unittest


class TestMetrics(unittest.TestCase):

    def test_root_mean_squared_error(self):
        prediction = array([1, 2, 3])
        target = array([4, 5, 6])
        value = RootMeanSquaredError.evaluate(prediction, target)
        self.assertEqual(value, 3)
        self.assertTrue(is_better(1, 2, RootMeanSquaredError))

    def test_accuracy(self):
        prediction = array([1, 1, 0, 1])
        target = array([1, 0, 0, 1])
        value = Accuracy.evaluate(prediction, target)
        self.assertEqual(value, 0.75)
        self.assertTrue(is_better(2, 1, Accuracy))
    
    def test_weighted_root_mean_squared_error(self):
        prediction = array([1, 0, 1, 1])
        target = array([1, 0, 1, 1])
        weight_vector = array([0.5, 0.5, 0.5, 0.5])
        metric = WeightedRootMeanSquaredError(weight_vector)
        value = metric.evaluate(prediction, target)
        self.assertEqual(value, 0)
        self.assertTrue(is_better(1, 2, WeightedRootMeanSquaredError))
        

if __name__ == '__main__':
    unittest.main()
