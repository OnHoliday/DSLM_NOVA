import timeit

from common.metrics import calculate_accuracy
import cProfile, pstats, io, os

def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner


@profile
def fit_and_predict(estimator, X_train, y_train, X_test, y_test):
    model = estimator.fit(X_train, y_train)
    y_pred_train = model.get_predictions()
    y_pred_train = y_pred_train.argmax(axis=1)

    start_time = timeit.default_timer()
    y_pred_test = estimator.predict(X_test)
    y_pred_test = y_pred_test.argmax(axis=1)
    time = timeit.default_timer() - start_time
    print('\tPredict time for test data:\t%.3f seconds in total, %.6f seconds per instance, %f milliseconds per instance' % (time, time / X_test.shape[0], time / X_test.shape[0] / 10 ** -3))

    print('\nAccuracy (train vs. test)\t\t%.2f%% vs. %.2f%%' % (calculate_accuracy(y_train, y_pred_train) * 100, calculate_accuracy(y_test, y_pred_test) * 100))
