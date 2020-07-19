# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelBinarizer
#
# from algorithms.common.metric import Cross_entropy
#
#
# def fit_and_predict(estimator, X_train, y_train, X_test, y_test):
#     model = estimator.fit(X_train, y_train, Cross_entropy)
#
#     y_pred_ce_train, y_pred_labels_train = model.predict(X_train)
#     y_train_ce = LabelBinarizer().fit_transform(y_train)
#     ce_train = Cross_entropy.evaluate(y_pred_ce_train, y_train_ce)
#     accuracy_train = accuracy_score(y_train, y_pred_labels_train)
#     print('Training set: CE loss %.5f, accuracy %.5f%%' % (ce_train, accuracy_train * 100))
#
#     y_pred_ce_test, y_pred_labels_test = model.predict(X_test)
#     y_test_ce = LabelBinarizer().fit_transform(y_test)
#     ce_test = Cross_entropy.evaluate(y_pred_ce_test, y_test_ce)
#     accuracy_test = accuracy_score(y_test, y_pred_labels_test)
#     print('Test set: CE loss %.5f, accuracy %.5f%%' % (ce_test, accuracy_test * 100))
