from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler


def load(scale=True):
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    print('Returning from CIFAR-10 load')
    
    return X_train, y_train, X_test, y_test
