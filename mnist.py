from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler


def load(scale=True):
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    print('Returning from MNIST load')
    
    return X_train, y_train, X_test, y_test
