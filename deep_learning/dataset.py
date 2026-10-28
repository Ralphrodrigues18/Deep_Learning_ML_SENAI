import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten



def load_dataset():
    #Carregando o dataset CIFAR-10
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    return X_train, y_train, X_test, y_test
