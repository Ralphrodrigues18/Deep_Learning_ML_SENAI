import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization

def create_model():
    model = keras.Sequential()

    # Adicionando a camada de entrada
    model.add(Input(shape=(32, 32, 3)))

    # Adicionando a primeira camada convolucional
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Adicionando a primeira camada de pooling
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Adicionando uma camada de dropout para reduzir overfitting


    # Adicionando a segunda camada convolucional
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Adicionando a segunda camada de pooling
    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.25))  # Adicionando uma camada de dropout para reduzir overfitting


    # Adicionando a terceira camada convolucional
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # Adicionando a terceira camada de pooling
    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.25))  # Adicionando uma camada de dropout para reduzir overfitting


    # Transição CNN -> Dense (totalmente conectada)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Adicionando uma camada de dropout para reduzir overfitting
    model.add(Dense(10, activation='softmax'))


    #Compiler
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # Resumo da arquitetura do modelo
    model.summary()
    return model