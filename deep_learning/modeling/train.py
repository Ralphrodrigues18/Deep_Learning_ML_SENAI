import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from deep_learning.modeling.pipeline import preprocess_data
from deep_learning.modeling.model import create_model

X_train, y_train, X_test, y_test, class_names = preprocess_data()
#model = create_model()

def train_model (model, X_train, y_train, epochs=30):
  
    # Definindo o callback de early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Treinando o modelo
    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    return history