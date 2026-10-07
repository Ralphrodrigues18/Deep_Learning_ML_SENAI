import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test, history, class_names):
    # Avaliando o modelo no conjunto de teste   
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")


    # Plotting Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title("Acurácia por época")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.show()


    # Plotting Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title("Loss por época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    pred_probs = model.predict(X_test)
    pred_labels = np.argmax(pred_probs, axis=1)

    cm = confusion_matrix(y_test, pred_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))

    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)

    plt.title("Matriz de Confusão - CIFAR-10")
    plt.show()


