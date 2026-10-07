from deep_learning.dataset import load_dataset

def preprocess_data():
    X_train, y_train, X_test, y_test = load_dataset()

    #Normalizando os dados
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Definindo os nomes das classes do dataset CIFAR-10
    class_names = [
        "avião",
        "automóvel",
        "pássaro",
        "gato",
        "cervo",
        "cachorro",
        "sapo",
        "cavalo",
        "navio",
        "caminhão"
    ]

    print("formato de X_train:", X_train.shape)
    print("formato de y_train:", y_train.shape)
    print("formato de X_test:", X_test.shape)
    print("formato de y_test:", y_test.shape)

    return X_train, y_train, X_test, y_test, class_names