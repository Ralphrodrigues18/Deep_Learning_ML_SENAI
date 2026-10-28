from dataset import load_dataset
from deep_learning.modeling.model import create_model
from deep_learning.modeling.train import train_model
from deep_learning.modeling.evaluate import evaluate_model
from deep_learning.modeling.pipeline import preprocess_data


def main():
    # Pré-processamento dos dados

    X_train, y_train, X_test, y_test, class_names = preprocess_data()

    model = create_model()

    history = train_model(model, X_train, y_train, epochs=30)

    evaluate_model(model, X_test, y_test, history, class_names)

if __name__ == "__main__":
    main()