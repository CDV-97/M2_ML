import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from decision_tree import DecisionTree


def main():
    # Cargar dataset
    data = pd.read_csv("dataset.csv")

    print("Primeras filas del dataset:")
    print(data.head())

    # Separar variables predictoras y variable objetivo
    X = data.iloc[:, :-1].values.tolist()
    y = data.iloc[:, -1].values.tolist()

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Crear y entrenar nuestro árbol implementado desde cero
    tree = DecisionTree(
        max_depth=5,
        min_samples_split=2
    )

    tree.fit(X_train, y_train)

    # Realizar predicciones
    predictions = tree.predict(X_test)

    # Evaluar resultados
    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, predictions))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
