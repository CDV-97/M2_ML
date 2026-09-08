import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

def main():
    # Cargar dataset
    data = pd.read_csv("../water_potability.csv")

    # Rellenar valores faltantes con la media
    data = data.fillna(data.mean())

    print("Primeras filas del dataset:")
    print(data.head())
    # Separar variables predictoras y variable objetivo
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Crear el modelo usando scikit-learn
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        min_samples_split=2,
        random_state=42
    )
    # Entrenar modelo
    model.fit(X_train, y_train)

    # Realizar predicciones
    predictions = model.predict(X_test)

    # Mostrar ejemplos de predicciones
    print("\nEjemplos de predicciones:")
    for i in range(5):
        print(
            f"Ejemplo {i + 1}: "
            f"Real = {y_test.iloc[i]}, "
            f"Predicción = {predictions[i]}"
        )
    # Evaluar resultados
    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nF1-score de la clase potable:")
    print(f1_score(y_test, predictions))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, predictions))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()

