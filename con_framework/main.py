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
