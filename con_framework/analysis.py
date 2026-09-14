# Análisis de desempeño del Árbol de Decisión
import pandas as pd

from sklearn.model_selection import train_test_split


# Cargar dataset
data = pd.read_csv("../water_potability.csv")

# Rellenar valores faltantes con la media
data = data.fillna(data.mean())

# Separar variables predictoras y variable objetivo
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Primera división: 70% train, 30% temporal
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
