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
# Segunda división: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Tamaños de los conjuntos:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Modelo base
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=2,
    random_state=42
)
