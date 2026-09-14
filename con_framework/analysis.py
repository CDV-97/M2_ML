# Análisis de desempeño del Árbol de Decisión
import pandas as pd

from sklearn.model_selection import train_test_split


# Cargar dataset
data = pd.read_csv("../water_potability.csv")

# Rellenar valores faltantes con la media
data = data.fillna(data.mean())
