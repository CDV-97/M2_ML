# Árbol de Decisión - Con y Sin Framework

Este repositorio contiene dos implementaciones de un algoritmo de Árbol de Decisión para clasificación utilizando el dataset **Water Potability**:

- Una implementación desarrollada manualmente **sin framework de Machine Learning**.
- Una implementación utilizando `DecisionTreeClassifier` de **scikit-learn**.

## Objetivo

Implementar y evaluar un Árbol de Decisión utilizando dos enfoques diferentes: una implementación manual desde cero y una implementación utilizando un framework de Machine Learning.

## Dataset

Se utilizó el dataset **Water Potability**, disponible en Kaggle.

El objetivo es predecir si una muestra de agua es potable:

- `0`: No potable
- `1`: Potable

El dataset contiene variables relacionadas con propiedades físico-químicas del agua, entre ellas:

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic carbon
- Trihalomethanes
- Turbidity

## Implementación

### Sin framework

La implementación manual se encuentra en la carpeta:

`sin_framework/`

El Árbol de Decisión fue desarrollado desde cero en `decision_tree.py`.

El algoritmo incluye:

- Cálculo de impureza Gini
- Búsqueda de la mejor división
- División de observaciones
- Construcción recursiva del árbol
- Creación de nodos hoja
- Predicción de nuevas observaciones

No se utiliza ningún modelo de Machine Learning previamente implementado.

### Con framework

La implementación utilizando framework se encuentra en la carpeta:

`con_framework/`

Se utiliza `DecisionTreeClassifier` de `scikit-learn`.

El modelo fue configurado utilizando parámetros como:

- `criterion`
- `max_depth`
- `min_samples_split`
- `random_state`

## Librerías utilizadas

Se utilizaron principalmente:

- `pandas` para lectura y manipulación del dataset.
- `scikit-learn` para división de entrenamiento y prueba, métricas de evaluación y la implementación del Árbol de Decisión con framework.

En la versión **sin framework**, `scikit-learn` no se utiliza para construir el modelo.

## Ejecución

### Sin framework

```bash
cd sin_framework
python main.py

```bash
cd con_framework
python main.py
