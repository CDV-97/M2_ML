# Árbol de Decisión sin Framework

Implementación desde cero de un algoritmo de Árbol de Decisión para clasificación, sin utilizar frameworks de Machine Learning para la construcción del modelo.

## Objetivo

Implementar manualmente un algoritmo de aprendizaje máquina y evaluar su funcionamiento utilizando un conjunto de datos real.

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

El Árbol de Decisión fue implementado manualmente en `decision_tree.py`.

El algoritmo incluye:

- Cálculo de impureza Gini
- Búsqueda de la mejor división
- División de observaciones
- Construcción recursiva del árbol
- Creación de nodos hoja
- Predicción de nuevas observaciones

No se utiliza ningún modelo de Machine Learning previamente implementado.

## Librerías utilizadas

Las librerías externas se utilizan únicamente para:

- Lectura y manipulación del dataset
- División de entrenamiento y prueba
- Evaluación del modelo

El modelo de Árbol de Decisión no utiliza frameworks de Machine Learning.

## Ejecución

Instalar las dependencias:

```bash
pip install -r requirements.txt
