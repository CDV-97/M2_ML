import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. CARGA Y PREPARACIÓN DE DATOS

data = pd.read_csv("../water_potability.csv")

# Rellenar valores faltantes con la media
data = data.fillna(data.mean())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]



# 2. DIVISIÓN TRAIN / VALIDATION / TEST
# 70% entrenamiento, 15% validación, 15% prueba

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

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

# 3. FUNCIÓN PARA EVALUAR UN MODELO

def evaluar_modelo(modelo, nombre):
    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)
    pred_test = modelo.predict(X_test)

    resultados = {
        "train_acc": accuracy_score(y_train, pred_train),
        "val_acc": accuracy_score(y_val, pred_val),
        "test_acc": accuracy_score(y_test, pred_test),
        "train_f1": f1_score(y_train, pred_train),
        "val_f1": f1_score(y_val, pred_val),
        "test_f1": f1_score(y_test, pred_test)
    }

    print(f"\n{nombre}")
    print(
        f"Accuracy -> "
        f"Train: {resultados['train_acc']:.4f} | "
        f"Validation: {resultados['val_acc']:.4f} | "
        f"Test: {resultados['test_acc']:.4f}"
    )

    print(
        f"F1-score -> "
        f"Train: {resultados['train_f1']:.4f} | "
        f"Validation: {resultados['val_f1']:.4f} | "
        f"Test: {resultados['test_f1']:.4f}"
    )

    return resultados


# 4. MODELO BASE

modelo_base = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=2,
    random_state=42
)

modelo_base.fit(X_train, y_train)

base = evaluar_modelo(
    modelo_base,
    "MODELO BASE"
)

# 5. ANÁLISIS DE SESGO / VARIANZA SEGÚN PROFUNDIDAD

depths = range(1, 16)

train_f1_depth = []
val_f1_depth = []

for depth in depths:
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    train_f1_depth.append(
        f1_score(
            y_train,
            modelo.predict(X_train)
        )
    )

    val_f1_depth.append(
        f1_score(
            y_val,
            modelo.predict(X_val)
        )
    )


plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_f1_depth,
    marker="o",
    label="Train"
)

plt.plot(
    depths,
    val_f1_depth,
    marker="o",
    label="Validation"
)

plt.xlabel("Max depth")
plt.ylabel("F1-score")
plt.title("Train vs Validation según profundidad")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 6. MODELO AJUSTADO / REGULARIZADO

modelo_ajustado = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

modelo_ajustado.fit(X_train, y_train)

ajustado = evaluar_modelo(
    modelo_ajustado,
    "MODELO AJUSTADO"
)


# 7. COMPARACIÓN ANTES VS DESPUÉS

modelos = ["Base", "Ajustado"]

validation_f1 = [
    base["val_f1"],
    ajustado["val_f1"]
]

test_f1 = [
    base["test_f1"],
    ajustado["test_f1"]
]


plt.figure(figsize=(7, 5))

x = range(len(modelos))

plt.plot(
    x,
    validation_f1,
    marker="o",
    label="Validation F1"
)

plt.plot(
    x,
    test_f1,
    marker="o",
    label="Test F1"
)

plt.xticks(x, modelos)

plt.ylabel("F1-score")
plt.title("Comparación antes y después del ajuste")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 7.5 GRÁFICA EXTRA: COMPARACIÓN DE F1 POR CONJUNTO

conjuntos = ["Train", "Validation", "Test"]

base_f1 = [
    base["train_f1"],
    base["val_f1"],
    base["test_f1"]
]

ajustado_f1 = [
    ajustado["train_f1"],
    ajustado["val_f1"],
    ajustado["test_f1"]
]

x = range(len(conjuntos))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width / 2 for i in x],
    base_f1,
    width=width,
    label="Base"
)

plt.bar(
    [i + width / 2 for i in x],
    ajustado_f1,
    width=width,
    label="Ajustado"
)

plt.xticks(list(x), conjuntos)
plt.ylabel("F1-score")
plt.title("Comparación de F1-score por conjunto")
plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.show()
# 8. DIAGNÓSTICO AUTOMÁTICO SIMPLE

print("\nDIAGNÓSTICO DEL MODELO BASE")

gap = base["train_f1"] - base["val_f1"]

if base["train_f1"] < 0.50:
    print("Bias / sesgo: ALTO")
elif base["train_f1"] < 0.70:
    print("Bias / sesgo: MEDIO")
else:
    print("Bias / sesgo: BAJO")

if gap > 0.20:
    print("Varianza: ALTA")
elif gap > 0.10:
    print("Varianza: MEDIA")
else:
    print("Varianza: BAJA")

if base["train_f1"] < 0.50 and gap < 0.15:
    print("Nivel de ajuste: UNDERFITTING")
elif gap > 0.20:
    print("Nivel de ajuste: OVERFITTING")
else:
    print("Nivel de ajuste: FIT ADECUADO")
