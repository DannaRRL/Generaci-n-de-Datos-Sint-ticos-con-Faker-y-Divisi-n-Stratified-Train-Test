
from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# PRACTICA 4: DATOS SINTETICOS Y
# PARTICION ESTRATIFICADA
# ==========================================

# Configuracion de Faker para Mexico
fake = Faker("es_MX")

# Semillas para reproducibilidad
Faker.seed(42)
np.random.seed(42)

# ==========================================
# 1. GENERACION DE 1000 REGISTROS
# ==========================================

registros = []

for i in range(1000):
    registros.append({
        "id_cliente": i + 1,
        "nombre": fake.name(),
        "email": fake.email(),

        # Campos adicionales solicitados
        "direccion": fake.address().replace("\n", ", "),
        "telefono": fake.phone_number(),

        "estado_civil": np.random.choice(
            ["Soltero", "Casado", "Divorciado"],
            p=[0.5, 0.4, 0.1]
        ),

        "ingreso_anual": round(
            np.random.uniform(100000, 1000000), 2
        ),

        "churn": np.random.choice(
            [0, 1],
            p=[0.8, 0.2]
        )
    })

df_sintetico = pd.DataFrame(registros)

print("\n=== DATASET SINTETICO ORIGINAL ===")
print(df_sintetico.head(10).to_string(index=False))

print("\nTotal de registros:", len(df_sintetico))

print("\n=== INFORMACION DEL DATASET ===")
print(df_sintetico.dtypes)

# ==========================================
# 2. DISTRIBUCION ORIGINAL DE CHURN
# ==========================================

print("\n=== DISTRIBUCION ORIGINAL DE CHURN ===")

print(df_sintetico["churn"].value_counts().sort_index())

print("\nProporciones originales:")
print(
    df_sintetico["churn"]
    .value_counts(normalize=True)
    .sort_index()
)

# ==========================================
# 3. SEPARACION DE VARIABLES X E Y
# ==========================================

# Excluir identificadores y datos personales
X = df_sintetico.drop(
    columns=[
        "id_cliente",
        "nombre",
        "email",
        "direccion",
        "telefono",
        "churn"
    ]
)

# Variable objetivo
y = df_sintetico["churn"]

print("\n=== VARIABLES PREDICTORAS X ===")
print(X.head())

print("\n=== VARIABLE OBJETIVO Y ===")
print(y.head(10))

# ==========================================
# 4. PARTICION ESTRATIFICADA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== PARTICION TRAIN-TEST ===")

print("Dimensiones Train:", X_train.shape)
print("Dimensiones Test:", X_test.shape)

print("Registros Train:", len(y_train))
print("Registros Test:", len(y_test))

# ==========================================
# 5. CODIFICACION DE VARIABLES CATEGORICAS
# ==========================================

# Crear variables numericas con One-Hot Encoding.
# Se ajusta usando solo el conjunto de entrenamiento.
X_train_encoded = pd.get_dummies(
    X_train,
    columns=["estado_civil"],
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=["estado_civil"],
    dtype=int
)

# Asegurar que ambas particiones tengan
# las mismas columnas, en el mismo orden
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

print("\n=== TRAIN CODIFICADO ===")
print(X_train_encoded.head())

print("\n=== TEST CODIFICADO ===")
print(X_test_encoded.head())

# ==========================================
# 6. COMPARACION DE PROPORCIONES
# ==========================================

comparacion = pd.DataFrame({
    "Dataset completo": (
        y.value_counts(normalize=True).sort_index()
    ),
    "Train": (
        y_train.value_counts(normalize=True).sort_index()
    ),
    "Test": (
        y_test.value_counts(normalize=True).sort_index()
    )
})

comparacion.index.name = "churn"

print("\n=== COMPARACION DE CHURN ===")
print((comparacion * 100).round(2).to_string())

print("\n=== PORCENTAJE DE ABANDONO ===")

print(f"Dataset completo: {y.mean() * 100:.2f}%")
print(f"Train: {y_train.mean() * 100:.2f}%")
print(f"Test: {y_test.mean() * 100:.2f}%")

# ==========================================
# 7. VALIDACION
# ==========================================

print("\n=== VALIDACION FINAL ===")

assert len(df_sintetico) == 1000
assert len(X_train) == 800
assert len(X_test) == 200
assert len(y_train) == 800
assert len(y_test) == 200

assert X_train_encoded.columns.equals(
    X_test_encoded.columns
)

print("Dataset generado correctamente.")
print("Particion estratificada completada.")
print("Variables categoricas codificadas.")
print("Validacion correcta.")

# ==========================================
# 8. EXPORTACION DE RESULTADOS
# ==========================================

df_sintetico.to_csv(
    "dataset_sintetico.csv",
    index=False
)

train_final = X_train_encoded.copy()
train_final["churn"] = y_train

test_final = X_test_encoded.copy()
test_final["churn"] = y_test

train_final.to_csv(
    "train_dataset.csv",
    index=False
)

test_final.to_csv(
    "test_dataset.csv",
    index=False
)

comparacion.to_csv(
    "comparacion_churn.csv"
)

print("\n=== ARCHIVOS GENERADOS ===")
print("dataset_sintetico.csv")
print("train_dataset.csv")
print("test_dataset.csv")
print("comparacion_churn.csv")

print("\nPRACTICA 4 FINALIZADA CORRECTAMENTE")
