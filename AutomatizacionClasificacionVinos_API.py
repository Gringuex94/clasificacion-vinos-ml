#   PROYECTO FINAL DE APRENDIZAJE AUTOMÁTICO VINOS - CONTINUACIÓN   #
# ESTA SEGUNDA ETAPA TOMA EL MODELO ENTRENADO EN LA PRIMERA PARTE
# (SVM, QUE FUE EL DE MEJOR RENDIMIENTO) Y LO PONE A DISPOSICIÓN
# COMO UNA API REST CON FASTAPI, PARA QUE PUEDA SER CONSUMIDO
# DESDE CUALQUIER CLIENTE (POSTMAN, UN FRONTEND, OTRO SCRIPT, ETC.)
# -*- coding: utf-8 -*-

"""
==========================================================================
PARTE 1 - ENTRENAMIENTO Y GUARDADO DEL MODELO
==========================================================================
Esta parte es prácticamente la misma que en el notebook original. La
diferencia clave es que, al final, en vez de solo imprimir métricas por
pantalla, GUARDAMOS el modelo entrenado y el scaler en disco usando
joblib. Esto es necesario porque una API no puede reentrenar el modelo
en cada petición: lo entrena una vez (o cada tanto) y después solo
lo carga para predecir.

Ejecutar este bloque genera dos archivos:
  - modelo_vino.pkl   -> el SVM ya entrenado
  - scaler_vino.pkl   -> el StandardScaler ya ajustado (fit) a los datos
==========================================================================
"""

import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Cargar datos (igual que en la primera parte)
wine_data = load_wine()
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wine_df["target"] = wine_data.target

# Separar features y etiqueta
X = wine_df[wine_data.feature_names].copy()
y = wine_df["target"].copy()

# Escalado
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X.values)

# Split entrenamiento/prueba (mismos parámetros que usaste antes)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.7, random_state=25
)

# Entrenamos SOLO el SVM, porque fue el que dio accuracy = 1.00
# en la comparación de los tres modelos de la primera parte.
svm = SVC(probability=True)  # probability=True permite devolver
                              # también la confianza de la predicción
svm.fit(X_train, y_train)

# Verificación rápida de que el modelo sigue funcionando igual que antes
svm_preds = svm.predict(X_test)
print("Verificación del modelo antes de guardarlo:\n")
print(classification_report(y_test, svm_preds))

# Guardamos modelo y scaler en disco. Esto es lo que la API va a leer.
joblib.dump(svm, "modelo_vino.pkl")
joblib.dump(scaler, "scaler_vino.pkl")
print("Modelo y scaler guardados como 'modelo_vino.pkl' y 'scaler_vino.pkl'")


"""
==========================================================================
PARTE 2 - API CON FASTAPI
==========================================================================
Acá es donde el proyecto deja de ser "solo un notebook" y pasa a ser
algo que se puede usar en la práctica. Levantamos un servidor con
FastAPI que expone un endpoint POST /predict: recibe las 13
características químicas de una muestra de vino y devuelve a qué
clase pertenece (0, 1 o 2), junto con la probabilidad de cada clase.

Para correr esta parte (desde la terminal, en la carpeta del archivo):
    uvicorn AutomatizacionClasificacionVinos_API:app --reload

Y probarlo entrando a:
    http://127.0.0.1:8000/docs
(FastAPI genera automáticamente una interfaz interactiva ahí, no hace
falta Postman para probarlo la primera vez)
==========================================================================
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="API Clasificación de Vinos")

# Cargamos el modelo y el scaler UNA sola vez, al arrancar la API,
# no en cada petición (por eso están fuera de la función).
modelo = joblib.load("modelo_vino.pkl")
scaler_cargado = joblib.load("scaler_vino.pkl")

# Nombres de las clases, para devolver algo legible en vez de solo 0/1/2
nombres_clases = {
    0: wine_data.target_names[0],
    1: wine_data.target_names[1],
    2: wine_data.target_names[2],
}


# Con Pydantic definimos la "forma" que debe tener el JSON que llega
# en la petición. FastAPI valida automáticamente los tipos de datos:
# si falta un campo o viene con el tipo equivocado, devuelve un error
# claro sin que nosotros tengamos que validarlo a mano.
class MuestraVino(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


@app.get("/")
def inicio():
    """Endpoint simple para confirmar que la API está corriendo."""
    return {"mensaje": "API de clasificación de vinos activa"}


@app.post("/predict")
def predecir(muestra: MuestraVino):
    """
    Recibe las características químicas de una muestra de vino
    y devuelve la clase predicha por el SVM entrenado.
    """
    # Convertimos los datos recibidos en el mismo orden de columnas
    # que se usó para entrenar el modelo (wine_data.feature_names)
    datos_entrada = np.array([[
        muestra.alcohol,
        muestra.malic_acid,
        muestra.ash,
        muestra.alcalinity_of_ash,
        muestra.magnesium,
        muestra.total_phenols,
        muestra.flavanoids,
        muestra.nonflavanoid_phenols,
        muestra.proanthocyanins,
        muestra.color_intensity,
        muestra.hue,
        muestra.od280_od315_of_diluted_wines,
        muestra.proline,
    ]])

    # Aplicamos el MISMO escalado que se usó en el entrenamiento.
    # Esto es crítico: si se predice con datos sin escalar, el
    # modelo va a dar resultados incorrectos.
    datos_escalados = scaler_cargado.transform(datos_entrada)

    # Predicción de clase y de probabilidades por clase
    prediccion = modelo.predict(datos_escalados)[0]
    probabilidades = modelo.predict_proba(datos_escalados)[0]

    return {
        "clase_predicha": int(prediccion),
        "nombre_clase": nombres_clases[int(prediccion)],
        "probabilidades": {
            nombres_clases[i]: round(float(p), 4)
            for i, p in enumerate(probabilidades)
        },
    }
