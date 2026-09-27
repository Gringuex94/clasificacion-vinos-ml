# Clasificación de Vinos con Machine Learning + API

📊 Ver infografía interactiva (https://gringuex94.github.io/clasificacion-vinos-ml/vinos-infografia.html)
Proyecto final de la diplomatura en Python (orientación a Inteligencia Artificial). Clasifica muestras de vino en 3 categorías a partir de sus características químicas, y expone el modelo entrenado como una API REST.

## ¿Qué hace?

1. **Entrenamiento**: usa el dataset `load_wine` de scikit-learn (178 muestras, 13 características químicas, 3 clases). Se prueban tres modelos — Regresión Logística, SVM y Árbol de Decisión — y se comparan con `classification_report`.
2. **Selección de modelo**: el SVM obtuvo el mejor desempeño (accuracy = 1.00 en el conjunto de prueba), por lo que es el que se guarda y se sirve en la API.
3. **API con FastAPI**: expone un endpoint `POST /predict` que recibe las 13 características químicas de una muestra y devuelve la clase predicha junto con las probabilidades de cada clase.

## Resultados de la comparación de modelos

| Modelo | Accuracy |
|---|---|
| Regresión Logística | 0.96 |
| **SVM** | **1.00** |
| Árbol de Decisión | 0.93 |

## Cómo correrlo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar y guardar el modelo (genera modelo_vino.pkl y scaler_vino.pkl)
python AutomatizacionClasificacionVinos_API.py

# 3. Levantar la API
uvicorn AutomatizacionClasificacionVinos_API:app --reload

# 4. Probarla en el navegador
# http://127.0.0.1:8000/docs
```

## Ejemplo de uso del endpoint

**Request:**
```json
POST /predict
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050
}
```

**Response:**
```json
{
  "clase_predicha": 0,
  "nombre_clase": "class_0",
  "probabilidades": {
    "class_0": 0.963,
    "class_1": 0.0241,
    "class_2": 0.0129
  }
}
```

## Nota sobre el dataset

Las clases (`class_0`, `class_1`, `class_2`) son etiquetas genéricas del dataset original de scikit-learn — no corresponden a variedades comerciales de vino (Malbec, Cabernet, etc.). El dataset proviene de un análisis químico de tres cultivares de una misma región de Italia.

## Próximos pasos

- Agregar matriz de confusión y validación cruzada para una evaluación más robusta
- Ajustar hiperparámetros del SVM con `GridSearchCV`
- Migrar a un dataset con variedades y regiones reales para un análisis con mayor contexto de negocio

## Stack

- Python 3
- scikit-learn
- pandas
- FastAPI + Pydantic
- uvicorn
- joblib
