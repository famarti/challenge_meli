# Predicción de Estado de Producto en MercadoLibre

## Descripción
Clasificación de productos en **MercadoLibre** como **nuevos** o **usados**, aplicando **EDA**, **selección de características** y **modelado con LightGBM** optimizado con **Optuna**.

## Flujo del Proyecto
1. **Carga de Datos** desde `MLA_100k_checked_v3.jsonlines`.
2. **Preprocesamiento**: Transformaciones, codificación y generación de variables.
3. **Selección de Características** a partir de análisis EDA previo, revisado con **RFE** basado en `RandomForestClassifier`.
4. **Entrenamiento y Optimización** con **LightGBM** + **Optuna**.
5. **Evaluación**: AUC Final **0.8834** (superando el umbral de 0.86).

## Tecnologías
- **Python, Pandas, NumPy, Scikit-learn** (Procesamiento)
- **LightGBM, Optuna** (Modelado y Optimización)
- **Logging** (Seguimiento del pipeline)

## Autor
📌 **Facundo Javier Martinez**
