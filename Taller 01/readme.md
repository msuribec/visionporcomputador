
Autores
- María Sofía Uribe
- Javier Javier Daza

# Taller 01 — Clasificación de hojas de mango (HOG + clasificadores tradicionales)


Resumen
- Propósito: Extraer descriptores HOG de imágenes de hojas de mango y comparar clasificadores tradicionales (SVM con RBF, SVM lineal, MLP, KNN, Random Forest).
- Resultado principal mostrado: MLP entrenado con features HOG alcanza ≈ 79% de exactitud en el conjunto de prueba.

Estructura del notebook
1. Funciones/clases clave:
   - `load_image_paths_and_labels(DATA_ROOT)`: recorre carpetas por clase y devuelve rutas, etiquetas y nombres de clase.
   - `show_examples(...)` / `show_examples_test(...)`: muestra ejemplos por clase y predicciones.
   - `HOGParams` (dataclass): parámetros para extracción HOG.
   - `MangoHOGFeatureExtractor`: preprocesado (gris, resize) y extracción de HOG.
   - `MangoHOGClassifier`: wrapper para entrenar/evaluar distintos clasificadores dentro de un `Pipeline` (opcionalmente con `StandardScaler`).
   - `plot_cm(cm, title)`: dibuja matriz de confusión.
4. Flujo experimental:
   - Carga de rutas y etiquetas desde la estructura de carpetas.
   - División train/test (80/20, estratificada).
   - Extracción masiva de features HOG para train y test.
   - Validación cruzada (cv=5) para SVM-RBF y MLP.
   - Entrenamiento final del mejor modelo (MLP), guardado (`.save`) y carga (`.load`).
   - Evaluación en test: accuracy, F1 macro, reporte de clasificación y matriz de confusión.
   - Visualización de predicciones sobre imágenes.

Requisitos (librerías principales)
- Python 3.8+
- numpy, pandas
- scikit-image
- scikit-learn
- pillow (PIL)
- matplotlib
- joblib
- tqdm

Ejecución
- Abrir el notebook en Jupyter:
  $ jupyter lab
  o
  $ jupyter notebook
- Ajustar la variable `root` (DATA_ROOT) al directorio local que contiene subcarpetas por clase con las imágenes (ej. `/ruta/al/dataset`).
- Ejecutar las celdas en orden.