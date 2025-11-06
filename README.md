# Talleres visión por computador


Este repositorio constituye un conjunto de cuadernos (notebooks) y materiales de apoyo para el curso de visión por computador. Está organizado de modo secuencial (Taller 01, Taller 02, etc.)


Autores
- María Sofía Uribe
- Javier Javier Daza

## Taller 01 — Clasificación de hojas de mango (HOG + clasificadores tradicionales)

Archivos:
- `Taller 01/taller1.ipynb` — notebook principal con todo el código y análisis.

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



## Taller 2 - Clasificación de Imágenes con Features Pre-entrenadas y AutoGluon

Este notebook implementa un sistema de clasificación de imágenes usando características extraídas de modelos pre-entrenados (ResNet50 y MobileNetV2) y clasificadores optimizados con AutoGluon.

### Dataset
Se utiliza el dataset LabelMe-12-50k que contiene imágenes clasificadas en 13 categorías:
- person
- car
- building
- window
- tree 
- sign
- door
- bookshelf
- chair
- table
- keyboard
- head
- clutter (regiones aleatorias sin objeto centrado)

###Metodología

#### 1. Extracción de Características
- Se utilizan dos modelos pre-entrenados como extractores de features:
  - ResNet50
  - MobileNetV2
- Las imágenes se redimensionan a 224x224 píxeles
- Se aplica normalización usando la media y desviación estándar de ImageNet
- Las características extraídas se reducen a 256 dimensiones usando TruncatedSVD

#### 2. Entrenamiento con AutoGluon
- Se entrenan modelos tabulares usando las características extraídas
- Se exploran diferentes algoritmos:
  - LightGBM (gradient boosting)
  - Modelos lineales
  - Ensambles ponderados
- Búsqueda de hiperparámetros con:
  - 5 trials
  - Scheduler local
  - Early stopping después de 25 rondas sin mejora

### Resultados Principales

#### ResNet50
- Accuracy global: 86%
- Mejor desempeño en clases mayoritarias:
  - person (F1: 0.91)
  - car (F1: 0.91)
  - clutter (F1: 0.90)
- Desempeño moderado en clases minoritarias:
  - keyboard (F1: 0.92)
  - bookshelf (F1: 0.84)
  - table (F1: 0.24)

#### MobileNetV2
- Accuracy global: 84%
- Buen desempeño en clases mayoritarias:
  - car (F1: 0.90)
  - keyboard (F1: 0.90)
  - clutter (F1: 0.88)
- Desempeño inferior en clases minoritarias:
  - table (F1: 0.27)
  - door (F1: 0.51)
  - building (F1: 0.55)

### Conclusiones
- ResNet50 ofrece mejor capacidad de representación, alcanzando métricas superiores
- MobileNetV2 proporciona un buen balance entre precisión y eficiencia computacional
- Las clases con más ejemplos y patrones visuales distintivos son mejor clasificadas
- El desbalance de clases afecta el desempeño en categorías minoritarias
- Los ensambles mejoran ligeramente el rendimiento sobre modelos individuales

### Requisitos
```python
pip install --no-cache-dir -U "numpy<2.0" "pandas<2.2" "autogluon==1.3.1"

