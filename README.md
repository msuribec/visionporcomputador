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

### Metodología

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
```
## Taller 3 – Sistema de Recuperación Texto-Imagen con CLIP y Caltech-256

Este taller implementa un **sistema de recuperación de imágenes a partir de texto** utilizando el modelo **CLIP (Contrastive Language–Image Pre-training)** de OpenAI, aplicado sobre el dataset **Caltech-256**.  
El objetivo es que, dada una consulta textual (ej. *"guitarra eléctrica"*, *"guantes de boxeo"*), el sistema recupere las imágenes más similares semánticamente dentro del conjunto de datos.

---

### Contenido del Notebook


#### Configuración Inicial
- Se define el modelo CLIP a utilizar y el dispositivo (`CPU`/`GPU`).
- Se establecen parámetros como tamaño de batch y cantidad de imágenes a muestrear.
- Se configura la visualización del top-k de resultados (top 12).

#### Carga del Dataset Caltech-256
- Se descarga el dataset desde el repositorio oficial.
- Se organizan las rutas de las imágenes, nombres de clases y etiquetas.
- Se valida la estructura de directorios y conteo total de imágenes: **~29.780 imágenes en 257 clases**.

#### Muestreo y Verificación del Dataset
- Se selecciona una muestra balanceada de ~5.900 imágenes (256 clases, excluyendo "clutter").
- Se comprueba que los paths sean válidos y se analiza la distribución de clases.

#### Generación de Embeddings con CLIP
- Se generan los embeddings de las imágenes usando la red visual de CLIP.
- Se codifican prompts textuales para establecer un espacio latente común entre texto e imagen.

#### Construcción del Índice de Búsqueda
- Se crea un **índice vectorial con FAISS** para realizar búsquedas rápidas por similitud coseno.
- Se almacena:  
  ✔ embeddings de imágenes  
  ✔ etiquetas de clase  
  ✔ rutas de archivo

#### Recuperación Texto → Imagen
Se implementa el flujo completo:
1. El usuario ingresa una consulta textual.
2. El texto se codifica con CLIP.
3. Se buscan los *k* vectores más similares en el índice FAISS.
4. Se muestran las imágenes recuperadas.

Ejemplos incluidos:
- `"guantes"` → guantes de boxeo, béisbol  
- `"guitarras"` → guitarras acústicas y eléctricas  
- `"animales"` → gorilas, jirafas, ballenas, etc.  
- `"deportes"` → pelotas, raquetas, ropa deportiva

#### Evaluación del Sistema
- Se utilizan captions para evaluar el sistema
- Se calcula precisión top-k verificando si alguna imagen del mismo label aparece en el resultado.
- Resultados muestran que CLIP alinea correctamente conceptos visuales con descripciones textuales.

#### Exportación para Uso en Otros Entornos
Se muestra cómo guardar:
- los embeddings generados,  
- el índice FAISS,  
- y los metadatos del dataset  
para reutilizarlos posteriormente sin recalcular todo.

---

### Requisitos

| Requisito | Versión / Recomendación |
|-----------|---------------------------|
| Python    | 3.8+                     |
| PyTorch   | GPU recomendado          |
| Librerías | clip, faiss, numpy, PIL, matplotlib |

---

## Taller 04: Tareas de Pretexto en Visión por Computador


### Objetivo del Taller

El propósito de este taller es implementar una **tarea pretexto** utilizando un modelo de visión por computador. Para esto se propone:

- Implementar un modelo basado en **CNN** (por ejemplo ResNet18) o **ViT**.
- Diseñar e implementar una **tarea de pretexto**.
- Entrenar el modelo usando un subconjunto de **ImageNet** (Tiny ImageNet en este caso).

---

### Tarea de Pretexto Elegida: Clasificación de Rotaciones

Se plantea como tarea pretexto la **predicción del ángulo de rotación** de una imagen.  
Los ángulos utilizados son:

- **0°**
- **90°**
- **180°**
- **270°**

#### ¿Por qué esta tarea?
Esta tarea obliga al modelo a aprender representaciones útiles sobre el contenido de la imagen (formas, objetos, texturas), sin necesidad de etiquetas reales.

---

### Dataset

- Se utiliza **Tiny ImageNet** como base.
- A cada imagen se le aplica una rotación aleatoria correspondiente a uno de los 4 ángulos.
- Se implementa un `__getitem__` personalizado para devolver:
  - La imagen transformada
  - La etiqueta de rotación (0, 1, 2, 3)

---

### Transformaciones y Preprocesamiento

Se utiliza `torchvision.transforms.Compose` para aplicar:

- Redimensionamiento de imágenes  
- Normalización usando los valores estándar del dataset Tiny ImageNet  
- Rotación según el ángulo correspondiente a la etiqueta de la tarea de pretexto

---

### Modelo Utilizado

Se usa **ResNet18** como backbone del modelo.

#### Modificaciones realizadas:

- Se descongelan las **últimas capas** del modelo.
- Se reemplaza la capa lineal final por una capa adaptada para **4 clases (rotaciones)**.

---

### Entrenamiento

- **Función de pérdida:** CrossEntropyLoss  
- **Optimizador:** Adam / SGD (según implementación en el código)  
- **Tarea de predicción:** Determinar el ángulo de rotación aplicado  
- **Entrenamiento en GPU:** Si está disponible  

---

### Estructura del Notebook

| Sección | Descripción |
|---------|-------------|
| Introducción | Explicación del taller y objetivos |
| Definición de tarea de pretexto | Rotaciones sobre imágenes |
| Dataset y DataLoader | Implementación de rotaciones y carga de datos |
| Modelo | Modificación de ResNet18 para 4 clases |
| Entrenamiento | Ciclo de entrenamiento y validación |
| Resultados | Cálculo de accuracy y análisis |

---

## Taller 05: Detección de Objetos

Este proyecto corresponde al Taller 05 del curso, enfocado en la **detección de objetos utilizando redes neuronales convolucionales (CNN)**.  
El objetivo principal fue entrenar un modelo capaz de predecir **cajas delimitadoras (bounding boxes)** para objetos presentes en imágenes.

---

### Contenido del Notebook

El notebook incluye las siguientes secciones principales:

#### Exploración de datos
- Visualización del formato de las anotaciones del dataset.
- Visualización de imágenes de ejemplo (como la clase *airplanes*) junto con sus cajas delimitadoras reales.

#### Construcción del modelo
Se parte de un modelo **ResNet50 preentrenado en ImageNet**, al cual se le modifica la parte final de clasificación para adaptarlo a detección de objetos.

**Ajustes realizados:**
- Se reemplaza la última capa de clasificación usando `nn.Identity()`.
- Se agrega una red personalizada para predecir coordenadas de bounding boxes:
  - Capa lineal → 256 neuronas (bottleneck) + ReLU + Dropout (50%)
  - Capa lineal → 128 neuronas + ReLU + Dropout (30%)
  - Capa final lineal con **4 salidas** (x_min, y_min, x_max, y_max)
- En el método `forward`, se aplica **normalización con función sigmoide** para restringir salidas al rango [0,1].

#### Evaluación del modelo

**Resultados de prueba:**
- *Loss:* `0.0012`  
- *IOU (Intersection over Union):* `86%`

---

#### Conclusiones

- El modelo logra un buen desempeño 
- La combinación de regularización (Dropout) y ajuste de capas permite evitar sobreajuste y generalizar mejor.
- Con una IOU del 86%, el modelo demuestra ser efectivo prediciendo bounding boxes sobre el dataset utilizado.

---

#### Requisitos y Tecnologías

| Herramienta | Uso |
|-------------|-----|
| Python (PyTorch) | Construcción y entrenamiento del modelo |
| ResNet50 | Modelo base preentrenado |
| Matplotlib | Visualización de imágenes y predicciones |
| Annotations Dataset | Coordenadas reales de bounding boxes |




## Taller 06: Segmentación con GroundingDINO + Segment Anything

Este proyecto implementa un sistema para detectar, y segmentar objetos en imágenes usando los modelos pre entrenados de **GroundingDINO** y **Segment Anything (SAM)**.

### Descripción

El notebook permite:
- Detectar objetos en imágenes a partir de prompts de texto con GroundingDINO.
- Generar máscaras precisas de los objetos detectados con Segment Anything a partir de las bounding boxes generadas en el paso anterior.

### Tecnologías

| Componente | Uso |
|------------|-----|
| Python | Lenguaje principal |
| PyTorch | Framework para deep learning |
| GroundingDINO | Detección basada en texto |
| Segment Anything | Segmentación automática de objetos |
| OpenCV, Matplotlib | Visualización |
| Pandas | Manejo de datos |

### Estructura del Notebook

1. Instalación de librerías y modelos.
2. Configuración de GroundingDINO y SAM.
3. Carga de imágenes y creación de DataFrame.
4. Detección y segmentación.
5. Visualización de resultados.
6. Exportación de datos y conteos.

### Cómo usar

```bash
pip install groundingdino-py segment-anything-py supervision opencv-python pandas tqdm
```

Luego:
1. Carga el notebook.
2. Coloca tus imágenes en la carpeta asignada.
3. Define las clases a buscar:
   ```python
   class_list = ["person", "dog", "car"]
   ```
4. Ejecuta el notebook
5. Visualiza resultados y exporta si lo deseas.

### Resultados

Genera:
- Imágenes con bounding boxes, máscaras y etiquetas.
- DataFrame con clase, puntaje, coordenadas, ruta de imagen.
