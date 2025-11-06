# Taller 04: Detección de Objetos

Este proyecto corresponde al Taller 04 del curso, enfocado en la **detección de objetos utilizando redes neuronales convolucionales (CNN)**.  
El objetivo principal fue entrenar un modelo capaz de predecir **cajas delimitadoras (bounding boxes)** para objetos presentes en imágenes.

---

## Autores

- María Sofía Uribe  
- Javier Daza Olivella

---

## Contenido del Notebook

El notebook incluye las siguientes secciones principales:

### 📁 Exploración de datos
- Visualización del formato de las anotaciones del dataset.
- Visualización de imágenes de ejemplo (como la clase *airplanes*) junto con sus cajas delimitadoras reales.

### Construcción del modelo
Se parte de un modelo **ResNet50 preentrenado en ImageNet**, al cual se le modifica la parte final de clasificación para adaptarlo a detección de objetos.

**Ajustes realizados:**
- Se reemplaza la última capa de clasificación usando `nn.Identity()`.
- Se agrega una red personalizada para predecir coordenadas de bounding boxes:
  - Capa lineal → 256 neuronas (bottleneck) + ReLU + Dropout (50%)
  - Capa lineal → 128 neuronas + ReLU + Dropout (30%)
  - Capa final lineal con **4 salidas** (x_min, y_min, x_max, y_max)
- En el método `forward`, se aplica **normalización con función sigmoide** para restringir salidas al rango [0,1].

### Evaluación del modelo

**Resultados de prueba:**
- 📉 *Loss:* `0.0012`  
- 🎯 *IOU (Intersection over Union):* `86%`

---

## Conclusiones

- El modelo logra un desempeño notable para una arquitectura basada en transfer learning.
- La combinación de regularización (Dropout) y ajuste de capas permite evitar sobreajuste y generalizar mejor.
- Con una IOU del 86%, el modelo demuestra ser efectivo prediciendo bounding boxes sobre el dataset utilizado.

---

## Requisitos y Tecnologías

| Herramienta | Uso |
|-------------|-----|
| Python (PyTorch) | Construcción y entrenamiento del modelo |
| ResNet50 | Modelo base preentrenado |
| Matplotlib | Visualización de imágenes y predicciones |
| Annotations Dataset | Coordenadas reales de bounding boxes |


