# Taller 04: Tareas de Pretexto en Visión por Computador

**Autores:**
- María Sofía Uribe  
- Javier Daza Olivella  

---

## Objetivo del Taller

El propósito de este taller es implementar una **tarea pretexto** utilizando un modelo de visión por computador. Para esto se propone:

- Implementar un modelo basado en **CNN** (por ejemplo ResNet18) o **ViT**.
- Diseñar e implementar una **tarea de pretexto**.
- Entrenar el modelo usando un subconjunto de **ImageNet** (Tiny ImageNet en este caso).

---

## Tarea de Pretexto Elegida: Clasificación de Rotaciones

Se plantea como tarea pretexto la **predicción del ángulo de rotación** de una imagen.  
Los ángulos utilizados son:

- **0°**
- **90°**
- **180°**
- **270°**

### ¿Por qué esta tarea?
Esta tarea obliga al modelo a aprender representaciones útiles sobre el contenido de la imagen (formas, objetos, texturas), sin necesidad de etiquetas reales.

---

## Dataset

- Se utiliza **Tiny ImageNet** como base.
- A cada imagen se le aplica una rotación aleatoria correspondiente a uno de los 4 ángulos.
- Se implementa un `__getitem__` personalizado para devolver:
  - La imagen transformada
  - La etiqueta de rotación (0, 1, 2, 3)

---

## Transformaciones y Preprocesamiento

Se utiliza `torchvision.transforms.Compose` para aplicar:

- Redimensionamiento de imágenes  
- Normalización usando los valores estándar del dataset Tiny ImageNet  
- Rotación según el ángulo correspondiente a la etiqueta de la tarea de pretexto

---

## Modelo Utilizado

Se usa **ResNet18** como backbone del modelo.

### Modificaciones realizadas:

- Se descongelan las **últimas capas** del modelo.
- Se reemplaza la capa lineal final por una capa adaptada para **4 clases (rotaciones)**.

---

## Entrenamiento

- **Función de pérdida:** CrossEntropyLoss  
- **Optimizador:** Adam / SGD (según implementación en el código)  
- **Tarea de predicción:** Determinar el ángulo de rotación aplicado  
- **Entrenamiento en GPU:** Si está disponible  

---

## Estructura del Notebook

| Sección | Descripción |
|---------|-------------|
| Introducción | Explicación del taller y objetivos |
| Definición de tarea de pretexto | Rotaciones sobre imágenes |
| Dataset y DataLoader | Implementación de rotaciones y carga de datos |
| Modelo | Modificación de ResNet18 para 4 clases |
| Entrenamiento | Ciclo de entrenamiento y validación |
| Resultados | Cálculo de accuracy y análisis |

---