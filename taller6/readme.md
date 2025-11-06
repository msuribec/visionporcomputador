
# Taller 6: Segmentación con GroundingDINO + Segment Anything

Este proyecto implementa un sistema para detectar, y segmentar objetos en imágenes usando los modelos pre entrenados de **GroundingDINO** y **Segment Anything (SAM)**.

## Descripción

El notebook permite:
- Detectar objetos en imágenes a partir de prompts de texto con GroundingDINO.
- Generar máscaras precisas de los objetos detectados con Segment Anything a partir de las bounding boxes generadas en el paso anterior.

## Tecnologías

| Componente | Uso |
|------------|-----|
| Python | Lenguaje principal |
| PyTorch | Framework para deep learning |
| GroundingDINO | Detección basada en texto |
| Segment Anything | Segmentación automática de objetos |
| OpenCV, Matplotlib | Visualización |
| Pandas | Manejo de datos |

## Estructura del Notebook

1. Instalación de librerías y modelos.
2. Configuración de GroundingDINO y SAM.
3. Carga de imágenes y creación de DataFrame.
4. Detección y segmentación.
5. Visualización de resultados.
6. Exportación de datos y conteos.

## Cómo usar

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

## Resultados

Genera:
- Imágenes con bounding boxes, máscaras y etiquetas.
- DataFrame con clase, puntaje, coordenadas, ruta de imagen.
